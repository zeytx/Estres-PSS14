import logging
import numpy as np
import openai
import sqlite3
import pandas as pd
import json
from typing import Dict, List, Optional
import os
from datetime import datetime
import re
import time
from pss_scoring import INVERTED_INDICES, DIRECT_INDICES

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False

class GPTStressAnalyzer:
    def __init__(self, api_key: str, stress_predictor=None):
        """Inicializa el analizador con la API key de OpenAI."""
        openai.api_key = api_key
        self.db_path = '../datos/pss_database.db'
        self.last_analysis_cache = {}  # Cache para evitar llamadas redundantes a GPT
        self.timeout = 30  # Timeout en segundos para las llamadas a la API
        self.stress_predictor = stress_predictor  # Reusar instancia global
        self._initialize_db()


    def _initialize_db(self):
        """Crea las tablas necesarias si no existen."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Tabla para análisis GPT
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS gpt_analyses (
                        test_id INTEGER PRIMARY KEY,
                        insight TEXT,
                        recommendations TEXT,
                        patterns TEXT,
                        professional_advice TEXT,
                        ml_comparison TEXT,
                        demographic_comparison TEXT,    
                        timestamp TEXT
                    )
                ''')

                # MIGRAR datos existentes si la columna no existe
                try:
                    cursor.execute("SELECT demographic_comparison FROM gpt_analyses LIMIT 1")
                except sqlite3.OperationalError:
                    cursor.execute("ALTER TABLE gpt_analyses ADD COLUMN demographic_comparison TEXT DEFAULT 'Análisis demográfico no disponible para registros anteriores'")
                    print("Columna demographic_comparison añadida a registros existentes")

                conn.commit()
            print("Tablas de base de datos inicializadas correctamente")
            
        except Exception as e:
            print(f"Error inicializando base de datos: {str(e)}")

    def _get_test_data(self, test_id: int) -> Optional[Dict]:
        """Obtiene los datos de un test específico desde SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Datos básicos del test
            cursor.execute('''
                SELECT age, profession, total_score, stress_level, timestamp, free_text
                FROM tests WHERE test_id = ?
            ''', (test_id,))
            test_row = cursor.fetchone()

            if not test_row:
                return None

            # Respuestas individuales
            cursor.execute('''
                SELECT question_number, original_value, processed_value 
                FROM responses WHERE test_id = ? ORDER BY question_number
            ''', (test_id,))
            responses = cursor.fetchall()

        return {
            "test_id": test_id,
            "age": test_row[0],
            "profession": test_row[1],
            "score": test_row[2],
            "stress_level": test_row[3],
            "timestamp": test_row[4],
            "free_text": test_row[5],
            "responses": [r[1] for r in responses],  # Valores originales
            "processed_responses": [r[2] for r in responses]  # Valores procesados
        }

    def _get_historical_data(self, limit: int = 100) -> List[Dict]:
        """Obtiene datos históricos para análisis comparativo."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(f'''
                SELECT age, profession, total_score, stress_level 
                FROM tests ORDER BY timestamp DESC LIMIT {limit}
            ''', conn)
        return df.to_dict('records')

    def _generate_gpt_prompt(self, test_data: Dict, historical_data: List[Dict]) -> str:
        """Genera el prompt para GPT basado en los datos."""
        
        # Comparación score real vs predicho
        ml_score = test_data.get('ml_prediction', {}).get('predicted_score', None)
        real_score = test_data['score']
        score_diff = "N/A"
        
        if ml_score is not None and ml_score != 'N/A':
            score_diff = round(float(real_score) - float(ml_score), 1)
            if score_diff > 0:
                score_comparison = f"El score real es {abs(score_diff)} puntos MAYOR que la predicción"
            elif score_diff < 0:
                score_comparison = f"El score real es {abs(score_diff)} puntos MENOR que la predicción"
            else:
                score_comparison = "El score real y la predicción coinciden exactamente"
            
            # Evaluar significancia de la diferencia
            if abs(score_diff) > 5:
                score_comparison += " (diferencia significativa)"
        else:
            score_comparison = "No hay datos suficientes para comparar con ML"


        # 2. Contexto demográfico enriquecido
        demographic_insight = self._generate_demographic_insight(test_data)
        
        # 3. Formato de respuestas mejorado
        responses_formatted = self._format_responses(test_data['responses'])

        # 4. Análisis de respuestas críticas
        critical_stressors = self._identify_critical_stressors(test_data['responses'])
        critical_context = "\n".join([f"- {s}" for s in critical_stressors]) if critical_stressors else "Sin respuestas críticas extremas."

        # 4b. Pre-calcular señales de alarma clínica
        alarm_signals = self._detect_alarm_signals(test_data['responses'])
        alarm_context = "\n".join([f"- ⚠️ {a}" for a in alarm_signals]) if alarm_signals else "Sin señales de alarma detectadas."

        # 5. Contexto emocional del usuario (texto libre)
        free_text = test_data.get('free_text', '')
        emotional_context = ""
        if free_text and len(free_text.strip()) > 0:
            emotional_context = f"""
        **Contexto Emocional Reportado por el Usuario:**
        "{free_text.strip()}"
        - Analiza el sentimiento y las emociones expresadas en este texto.
        - Evalúa la INTENSIDAD emocional (leve, moderada, severa).
        - Integra este contexto emocional con las respuestas del PSS-14.
        - Busca coherencia o contradicciones entre lo que expresa y sus respuestas numéricas.
        - Si detectas señales de malestar severo (ideación de desesperanza, aislamiento, agotamiento extremo), menciónalo explícitamente.
        """
        else:
            emotional_context = "**Contexto Emocional:** No proporcionado por el usuario."

        # 6. Calcular subtotales dimensionales para facilitar el análisis
        responses = test_data['responses']
        helplessness_items = [responses[i] for i in [0, 1, 2, 7, 10, 11, 13] if i < len(responses)]
        efficacy_items = [responses[i] for i in [3, 4, 5, 6, 8, 9, 12] if i < len(responses)]
        helplessness_total = sum(helplessness_items)
        efficacy_total = sum(efficacy_items)
        # Los ítems de autoeficacia se invierten: valor alto original = bajo estrés
        efficacy_stress_contribution = sum(4 - v for v in efficacy_items)

        prompt = f"""
        ## Datos del Paciente para Análisis PSS-14
        - Edad: {test_data['age']} años ({self._get_life_stage(test_data['age'])})
        - Profesión: {test_data['profession']} ({self._get_profession_category(test_data['profession'])})
        - Puntuación real: {test_data['score']}/56 ({test_data['stress_level']})
        - Predicción ML: {test_data.get('ml_prediction', {}).get('predicted_stress_level', 'N/A')}
        - Confianza ML: {test_data.get('ml_prediction', {}).get('confidence_interpretation', 'N/A')}
        - Comparación: {score_comparison}

        **Subtotales Dimensionales (pre-calculados para tu análisis):**
        - Indefensión (Q1+Q2+Q3+Q8+Q11+Q12+Q14): {helplessness_total}/28
        - Autoeficacia invertida (contribución al estrés): {efficacy_stress_contribution}/28
        - Dimensión dominante: {"Indefensión" if helplessness_total > efficacy_stress_contribution else "Déficit de autoeficacia" if efficacy_stress_contribution > helplessness_total else "Equilibradas"}

        **Contextualización Demográfica:**
        {demographic_insight}

        **Señales de Alarma Clínica Detectadas:**
        {alarm_context}

        **Factores Críticos (Respuestas indicativas de estrés):**
        {critical_context}

        {emotional_context}

        **Respuestas del Test (VALORES ORIGINALES antes de inversión):**
        {responses_formatted}

        **Tareas — genera el JSON con estos campos:**
        
        1. **insight** (120-180 palabras): Integra ambas dimensiones, menciona señales de alarma si las hay, conecta con la profesión.
        
        2. **patterns** (3-5): Cada patrón DEBE citar Qn=valor y conectar con "{test_data['profession']}".
        
        3. **recommendations** (3, cada una ≥15 palabras): Cada una aborda un factor crítico específico, adaptada a "{test_data['profession']}" y {self._get_life_stage(test_data['age'])}.
        
        4. **professional_advice**: Consejo táctico concreto para un/a {test_data['profession']} que pueda implementar mañana.
        
        5. **ml_comparison**: Análisis cualitativo de la discrepancia (sin revelar score ML exacto).
        
        6. **demographic_comparison**: Contextualización usando comparativas relativas.
        
        7. **emotional_analysis**: {"Analiza el texto libre proporcionado, evaluando intensidad emocional y coherencia con las respuestas numéricas." if free_text and len(free_text.strip()) > 0 else "'No proporcionado'"}

        NO incluyas texto antes o después del JSON. NO uses comillas simples.
        """

        return prompt
    
    def analyze_with_gpt(self, test_data: dict) -> Dict:
        """Versión actualizada para OpenAI API v1.0+ con validación estricta"""
        try:
            # Validación de datos de entrada - MANTENER TIPOS ESTRICTOS
            required_keys = {
                'test_id': int,           # Solo int - coincide con BD
                'age': int,               # Solo int - coincide con BD  
                'profession': str,        # Solo str - coincide con BD
                'score': (int, float),    # int o float - score puede calcularse como float
                'stress_level': str,      # Solo str - coincide con BD
                'responses': list         # Solo list - coincide con estructura esperada
            }
            
            for key, expected_types in required_keys.items():
                if key not in test_data:
                    raise ValueError(f"Falta clave requerida: {key}")
                
                # Manejo de múltiples tipos solo donde sea necesario (score)
                if isinstance(expected_types, tuple):
                    if not any(isinstance(test_data[key], t) for t in expected_types):
                        raise ValueError(f"Tipo incorrecto para {key}. Esperado: {[t.__name__ for t in expected_types]}")
                else:
                    if not isinstance(test_data[key], expected_types):
                        raise ValueError(f"Tipo incorrecto para {key}. Esperado: {expected_types.__name__}")

            if len(test_data['responses']) != 14:
                raise ValueError("Se requieren exactamente 14 respuestas")

            # Obtener contexto
            historical_data = self._get_historical_data()
            prompt = self._generate_gpt_prompt(test_data, historical_data)

            # Versión actualizada para OpenAI v1.0+
            client = openai.OpenAI(api_key=openai.api_key)
            
            # Llamada a API con configuración clínica optimizada
            response = self._call_gpt_api(client, prompt, test_data)

            # Validar respuesta
            if not response.choices:
                raise ValueError("No se recibieron opciones en la respuesta")

            # Extraer contenido y usar el parser robusto
            content = response.choices[0].message.content.strip()
            
            try:
                # Usar el método _parse_response mejorado
                analysis = self._parse_response(content)
            except ValueError as e:
                self._log_error(f"Error al parsear respuesta JSON: {str(e)}", content)
                # Intento final de reparación automática
                try:
                    content_repaired = (content.replace("'", '"')  # Corrige comillas simples
                                      .replace("True", "true")     # Corrige booleanos
                                      .replace("False", "false"))
                    analysis = self._parse_response(content_repaired)
                except ValueError:
                    raise ValueError(f"No se pudo parsear la respuesta después de múltiples intentos: {str(e)}")

            # Enriquecimiento de datos de salida
            return {
                **test_data,
                **analysis,
                "model_used": response.model,
                "api_version": "v1.0+",
                "timestamp": datetime.now().isoformat()
            }

        except openai.APIConnectionError as e:
            print(f"Error de conexión: {str(e)}")
            return {"error": f"Error al conectar con OpenAI: {str(e)}"}
        except openai.RateLimitError as e:
            print(f"Límite de tasa excedido: {str(e)}")
            return {"error": "Límite de solicitudes excedido. Intenta nuevamente en unos minutos."}
        except Exception as e:
            print(f"Error inesperado: {str(e)}")
            return {"error": f"Error en análisis: {str(e)}"}

    def _build_system_prompt(self) -> str:
        """Construye el system prompt clínico fundamentado en PSS-14"""
        return (
            "Eres un psicólogo clínico especializado en estrés percibido, con dominio del instrumento PSS-14 "
            "(Cohen, Kamarck & Mermelstein, 1983). Tu formación incluye psicometría, psicología laboral, "
            "neurociencia del estrés y psicología ocupacional diferencial.\n\n"
            
            "## CONOCIMIENTO DEL PSS-14\n"
            "El PSS-14 mide la percepción de estrés en el último mes a través de dos dimensiones:\n\n"
            
            "**Dimensión 1 — Perceived Helplessness (Indefensión Percibida):**\n"
            "Ítems directos (puntuación directa indica más estrés):\n"
            "- Q1: Frecuencia de molestia por imprevistos\n"
            "- Q2: Sensación de incapacidad para controlar cosas importantes\n"
            "- Q3: Nerviosismo y sentirse estresado\n"
            "- Q8: Sensación de no poder afrontar todas las tareas pendientes\n"
            "- Q11: Enfado por cosas fuera de control\n"
            "- Q12: Pensar constantemente en cosas pendientes por resolver\n"
            "- Q14: Sensación de que las dificultades se acumulan sin poder superarlas\n\n"
            
            "**Dimensión 2 — Perceived Self-Efficacy (Autoeficacia Percibida):**\n"
            "Ítems invertidos (se invierte la puntuación: 4→0, 3→1, 2→2, 1→3, 0→4):\n"
            "- Q4: Capacidad percibida para manejar problemas personales\n"
            "- Q5: Sensación de que las cosas van bien / afrontamiento efectivo\n"
            "- Q6: Confianza en la capacidad personal para manejar problemas\n"
            "- Q7: Sensación de que las cosas van por buen camino\n"
            "- Q9: Capacidad para controlar las dificultades de la vida\n"
            "- Q10: Sentir que se tiene todo bajo control\n"
            "- Q13: Capacidad para decidir cómo pasar el tiempo\n\n"
            
            "**Escala Likert:** 0=Nunca, 1=Casi nunca, 2=A veces, 3=Frecuentemente, 4=Muy frecuentemente\n"
            "**Rango total:** 0-56 puntos (después de inversión)\n"
            "**Umbrales clínicos:** 0-28 Bajo | 29-42 Moderado | 43-56 Alto\n\n"
            
            "## SEÑALES DE ALARMA CLÍNICA\n"
            "Evalúa y reporta si detectas alguna de estas combinaciones de riesgo:\n"
            "- **Desesperanza aprendida:** Q2≥3 + Q14≥3 + Q4≤1 → patrón de indefensión persistente\n"
            "- **Agotamiento de recursos:** Q8≥3 + Q12≥3 + Q5≤1 → sobrecarga con afrontamiento colapsado\n"
            "- **Pérdida de control generalizada:** Q2≥3 + Q9≤1 + Q10≤1 → sensación de que nada está bajo control\n"
            "- **Hiperactivación sostenida:** Q3≥3 + Q11≥3 + Q1≥3 → activación simpática crónica\n"
            "- **Desconexión autoeficacia:** Q4≤1 + Q5≤1 + Q6≤1 → autoeficacia severamente comprometida\n"
            "Si detectas alguno, menciónalo EXPLÍCITAMENTE en el insight con su nombre y las preguntas involucradas.\n\n"
            
            "## RAZONAMIENTO POR PROFESIÓN\n"
            "No basta con nombrar la profesión. DEBES razonar sobre:\n"
            "1. **Demandas específicas:** ¿Qué exige esa profesión cognitiva, emocional o físicamente?\n"
            "2. **Recursos disponibles:** ¿Qué recursos de afrontamiento suele tener esa profesión?\n"
            "3. **Estresores ocupacionales típicos:** ¿Cuáles son los estresores laborales documentados para esa profesión?\n"
            "4. **Conexión con respuestas:** ¿Cómo los estresores ocupacionales se reflejan en las respuestas del PSS-14?\n"
            "Ejemplo: Un docente con Q8=4 no solo 'tiene tareas pendientes' — las demandas de planificación de clases, "
            "evaluación de estudiantes y trabajo administrativo generan una carga cognitiva continua que se manifiesta "
            "como rumiación sobre pendientes (Q12) y sensación de no poder manejar todo (Q14).\n\n"
            
            "## PROCESO DE ANÁLISIS INTERNO (Chain-of-Thought)\n"
            "Antes de generar tu respuesta JSON, sigue estos pasos internamente:\n"
            "1. Calcula mentalmente el subtotal de Indefensión (Q1+Q2+Q3+Q8+Q11+Q12+Q14) y Autoeficacia (Q4+Q5+Q6+Q7+Q9+Q10+Q13 invertidos)\n"
            "2. Identifica qué dimensión contribuye más al estrés total\n"
            "3. Evalúa si hay señales de alarma clínica según la tabla anterior\n"
            "4. Busca contradicciones: ¿reporta alta autoeficacia pero alta indefensión? ¿O viceversa?\n"
            "5. Conecta los hallazgos con la profesión y etapa vital específica del usuario usando el marco de razonamiento ocupacional\n"
            "6. Si hay texto libre, analiza coherencia entre lo verbal y lo numérico\n"
            "7. Solo entonces genera el JSON con hallazgos específicos y fundamentados\n\n"
            
            "## FEW-SHOT EXAMPLES\n\n"
            
            "### Ejemplo 1: Ingeniero 28 años, score 35/56 (Moderado)\n"
            "Respuestas: Q1=3, Q2=4, Q3=3, Q4=1, Q5=1, Q6=1, Q7=2, Q8=3, Q9=2, Q10=2, Q11=2, Q12=3, Q13=2, Q14=2\n"
            '{\n'
            '  "insight": "El perfil revela una tensión significativa entre las demandas percibidas y los recursos '
            'de afrontamiento. La dimensión de indefensión muestra activación elevada, especialmente en la percepción '
            'de falta de control (Q2=4) y nerviosismo sostenido (Q3=3). Paralelamente, la autoeficacia presenta debilitamiento '
            'crítico: la baja sensación de afrontamiento efectivo (Q5=1) y la inseguridad en capacidades personales (Q6=1) '
            'configuran un patrón de Desconexión de Autoeficacia que requiere atención. En el contexto de ingeniería, donde '
            'la resolución de problemas complejos es constante y la validación viene de resultados técnicos medibles, esta '
            'combinación sugiere que la persona resuelve exitosamente en lo técnico pero siente que su vida personal escapa '
            'a su control, erosionando la autoeficacia global. La rumiación sobre pendientes (Q12=3) en una profesión con '
            'deadlines y entregables múltiples amplifica la percepción de sobrecarga.",\n'
            '  "patterns": [\n'
            '    "Señal de alarma: Desconexión de Autoeficacia — Q4=1, Q5=1, Q6=1 configuran un patrón donde la confianza '
            'en las propias capacidades está severamente comprometida, lo cual en ingeniería contrasta con la competencia '
            'técnica objetiva y sugiere que el estrés ha disociado el rendimiento real de la percepción subjetiva",\n'
            '    "Disociación competencia-control: Q2=4 (falta de control) coexiste con Q9=2 (control moderado de dificultades), '
            'indicando que la falta de control se focaliza en áreas no-técnicas de la vida, posiblemente relaciones o finanzas, '
            'mientras mantiene funcionalidad en el ámbito laboral",\n'
            '    "Rumiación ocupacional: Q12=3 (pendientes constantes) + Q8=3 (sobrecarga) en el contexto de múltiples '
            'proyectos simultáneos típico de ingeniería genera un ciclo donde la hipervigilancia sobre tareas pendientes '
            'impide la desconexión cognitiva fuera del horario laboral"\n'
            '  ]\n'
            '}\n\n'
            
            "### Ejemplo 2: Docente 45 años, score 18/56 (Bajo)\n"
            "Respuestas: Q1=1, Q2=1, Q3=2, Q4=3, Q5=3, Q6=4, Q7=3, Q8=1, Q9=3, Q10=3, Q11=1, Q12=2, Q13=3, Q14=0\n"
            '{\n'
            '  "insight": "El perfil refleja un equilibrio sólido entre demandas percibidas y recursos de afrontamiento. '
            'La autoeficacia percibida es notablemente alta: la confianza en capacidades personales (Q6=4) y la sensación '
            'de control sobre dificultades (Q9=3, Q10=3) indican un repertorio de afrontamiento bien desarrollado. Esto es '
            'particularmente relevante en la docencia, donde las demandas emocionales y administrativas son continuas. La '
            'baja puntuación en acumulación de dificultades (Q14=0) sugiere una capacidad de procesamiento y resolución '
            'eficiente de problemas, mientras que Q12=2 (nivel moderado de rumiación sobre pendientes) refleja una conciencia '
            'saludable de responsabilidades sin que estas generen sobrecarga. A los 45 años en la etapa de adulto medio '
            'temprano, esta configuración sugiere que la experiencia docente ha consolidado estrategias de regulación emocional '
            'efectivas y una identidad profesional estable.",\n'
            '  "patterns": [\n'
            '    "Fortaleza de autoeficacia consolidada: Q4=3, Q5=3, Q6=4 configuran un núcleo de confianza sólido que en '
            'docencia permite manejar la impredecibilidad del aula y las demandas administrativas sin erosión del bienestar",\n'
            '    "Regulación emocional madura: Q11=1 (bajo enfado por imprevistos) + Q1=1 (baja molestia por imprevistos) '
            'sugiere alta tolerancia a la frustración, recurso crítico para una profesión donde los planes se modifican '
            'constantemente por dinámicas grupales",\n'
            '    "Conciencia sin sobrecarga: Q12=2 (rumiación moderada) coexiste con Q8=1 (baja sensación de sobrecarga), '
            'indicando que la persona monitorea sus pendientes sin que esto genere ansiedad anticipatoria"\n'
            '  ]\n'
            '}\n\n'
            
            "## CONSIDERACIONES ÉTICAS\n"
            "- Este análisis es orientativo, NO diagnóstico. Siempre aclarar que no sustituye evaluación profesional presencial.\n"
            "- Si detectas patrones de riesgo severo (score ≥48 o múltiples señales de alarma), incluye en professional_advice "
            "la recomendación explícita de buscar ayuda profesional inmediata.\n"
            "- No patologices puntuaciones bajas — analiza las fortalezas y recursos del individuo.\n"
            "- Respeta la diversidad cultural y profesional en tus interpretaciones.\n\n"
            
            "## FORMATO DE RESPUESTA\n"
            "Responde EXCLUSIVAMENTE en JSON válido con estos campos:\n"
            "- insight (string, 120-180 palabras, integra ambas dimensiones PSS-14, menciona señales de alarma si las hay)\n"
            "- recommendations (array de 3 strings, cada uno ≥15 palabras, específicos al perfil y profesión)\n"
            "- patterns (array de 3-5 strings, cada uno DEBE citar al menos 1 pregunta como Q1=valor, vinculado a profesión)\n"
            "- professional_advice (string, consejo táctico para su profesión concreta, accionable mañana)\n"
            "- ml_comparison (string, análisis cualitativo de discrepancia ML/real sin revelar score ML exacto)\n"
            "- demographic_comparison (string, contextualización sin números crudos)\n"
            "- emotional_analysis (string, análisis del texto libre o 'No proporcionado')\n\n"
            
            "## REGLAS ANTI-GENERICIDAD\n"
            "1. PROHIBIDO: 'Es importante manejar el estrés' → USA: 'Dado que Q8=3 indica sobrecarga de tareas, implementar la técnica de time-blocking...'\n"
            "2. PROHIBIDO: 'Se recomienda buscar ayuda' → USA: 'La combinación de Q2=4 y Q14=3 configura un patrón de desesperanza aprendida; consultar con un psicólogo especializado en burnout laboral'\n"
            "3. PROHIBIDO: patrones sin evidencia → cada patrón DEBE citar Qn=valor y conectar con la profesión\n"
            "4. PROHIBIDO: consejos profesionales genéricos → DEBE nombrar la profesión, una situación laboral concreta, y una técnica específica\n"
            "5. PROHIBIDO: 'Practicar técnicas de relajación' → USA: 'Implementar respiración 4-7-8 entre reuniones de sprint, especialmente cuando Q3=3 señala nerviosismo sostenido'\n"
            "6. PROHIBIDO: repetir el mismo patrón con distinta redacción → cada patrón debe aportar información NUEVA\n"
            "7. Usa solo comillas dobles en JSON\n"
            "8. Sin texto fuera del JSON\n"
            "9. No inventes campos adicionales\n"
            "10. PROHIBIDO mencionar promedios numéricos de la base de datos\n"
            "11. USA comparativas relativas ('por encima de lo típico para su grupo')\n"
            "12. Si el score es bajo (≤28), el insight DEBE enfocarse en FORTALEZAS y recursos, no en riesgos inexistentes"
        )

    def _call_gpt_api(self, client, prompt: str, test_data: dict, max_retries: int = 3):
        """Llama a la API de GPT con retry, backoff exponencial, y validación de calidad"""
        system_prompt = self._build_system_prompt()
        last_error = None

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4,   # Más bajo para análisis clínico consistente
                    top_p=0.85,
                    max_tokens=2500,   # Más espacio para emotional_analysis detallado
                    timeout=self.timeout
                )

                # Validar calidad de la respuesta
                if response.choices:
                    content = response.choices[0].message.content.strip()
                    quality_issues = self._validate_gpt_quality(content)

                    if quality_issues and attempt < max_retries - 1:
                        logging.warning(f"Intento {attempt+1}: Calidad insuficiente - {quality_issues}")
                        time.sleep(2 ** attempt)  # Backoff exponencial: 1s, 2s, 4s
                        continue

                return response

            except openai.RateLimitError as e:
                last_error = e
                wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s
                logging.warning(f"Rate limit alcanzado, reintentando en {wait_time}s (intento {attempt+1}/{max_retries})")
                time.sleep(wait_time)

            except openai.APIConnectionError as e:
                last_error = e
                wait_time = 2 ** attempt
                logging.warning(f"Error de conexión, reintentando en {wait_time}s (intento {attempt+1}/{max_retries})")
                time.sleep(wait_time)

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        if last_error:
            raise last_error

    def _validate_gpt_quality(self, content: str) -> Optional[str]:
        """Valida la calidad de la respuesta GPT antes de aceptarla"""
        try:
            data = json.loads(content)

            # Validar longitud del insight (mínimo 100 palabras para calidad clínica)
            insight = data.get('insight', '')
            word_count = len(insight.split())
            if word_count < 100:
                return f"Insight muy corto: {word_count} palabras (mínimo 100)"

            # Validar que recommendations tenga contenido sustancial
            recommendations = data.get('recommendations', [])
            if isinstance(recommendations, list):
                if len(recommendations) < 3:
                    return f"Solo {len(recommendations)} recomendaciones (mínimo 3)"
                for i, rec in enumerate(recommendations):
                    if isinstance(rec, str) and len(rec.split()) < 12:
                        return f"Recomendación {i+1} muy corta: '{rec}'"

            # Validar que patterns referencien preguntas con VALORES (Q1=3, no solo Q1)
            patterns = data.get('patterns', [])
            if isinstance(patterns, list):
                if len(patterns) < 3:
                    return f"Solo {len(patterns)} patrones (mínimo 3)"

                patterns_with_values = sum(
                    1 for p in patterns if isinstance(p, str) and re.search(r'Q\d+=\d', p, re.IGNORECASE)
                )
                if patterns_with_values < 2:
                    return f"Solo {patterns_with_values} patrones citan Qn=valor (mínimo 2)"

            # Validar que professional_advice no sea genérico
            advice = data.get('professional_advice', '')
            generic_phrases = [
                'es importante', 'se recomienda buscar', 'practicar técnicas de relajación',
                'manejar el estrés', 'buscar ayuda profesional', 'cuidar la salud mental'
            ]
            if isinstance(advice, str):
                advice_lower = advice.lower()
                for phrase in generic_phrases:
                    if phrase in advice_lower and len(advice.split()) < 20:
                        return f"Consejo profesional genérico detectado: contiene '{phrase}'"

            return None  # Sin problemas de calidad

        except json.JSONDecodeError:
            return "Respuesta no es JSON válido"

    def analyze_test(self, test_id: int) -> Dict:
        """Genera análisis psicológico con contexto demográfico integrado"""
        try:
            # 1. Obtener datos básicos del test
            test_data = self._get_test_data(test_id)
            if not test_data:
                raise ValueError(f"Test ID {test_id} no encontrado")
            
            # 2. Predicción del modelo ML (reusar instancia inyectada)
            if self.stress_predictor is None:
                from ml_extension import StressPredictor
                self.stress_predictor = StressPredictor()

            ml_prediction = self.stress_predictor.predict_stress(
                age=test_data['age'],
                profession=test_data['profession'],
                responses=test_data['responses']
            )
            
            # 3. Enriquecer datos con contexto y predicción
            enriched_data = {
                **test_data,
                'ml_prediction': ml_prediction
            }
            
            # 4. Generar análisis con GPT
            result = self.analyze_with_gpt(enriched_data)

            self.save_gpt_analysis(test_id, result)
            
            return {
                **result,
                'metadata': {
                    'test_id': test_id,
                    'timestamp': datetime.now().isoformat(),
                    'model_version': '1.2.0'
                }
            }

        except Exception as e:
            logging.error(f"Error en analyze_test: {str(e)}", exc_info=True)
            return {
                'error': 'Error en el análisis',
                'details': str(e),
                'status_code': 500
            }
    
    

    def get_stress_trends(self, profession: str = None) -> Dict:
        """Versión actualizada para OpenAI API v1.0+"""
        try:
            historical_data = self._get_historical_data(limit=500)
            
            if profession:
                filtered_data = [d for d in historical_data if str(d['profession']).lower() == profession.lower()]
                dataset = filtered_data if filtered_data else historical_data
            else:
                dataset = historical_data

            df = pd.DataFrame(dataset)
            trends = {
                "average_score": round(df['total_score'].mean(), 1),
                "common_levels": df['stress_level'].value_counts().to_dict(),
                "top_professions": df['profession'].value_counts().head(5).to_dict()
            }

            # Calcular distribución por nivel para contexto
            total = len(df)
            level_pcts = {k: f"{v/total*100:.1f}%" for k, v in trends['common_levels'].items()} if total > 0 else {}

            prompt = f"""
            Analiza estas tendencias de estrés percibido (PSS-14) de una muestra de {total} evaluaciones:
            - Puntuación promedio: {trends['average_score']}/56 (Umbrales: 0-28 Bajo, 29-42 Moderado, 43-56 Alto)
            - Distribución por nivel: {json.dumps(level_pcts)}
            - Profesiones más frecuentes: {json.dumps(trends['top_professions'])}
            {"- Filtrado por profesión: " + profession if profession else "- Vista general (todas las profesiones)"}

            Genera exactamente 3 insights en formato JSON:
            {{
                "insights": [
                    "Insight 1: Análisis de la distribución de niveles de estrés y qué implica para la población evaluada",
                    "Insight 2: Relación entre las profesiones más frecuentes y los niveles de estrés observados",
                    "Insight 3: Recomendación institucional basada en los patrones detectados"
                ]
            }}
            
            Cada insight debe ser específico a los datos mostrados, NO genérico.
            Cita los datos concretos en cada insight.
            """
            
            # Versión actualizada para OpenAI v1.0+
            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": (
                        "Eres un epidemiólogo especializado en salud mental ocupacional. "
                        "Analizas datos poblacionales del PSS-14 (Perceived Stress Scale) para "
                        "identificar tendencias, factores de riesgo ocupacionales y generar "
                        "recomendaciones de intervención basadas en evidencia. "
                        "Responde SOLO en JSON válido."
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                timeout=self.timeout
            )
            
            try:
                insights_data = json.loads(response.choices[0].message.content)
                trends["gpt_insights"] = insights_data.get("insights", [response.choices[0].message.content])
            except json.JSONDecodeError:
                trends["gpt_insights"] = response.choices[0].message.content

            return trends

        except Exception as e:
            print(f"Error en get_stress_trends: {str(e)}")
            return {"error": str(e)}

    def save_gpt_analysis(self, test_id: int, analysis: Dict) -> None:
        """Guarda el análisis de GPT en la base de datos."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Asegurar que la columna emotional_analysis exista
                try:
                    cursor.execute("SELECT emotional_analysis FROM gpt_analyses LIMIT 1")
                except sqlite3.OperationalError:
                    cursor.execute("ALTER TABLE gpt_analyses ADD COLUMN emotional_analysis TEXT DEFAULT 'No proporcionado'")

                cursor.execute('''
                    INSERT OR REPLACE INTO gpt_analyses 
                    (test_id, insight, recommendations, patterns, professional_advice, ml_comparison, demographic_comparison, emotional_analysis, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    test_id,
                    analysis.get('insight', ''),
                    json.dumps(analysis.get('recommendations', [])),
                    json.dumps(analysis.get('patterns', [])),
                    analysis.get('professional_advice', ''),
                    analysis.get('ml_comparison', 'Comparación ML no disponible'),
                    analysis.get('demographic_comparison', 'Comparación demográfica no disponible'),
                    analysis.get('emotional_analysis', 'No proporcionado'),
                    datetime.now().isoformat()
                ))

                conn.commit()
            logging.info(f"Análisis GPT guardado para test_id {test_id}")
        except sqlite3.Error as e:
            logging.error(f"Error SQLite al guardar análisis: {str(e)}")
            raise

    def _parse_response(self, content: str) -> Dict:
        """
        Parsea y valida estrictamente la respuesta JSON de GPT.
        Implementa múltiples estrategias de recuperación ante errores.
        """
        required_fields = ['insight', 'recommendations', 'patterns', 'professional_advice', 'ml_comparison', 'demographic_comparison', 'emotional_analysis']

        try:
            # Limpiar contenido inicial
            content = content.strip()
            
            # Si no empieza con {, buscar JSON embebido
            if not content.startswith('{'):
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group()
                else:
                    raise ValueError("No se encontró JSON válido en la respuesta")    

            # Intento principal de parsing
            data = json.loads(content)
            
            # VALIDACIÓN Y CORRECCIÓN MEJORADA
            return self._validate_and_fix_data(data, required_fields)
            
        except json.JSONDecodeError as e:
            # MANTENER ESTRATEGIA DE RECUPERACIÓN 1: Extraer JSON embebido
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    extracted_json = json_match.group()
                    data = json.loads(extracted_json)
                    return self._validate_and_fix_data(data, required_fields)
                except (json.JSONDecodeError, ValueError):
                    pass
                    
            # MANTENER  ESTRATEGIA DE RECUPERACIÓN 2: Limpiar JSON malformado
            content_clean = re.sub(r',(\s*[\]}])', r'\1', content)  # Elimina comas finales
            content_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content_clean)  # Elimina caracteres no imprimibles
            
            try:
                data = json.loads(content_clean)
                return self._validate_and_fix_data(data, required_fields)
            except json.JSONDecodeError:
                # ESTRATEGIA 3: Intentar reparaciones adicionales
                content_repaired = (content_clean.replace("'", '"')      # Corrige comillas simples
                                .replace("True", "true")               # Corrige booleanos Python
                                .replace("False", "false")
                                .replace("None", "null"))
                
                try:
                    data = json.loads(content_repaired)
                    return self._validate_and_fix_data(data, required_fields)
                except json.JSONDecodeError:
                    self._log_error(f"Respuesta no es JSON válido después de limpieza: {str(e)}", content)
                    # FALLBACK FINAL: Respuesta de emergencia
                    return self._generate_emergency_fallback()

    def _validate_and_fix_data(self, data: Dict, required_fields: List[str]) -> Dict:
        """Valida y corrige la estructura de datos JSON"""
    
        # ACTUALIZAR campos requeridos
        all_required_fields = ['insight', 'recommendations', 'patterns', 'professional_advice', 'ml_comparison', 'demographic_comparison', 'emotional_analysis']

        # Verificar campos faltantes
        missing_fields = [field for field in all_required_fields if field not in data]
        if missing_fields:
            # GENERAR RESPUESTA DE FALLBACK PARCIAL
            return self._generate_fallback_response(data, missing_fields)
            
        # Validar y corregir tipos
        if not isinstance(data.get('recommendations'), list):
            data['recommendations'] = self._extract_list_from_string(
                data.get('recommendations', ''), 
                ["Buscar apoyo profesional", "Practicar técnicas de relajación", "Evaluar factores estresantes"]
            )
            
        if not isinstance(data.get('patterns'), list):
            data['patterns'] = self._extract_list_from_string(
                data.get('patterns', ''),
                ["Patrón de estrés identificado", "Respuestas variables observadas", "Necesita evaluación adicional"]
            )
            
        # Asegurar valores por defecto para campos de texto
        if not data.get('demographic_comparison'):
            data['demographic_comparison'] = "Contextualización demográfica requiere análisis adicional."
            
        # Asegurar mínimos
        if len(data['recommendations']) < 3:
            data['recommendations'].extend([
                "Consultar con profesional de salud mental",
                "Implementar rutinas de autocuidado",
                "Monitorear niveles de estrés regularmente"
            ][:3-len(data['recommendations'])])
            
        if len(data['patterns']) < 3:
            data['patterns'].extend([
                "Patrón adicional identificado",
                "Respuesta requiere análisis detallado",
                "Variabilidad en percepciones observada"
            ][:3-len(data['patterns'])])
            
        return data

    def _extract_list_from_string(self, text: str, fallback: List[str]) -> List[str]:
        """Intenta extraer una lista de un string mal formateado"""
        if isinstance(text, list):
            return text
            
        if not isinstance(text, str):
            return fallback
            
        # Intentar extraer elementos separados por comas, puntos, o saltos de línea
        items = []
        for separator in ['\n-', '\n•', '\n*', '\n', ',', ';']:
            if separator in text:
                items = [item.strip().lstrip('-•*').strip() for item in text.split(separator)]
                items = [item for item in items if item and len(item) > 5]  # Filtrar items muy cortos
                break
        
        return items if len(items) >= 2 else fallback

    # MANTENER MÉTODOS DE FALLBACK EXISTENTES:
    def _generate_fallback_response(self, partial_data: Dict, missing_fields: List[str]) -> Dict:
        """Genera respuesta de fallback cuando faltan campos"""
        fallback_data = {
            'insight': "Análisis completado con datos disponibles. Se requiere evaluación adicional.",
            'recommendations': [
                "Consultar con profesional de salud mental",
                "Implementar técnicas de manejo del estrés",
                "Realizar seguimiento regular"
            ],
            'patterns': [
                "Patrones de respuesta identificados",
                "Variabilidad en percepciones de estrés",
                "Necesita análisis más profundo"
            ],
            'professional_advice': "Se recomienda evaluación profesional adicional para análisis completo.",
            'ml_comparison': "Comparación ML no disponible o incompleta.",
            'demographic_comparison': "Contextualización demográfica no disponible.",
            'emotional_analysis': "No proporcionado"
        }
        
        # Mantener datos existentes válidos
        for key, value in partial_data.items():
            if key in fallback_data and value:
                fallback_data[key] = value
        
        return fallback_data

    def _generate_emergency_fallback(self) -> Dict:
        """Respuesta de emergencia cuando todo falla"""
        return {
            'insight': "Análisis automático completado. Los resultados indican niveles de estrés que requieren atención.",
            'recommendations': [
                "Consultar con un profesional de salud mental",
                "Practicar técnicas de relajación y mindfulness",
                "Evaluar y modificar factores estresantes identificables"
            ],
            'patterns': [
                "Respuestas del test indican presencia de factores estresantes",
                "Variabilidad en percepciones de control y afrontamiento",
                "Patrón sugiere necesidad de intervención profesional"
            ],
            'professional_advice': "Los resultados sugieren la conveniencia de una evaluación más detallada con un profesional de la salud mental.",
            'ml_comparison': "Sistema de comparación ML temporalmente no disponible.",
            'demographic_comparison': "Análisis demográfico no disponible en este momento.",
            'emotional_analysis': "No proporcionado"
        }

    # MANTENER MÉTODO DE LOGGING:
    def _log_error(self, error_message: str, content: str = None) -> None:
        """Registra errores en el sistema de logging."""
        logging.error(error_message)
        if content:
            logging.error(f"Contenido problemático: {content[:500]}...")  # Solo primeros 500 caracteres


    def _generate_demographic_insight(self, test_data: Dict) -> str:
        """Genera contexto demográfico sin números crudos"""
        context_lines = []
        demo = test_data.get('demographic_context', {})
        
        if demo.get('profession_stats'):
            prof_ctx = self._get_profession_context(
                test_data['score'],
                demo['profession_stats']
            )
            context_lines.append(f"- Contexto Profesional: {prof_ctx}")
        
        if demo.get('age_group_stats'):
            age_ctx = self._get_age_context(
                test_data['score'],
                demo['age_group_stats']
            )
            context_lines.append(f"- Contexto Generacional: {age_ctx}")
        
        return "\n".join(context_lines) if context_lines else "- Sin datos contextuales adicionales"

    def _get_profession_context(self, score: int, stats: Dict) -> str:
        """Genera descripción cualitativa para profesión"""
        if stats.get('sample_size', 0) < 5:
            return "Datos limitados para comparación profesional"
        
        diff = score - stats['avg_score']
        if diff > 5: 
            return "Nivel significativamente alto para su campo profesional"
        elif diff > 2: 
            return "Ligeramente alto para su profesión"
        elif diff < -5: 
            return "Nivel notablemente bajo para su área profesional"
        elif diff < -2: 
            return "Ligeramente bajo para su ámbito profesional"
        return "Nivel típico para su profesión"

    def _get_age_context(self, score: int, stats: Dict) -> str:
        """Genera descripción cualitativa para edad"""
        if stats.get('sample_size', 0) < 5:
            return "Datos limitados para comparación generacional"
        
        diff = score - stats['avg_score']
        age_range = stats.get('age_range', 'su grupo de edad')
        
        if diff > 5: 
            return f"Nivel significativamente alto para {age_range}"
        elif diff > 2: 
            return f"Ligeramente alto para {age_range}"
        elif diff < -5: 
            return f"Nivel notablemente bajo para {age_range}"
        elif diff < -2: 
            return f"Ligeramente bajo para {age_range}"
        return f"Nivel típico para {age_range}"

    def _get_life_stage(self, age: int) -> str:
        """Determina la etapa vital"""
        if age < 25: 
            return "Adulto joven temprano"
        elif age < 35: 
            return "Adulto joven"
        elif age < 45: 
            return "Adulto medio temprano"
        elif age < 55: 
            return "Adulto medio"
        elif age < 65: 
            return "Adulto medio tardío"
        else: 
            return "Adulto mayor"

    def _get_generation(self, age: int) -> str:
        """Determina la generación"""
        if age >= 60: 
            return "Baby Boomer"
        elif age >= 40: 
            return "Generación X"
        elif age >= 25: 
            return "Millennial"
        else: 
            return "Generación Z"

    def _get_profession_category(self, profession: str) -> str:
        """Categoriza profesiones basado en la lista predefinida del frontend"""
        profession_lower = profession.lower()
        
        # Mapeo exacto de categorías basado en app.js
        category_mapping = {
            "Ingenierías": ["ingenier", "sistem", "mecán", "civil", "electr", "quím", "petrol", "naval", "aeronáut", "agrónom", "ambient", "bioméd", "comput", "telecom", "industrial", "mecatr", "aliment", "forest", "geológ", "miner", "metalúrg", "pesqu", "textil", "acúst", "automotriz", "materiales", "genét", "energías renov", "robót", "fintech"],
            "Ciencias de la Salud": ["médic", "cirujan", "pediatr", "cardiólog", "neurólog", "psiquiatr", "enfermer", "odontólog", "veterinar", "nutriólog", "fisioterap", "biólog", "farmacéut", "técnico en enfermer", "parter", "comadron", "logoped", "terapeuta ocupacional", "biotecnolog", "masoterap"],
            "Tecnología y Ciencias de la Computación": ["program", "desarrollad", "analista de datos", "inteligencia artificial", "administrador de redes", "diseñador ux/ui", "cibersegurid", "técnico en comput", "técnico en telecom", "videojuegos", "big data", "software", "arquitecto de comput", "e-commerce", "aplicaciones móviles"],
            "Ciencias Sociales y Humanidades": ["psicólog", "sociólog", "antropólog", "economista", "abogad", "profesor", "trabajador social", "historiad", "politólog", "geógraf", "pedagog", "notari", "juez", "criminólog", "lingüístic"],
            "Artes y Humanidades": ["arquitect", "diseñador gráfic", "escritor", "músic", "actor", "artista plást", "diseñador de mod", "fotógraf", "compositor", "director de muse", "model", "escultor", "pintor", "bailarín", "curador de arte"],
            "Oficios Técnicos y Manuales": ["técnico electric", "técnico mecán", "carpinter", "plomer", "soldad", "albañil", "mecánico automotriz", "refrigeración", "electrónic", "energías renov", "operador de maquinaria", "paneles solar", "climatizació"],
            "Educación y Formación": ["maestro", "educador infantil", "orientador educ", "instructor", "docente de educación especial", "formador de adultos", "tutor en línea", "diseñador instruccional"],
            "Ciencias Naturales y Exactas": ["físic", "químic", "biólog", "matemát", "geólog", "astrónom", "bioquím", "oceanógraf", "meteorólog", "estadístic"],
            "Administración y Negocios": ["administrador", "contador", "auditor", "analista financier", "asesor financier", "gerente de proyect", "consultor de negoci", "recursos humanos", "agente de seguros", "corredor de bols", "logístic", "marketing", "riesgos", "planificador estratég"],
            "Comunicación y Medios": ["periodist", "comunicador social", "relaciones públicas", "locutor", "presentador", "editor", "redactor", "guionist", "productor audiovis", "community manager", "marketing digital", "diseñador de contenido"],
            "Transporte y Logística": ["pilot", "controlador aére", "conductor", "operador de grú", "logístic", "despachador de vuel", "capitán de barc", "mantenimiento aeronáut", "tráfic", "coordinador de logístic"],
            "Servicios y Atención al Cliente": ["recepcionist", "cajer", "asistente administrat", "atención al cliente", "call center", "anfitrión", "azafat", "auxiliar de vuel", "conserj", "guía turístic", "agente de viajes", "barist", "meser", "bartender"],
            "Seguridad y Defensa": ["policí", "bomber", "militar", "guardia de segurid", "detective privad", "agente de aduan", "protección civil", "seguridad informát", "defensa personal", "analista de inteligenci"],
            "Agricultura, Ganadería y Pesca": ["agricultor", "ganader", "pescad", "agrónom", "agropecuari", "apicultor", "silvicultor", "maquinaria agrícol", "acuicult", "calidad agroalimentari"],
            "Ciencias Jurídicas y Políticas": ["abogad", "juez", "fiscal", "notari", "defensor públic", "procurad", "asesor legal", "diplomát", "funcionario públic", "analista polít"],
            "Ciencias Económicas y Financieras": ["economist", "contador", "auditor", "analista financier", "asesor de invers", "corredor de bols", "comercio internacion", "consultor económic", "gestor de patrimon", "investigador económic"],
            "Ciencias de la Información y Documentación": ["bibliotecari", "archivist", "documentalist", "gestor de informació", "gestión del conocimient", "creador de contenido", "curador de contenido", "analista de informació", "gestión document", "museolog", "preservación digital"]
        }
        
        # Buscar coincidencias
        for category, keywords in category_mapping.items():
            if any(keyword in profession_lower for keyword in keywords):
                return category
        
        return "Otros sectores"

    def _format_responses(self, responses: List[int]) -> str:
        """Formatea las respuestas de manera más legible"""
        scale = {0: "Nunca", 1: "Casi nunca", 2: "A veces", 3: "Frecuentemente", 4: "Muy frecuentemente"}
        
        formatted = []
        for i, resp in enumerate(responses):
            formatted.append(f"Q{i+1}: {resp} ({scale.get(resp, 'N/A')})")
        
        # Agrupar en líneas de 7 para mejor legibilidad
        line1 = " | ".join(formatted[:7])
        line2 = " | ".join(formatted[7:])
        
        return f"{line1}\n{line2}"

    def _get_profession_sector(self, profession: str) -> str:
        """Alias para compatibilidad"""
        return self._get_profession_category(profession)

    def _detect_alarm_signals(self, responses: List[int]) -> List[str]:
        """
        Detecta combinaciones de respuestas que configuran señales de alarma clínica.
        Basado en patrones empíricos de riesgo en PSS-14.
        """
        if len(responses) < 14:
            return []

        signals = []
        r = responses  # Alias para brevedad (valores originales, 0-indexed)

        # Desesperanza aprendida: Q2≥3 + Q14≥3 + Q4≤1
        if r[1] >= 3 and r[13] >= 3 and r[3] <= 1:
            signals.append(
                f"DESESPERANZA APRENDIDA: Q2={r[1]} (falta de control) + Q14={r[13]} "
                f"(dificultades insuperables) + Q4={r[3]} (baja capacidad de manejo) — "
                f"patrón de indefensión persistente que requiere atención profesional"
            )

        # Agotamiento de recursos: Q8≥3 + Q12≥3 + Q5≤1
        if r[7] >= 3 and r[11] >= 3 and r[4] <= 1:
            signals.append(
                f"AGOTAMIENTO DE RECURSOS: Q8={r[7]} (sobrecarga) + Q12={r[11]} "
                f"(rumiación sobre pendientes) + Q5={r[4]} (bajo afrontamiento) — "
                f"el sistema de afrontamiento está colapsado"
            )

        # Pérdida de control generalizada: Q2≥3 + Q9≤1 + Q10≤1
        if r[1] >= 3 and r[8] <= 1 and r[9] <= 1:
            signals.append(
                f"PÉRDIDA DE CONTROL GENERALIZADA: Q2={r[1]} (falta de control) + Q9={r[8]} "
                f"(sin control de dificultades) + Q10={r[9]} (nada bajo control) — "
                f"sensación pervasiva de falta de agencia"
            )

        # Hiperactivación sostenida: Q3≥3 + Q11≥3 + Q1≥3
        if r[2] >= 3 and r[10] >= 3 and r[0] >= 3:
            signals.append(
                f"HIPERACTIVACIÓN SOSTENIDA: Q1={r[0]} (molestia por imprevistos) + Q3={r[2]} "
                f"(nerviosismo) + Q11={r[10]} (enfado) — activación simpática crónica "
                f"con riesgo de somatización"
            )

        # Desconexión autoeficacia: Q4≤1 + Q5≤1 + Q6≤1
        if r[3] <= 1 and r[4] <= 1 and r[5] <= 1:
            signals.append(
                f"DESCONEXIÓN DE AUTOEFICACIA: Q4={r[3]} + Q5={r[4]} + Q6={r[5]} — "
                f"autoeficacia severamente comprometida en múltiples dominios, "
                f"posible erosión de la identidad de competencia"
            )

        return signals

    def _identify_critical_stressors(self, responses: List[int]) -> List[str]:
        """Identifica respuestas que indican alto estrés (considerando items invertidos)"""
        stressors = []
        
        # Usar constantes centralizadas de pss_scoring
        # INVERTED_INDICES (positivos, se invierten): Q4,Q5,Q6,Q7,Q9,Q10,Q13 → [3,4,5,6,8,9,12]
        # DIRECT_INDICES (negativos, directos): Q1,Q2,Q3,Q8,Q11,Q12,Q14 → [0,1,2,7,10,11,13]

        # Descripción de cada pregunta para contexto de GPT
        q_map = [
            "Molestia por imprevistos",                  # Q1 - Directo
            "Falta de control sobre vida",               # Q2 - Directo
            "Nerviosismo general",                       # Q3 - Directo
            "Capacidad de manejar problemas",            # Q4 - Invertido
            "Sensación de afrontamiento efectivo",       # Q5 - Invertido
            "Seguridad en capacidad personal",           # Q6 - Invertido
            "Sensación de que las cosas van bien",       # Q7 - Invertido
            "No poder afrontar todas las tareas",        # Q8 - Directo
            "Control de dificultades en la vida",        # Q9 - Invertido
            "Sentir que tiene todo bajo control",        # Q10 - Invertido
            "Enfado por cosas fuera de control",         # Q11 - Directo
            "Pensar en cosas pendientes",                # Q12 - Directo
            "Control de la forma de pasar el tiempo",    # Q13 - Invertido
            "Dificultades acumuladas insuperables"       # Q14 - Directo
        ]

        for i, val in enumerate(responses):
            is_stressor = False
            val_desc = ""
            
            if i in INVERTED_INDICES:
                # Ítems positivos: valor bajo (0,1) indica estrés
                if val <= 1:
                    is_stressor = True
                    val_desc = "Muy bajo/Bajo (Indica Estrés)"
            elif i in DIRECT_INDICES:
                # Ítems negativos: valor alto (3,4) indica estrés
                if val >= 3:
                    is_stressor = True
                    val_desc = "Alto/Muy Alto (Indica Estrés)"
            
            if is_stressor:
                question_content = q_map[i] if i < len(q_map) else "Pregunta"
                stressors.append(f"Pregunta {i+1} ('{question_content}'): Valor {val} - {val_desc}")
                
        return stressors
