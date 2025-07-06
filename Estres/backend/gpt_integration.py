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

class GPTStressAnalyzer:
    def __init__(self, api_key: str):
        """Inicializa el analizador con la API key de OpenAI."""
        openai.api_key = api_key
        self.db_path = '../datos/pss_database.db'
        self.last_analysis_cache = {}  # Cache para evitar llamadas redundantes a GPT
        self.timeout = 30  # Timeout en segundos para las llamadas a la API
        self._initialize_db()


    def _initialize_db(self):
        """Crea las tablas necesarias si no existen."""
        try:
            conn = sqlite3.connect(self.db_path)
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
                # La columna no existe, añadirla
                cursor.execute("ALTER TABLE gpt_analyses ADD COLUMN demographic_comparison TEXT DEFAULT 'Análisis demográfico no disponible para registros anteriores'")
                print("Columna demographic_comparison añadida a registros existentes")

            conn.commit()
            conn.close()
            print("Tablas de base de datos inicializadas correctamente")
            
        except Exception as e:
            print(f"Error inicializando base de datos: {str(e)}")

    def _get_test_data(self, test_id: int) -> Optional[Dict]:
        """Obtiene los datos de un test específico desde SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Datos básicos del test
        cursor.execute('''
            SELECT age, profession, total_score, stress_level, timestamp 
            FROM tests WHERE test_id = ?
        ''', (test_id,))
        test_row = cursor.fetchone()

        if not test_row:
            conn.close()
            return None

        # Respuestas individuales
        cursor.execute('''
            SELECT question_number, original_value, processed_value 
            FROM responses WHERE test_id = ? ORDER BY question_number
        ''', (test_id,))
        responses = cursor.fetchall()

        conn.close()

        return {
            "test_id": test_id,
            "age": test_row[0],
            "profession": test_row[1],
            "score": test_row[2],
            "stress_level": test_row[3],
            "timestamp": test_row[4],
            "responses": [r[1] for r in responses],  # Valores originales
            "processed_responses": [r[2] for r in responses]  # Valores procesados
        }

    def _get_historical_data(self, limit: int = 100) -> List[Dict]:
        """Obtiene datos históricos para análisis comparativo."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(f'''
            SELECT age, profession, total_score, stress_level 
            FROM tests ORDER BY timestamp DESC LIMIT {limit}
        ''', conn)
        conn.close()
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

        prompt = f"""
        ## Instrucciones para el análisis PSS-14
        **Datos del paciente:**
        - Edad: {test_data['age']} años ({self._get_life_stage(test_data['age'])})
        - Profesión: {test_data['profession']} ({self._get_profession_category(test_data['profession'])})
        - Puntuación real: {test_data['score']}/56 ({test_data['stress_level']})
        - Puntuación ML: {test_data.get('ml_prediction', {}).get('predicted_score', 'N/A')}
        - Nivel predicho: {test_data.get('ml_prediction', {}).get('predicted_stress_level', 'N/A')}
        - Comparación: {score_comparison}
        - Esta diferencia podría indicar factores situacionales o de percepción subjetiva

        **Contextualización (Solo para tu Análisis):**
        {demographic_insight}
        **Respuestas del Test:**
        {responses_formatted}

        **Tareas requeridas:**
        1. Análisis comparativo (real vs ML):
        - Explicar diferencias >5 puntos
        - Considerar factores psicosociales/contextuales
        2. Patrones:
        - 3-5 patrones comportamentales
        - Basados en respuestas + contexto demográfico
        3. Recomendaciones:
        - 3 sugerencias personalizadas
        - Accionables para su profesión/edad
        4. Consejo profesional:
        - Consejos específicos para su profesión

        **Restricciones ESTRICTAS:**
        - PROHIBIDO mencionar promedios numéricos
        - Usar comparativas relativas ("mayor/menor que lo típico")
        - PROHIBIDO indicar el score real de la predicción ML
        - PROHIBIDO indicar score ficticio o inventado
        - Patrones deben ser descriptivos, NO numéricos
        - Enfoque en situación actual (NO histórico)
        - NO usar términos como "mejoría" o "empeoramiento"
        - Recomendaciones deben ser accionables

        **Formato JSON requerido:**
        {{
            "insight": "Análisis de 120-150 palabras que integre contexto demográfico...",
            "recommendations": [
                "Recomendación 1 (específica para {test_data['profession']})",
                "Recomendación 2 (adaptada a su etapa vital)",
                "Recomendación 3 (enfoque práctico)"
            ],
            "patterns": [
                "Patrón 1 (ej: 'Dificultad para desconectar del trabajo')",
                "Patrón 2 (relacionado con su edad/profesión)",
                "Patrón 3 (ej: 'Dificultad para establecer límites saludables')"
            ],
            "professional_advice": "Consejo concreto para un {test_data['profession']}....",
            "ml_comparison": "Análisis de discrepancia en términos cualitativos",
            "demographic_comparison": "Contextualización demográfica sin números específicos"
        }}
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
            
            # Llamada a API con configuración mejorada
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},  # Fuerza formato JSON
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "Eres un psicólogo experto en el test PSS-14 con contexto demográfico. Responde EXCLUSIVAMENTE en JSON válido con estos campos:\n"
                            "- insight (análisis de 120-150 palabras)\n"
                            "- recommendations (array de 3 strings)\n"
                            "- patterns (array de 3-5 strings descriptivos)\n"
                            "- professional_advice (consejo específico para su profesión)\n"
                            "- ml_comparison (análisis de discrepancia ML/real)\n"
                            "- demographic_comparison\n\n"
                            "CONTEXTO DISPONIBLE:\n"
                            "- Predicción ML cuantitativa ya calculada\n"
                            "- Comparaciones demográficas cualitativas\n"
                            "- Etapa vital y sector profesional\n\n"
                            "INSTRUCCIONES:\n"
                            "- Analiza discrepancias ML vs real cualitativamente\n"
                            "- Integra contexto generacional y profesional\n"
                            "- NO menciones números específicos de promedios\n"
                            "- USA comparativas relativas (ej: 'por encima de lo típico')\n"
                            "- Personaliza según etapa vital y sector profesional"
                            "REGLAS ESTRICTAS:\n"
                            "1. Usa solo comillas dobles\n"
                            "2. Sin texto fuera del JSON\n"
                            "3. No inventes campos adicionales\n"
                            "4. Patrones deben ser descriptivos (ej: 'tendencia a sobreanalizar situaciones')"
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                top_p=0.9,
                max_tokens=1200,
                timeout=self.timeout
            )

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


    def analyze_test(self, test_id: int) -> Dict:
        """Genera análisis psicológico con contexto demográfico integrado"""
        try:
            # 1. Obtener datos básicos del test
            test_data = self._get_test_data(test_id)
            if not test_data:
                raise ValueError(f"Test ID {test_id} no encontrado")
            
            
            # 2. Predicción del modelo ML
            from ml_extension import StressPredictor
            predictor = StressPredictor()
            
            ml_prediction = predictor.predict_stress(
                age=test_data['age'],
                profession=test_data['profession'],
                responses=test_data['responses']
            )
            
            # 3. Enriquecer datos con contexto y predicción
            enriched_data = {
            **test_data,
            'ml_prediction': ml_prediction
            }
            
            # 4. Generar análisis con GPT (tu método existente)
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

            prompt = f"""
            Analiza estas tendencias de estrés:
            - Puntuación promedio: {trends['average_score']}/56
            - Niveles más comunes: {json.dumps(trends['common_levels'])}
            - Profesiones más frecuentes: {json.dumps(trends['top_professions'])}

            Genera 3 insights clave sobre estos datos.
            """
            
            # Versión actualizada para OpenAI v1.0+
            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", # Usar gpt-3.5-turbo en lugar de gpt-4 si tienes problemas
                messages=[
                    {"role": "system", "content": "Eres un analista de datos de salud mental."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                timeout=self.timeout
            )
            
            trends["gpt_insights"] = response.choices[0].message.content
            return trends

        except Exception as e:
            print(f"Error en get_stress_trends: {str(e)}")
            return {"error": str(e)}

    def save_gpt_analysis(self, test_id: int, analysis: Dict) -> None:
        """Guarda el análisis de GPT en la base de datos."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO gpt_analyses 
                (test_id, insight, recommendations, patterns, professional_advice, ml_comparison, demographic_comparison, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                test_id, 
                analysis.get('insight', ''),
                json.dumps(analysis.get('recommendations', [])),
                json.dumps(analysis.get('patterns', [])),
                analysis.get('professional_advice', ''),
                analysis.get('ml_comparison', 'Comparación ML no disponible'),
                analysis.get('demographic_comparison', 'Comparación demográfica no disponible'), 
                datetime.now().isoformat()
            ))
            
            conn.commit()
            logging.info(f"Análisis GPT guardado para test_id {test_id}")
        except sqlite3.Error as e:
            logging.error(f"Error SQLite al guardar análisis: {str(e)}")
            raise  # Relanza la excepción para manejo superior
        finally:
            conn.close() if conn else None

    def _parse_response(self, content: str) -> Dict:
        """
        Parsea y valida estrictamente la respuesta JSON de GPT.
        Implementa múltiples estrategias de recuperación ante errores.
        """
        required_fields = ['insight', 'recommendations', 'patterns', 'professional_advice', 'ml_comparison', 'demographic_comparison']
        
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
        all_required_fields = ['insight', 'recommendations', 'patterns', 'professional_advice', 'ml_comparison', 'demographic_comparison']
        
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
            'demographic_comparison': "Contextualización demográfica no disponible."
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
            'demographic_comparison': "Análisis demográfico no disponible en este momento."
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

    def _get_profession_sector(self, profession: str) -> str:
        """Alias para compatibilidad"""
        return self._get_profession_category(profession)

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

    
    
    