import openai
import sqlite3
import pandas as pd
import json
from typing import Dict, List, Optional
import os
from datetime import datetime

class GPTStressAnalyzer:
    def __init__(self, api_key: str):
        """Inicializa el analizador con la API key de OpenAI."""
        openai.api_key = api_key
        self.db_path = '../datos/pss_database.db'
        self.last_analysis_cache = {}  # Cache para evitar llamadas redundantes a GPT
        self.timeout = 30  # Timeout en segundos para las llamadas a la API

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
        prompt = f"""
        ## Análisis de Test de Estrés PSS-14
        **Perfil del paciente:**
        - Edad: {test_data['age']}
        - Profesión: {test_data['profession']}
        - Puntuación total: {test_data['score']}/56
        - Nivel de estrés: {test_data['stress_level']}

         **Predicción del modelo ML:**
            - Puntuación predicha: {test_data.get('ml_prediction', {}).get('predicted_score', 'N/A')}
            - Nivel predicho: {test_data.get('ml_prediction', {}).get('predicted_stress_level', 'N/A')}
        **Respuestas:**
        {json.dumps(test_data['responses'], indent=2)}

        **Contexto histórico (últimos {len(historical_data)} tests):**
        {json.dumps(historical_data[:5], indent=2)}

        **Instrucciones importantes:**
        1. Compara los resultados calculados con la predicción del modelo
        2. Analiza patrones considerando ambos resultados
        3. Proporciona recomendaciones específicas
        4. Este es un análisis INDIVIDUAL y NO debes compararlo con tests anteriores.
        5. NO menciones "mejorías" o "cambios" respecto a tests previos.
        6. Cada análisis debe tratarse como un caso independiente.
        7. Evita frases como "has mejorado" o "continúas con".
        8. Para la sección "patterns", NO LISTES LAS RESPUESTAS NUMÉRICAS, sino patrones como "tendencia a sentirse sobrepasado", "dificultad para controlar irritabilidad", etc.

        **Formato de respuesta (JSON):**
        Proporciona un análisis JSON con:
        1. "insight": Análisis profesional del nivel actual de estrés (máx. 150 palabras)
        2. "recommendations": Array con 3 recomendaciones personalizadas según edad, profesión y nivel de estrés
        3. "patterns": Array con 3-5 PATRONES VERBALES DESCRIPTIVOS detectados en las respuestas, NO los valores numéricos
        4. "professional_advice": Consejo específico para su profesión
        """

        return prompt
    
    def analyze_with_gpt(self, test_data: dict) -> Dict:
        """Versión actualizada para OpenAI API v1.0+"""
        try:
            # Validación de datos de entrada
            required_keys = ['test_id', 'age', 'profession', 'score', 'stress_level', 'responses']
            for key in required_keys:
                if key not in test_data:
                    raise ValueError(f"Falta clave requerida: {key}")

            if len(test_data['responses']) != 14:
                raise ValueError("Se requieren exactamente 14 respuestas")

            # Obtener contexto
            historical_data = self._get_historical_data()
            prompt = self._generate_gpt_prompt(test_data, historical_data)

            # Versión actualizada para OpenAI v1.0+
            client = openai.OpenAI(api_key=openai.api_key)
            
            # Llamada a API con timeout - sintaxis actualizada
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", # Usar gpt-3.5-turbo en lugar de gpt-4 si tienes problemas
                messages=[
                    {"role": "system", "content": "Eres un psicólogo experto en el test PSS-14. Responde SIEMPRE en formato JSON con las claves 'insight', 'recommendations', 'patterns' y 'professional_advice'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                timeout=self.timeout
            )

            # Validar respuesta (sintaxis actualizada)
            if not response.choices:
                raise ValueError("No se recibieron opciones en la respuesta")

            # Intentar extraer JSON de la respuesta
            content = response.choices[0].message.content.strip()
            
            try:
                analysis = json.loads(content)
            except json.JSONDecodeError:
                # Intenta extraer solo la parte JSON si está envuelto en explicaciones
                try:
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx >= 0 and end_idx > 0:
                        json_str = content[start_idx:end_idx]
                        analysis = json.loads(json_str)
                    else:
                        raise ValueError("No se pudo extraer JSON de la respuesta")
                except Exception:
                    print(f"Error al parsear respuesta: {content}")
                    raise ValueError("No se pudo extraer JSON de la respuesta")

            # Validar estructura de análisis
            for key in ['insight', 'recommendations']:
                if key not in analysis:
                    raise ValueError(f"Falta clave requerida en análisis: {key}")

            return {
                **test_data,
                **analysis,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            if "openai" in str(e).lower():
                print(f"Error de OpenAI API: {str(e)}")
                return {"error": f"Error de OpenAI: {str(e)}"}
            elif isinstance(e, json.JSONDecodeError):
                print(f"Error decodificando JSON: {str(e)}")
                return {"error": "Error procesando respuesta de GPT"}
            else:
                print(f"Error inesperado: {str(e)}")
                return {"error": f"Error en análisis: {str(e)}"}


    def analyze_test(self, test_id: int) -> Dict:
        """Versión con caché y manejo de errores mejorado"""
        try:
            if test_id in self.last_analysis_cache:
                return self.last_analysis_cache[test_id]

            test_data = self._get_test_data(test_id)
            if not test_data:
                raise ValueError(f"Test ID {test_id} no encontrado")

            result = self.analyze_with_gpt(test_data)
            self.last_analysis_cache[test_id] = result
            return result

        except Exception as e:
            print(f"Error en analyze_test: {str(e)}")
            return {"error": str(e)}
    
    

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