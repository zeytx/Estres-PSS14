import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import TestResponse, AnalysisRequest, GPTAnalysis
from gpt_integration import GPTStressAnalyzer
from database import save_to_db
from csv_handler import save_to_csv
import os
import uvicorn

app = FastAPI(title="API de Análisis de Estrés PSS-14")

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configura GPT (usar variable de entorno)
gpt_analyzer = GPTStressAnalyzer(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/submit-test", response_model=GPTAnalysis)
async def submit_test(test_data: TestResponse):
    """Endpoint para enviar nuevos tests"""
    try:
        # Guardar en base de datos
        test_id, score, level = save_to_db(test_data.age, test_data.profession, test_data.responses)
        save_to_csv(test_data.age, test_data.profession, test_data.responses)
        
        # Analizar con GPT
        analysis = gpt_analyzer.analyze_with_gpt({
            "test_id": test_id,
            "age": test_data.age,
            "profession": test_data.profession,
            "score": score,
            "stress_level": level,
            "responses": test_data.responses
        })
        
        return {
            "test_id": test_id,
            "stress_level": level,
            "score": score,
            **analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/{test_id}", response_model=GPTAnalysis)
async def analyze_existing(test_id: int, request: AnalysisRequest):
    """Endpoint para re-analizar tests existentes"""
    try:
        # Obtener datos de la base de datos
        conn = sqlite3.connect('../datos/pss_database.db')
        c = conn.cursor()
        
        c.execute('''SELECT age, profession, total_score, stress_level 
                     FROM tests WHERE test_id = ?''', (test_id,))
        test_data = c.fetchone()
        
        if not test_data:
            raise HTTPException(status_code=404, detail="Test no encontrado")
            
        c.execute('''SELECT question_number, original_value, processed_value
                     FROM responses WHERE test_id = ?''', (test_id,))
        responses = c.fetchall()
        conn.close()
        
        # Formatear datos para GPT
        data = {
            "test_id": test_id,
            "age": test_data[0],
            "profession": test_data[1],
            "score": test_data[2],
            "stress_level": test_data[3],
            "responses": [r[1] for r in sorted(responses, key=lambda x: x[0])],
            "additional_context": request.additional_context
        }
        
        # Analizar con GPT
        analysis = gpt_analyzer.analyze_with_gpt(data)
        return {
            "test_id": test_id,
            "stress_level": test_data[3],
            "score": test_data[2],
            **analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)