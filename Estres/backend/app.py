from http.client import responses
import json
import shutil
import sqlite3
from turtle import pd
from venv import logger
from flask import Flask, request, jsonify, render_template, redirect, send_from_directory, url_for
from flask_cors import CORS
from database import save_to_db, init_db, get_test_results
from csv_handler import save_to_csv
from gpt_integration import GPTStressAnalyzer
import os
import threading
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from ml_extension import StressPredictor
import uuid
import logging 


load_dotenv('variable.env')
stress_predictor = StressPredictor(db_path='../datos/pss_database.db')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

os.makedirs('../datos', exist_ok=True)
os.makedirs('../modelos', exist_ok=True)
os.makedirs('../logs', exist_ok=True)
# Entrenar el modelo cuando se inicia la aplicación
try:
    print("Entrenando modelo de predicción de estrés...")
    result = stress_predictor.train_model()
    if isinstance(result, dict) and 'status' in result and result['status'] == 'success':
        print(f"Modelo entrenado con MAE: {result.get('mae', 'N/A')}")
    else:
        print(f"Entrenamiento del modelo: {result}")
except Exception as e:
    print(f"Error entrenando modelo: {str(e)}")

# Añade este endpoint para hacer predicciones
@app.route('/api/predict-stress', methods=['POST'])
def predict_stress():
    """Predice el nivel de estrés con manejo robusto de errores"""
    data = request.json
    age = data.get('age')
    profession = data.get('profession')
    responses = data.get('responses')
    
    # Llama al predictor del modelo
    prediction = stress_predictor.predict_stress(
        age=age,
        profession=profession,
        responses=responses
    )
    
    return jsonify(prediction)

# Verificar que la API key está cargada
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ADVERTENCIA: No se encontró la API key de OpenAI en las variables de entorno")
    # Cargar directamente del archivo como alternativa
    with open('variable.env', 'r') as f:
        for line in f:
            if line.startswith('OPENAI_API_KEY='):
                api_key = line.strip().split('=', 1)[1]
                break


# Configuración GPT (usa variable de entorno)
gpt_analyzer = GPTStressAnalyzer(api_key=api_key)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results/<int:test_id>')
def results(test_id):
    # Obtener resultados básicos de la base de datos
    test_data = get_test_results(test_id)
    if not test_data:
        return redirect(url_for('index'))
    # Obtener análisis GPT (sincrónico para asegurar que tenemos los datos)
    try:
        analysis = gpt_analyzer.analyze_test(test_id)
    except Exception as e:
        analysis = {"error": str(e)}
    
    
    return render_template(
        'results.html',
        test_id=test_id,
        stress_level=test_data['stress_level'],
        age=test_data['age'],
        profession=test_data['profession'],
        analysis=analysis  # Pasar el análisis completo al template
    )

@app.route('/api/submit-test', methods=['POST'])
def submit_test():
    data = request.json
    
    try:
        # Validación básica
        if len(data['responses']) != 14:
            return jsonify({
                'success': False,
                'error': 'Se requieren exactamente 14 respuestas'
            }), 400
        
        # Primero hacer predicción con mi modelo
        ml_prediction = stress_predictor.predict_stress(
            age=data['age'],
            profession=data['profession'],
            responses=data['responses']
        )
        
        # Guardar en base de datos
        init_db()
        test_id, total_score, stress_level = save_to_db(
            age=data['age'],
            profession=data['profession'],
            responses=data['responses']
        )
        
        save_to_csv(
            age=data['age'],
            profession=data['profession'],
            responses=data['responses'],
            ml_score=ml_prediction['predicted_score'],
            ml_stress_level=ml_prediction['predicted_stress_level']
        )
        
        # Preparar datos para GPT
        gpt_input = {
            'test_id': test_id,
            'age': data['age'],
            'profession': data['profession'],
            'score': total_score,
            'stress_level': stress_level,
            'responses': data['responses'],
            'ml_prediction': ml_prediction,  # Incluir predicción del modelo
            'timestamp': datetime.now().isoformat()
        }
        
        # Analizar con GPT en segundo plano
        def gpt_analysis_background():
            try:
                analysis = gpt_analyzer.analyze_with_gpt(gpt_input)
                
                # Guardar análisis completo
                conn = sqlite3.connect('../datos/pss_database.db')
                c = conn.cursor()
                c.execute('''UPDATE tests SET 
                          gpt_analysis = ?,
                          ml_score = ?,
                          ml_stress_level = ?
                          WHERE test_id = ?''',
                         (json.dumps(analysis),
                          ml_prediction['predicted_score'],
                          ml_prediction['predicted_stress_level'],
                          test_id))
                conn.commit()
                conn.close()
                
            except Exception as e:
                print(f"Error en análisis GPT: {str(e)}")

        threading.Thread(target=gpt_analysis_background).start()
        
        return jsonify({
            'success': True,
            'test_id': test_id,
            'stress_level': stress_level,
            'ml_prediction': ml_prediction,  # Devolver también la predicción
            'redirect_url': f'/results/{test_id}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    

# Proteger contra hotlinking
@app.before_request
def check_referrer():
    if request.path.startswith('/protected-static/'):
        referrer = request.headers.get("Referer")
        if not referrer or not referrer.startswith(request.host_url):
            return "Acceso no autorizado", 403
    
@app.route('/api/get-analysis/<int:test_id>', methods=['GET'])
def get_analysis(test_id):
    try:
        analysis = gpt_analyzer.analyze_test(test_id)
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-trends', methods=['GET'])
def get_trends():
    profession = request.args.get('profession')
    trends = gpt_analyzer.get_stress_trends(profession)
    return jsonify(trends)

if __name__ == '__main__':
    try:
        from pyngrok import ngrok
        import webbrowser
        
        # Configurar puerto
        port = 5000
        
        # Configurar ngrok (obtener de variable.env)
        ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
        if ngrok_token:
            ngrok.set_auth_token(ngrok_token)
            print(" ✓ Token de ngrok configurado correctamente")
        else:
            print(" ⚠️ Token de ngrok no encontrado en variable.env")
            print(" ⚠️ La conexión puede tener restricciones sin un token de autenticación")
        
        # Iniciar túnel ngrok
        public_url = ngrok.connect(port).public_url
        
        print(f"\n ✅ Test de Estrés PSS-14 disponible públicamente en:")
        print(f" 🌐 {public_url}")
        print(f"\n 🖥️ Localmente disponible en: http://127.0.0.1:{port}")
        print("\n 📋 Comparte el enlace público para que otros puedan acceder al test")
        print("\n ⚠️ El enlace estará activo mientras este programa se ejecute")
        print(" ⚠️ El enlace cambiará cada vez que reinicies la aplicación")
        
        # Ejecutar la aplicación Flask
        app.run(host='0.0.0.0', port=port, debug=False)
    
    except ImportError:
        print("\n ❌ Pyngrok no está instalado. Instalalo con: pip install pyngrok")
        print(" 🔗 Ejecutando solo en modo local...")
        app.run(debug=True)
    except Exception as e:
        print(f"\n ❌ Error al iniciar ngrok: {str(e)}")
        print(" 🔗 Ejecutando solo en modo local...")
        app.run(debug=True)