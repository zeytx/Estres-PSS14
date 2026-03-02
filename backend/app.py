import json
import shutil
import sqlite3
from flask import Flask, request, jsonify, render_template, redirect, send_from_directory, url_for, session, abort
from itsdangerous import URLSafeTimedSerializer
import secrets
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

app = Flask(__name__,
            static_folder="../frontend/dist/assets",
            template_folder="../frontend/dist",
            static_url_path="/assets")
CORS(app)

# Registrar panel de administración
from admin import admin_bp
app.register_blueprint(admin_bp)

app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))  # Clave secreta para sesiones
token_serializer = URLSafeTimedSerializer(app.secret_key)


# Nuevas configuraciones
app.config['JSON_AS_ASCII'] = False  # Para caracteres especiales
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True  # Para debugging

os.makedirs('../datos', exist_ok=True)
os.makedirs('../modelos', exist_ok=True)
os.makedirs('../logs', exist_ok=True)
# Cargar modelo pre-entrenado (NO re-entrenar en cada inicio)
# El modelo se entrena localmente con: python -c "from ml_extension import StressPredictor; s=StressPredictor(); s.train_model(force_retrain=True)"
# y los archivos .joblib se suben al repo para que Render los use directamente.
try:
    if stress_predictor.model is not None:
        print("✅ Modelo pre-entrenado cargado exitosamente")
    else:
        print("⚠️ No se encontró modelo pre-entrenado. Intentando entrenar...")
        result = stress_predictor.train_model()
        if isinstance(result, dict) and result.get('status') == 'success':
            print(f"Modelo entrenado con MAE: {result.get('metrics', {}).get('mae', 'N/A')}")
        else:
            print(f"Entrenamiento del modelo: {result}")
except Exception as e:
    print(f"⚠️ Error cargando/entrenando modelo: {str(e)} - Las predicciones ML no estarán disponibles")



# Configuración de tokens estáticos
STATIC_TOKEN = None
STATIC_TOKEN_EXPIRATION = 3600  # 1 hora en segundos
token_timer = None  # Variable para controlar el timer

def init_static_token():
    global STATIC_TOKEN, token_timer
    STATIC_TOKEN = secrets.token_urlsafe(32)
    logger.info(f"Token estático inicial generado")
    token_timer = threading.Timer(STATIC_TOKEN_EXPIRATION, renew_static_token)
    token_timer.daemon = True  # ← ESTO ES CLAVE
    token_timer.start()

def renew_static_token():
    global STATIC_TOKEN, token_timer
    STATIC_TOKEN = secrets.token_urlsafe(32)
    logger.info(f"Token estático renovado (válido por {STATIC_TOKEN_EXPIRATION}s)")
    token_timer = threading.Timer(STATIC_TOKEN_EXPIRATION, renew_static_token)
    token_timer.daemon = True  # ← ESTO ES CLAVE
    token_timer.start()

def shutdown_tokens():
    """Función para cancelar timers al cerrar la aplicación"""
    global token_timer
    if token_timer:
        token_timer.cancel()
        logger.info("Timer de tokens cancelado")

# Llamar al inicio de la aplicación
init_static_token()



# Ruta protegida para archivos estáticos
@app.context_processor
def inject_static_token():
    def generate_static_url(filename):
        return f'/protected-static/{filename}?token={STATIC_TOKEN}'
    return {'generate_static_url': generate_static_url}

# Generador de URLs protegidas
def get_protected_static_url(filename):
    return f'/protected-static/{filename}?token={STATIC_TOKEN}'

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

@app.route('/api/generate-new-token/<int:test_id>', methods=['GET'])
def generate_new_token(test_id):
    # AÑADIR ESTAS VALIDACIONES:
    logger.info(f"Solicitud de nuevo token para test_id: {test_id} desde IP: {request.remote_addr}")
    
    # 1. Verificar que el test existe
    test_data = get_test_results(test_id)
    if not test_data:
        logger.warning(f"Intento de generar token para test inexistente: {test_id} desde IP: {request.remote_addr}")
        return jsonify({
            'success': False,
            'error': 'Test no encontrado'
        }), 404
    
    # 2. Rate limiting básico (opcional)
    # Evitar spam de generación de tokens
    
    # 3. Validar origen de la petición
    referrer = request.headers.get('Referer')
    if not referrer or not any(allowed in referrer for allowed in ['localhost', '127.0.0.1', 'tu-dominio.com']):
        return jsonify({
            'success': False,
            'error': 'Origen no autorizado'
        }), 403
    
    # 4. Verificar que no hay un token reciente (opcional)
    # Para evitar generar tokens innecesarios
    
    try:
        # Generar nuevo token
        token = token_serializer.dumps({'test_id': test_id})
        
        # Log de seguridad
        logger.info(f"Nuevo token generado para test_id: {test_id}")
        
        return jsonify({
            'success': True,
            'new_token': token,
            'new_url': f'/results/{test_id}?token={token}',
            'expires_in': 86400  # 24 horas en segundos
        })
        
    except Exception as e:
        logger.error(f"Error generando token: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor'
        }), 500

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
gpt_analyzer = GPTStressAnalyzer(api_key=api_key, stress_predictor=stress_predictor)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path.startswith('api/') or path.startswith('protected-static/'):
        return abort(404)
        
    if path != "" and os.path.exists(os.path.join(app.template_folder, path)):
        return send_from_directory(app.template_folder, path)
        
    return render_template('index.html')

@app.route('/results/<int:test_id>')
def results(test_id):
    # Verificar el token
    token = request.args.get('token')
    if not token:
        logger.warning(f"Intento de acceso sin token a test_id: {test_id} desde IP: {request.remote_addr}")
        abort(403, description="Acceso no autorizado: token requerido")
    
    try:
        # Verificar el token (válido por 24 horas)
        token_data = token_serializer.loads(token, max_age=86400)
        if token_data['test_id'] != test_id:
            logger.warning(f"Token inválido para test_id: {test_id} desde IP: {request.remote_addr}")
            abort(403, description="Token inválido para este test")
    except Exception as e:
        # Log de token expirado o malformado
        logger.warning(f"Token inválido/expirado para test_id: {test_id} desde IP: {request.remote_addr} - Error: {str(e)}")
        abort(403, description="Token inválido o expirado")
    
    # Obtener datos del test
    test_data = get_test_results(test_id)
    analysis = {}
    try:
        # Solo verificamos que exista, el frontend hará el fetch completo
        test_data = get_test_results(test_id)
        if not test_data:
            return redirect('/')
    except Exception as e:
        logger.error(f"Error verificando test: {e}")

    # Servir la app de React y dejar que el router del frontend maneje la vista
    return serve('index.html')

@app.route('/api/test-results/<int:test_id>', methods=['GET'])
def get_test_results_api(test_id):
    # Verificar el token
    token = request.args.get('token')
    if not token:
        return jsonify({'error': 'Token requerido'}), 403
    
    try:
        token_data = token_serializer.loads(token, max_age=86400)
        if token_data['test_id'] != test_id:
            return jsonify({'error': 'Token inválido'}), 403
    except:
        return jsonify({'error': 'Token expirado o inválido'}), 403
    
    # Obtener datos del test
    test_data = get_test_results(test_id)
    if not test_data:
        return jsonify({'error': 'Test no encontrado'}), 404
    
    analysis = {}
    try:
        analysis = gpt_analyzer.analyze_test(test_id)
    except Exception as e:
        logger.error(f"Error cargando análisis GPT: {str(e)}")
        analysis = {"error": str(e)}

    # Combinar datos
    response_data = {
        **test_data,
        'analysis': analysis,
        # Asegurar que ml_score y ml_stress_level estén presentes si existen en test_data
        'ml_score': test_data.get('ml_score'),
        'ml_stress_level': test_data.get('ml_stress_level'),
        'ml_prediction': analysis.get('ml_prediction', {}) # En caso de que venga del análisis enriquecido
    }
    
    return jsonify(response_data)

@app.route('/api/submit-test', methods=['POST'])
def submit_test():
    # Log de inicio de envío
    logger.info(f"Nuevo test enviado desde IP: {request.remote_addr}")
    data = request.json
    
    try:
        # Validación básica
        if not data:
            return jsonify({
                'success': False,
                'error': 'No se recibieron datos'
            }), 400
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
        
        # Obtener texto libre opcional
        free_text = data.get('free_text', '').strip() or None

        # Guardar en base de datos
        init_db()
        test_id, total_score, stress_level = save_to_db(
            age=data['age'],
            profession=data['profession'],
            responses=data['responses'],
            free_text=free_text
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
            'free_text': free_text,  # Texto libre del usuario
            'timestamp': datetime.now().isoformat()
        }
        
        # Analizar con GPT en segundo plano
        def gpt_analysis_background():
            try:
                analysis = gpt_analyzer.analyze_with_gpt(gpt_input)
                
                # Guardar análisis completo con context manager
                with sqlite3.connect('../datos/pss_database.db') as conn:
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

            except Exception as e:
                print(f"Error en análisis GPT: {str(e)}")

        threading.Thread(target=gpt_analysis_background).start()
        # Generar token de un solo uso
        token = token_serializer.dumps({'test_id': test_id})
        
        return jsonify({
            'success': True,
            'test_id': test_id,
            'stress_level': stress_level,
            'ml_prediction': ml_prediction,  # Devolver también la predicción
            'redirect_url': f'/results/{test_id}?token={token}'
        })
        
    except Exception as e:
        print(f"Error en submit_test: {str(e)}")  # Para debugging
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    

# Proteger contra hotlinking
@app.before_request
def check_referrer():
    if request.path.startswith('/protected-static/'):
        referrer = request.headers.get("Referer")
        
        # Debug: mostrar información del referrer
        logger.debug(f"Acceso a {request.path} - Referrer: {referrer} - Host: {request.host}")
        
        # Permitir acceso si:
        # 1. No hay referrer (acceso directo)
        # 2. El referrer viene del mismo host
        # 3. El referrer viene de localhost/127.0.0.1
        if referrer:
            allowed_hosts = [
                request.host,  # Host actual de la petición
                'localhost:5000',
                '127.0.0.1:5000',
                'localhost',
                '127.0.0.1'
            ]
            
            # Verificar si el referrer viene de un host permitido
            referrer_valid = any(
                allowed_host in referrer 
                for allowed_host in allowed_hosts
            )
            
            if not referrer_valid:
                logger.warning(f"Referrer no válido: {referrer} - Host esperado: {request.host}")
                return "Acceso no autorizado", 403
        
        # Si llegamos aquí, el referrer es válido o no existe
        return None

# NUEVO ENDPOINT - Añadir aquí
@app.route('/protected-static/<filename>')
def protected_static(filename):
    """Servir archivos estáticos con protección por token mejorada"""
    # Verificar token
    token = request.args.get('token')
    if not token or token != STATIC_TOKEN:
        logger.warning(f"Token inválido para archivo estático: {filename} - IP: {request.remote_addr}")
        abort(403)
    
    # Lista blanca de archivos permitidos con tipos MIME
    allowed_files = {
        'app.js': 'application/javascript',
        'style.css': 'text/css'
    }
    
    if filename not in allowed_files:
        logger.warning(f"Archivo no permitido: {filename} - IP: {request.remote_addr}")
        abort(404)
    
    try:
        # Verificar que el archivo existe
        static_dir = os.path.join(app.root_path, 'static')
        file_path = os.path.join(static_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"Archivo no encontrado: {filename}")
            abort(404)
        
        # Servir con tipo MIME correcto
        response = send_from_directory('static', filename)
        response.headers['Content-Type'] = allowed_files[filename]
        response.headers['Cache-Control'] = 'private, max-age=3600'  # Cache por 1 hora
        
        logger.info(f"Archivo servido exitosamente: {filename} - IP: {request.remote_addr}")
        return response
        
    except Exception as e:
        logger.error(f"Error sirviendo archivo {filename}: {str(e)}")
        abort(500)
    
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

# Manejo global de errores
@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Error interno del servidor'
    }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint no encontrado'
    }), 404

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