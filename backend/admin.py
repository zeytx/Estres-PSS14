"""
Módulo de administración con autenticación.
Panel protegido para ver métricas del modelo, tests, y estadísticas.
"""

import os
import json
import sqlite3
import logging
import functools
import bcrypt
from datetime import datetime
from flask import Blueprint, request, jsonify, session
from dotenv import load_dotenv

load_dotenv('variable.env')
logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')

# ========== Credenciales ==========
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'change-this-password')

# Hash de la contraseña al iniciar (no se almacena en texto plano en memoria)
_password_hash = bcrypt.hashpw(ADMIN_PASSWORD.encode('utf-8'), bcrypt.gensalt())


def require_admin(f):
    """Decorador que protege rutas admin con autenticación por sesión"""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_authenticated'):
            return jsonify({'error': 'No autorizado', 'authenticated': False}), 401
        return f(*args, **kwargs)
    return decorated


# ========== Auth Endpoints ==========

@admin_bp.route('/login', methods=['POST'])
def admin_login():
    """Login del administrador"""
    data = request.json
    if not data:
        return jsonify({'error': 'Datos requeridos'}), 400

    username = data.get('username', '')
    password = data.get('password', '')

    if username == ADMIN_USERNAME and bcrypt.checkpw(password.encode('utf-8'), _password_hash):
        session['admin_authenticated'] = True
        session['admin_login_time'] = datetime.now().isoformat()
        logger.info(f"Admin login exitoso desde IP: {request.remote_addr}")
        return jsonify({
            'success': True,
            'message': 'Autenticación exitosa'
        })
    else:
        logger.warning(f"Admin login fallido desde IP: {request.remote_addr}")
        return jsonify({'error': 'Credenciales inválidas'}), 401


@admin_bp.route('/logout', methods=['POST'])
def admin_logout():
    """Logout del administrador"""
    session.pop('admin_authenticated', None)
    session.pop('admin_login_time', None)
    return jsonify({'success': True, 'message': 'Sesión cerrada'})


@admin_bp.route('/check', methods=['GET'])
def admin_check():
    """Verifica si el admin está autenticado"""
    if session.get('admin_authenticated'):
        return jsonify({
            'authenticated': True,
            'login_time': session.get('admin_login_time')
        })
    return jsonify({'authenticated': False}), 401


# ========== Dashboard Endpoints ==========

@admin_bp.route('/stats', methods=['GET'])
@require_admin
def get_stats():
    """Obtiene estadísticas generales para el dashboard"""
    try:
        from firebase_config import USE_FIREBASE

        if USE_FIREBASE:
            from firebase_config import FirestoreDatabase
            fb = FirestoreDatabase()
            stats = fb.get_admin_stats()
            stats['database'] = 'Firebase Firestore'
            return jsonify(stats)

        # Fallback: SQLite
        stats = _get_sqlite_stats()
        stats['database'] = 'SQLite (local)'
        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error obteniendo stats: {e}")
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/model-info', methods=['GET'])
@require_admin
def get_model_info():
    """Obtiene información del modelo ML entrenado"""
    try:
        from ml_extension import StressPredictor
        predictor = StressPredictor(db_path='../datos/pss_database.db')

        model_info = {
            'model_loaded': predictor.model is not None,
            'model_type': type(predictor.model).__name__ if predictor.model else 'No cargado',
            'last_training': predictor.last_training_time.isoformat() if predictor.last_training_time else 'Nunca',
            'features_count': len(predictor._get_engineered_feature_names()) + 16,  # engineered + base
        }

        # Feature importance
        if hasattr(predictor.model, 'feature_importances_') and predictor.preprocessor:
            try:
                importances = predictor.model.feature_importances_
                features = predictor.preprocessor.get_feature_names_out()
                importance_pairs = sorted(
                    zip(features, importances),
                    key=lambda x: -x[1]
                )
                model_info['feature_importance'] = [
                    {'feature': f, 'importance': round(float(imp), 4)}
                    for f, imp in importance_pairs[:15]
                ]
            except Exception:
                model_info['feature_importance'] = []

        # Métricas del último entrenamiento
        model_version_path = os.path.join(predictor.model_dir, 'model_versions.json')
        if os.path.exists(model_version_path):
            with open(model_version_path, 'r') as f:
                versions = json.load(f)
                if versions:
                    latest = versions[-1] if isinstance(versions, list) else versions
                    model_info['latest_metrics'] = latest.get('metrics', {})
                    model_info['model_version'] = latest.get('version', 'Unknown')

        # SHAP importance
        shap_path = os.path.join(predictor.model_dir, 'shap_importance.csv')
        if os.path.exists(shap_path):
            import pandas as pd
            shap_df = pd.read_csv(shap_path).head(10)
            model_info['shap_importance'] = shap_df.to_dict('records')

        # Session stats
        if hasattr(predictor, 'session_stats'):
            model_info['session_stats'] = predictor.session_stats

        return jsonify(model_info)

    except Exception as e:
        logger.error(f"Error obteniendo model info: {e}")
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/recent-tests', methods=['GET'])
@require_admin
def get_recent_tests():
    """Obtiene los últimos N tests con detalles"""
    try:
        limit = request.args.get('limit', 20, type=int)
        limit = min(limit, 100)  # Máximo 100

        from firebase_config import USE_FIREBASE

        if USE_FIREBASE:
            from firebase_config import FirestoreDatabase
            fb = FirestoreDatabase()
            tests = fb.get_all_tests(limit=limit)
            # Limpiar datos de respuestas para la vista
            for t in tests:
                if 'responses' in t:
                    t['responses_summary'] = len(t['responses'])
                    del t['responses']
            return jsonify({'tests': tests, 'count': len(tests)})

        # Fallback: SQLite
        tests = _get_sqlite_recent_tests(limit)
        return jsonify({'tests': tests, 'count': len(tests)})

    except Exception as e:
        logger.error(f"Error obteniendo recent tests: {e}")
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/retrain', methods=['POST'])
@require_admin
def retrain_model():
    """Reentrenar el modelo ML (admin only)"""
    try:
        from ml_extension import StressPredictor
        predictor = StressPredictor(db_path='../datos/pss_database.db')
        result = predictor.train_model(force_retrain=True)

        logger.info(f"Reentrenamiento ejecutado por admin: {result.get('status')}")
        return jsonify({
            'success': result.get('status') == 'success',
            'result': _make_json_serializable(result)
        })

    except Exception as e:
        logger.error(f"Error en reentrenamiento: {e}")
        return jsonify({'error': str(e)}), 500


# ========== Helpers SQLite ==========

def _get_sqlite_stats() -> dict:
    """Obtiene estadísticas desde SQLite"""
    db_path = '../datos/pss_database.db'
    if not os.path.exists(db_path):
        return {'total_tests': 0, 'avg_score': 0, 'stress_distribution': {},
                'top_professions': {}, 'tests_today': 0, 'recent_tests': []}

    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM tests")
        total = c.fetchone()[0]

        c.execute("SELECT AVG(total_score) FROM tests")
        avg = c.fetchone()[0] or 0

        c.execute("SELECT stress_level, COUNT(*) FROM tests GROUP BY stress_level")
        distribution = dict(c.fetchall())

        c.execute("SELECT profession, COUNT(*) FROM tests GROUP BY profession ORDER BY COUNT(*) DESC LIMIT 10")
        professions = dict(c.fetchall())

        today = datetime.now().strftime('%Y-%m-%d')
        c.execute("SELECT COUNT(*) FROM tests WHERE timestamp LIKE ?", (f"{today}%",))
        tests_today = c.fetchone()[0]

        c.execute("""SELECT test_id, age, profession, total_score, stress_level, timestamp 
                     FROM tests ORDER BY test_id DESC LIMIT 10""")
        recent = [{
            'test_id': r[0], 'age': r[1], 'profession': r[2],
            'score': r[3], 'level': r[4], 'timestamp': r[5]
        } for r in c.fetchall()]

    return {
        'total_tests': total,
        'avg_score': round(avg, 1),
        'stress_distribution': distribution,
        'top_professions': professions,
        'tests_today': tests_today,
        'recent_tests': recent
    }


def _get_sqlite_recent_tests(limit: int) -> list:
    """Obtiene tests recientes desde SQLite"""
    db_path = '../datos/pss_database.db'
    if not os.path.exists(db_path):
        return []

    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("""SELECT test_id, age, profession, total_score, stress_level, 
                     ml_score, ml_stress_level, timestamp
                     FROM tests ORDER BY test_id DESC LIMIT ?""", (limit,))
        return [{
            'test_id': r[0], 'age': r[1], 'profession': r[2],
            'total_score': r[3], 'stress_level': r[4],
            'ml_score': r[5], 'ml_stress_level': r[6],
            'timestamp': r[7]
        } for r in c.fetchall()]


def _make_json_serializable(obj):
    """Convierte objetos numpy/etc a tipos JSON serializables"""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy types
        return obj.item()
    elif hasattr(obj, '__float__'):
        return float(obj)
    return obj

