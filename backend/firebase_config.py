"""
Módulo de integración con Firebase Firestore.
Reemplaza SQLite para producción en la nube.

Uso:
- Localmente: Si no hay credenciales de Firebase, usa SQLite como fallback.
- Producción (Render): Usa Firestore con credenciales de servicio.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pss_scoring import score_and_classify

logger = logging.getLogger(__name__)

# Intentar inicializar Firebase
_firestore_client = None
USE_FIREBASE = False

try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    def _init_firebase():
        global _firestore_client, USE_FIREBASE

        if firebase_admin._apps:
            _firestore_client = firestore.client()
            USE_FIREBASE = True
            return

        # Opción 1: Credenciales desde variable de entorno (JSON string - para Render)
        cred_json = os.environ.get('FIREBASE_CREDENTIALS_JSON')
        if cred_json:
            try:
                cred_dict = json.loads(cred_json)
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                _firestore_client = firestore.client()
                USE_FIREBASE = True
                logger.info("Firebase inicializado desde variable de entorno")
                return
            except Exception as e:
                logger.warning(f"Error con credenciales JSON: {e}")

        # Opción 2: Archivo de credenciales local
        cred_path = os.environ.get('FIREBASE_CREDENTIALS', 'firebase-key.json')
        if os.path.exists(cred_path):
            try:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                _firestore_client = firestore.client()
                USE_FIREBASE = True
                logger.info(f"Firebase inicializado desde archivo: {cred_path}")
                return
            except Exception as e:
                logger.warning(f"Error con archivo de credenciales: {e}")

        logger.info("Firebase no configurado — usando SQLite como fallback")

    _init_firebase()

except ImportError:
    logger.info("firebase-admin no instalado — usando SQLite")


def get_db():
    """Retorna el cliente de Firestore si está disponible"""
    return _firestore_client


class FirestoreDatabase:
    """Capa de datos con Firestore como backend"""

    def __init__(self):
        self.db = get_db()
        if not self.db:
            raise RuntimeError("Firestore no está inicializado")

    def save_test(self, age: int, profession: str, responses: list,
                  free_text: Optional[str] = None) -> Tuple[str, int, str]:
        """Guarda un test completo en Firestore. Retorna (test_id, total_score, stress_level)"""

        processed, total, level = score_and_classify(responses)

        # Obtener siguiente ID numérico
        counter_ref = self.db.collection('_counters').document('tests')
        counter_doc = counter_ref.get()
        if counter_doc.exists:
            next_id = counter_doc.to_dict().get('next_id', 1)
        else:
            next_id = 1
        counter_ref.set({'next_id': next_id + 1})

        # Guardar test
        test_ref = self.db.collection('tests').document(str(next_id))
        test_data = {
            'test_id': next_id,
            'age': age,
            'profession': profession,
            'total_score': total,
            'stress_level': level,
            'free_text': free_text,
            'timestamp': datetime.now().isoformat(),
            'responses': []
        }

        # Guardar respuestas embebidas en el documento
        responses_data = []
        for i, (orig, proc) in enumerate(zip(responses, processed)):
            responses_data.append({
                'question_number': i + 1,
                'original_value': orig,
                'processed_value': proc
            })
        test_data['responses'] = responses_data

        test_ref.set(test_data)
        logger.info(f"Test {next_id} guardado en Firestore")

        return next_id, total, level

    def get_test_results(self, test_id: int) -> Optional[Dict]:
        """Obtiene los resultados de un test"""
        doc = self.db.collection('tests').document(str(test_id)).get()
        if not doc.exists:
            return None

        data = doc.to_dict()
        return {
            'age': data.get('age'),
            'profession': data.get('profession'),
            'total_score': data.get('total_score'),
            'stress_level': data.get('stress_level'),
            'free_text': data.get('free_text')
        }

    def get_test_full(self, test_id: int) -> Optional[Dict]:
        """Obtiene test completo con respuestas"""
        doc = self.db.collection('tests').document(str(test_id)).get()
        if not doc.exists:
            return None
        return doc.to_dict()

    def update_test(self, test_id: int, updates: Dict) -> None:
        """Actualiza campos de un test"""
        self.db.collection('tests').document(str(test_id)).update(updates)

    def get_all_tests(self, limit: int = 500) -> List[Dict]:
        """Obtiene todos los tests (para entrenamiento y trends)"""
        docs = (self.db.collection('tests')
                .order_by('timestamp', direction=firestore.Query.DESCENDING)
                .limit(limit)
                .stream())
        return [doc.to_dict() for doc in docs]

    def get_test_count(self) -> int:
        """Cuenta total de tests"""
        counter = self.db.collection('_counters').document('tests').get()
        if counter.exists:
            return counter.to_dict().get('next_id', 1) - 1
        return 0

    def save_gpt_analysis(self, test_id: int, analysis: Dict) -> None:
        """Guarda el análisis GPT en el documento del test"""
        self.db.collection('tests').document(str(test_id)).update({
            'gpt_analysis': analysis,
            'analysis_timestamp': datetime.now().isoformat()
        })

    # ========== ADMIN: Métricas y estadísticas ==========

    def get_admin_stats(self) -> Dict:
        """Obtiene estadísticas para el panel de admin"""
        tests = self.get_all_tests(limit=1000)

        if not tests:
            return {
                'total_tests': 0,
                'avg_score': 0,
                'stress_distribution': {},
                'top_professions': {},
                'tests_today': 0,
                'recent_tests': []
            }

        total = len(tests)
        scores = [t.get('total_score', 0) for t in tests]
        levels = [t.get('stress_level', 'Unknown') for t in tests]
        professions = [t.get('profession', 'Unknown') for t in tests]

        # Distribución de niveles
        level_counts = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1

        # Top profesiones
        prof_counts = {}
        for prof in professions:
            prof_counts[prof] = prof_counts.get(prof, 0) + 1
        top_profs = dict(sorted(prof_counts.items(), key=lambda x: -x[1])[:10])

        # Tests de hoy
        today = datetime.now().strftime('%Y-%m-%d')
        tests_today = sum(1 for t in tests if t.get('timestamp', '').startswith(today))

        # Últimos 10 tests
        recent = tests[:10]
        recent_formatted = [{
            'test_id': t.get('test_id'),
            'age': t.get('age'),
            'profession': t.get('profession'),
            'score': t.get('total_score'),
            'level': t.get('stress_level'),
            'timestamp': t.get('timestamp')
        } for t in recent]

        return {
            'total_tests': total,
            'avg_score': round(sum(scores) / total, 1) if total > 0 else 0,
            'stress_distribution': level_counts,
            'top_professions': top_profs,
            'tests_today': tests_today,
            'recent_tests': recent_formatted
        }


# ========== Capa de compatibilidad ==========
# Funciones que usa app.py — detectan automáticamente Firebase o SQLite

def init_db(db_path='../datos/pss_database.db'):
    """Inicializa la base de datos (compatible con ambos backends)"""
    if USE_FIREBASE:
        logger.info("Usando Firebase Firestore — no se necesita init_db")
        return

    # Fallback: SQLite original
    from database import init_db as sqlite_init_db
    sqlite_init_db(db_path)


def save_to_db(age, profession, responses, free_text=None, db_path='../datos/pss_database.db'):
    """Guarda test (compatible con ambos backends)"""
    if USE_FIREBASE:
        fb = FirestoreDatabase()
        return fb.save_test(age, profession, responses, free_text)

    from database import save_to_db as sqlite_save
    return sqlite_save(age, profession, responses, free_text, db_path)


def get_test_results(test_id, db_path='../datos/pss_database.db'):
    """Obtiene resultados (compatible con ambos backends)"""
    if USE_FIREBASE:
        fb = FirestoreDatabase()
        return fb.get_test_results(test_id)

    from database import get_test_results as sqlite_get
    return sqlite_get(test_id, db_path)

