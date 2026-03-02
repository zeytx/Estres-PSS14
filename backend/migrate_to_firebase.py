"""
Migracion unica: SQLite -> Firebase Firestore
Ejecutar una sola vez desde la carpeta backend/

Uso:
    cd backend
    python migrate_to_firebase.py
"""

import sqlite3
import os
import sys
import json

# Importar Firebase
import firebase_admin
from firebase_admin import credentials, firestore


def connect_firebase():
    """Inicializa Firebase usando el archivo firebase-key.json local"""
    if firebase_admin._apps:
        return firestore.client()

    cred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'firebase-key.json')
    if not os.path.exists(cred_path):
        print("ERROR: No se encontro: " + cred_path)
        sys.exit(1)

    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    print("OK Firebase conectado desde: " + cred_path)
    return firestore.client()


def migrate():
    """Migra todos los datos de SQLite a Firestore"""

    db_path = '../datos/pss_database.db'
    if not os.path.exists(db_path):
        print("ERROR: No se encontro la base de datos: " + db_path)
        return

    # Conectar a ambas bases
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    db = connect_firebase()

    # 1. Ver que tablas existen en SQLite
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row['name'] for row in cursor.fetchall()]
    print("\nTablas encontradas en SQLite: " + str(tables))

    # 2. Migrar TESTS
    cursor.execute("SELECT * FROM tests ORDER BY test_id")
    tests = cursor.fetchall()
    print("\nMigrando " + str(len(tests)) + " tests...")

    max_test_id = 0
    migrated_tests = 0

    for test in tests:
        test_dict = dict(test)
        test_id = test_dict['test_id']
        max_test_id = max(max_test_id, test_id)

        # Obtener las respuestas de este test
        cursor.execute(
            "SELECT question_number, original_value, processed_value FROM responses WHERE test_id = ? ORDER BY question_number",
            (test_id,)
        )
        responses = [dict(r) for r in cursor.fetchall()]

        # Construir documento Firestore
        doc = {
            'test_id': test_id,
            'age': test_dict['age'],
            'profession': test_dict['profession'],
            'total_score': test_dict['total_score'],
            'stress_level': test_dict['stress_level'],
            'free_text': test_dict.get('free_text'),
            'timestamp': test_dict.get('timestamp', ''),
            'ml_score': test_dict.get('ml_score'),
            'ml_stress_level': test_dict.get('ml_stress_level'),
            'gpt_analysis': test_dict.get('gpt_analysis'),
            'responses': responses
        }

        # Parsear gpt_analysis si es JSON string
        if doc['gpt_analysis'] and isinstance(doc['gpt_analysis'], str):
            try:
                doc['gpt_analysis'] = json.loads(doc['gpt_analysis'])
            except (json.JSONDecodeError, TypeError):
                pass

        # Guardar en Firestore usando test_id como document ID
        db.collection('tests').document(str(test_id)).set(doc)
        migrated_tests += 1

        if migrated_tests % 25 == 0:
            print("  " + str(migrated_tests) + "/" + str(len(tests)) + " tests migrados...")

    print("  OK " + str(migrated_tests) + "/" + str(len(tests)) + " tests migrados exitosamente")

    # 3. Crear contador para IDs auto-incrementales
    db.collection('_counters').document('tests').set({
        'next_id': max_test_id + 1
    })
    print("  OK Contador inicializado: next_id = " + str(max_test_id + 1))

    # 4. Migrar GPT analyses (si existe tabla separada)
    if 'gpt_analyses' in tables:
        cursor.execute("SELECT * FROM gpt_analyses")
        analyses = cursor.fetchall()
        print("\nMigrando " + str(len(analyses)) + " analisis GPT...")

        for analysis in analyses:
            a_dict = dict(analysis)
            test_id = str(a_dict.get('test_id', ''))
            if test_id:
                db.collection('tests').document(test_id).update({
                    'gpt_analysis_detail': a_dict
                })

        print("  OK " + str(len(analyses)) + " analisis GPT migrados")

    # 5. Verificacion
    print("\nVerificando migracion...")

    docs = list(db.collection('tests').limit(5).stream())
    print("  Documentos de ejemplo en Firestore: " + str(len(docs)))

    if docs:
        sample = docs[0].to_dict()
        print("  Ejemplo (test_id=" + str(sample.get('test_id')) + "):")
        print("     Edad: " + str(sample.get('age')))
        print("     Profesion: " + str(sample.get('profession')))
        print("     Score: " + str(sample.get('total_score')))
        print("     Nivel: " + str(sample.get('stress_level')))
        print("     Respuestas: " + str(len(sample.get('responses', []))) + " items")

    conn.close()

    print("\nMIGRACION COMPLETADA")
    print("   Tests migrados: " + str(migrated_tests))
    print("   Siguiente ID: " + str(max_test_id + 1))
    print("\n   Verifica en: https://console.firebase.google.com")
    print("   Proyecto: pss14-estres -> Firestore Database")


if __name__ == '__main__':
    print("=" * 50)
    print("  MIGRACION SQLite -> Firebase Firestore")
    print("=" * 50)
    migrate()

