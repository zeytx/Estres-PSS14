import sqlite3
from datetime import datetime
import os

def init_db(db_path='../datos/pss_database.db'):
    """Inicializa la base de datos"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS tests (
               test_id INTEGER PRIMARY KEY AUTOINCREMENT,
               age INTEGER NOT NULL,
               profession TEXT NOT NULL,
               total_score INTEGER NOT NULL,
               stress_level TEXT NOT NULL,
               timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS responses (
               response_id INTEGER PRIMARY KEY AUTOINCREMENT,
               test_id INTEGER NOT NULL,
               question_number INTEGER NOT NULL,
               original_value INTEGER NOT NULL,
               processed_value INTEGER NOT NULL,
               FOREIGN KEY(test_id) REFERENCES tests(test_id))''')
    
    # Verificar si la columna ya existe
    c.execute('''PRAGMA table_info(tests)''')
    columns = [col[1] for col in c.fetchall()]

    # Añadir columnas para ML si no existen
    if 'ml_score' not in columns:
        c.execute('''ALTER TABLE tests ADD COLUMN ml_score REAL''')
    
    if 'ml_stress_level' not in columns:
        c.execute('''ALTER TABLE tests ADD COLUMN ml_stress_level TEXT''')
    
    if 'gpt_analysis' not in columns:
        c.execute('''ALTER TABLE tests ADD COLUMN gpt_analysis TEXT''')
    
    conn.commit()
    conn.close()

def save_to_db(age, profession, responses, db_path='../datos/pss_database.db'):
    """Guarda los datos del test en la base de datos"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Procesar respuestas
    processed = []
    for i, resp in enumerate(responses):
        processed.append(4 - resp if i in [3,4,6,7,10,13] else resp)
    
    total = sum(processed)
    level = "Bajo" if total <= 28 else "Moderado" if total <= 42 else "Alto"
    
    # Insertar test
    c.execute('''INSERT INTO tests (age, profession, total_score, stress_level)
               VALUES (?, ?, ?, ?)''', (age, profession, total, level))
    test_id = c.lastrowid
    
    # Insertar respuestas
    for i, (orig, proc) in enumerate(zip(responses, processed)):
        c.execute('''INSERT INTO responses 
                   (test_id, question_number, original_value, processed_value)
                   VALUES (?, ?, ?, ?)''',
                (test_id, i+1, orig, proc))
    
    conn.commit()
    conn.close()
    return test_id, total, level

def get_test_results(test_id, db_path='../datos/pss_database.db'):
    """Obtiene los resultados de un test específico"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''SELECT age, profession, total_score, stress_level 
               FROM tests WHERE test_id = ?''', (test_id,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        return None
    
    return {
        'age': result[0],
        'profession': result[1],
        'total_score': result[2],
        'stress_level': result[3]
    }