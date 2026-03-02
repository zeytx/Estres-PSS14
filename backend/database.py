import re
import sqlite3
from datetime import datetime
import os
from pss_scoring import process_responses, calculate_score, determine_stress_level, score_and_classify


def clean_profession_input(profession):
    """Versión simplificada ya que las profesiones vienen predefinidas"""
    if not isinstance(profession, str):
        return 'unknown'
    
    # Solo normalización básica
    profession = profession.strip().title()
    
    return profession if len(profession) >= 2 else 'unknown'



def init_db(db_path='../datos/pss_database.db'):
    """Inicializa la base de datos"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()

        c.execute('''CREATE TABLE IF NOT EXISTS tests (
                   test_id INTEGER PRIMARY KEY AUTOINCREMENT,
                   age INTEGER NOT NULL,
                   profession TEXT NOT NULL,
                   total_score INTEGER NOT NULL,
                   stress_level TEXT NOT NULL,
                   free_text TEXT,
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

        if 'free_text' not in columns:
            c.execute('''ALTER TABLE tests ADD COLUMN free_text TEXT''')

        conn.commit()

def save_to_db(age, profession, responses, free_text=None, db_path='../datos/pss_database.db'):
    """Guarda los datos del test en la base de datos"""
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()

        # Procesar respuestas usando módulo centralizado
        processed, total, level = score_and_classify(responses)

        # Insertar test
        c.execute('''INSERT INTO tests (age, profession, total_score, stress_level, free_text)
                   VALUES (?, ?, ?, ?, ?)''', (age, profession, total, level, free_text))
        test_id = c.lastrowid

        # Insertar respuestas
        for i, (orig, proc) in enumerate(zip(responses, processed)):
            c.execute('''INSERT INTO responses 
                       (test_id, question_number, original_value, processed_value)
                       VALUES (?, ?, ?, ?)''',
                    (test_id, i+1, orig, proc))

        conn.commit()
    return test_id, total, level

def get_test_results(test_id, db_path='../datos/pss_database.db'):
    """Obtiene los resultados de un test específico"""
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()

        c.execute('''SELECT age, profession, total_score, stress_level, free_text
                   FROM tests WHERE test_id = ?''', (test_id,))
        result = c.fetchone()

    if not result:
        return None
    
    return {
        'age': result[0],
        'profession': result[1],
        'total_score': result[2],
        'stress_level': result[3],
        'free_text': result[4]
    }