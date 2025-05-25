import csv
from datetime import datetime
import os

def save_to_csv(age, profession, responses, ml_score=None, ml_stress_level=None, csv_path='../datos/respuestas.csv'):
    """Guarda los datos del test en un archivo CSV"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Procesar respuestas
    processed_responses = []
    for i, response in enumerate(responses):
        processed = (4 - response) if i in [3,4,6,7,10,13] else response
        processed_responses.append(processed)
    
    total_score = sum(processed_responses)
    stress_level = "Bajo" if total_score <= 28 else "Moderado" if total_score <= 42 else "Alto"
    
    # Preparar datos para CSV
    data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'age': age,
        'profession': profession,
        'total_score': total_score,
        'stress_level': stress_level
    }

    # Añadir predicciones de ML si están disponibles
    if ml_score is not None:
        data['ml_score'] = ml_score
    if ml_stress_level is not None:
        data['ml_stress_level'] = ml_stress_level
    
    # Agregar respuestas individuales
    for i, (orig, proc) in enumerate(zip(responses, processed_responses)):
        data[f'q{i+1}_original'] = orig
        data[f'q{i+1}_processed'] = proc
    
    # Escribir en CSV
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)
    
    return total_score, stress_level