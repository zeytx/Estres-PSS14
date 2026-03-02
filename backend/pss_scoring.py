"""
Módulo centralizado para el scoring del PSS-14 (Perceived Stress Scale).

Referencia: Cohen, S., Kamarck, T., & Mermelstein, R. (1983).
Ítems positivos (invertidos): 4, 5, 6, 7, 9, 10, 13 (1-based)
Ítems negativos (directos):   1, 2, 3, 8, 11, 12, 14 (1-based)

Inversión: valor_invertido = 4 - valor_original
Rango total: 0-56
"""

from typing import List, Tuple

# Índices 0-based de los ítems que se INVIERTEN (positivos)
# PSS-14 estándar: Q4, Q5, Q6, Q7, Q9, Q10, Q13
INVERTED_INDICES = [3, 4, 5, 6, 8, 9, 12]

# Índices 0-based de los ítems directos (negativos / indicadores de estrés)
DIRECT_INDICES = [0, 1, 2, 7, 10, 11, 13]

# Umbrales de nivel de estrés
STRESS_THRESHOLDS = {
    'low': 28,       # <= 28: Bajo
    'moderate': 42,  # <= 42: Moderado
    # > 42: Alto
}

NUM_QUESTIONS = 14
MAX_SCORE = 56
MIN_SCORE = 0


def process_responses(responses: List[int]) -> List[int]:
    """
    Procesa las respuestas invirtiendo los ítems positivos del PSS-14.

    Args:
        responses: Lista de 14 respuestas (valores 0-4)

    Returns:
        Lista de 14 respuestas procesadas

    Raises:
        ValueError: Si las respuestas no son válidas
    """
    if len(responses) != NUM_QUESTIONS:
        raise ValueError(f"Se requieren exactamente {NUM_QUESTIONS} respuestas, se recibieron {len(responses)}")

    if any(not isinstance(r, int) or r < 0 or r > 4 for r in responses):
        raise ValueError("Cada respuesta debe ser un entero entre 0 y 4")

    processed = []
    for i, resp in enumerate(responses):
        processed.append(4 - resp if i in INVERTED_INDICES else resp)
    return processed


def calculate_score(responses: List[int]) -> int:
    """
    Calcula el score total PSS-14 a partir de respuestas crudas.

    Args:
        responses: Lista de 14 respuestas originales (valores 0-4)

    Returns:
        Score total (0-56)
    """
    processed = process_responses(responses)
    return sum(processed)


def determine_stress_level(score: float) -> str:
    """
    Determina el nivel de estrés basado en el score.

    Args:
        score: Puntuación total (0-56)

    Returns:
        "Bajo", "Moderado" o "Alto"
    """
    score = round(score)
    if score <= STRESS_THRESHOLDS['low']:
        return "Bajo"
    elif score <= STRESS_THRESHOLDS['moderate']:
        return "Moderado"
    else:
        return "Alto"


def score_and_classify(responses: List[int]) -> Tuple[List[int], int, str]:
    """
    Procesa, calcula score y clasifica en un solo paso.

    Args:
        responses: Lista de 14 respuestas originales (valores 0-4)

    Returns:
        Tupla (respuestas_procesadas, score_total, nivel_estrés)
    """
    processed = process_responses(responses)
    total = sum(processed)
    level = determine_stress_level(total)
    return processed, total, level

