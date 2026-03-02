from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class TestResponse(BaseModel):
    age: int = Field(..., ge=12, le=120, description="Edad del usuario")
    profession: str = Field(..., min_length=2, description="Profesión del usuario")
    responses: List[int] = Field(..., description="14 respuestas del PSS-14 (valores 0-4)")
    free_text: Optional[str] = Field(None, max_length=2000, description="Descripción libre de cómo se siente el usuario")

    @field_validator('responses')
    @classmethod
    def validate_responses(cls, v):
        if len(v) != 14:
            raise ValueError('Se requieren exactamente 14 respuestas')
        if any(r < 0 or r > 4 for r in v):
            raise ValueError('Cada respuesta debe estar entre 0 y 4')
        return v

class AnalysisRequest(BaseModel):
    test_id: int
    additional_context: Optional[str] = None

class GPTAnalysis(BaseModel):
    test_id: int
    stress_level: str
    score: int
    professional_insight: str
    recommendations: List[str]
    patterns_detected: List[str]

class FreeTextAnalysis(BaseModel):
    """Modelo para análisis de texto libre del usuario"""
    text: str = Field(..., min_length=10, max_length=2000, description="Texto del usuario describiendo cómo se siente")
    test_id: Optional[int] = Field(None, description="ID del test asociado (opcional)")
