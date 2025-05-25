from pydantic import BaseModel
from typing import List

class TestResponse(BaseModel):
    age: int
    profession: str
    responses: List[int]

class AnalysisRequest(BaseModel):
    test_id: int
    additional_context: str = None

class GPTAnalysis(BaseModel):
    test_id: int
    stress_level: str
    score: int
    professional_insight: str
    recommendations: List[str]
    patterns_detected: List[str]