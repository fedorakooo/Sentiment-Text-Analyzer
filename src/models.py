from enum import StrEnum
from datetime import datetime

from pydantic import BaseModel, Field


class SentimentLabel(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class SentimentAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")


class SentimentResult(BaseModel):
    label: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str | None = None


class AnalysisResponse(BaseModel):
    text: str
    sentiment: SentimentResult
    cached: bool = False
    processing_time: float | None = None
    model_used: str | None = None
    timestamp: datetime


class ErrorResponse(BaseModel):
    error: str
    details: str | None = None
    code: int
