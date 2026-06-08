"""Slot do approach (Fase E): a única caixa que muda entre abordagens.

Um approach é um callable `approach(ctx) -> list[str]`. Aqui fica o approach real (D3):
seed (Gemini one-shot) + refinamento iterativo por VLM-crítico (Gemini).
"""
from .refine import GeminiCritic, refine_approach

__all__ = [
    "GeminiCritic", "refine_approach",
]
