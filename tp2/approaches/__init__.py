"""Slot do approach (Fase E): a única caixa que muda entre abordagens.

Um approach é um callable `approach(ctx) -> list[str]`. Aqui ficam os baselines triviais
(para validar a pipeline) e o approach real (D3): seed (Gemini one-shot) + refinamento
iterativo por VLM-crítico (Gemini).
"""
from .base import constant_prompt, fixed_prompts
from .refine import GeminiCritic, refine_approach

__all__ = [
    "constant_prompt", "fixed_prompts",
    "GeminiCritic", "refine_approach",
]
