"""Configuração fixa do gerador LCM, dispositivo, dtype e caminhos do projeto."""
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class LCMConfig:
    """Contrato fixo de geração — tem de bater certo com o setup que gerou os alvos.

    A seed NÃO vem daqui: vem do nome do ficheiro do alvo (ver targets.seed_from_filename).
    O campo `seed` é só um fallback.
    """
    model_id: str = "SimianLuo/LCM_Dreamshaper_v7"
    seed: int = 2026
    num_inference_steps: int = 8
    guidance_scale: float = 8.0
    lcm_origin_steps: int = 50
    width: int = 768
    height: int = 768


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str) -> torch.dtype:
    """Escolhe o dtype do render.

    bfloat16 em CUDA: o float16 do notebook produz NaN/imagem preta na GTX série 16
    (bug de hardware fp16). O bf16 tem a mesma gama de expoente do fp32 → não estoura,
    e gasta metade da memória. Em GPUs sem o bug (ex.: Colab T4) podes forçar float16
    passando dtype explicitamente em load_pipeline (mais rápido lá).
    """
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


# Caminhos do projeto (raiz = pasta-pai de tp2/)
PROJECT_DIR = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_DIR / "tp2-chosen"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CACHE_DIR = PROJECT_DIR / ".cache" / "renders"
