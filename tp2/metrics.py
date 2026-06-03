"""Avaliador: métricas imagem-imagem entre alvo e gerada.

As três métricas obrigatórias do enunciado, todas calculadas entre o alvo e a imagem renderizada:
- CLIP image-image similarity (cosseno de embeddings, encoder fixo)  -> MAIOR = mais parecido
- LPIPS perceptual distance (AlexNet)                                -> MENOR = mais parecido
- Pixel RMSE (em [0,1])                                              -> MENOR = mais parecido

Corre na CPU (estratégia de memória: a GPU fica para o render). Não combina as métricas num
critério único — isso é a Fase B.
"""
import numpy as np
import torch
from PIL import Image

from .targets import load_image

DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"


def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return load_image(img)  # aceita um caminho


def pixel_rmse(img_a, img_b) -> float:
    """RMSE pixel a pixel em [0,1]. Menor = mais parecido."""
    a, b = _to_pil(img_a), _to_pil(img_b)
    if a.size != b.size:
        b = b.resize(a.size, Image.BICUBIC)
    arr_a = np.asarray(a, dtype=np.float32) / 255.0
    arr_b = np.asarray(b, dtype=np.float32) / 255.0
    return float(np.sqrt(np.mean((arr_a - arr_b) ** 2)))


class Evaluator:
    """Calcula as 3 métricas. Carrega CLIP e LPIPS preguiçosamente, na 1ª utilização."""

    def __init__(self, clip_model: str = DEFAULT_CLIP_MODEL, device: str = "cpu"):
        self.device = device
        self.clip_model_name = clip_model
        self._clip = None
        self._clip_proc = None
        self._lpips = None

    def _ensure_clip(self):
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor
            self._clip = CLIPModel.from_pretrained(self.clip_model_name).to(self.device).eval()
            self._clip_proc = CLIPProcessor.from_pretrained(self.clip_model_name)

    def _ensure_lpips(self):
        if self._lpips is None:
            import lpips
            self._lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()

    def clip_similarity(self, img_a, img_b) -> float:
        """Cosseno entre os embeddings de imagem do CLIP. Maior = mais parecido."""
        self._ensure_clip()
        a, b = _to_pil(img_a), _to_pil(img_b)
        inputs = self._clip_proc(images=[a, b], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._clip.get_image_features(**inputs)
            # transformers >=5 devolve BaseModelOutputWithPooling cujo pooler_output já é o
            # embedding de imagem projetado (versões antigas devolvem o tensor diretamente).
            feats = out if isinstance(out, torch.Tensor) else out.pooler_output
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return float((feats[0] @ feats[1]).item())

    def _pil_to_lpips(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0          # [0,1] HWC
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1CHW
        return (t * 2 - 1).to(self.device)                       # [-1,1]

    def lpips_distance(self, img_a, img_b) -> float:
        """Distância perceptual LPIPS (AlexNet) à resolução nativa. Menor = mais parecido."""
        self._ensure_lpips()
        a, b = _to_pil(img_a), _to_pil(img_b)
        if a.size != b.size:
            b = b.resize(a.size, Image.BICUBIC)
        with torch.inference_mode():
            return float(self._lpips(self._pil_to_lpips(a), self._pil_to_lpips(b)).item())

    def pixel_rmse(self, img_a, img_b) -> float:
        return pixel_rmse(img_a, img_b)

    def evaluate(self, target, generated) -> dict:
        """Devolve as 3 métricas entre alvo e imagem gerada."""
        return {
            "clip_sim": self.clip_similarity(target, generated),
            "lpips": self.lpips_distance(target, generated),
            "rmse": self.pixel_rmse(target, generated),
        }
