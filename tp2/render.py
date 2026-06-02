"""Carregamento do pipeline LCM e render de prompts, com cache em disco.

Renders são caros (~55s na GTX 1650), por isso guardamos cada resultado em disco com chave
(prompt + seed + parâmetros fixos + dtype). Render repetido = leitura instantânea do cache.
"""
import hashlib

import torch
from diffusers import DiffusionPipeline

from .config import CACHE_DIR, LCMConfig, default_device, default_dtype
from .targets import load_image, seed_from_filename

_PIPE = None  # singleton carregado preguiçosamente


def load_pipeline(config: LCMConfig = LCMConfig(), device: str | None = None,
                  dtype: torch.dtype | None = None) -> DiffusionPipeline:
    """Carrega o pipeline LCM com as otimizações de memória para caber em ~4GB."""
    device = device or default_device()
    dtype = dtype or default_dtype(device)
    pipe = DiffusionPipeline.from_pretrained(
        config.model_id, torch_dtype=dtype, use_safetensors=True,
    )
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    pipe.to(device)
    # otimizações de memória: fazem o 768x768 caber nos 4GB
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def get_pipeline(**kwargs) -> DiffusionPipeline:
    """Devolve o pipeline singleton (carrega na 1ª chamada). kwargs só valem na 1ª vez."""
    global _PIPE
    if _PIPE is None:
        _PIPE = load_pipeline(**kwargs)
    return _PIPE


def _cache_key(prompt: str, seed: int, config: LCMConfig, dtype) -> str:
    payload = "|".join(str(x) for x in [
        config.model_id, prompt, seed, config.num_inference_steps, config.guidance_scale,
        config.lcm_origin_steps, config.width, config.height, str(dtype),
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def render_prompt(prompt: str, seed: int, *, pipe: DiffusionPipeline = None,
                  config: LCMConfig = LCMConfig(), use_cache: bool = True):
    """Renderiza `prompt` com `seed` usando os parâmetros fixos. Devolve uma PIL.Image RGB."""
    pipe = pipe or get_pipeline(config=config)
    seed = int(seed)
    key = _cache_key(prompt, seed, config, pipe.dtype)
    cache_path = CACHE_DIR / f"{key}.png"
    if use_cache and cache_path.exists():
        return load_image(cache_path)

    gen_device = "cpu" if pipe.device.type == "mps" else pipe.device.type
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            lcm_origin_steps=config.lcm_origin_steps,
            width=config.width,
            height=config.height,
            output_type="pil",
            generator=generator,
        ).images[0]

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        image.save(cache_path)
    return image


def render_prompt_for_target(prompt: str, target_path, **kwargs):
    """Como render_prompt, mas tira a seed do nome do ficheiro do alvo."""
    seed = seed_from_filename(target_path, LCMConfig().seed)
    return render_prompt(prompt, seed=seed, **kwargs)
