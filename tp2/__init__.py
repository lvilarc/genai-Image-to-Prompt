"""TP2 Image-to-Prompt — utilitários partilhados (fundação invariante do pipeline)."""
from .config import (
    CACHE_DIR,
    LCMConfig,
    OUTPUT_DIR,
    PROJECT_DIR,
    TARGET_DIR,
    default_device,
    default_dtype,
)
from .render import (
    get_pipeline,
    load_pipeline,
    render_prompt,
    render_prompt_for_target,
)
from .targets import (
    list_target_images,
    load_image,
    safe_stem,
    seed_from_filename,
)

__all__ = [
    "CACHE_DIR", "LCMConfig", "OUTPUT_DIR", "PROJECT_DIR", "TARGET_DIR",
    "default_device", "default_dtype",
    "get_pipeline", "load_pipeline", "render_prompt", "render_prompt_for_target",
    "list_target_images", "load_image", "safe_stem", "seed_from_filename",
]
