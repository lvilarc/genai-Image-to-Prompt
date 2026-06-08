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
from .aggregate import aggregate_metrics, load_rows, summarize_run, write_summary
from .approaches import (
    GeminiCritic,
    constant_prompt,
    fixed_prompts,
    refine_approach,
)
from .io_utils import create_run_dir, save_run, save_target_results, write_csv
from .orchestrator import (
    TargetContext,
    evaluate_prompts_for_target,
    run_pipeline,
)
from .metrics import Evaluator, pixel_rmse
from .ranking import composite_score, rank_candidates, select_top_k
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
    "Evaluator", "pixel_rmse",
    "composite_score", "rank_candidates", "select_top_k",
    "create_run_dir", "save_run", "save_target_results", "write_csv",
    "aggregate_metrics", "load_rows", "summarize_run", "write_summary",
    "run_pipeline", "TargetContext", "evaluate_prompts_for_target",
    "constant_prompt", "fixed_prompts",
    "GeminiCritic", "refine_approach",
]
