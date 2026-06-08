"""Orquestrador (Fase E): liga toda a fundação para uma run completa.

Para cada alvo: pede prompts ao `approach` -> renderiza -> avalia -> ordena -> top-k.
No fim: persiste (Fase C) e agrega (Fase D).

Um *approach* é qualquer callable `approach(ctx) -> list[str]`, onde `ctx` (TargetContext) dá
acesso ao alvo, à seed, ao avaliador e a `ctx.render(prompt)`. É o único ponto que muda entre
abordagens — tudo o resto aqui é invariante.
"""
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .aggregate import write_summary
from .config import LCMConfig, TARGET_DIR
from .io_utils import create_run_dir, save_run
from .metrics import Evaluator
from .ranking import select_top_k
from .render import render_prompt
from .targets import list_target_images, load_image, seed_from_filename


@dataclass
class TargetContext:
    """Tudo o que um approach pode precisar para um alvo."""
    target_path: Path
    target_image: Image.Image
    seed: int
    evaluator: Evaluator
    config: LCMConfig

    def render(self, prompt: str) -> Image.Image:
        """Renderiza um prompt com a seed deste alvo (usa cache)."""
        return render_prompt(prompt, self.seed, config=self.config)


def _seed_everything(s: int) -> None:
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def evaluate_prompts_for_target(target_path, prompts, evaluator, config, k: int = 3) -> list:
    """Renderiza e avalia cada prompt contra o alvo, devolve os top-k candidatos ordenados."""
    target_path = Path(target_path)
    target_image = load_image(target_path)
    seed = seed_from_filename(target_path, config.seed)

    candidates, seen = [], set()
    for prompt in prompts:
        if prompt in seen:
            continue
        seen.add(prompt)
        image = render_prompt(prompt, seed, config=config)
        metrics = evaluator.evaluate(target_image, image)
        candidates.append({"prompt": prompt, "image": image, **metrics})
    return select_top_k(candidates, k=k)


def run_pipeline(approach, targets=None, *, k: int = 3, optimiser_seeds=(0,),
                 identity: str = "run", config: LCMConfig = None, evaluator: Evaluator = None):
    """Corre a pipeline completa para todos os alvos. Devolve (run_dir, rows, summary)."""
    config = config or LCMConfig()
    evaluator = evaluator or Evaluator()
    if targets is None:
        targets = list_target_images(TARGET_DIR)
    targets = [Path(t) for t in targets]

    results = []
    for target_path in targets:
        target_image = load_image(target_path)
        seed = seed_from_filename(target_path, config.seed)

        # recolhe prompts do approach (repetindo por seed de otimizador, se estocástico)
        prompts = []
        for s in optimiser_seeds:
            _seed_everything(s)
            ctx = TargetContext(target_path, target_image, seed, evaluator, config)
            prompts.extend(approach(ctx))

        top = evaluate_prompts_for_target(target_path, prompts, evaluator, config, k=k)
        results.append((target_path, top))

    run_dir = create_run_dir(identity=identity)
    rows = save_run(run_dir, results)
    summary = write_summary(run_dir, rows)
    return run_dir, rows, summary
