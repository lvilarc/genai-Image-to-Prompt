"""Persistência dos resultados (Fase C): imagens organizadas + tabelas de métricas.

Estrutura de uma run:
    outputs/<timestamp>_<identity>/
        metrics.csv, metrics.json          (todos os candidatos, todos os alvos)
        <target_stem>/
            target.png                     (cópia do alvo)
            rank_1.png, rank_2.png, ...     (imagens geradas, ordenadas pelo rank)
            metrics.csv                     (só os candidatos deste alvo)

Mapeia diretamente no que o enunciado pede por alvo: alvo + N prompts + N imagens + tabela.
"""
import csv
import json
from datetime import datetime
from pathlib import Path

from PIL import Image

from .config import OUTPUT_DIR
from .targets import safe_stem, seed_from_filename

TABLE_COLUMNS = ["target_name", "render_seed", "rank", "score",
                 "clip_sim", "lpips", "rmse", "prompt", "image"]


def create_run_dir(base_dir=OUTPUT_DIR, identity: str = "run") -> Path:
    """Cria uma pasta de run com timestamp: outputs/<YYYYmmdd-HHMMSS>_<identity>/."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / f"{timestamp}_{identity}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_csv(path, rows, columns=None) -> None:
    """Escreve `rows` (lista de dicts) em CSV. Colunas em falta no dict são ignoradas."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_image(img, path) -> Path:
    path = Path(path)
    pil = img if isinstance(img, Image.Image) else Image.open(img)
    pil.convert("RGB").save(path)
    return path


def save_target_results(run_dir, target_path, candidates, save_target_copy: bool = True) -> list:
    """Guarda imagens + tabela de um alvo. `candidates` já vem ordenado (rank 1 = melhor).

    Cada candidato é um dict: prompt, image (PIL ou caminho), clip_sim, lpips, rmse, score, rank.
    Devolve as linhas (dicts) prontas para a tabela combinada da run.
    """
    target_path = Path(target_path)
    run_dir = Path(run_dir)
    tdir = run_dir / safe_stem(target_path)
    tdir.mkdir(parents=True, exist_ok=True)
    seed = seed_from_filename(target_path)

    if save_target_copy:
        _save_image(target_path, tdir / "target.png")

    rows = []
    for i, cand in enumerate(candidates, start=1):
        rank = cand.get("rank", i)
        img_path = _save_image(cand["image"], tdir / f"rank_{rank}.png")
        rows.append({
            "target_name": target_path.name,
            "render_seed": seed,
            "rank": rank,
            "score": round(float(cand.get("score", float("nan"))), 6),
            "clip_sim": round(float(cand["clip_sim"]), 6),
            "lpips": round(float(cand["lpips"]), 6),
            "rmse": round(float(cand["rmse"]), 6),
            "prompt": cand["prompt"],
            "image": str(img_path.relative_to(run_dir)),
        })
    write_csv(tdir / "metrics.csv", rows, columns=TABLE_COLUMNS)
    return rows


def save_run(run_dir, results, save_target_copy: bool = True) -> list:
    """Guarda todos os alvos e escreve a tabela combinada (csv + json).

    `results`: iterável de (target_path, candidates), com candidates já ordenados por rank.
    Devolve todas as linhas.
    """
    run_dir = Path(run_dir)
    all_rows = []
    for target_path, candidates in results:
        all_rows.extend(
            save_target_results(run_dir, target_path, candidates, save_target_copy=save_target_copy)
        )
    write_csv(run_dir / "metrics.csv", all_rows, columns=TABLE_COLUMNS)
    (run_dir / "metrics.json").write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
    return all_rows
