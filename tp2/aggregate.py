"""Agregação das métricas no conjunto de alvos (Fase D).

O enunciado pede média e desvio-padrão das métricas no conjunto de teste. Agregamos:
- `best_per_target`: só o rank-1 de cada alvo (headline — a melhor reconstrução por alvo);
- `all_candidates`: todos os candidatos guardados (visão mais ampla).

Trabalha sobre as linhas produzidas pela Fase C (lista de dicts com clip_sim, lpips, rmse, score, rank).
"""
import json
import math
from pathlib import Path

import numpy as np

METRICS = ("clip_sim", "lpips", "rmse", "score")


def _coerce(value):
    try:
        v = float(value)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def aggregate_metrics(rows, metrics=METRICS, rank=None) -> dict:
    """Média/desvio (ddof=1) por métrica. Se `rank` for dado, filtra essas linhas (ex.: rank=1)."""
    rows = [r for r in rows if rank is None or int(r.get("rank", -1)) == rank]
    summary = {}
    for m in metrics:
        vals = [v for v in (_coerce(r.get(m)) for r in rows) if v is not None]
        if vals:
            arr = np.asarray(vals, dtype=float)
            summary[m] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "n": int(len(arr)),
            }
        else:
            summary[m] = {"mean": float("nan"), "std": float("nan"), "n": 0}
    return summary


def summarize_run(rows) -> dict:
    """Resumo da run: agregados do rank-1 por alvo e de todos os candidatos."""
    return {
        "best_per_target": aggregate_metrics(rows, rank=1),
        "all_candidates": aggregate_metrics(rows, rank=None),
    }


def load_rows(run_dir) -> list:
    """Lê as linhas de uma run a partir do metrics.json."""
    return json.loads((Path(run_dir) / "metrics.json").read_text())


def write_summary(run_dir, rows=None) -> dict:
    """Calcula e grava o resumo (summary.json + summary.csv) na pasta da run."""
    import csv

    run_dir = Path(run_dir)
    if rows is None:
        rows = load_rows(run_dir)
    summary = summarize_run(rows)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(run_dir / "summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "metric", "mean", "std", "n"])
        for group, agg in summary.items():
            for metric, stats in agg.items():
                writer.writerow([group, metric,
                                 round(stats["mean"], 6), round(stats["std"], 6), stats["n"]])
    return summary
