"""Ranking e seleção de candidatos por alvo (Fase B).

Combina as 3 métricas obrigatórias num score composto em [0,1] (maior = mais parecido com o
alvo), e ordena/seleciona os candidatos. Critério escolhido: média ponderada das "goodness".

Cada métrica é convertida para "goodness" (maior = melhor) em [0,1]:
    clip_sim -> clip_sim        (já é maior = melhor)
    lpips    -> 1 - lpips       (distância: menor = melhor)
    rmse     -> 1 - rmse        (distância: menor = melhor)
Pesos por omissão iguais (1/3 cada); são renormalizados, por isso não precisam de somar 1.
"""

DEFAULT_WEIGHTS = {"clip_sim": 0.4, "lpips": 0.3, "rmse": 0.3}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def composite_score(metrics: dict, weights: dict = DEFAULT_WEIGHTS) -> float:
    """Score composto em [0,1], maior = melhor. `metrics` tem clip_sim, lpips, rmse."""
    goodness = {
        "clip_sim": _clamp01(metrics["clip_sim"]),
        "lpips": _clamp01(1.0 - metrics["lpips"]),
        "rmse": _clamp01(1.0 - metrics["rmse"]),
    }
    total_w = sum(weights.get(k, 0.0) for k in goodness)
    if total_w == 0:
        return 0.0
    return sum(weights.get(k, 0.0) * g for k, g in goodness.items()) / total_w


def rank_candidates(candidates, weights: dict = DEFAULT_WEIGHTS, score_fn=composite_score) -> list:
    """Ordena candidatos (melhor primeiro). Cada candidato é um dict com as 3 métricas.

    Devolve novas cópias com `score` e `rank` (1 = melhor) anexados; não muta a entrada.
    """
    scored = []
    for cand in candidates:
        c = dict(cand)
        c["score"] = score_fn(cand, weights)
        scored.append(c)
    scored.sort(key=lambda c: c["score"], reverse=True)
    for i, c in enumerate(scored, start=1):
        c["rank"] = i
    return scored


def select_top_k(candidates, k: int = 3, weights: dict = DEFAULT_WEIGHTS,
                 score_fn=composite_score) -> list:
    """Ordena e devolve os k melhores (ou menos, se houver menos candidatos)."""
    return rank_candidates(candidates, weights=weights, score_fn=score_fn)[:k]
