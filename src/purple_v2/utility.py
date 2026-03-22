from __future__ import annotations

import math
from typing import Dict, Iterable


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_probs(values: Dict[object, float], eps: float = 1e-12) -> Dict[object, float]:
    total = sum(max(v, eps) for v in values.values())
    if total <= 0:
        uniform = 1.0 / max(len(values), 1)
        return {k: uniform for k in values}
    return {k: max(v, eps) / total for k, v in values.items()}


def softmax(scores: Iterable[float], temperature: float = 1.0) -> list[float]:
    temperature = max(temperature, 1e-6)
    score_list = list(scores)
    pivot = max(score_list)
    exps = [math.exp((score - pivot) / temperature) for score in score_list]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(score_list)] * len(score_list)
    return [v / total for v in exps]


def entropy(probabilities: Iterable[float], eps: float = 1e-12) -> float:
    value = 0.0
    for p in probabilities:
        p = max(p, eps)
        value -= p * math.log(p)
    return value
