"""Controles de validez del protocolo (Tabla 19), independientes del tipo de modelo.

`control_negativo` recibe una función que entrena y predice fuera de pliegue con etiquetas permutadas,
así que sirve igual para M0–M2 (scikit-learn) que para M3–M7 (PyTorch): solo cambia la función.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score


def control_negativo(predecir_permutado: Callable[[np.random.Generator], np.ndarray], y: np.ndarray,
                     n_permutaciones: int = 100, semilla: int = 0, tolerancia: float = 0.05,
                     auc_observado: float | None = None) -> dict:
    """Entrena `n_permutaciones` veces con las etiquetas de entrenamiento permutadas y mide el AUC contra las
    etiquetas reales de prueba. Sin fuga, el AUC medio debe ser ≈ 0,5 y el intervalo 2,5–97,5 % debe contener 0,5.

    Con `auc_observado`, devuelve además el p-valor de la prueba de permutación: la proporción de AUC con etiquetas
    permutadas que igualan o superan al observado (con la corrección +1 de Phipson y Smyth). Es la prueba formal de
    que el modelo predice mejor que el azar; en M1 responde a «¿la adquisición predice la etiqueta?».

    `predecir_permutado(rng)` debe devolver una probabilidad fuera de pliegue por nódulo, en el orden de `y`,
    entrenando con etiquetas permutadas mediante `rng` (sin tocar las de prueba).
    """
    rng = np.random.default_rng(semilla)
    aucs = np.array([roc_auc_score(y, predecir_permutado(rng)) for _ in range(n_permutaciones)])
    media, lo, hi = float(aucs.mean()), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
    out = dict(n_permutaciones=n_permutaciones, auc_media=round(media, 4), auc_p2_5=round(lo, 4), auc_p97_5=round(hi, 4),
               tolerancia=tolerancia, superado=bool(abs(media - 0.5) <= tolerancia and lo <= 0.5 <= hi),
               aucs_permutados=np.round(aucs, 4).tolist())
    if auc_observado is not None:
        out["auc_observado"] = round(auc_observado, 4)
        out["p_valor"] = float((1 + np.sum(aucs >= auc_observado)) / (n_permutaciones + 1))
    return out
