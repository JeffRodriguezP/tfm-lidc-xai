"""Agregación de las probabilidades de los cortes en una puntuación de nódulo (RF-M05; apartado 3.3.5.6).

La unidad de evaluación es el nódulo: todas las métricas principales se calculan sobre estas puntuaciones.
Media = opción principal; máximo y media recortada = análisis de sensibilidad.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

METODOS = ("media", "maximo", "media_recortada")


def _recortada(x: pd.Series, proporcion: float = 0.2) -> float:
    return float(stats.trim_mean(x, proporcion)) if len(x) >= 3 else float(x.mean())


def por_nodulo(nodule_id, prob, metodo: str = "media") -> pd.Series:
    """Puntuación de cada nódulo a partir de las de sus cortes (índice = nodule_id, en orden de aparición)."""
    s = pd.Series(np.asarray(prob, float), index=pd.Index(np.asarray(nodule_id), name="nodule_id"))
    g = s.groupby(level=0, sort=False)
    if metodo == "media":
        return g.mean()
    if metodo == "maximo":
        return g.max()
    if metodo == "media_recortada":
        return g.apply(_recortada)
    raise ValueError(f"Método de agregación desconocido: {metodo}")
