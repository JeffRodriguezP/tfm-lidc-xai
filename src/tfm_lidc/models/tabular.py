"""Modelos tabulares M0–M2. Cada uno se construye a partir de su configuración."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def construir(cfg_modelo: dict, semilla: int) -> Pipeline:
    """Pipeline sin estado previo: la estandarización se ajusta solo con el entrenamiento de cada pliegue."""
    est, par = cfg_modelo["estimador"], dict(cfg_modelo.get("parametros", {}))
    if est == "logistica":
        clf = LogisticRegression(random_state=semilla, **par)
    elif est == "bosque":
        clf = RandomForestClassifier(random_state=semilla, n_jobs=-1, **par)
    else:
        raise ValueError(f"Estimador no soportado: {est}")
    return Pipeline([("escalado", StandardScaler()), ("clf", clf)])
