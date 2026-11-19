"""Modelos tabulares M0–M2. Cada uno se construye a partir de su configuración."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def columnas(cfg_modelo: dict) -> list[str]:
    """Columnas de entrada del modelo. `variables` puede ser una lista (todas numéricas)
    o un diccionario con `numericas` y `categoricas`."""
    v = cfg_modelo["variables"]
    return list(v) if isinstance(v, list) else [*v.get("numericas", []), *v.get("categoricas", [])]


def construir(cfg_modelo: dict, semilla: int) -> Pipeline:
    """Pipeline sin estado previo: el escalado y la codificación se ajustan solo con el entrenamiento de cada pliegue."""
    est, par = cfg_modelo["estimador"], dict(cfg_modelo.get("parametros", {}))
    if est == "logistica":
        clf = LogisticRegression(random_state=semilla, **par)
    elif est == "bosque":
        clf = RandomForestClassifier(random_state=semilla, n_jobs=-1, **par)
    elif est == "mlp":
        if "hidden_layer_sizes" in par:
            par["hidden_layer_sizes"] = tuple(par["hidden_layer_sizes"])
        clf = MLPClassifier(random_state=semilla, **par)
    else:
        raise ValueError(f"Estimador no soportado: {est}")
    v = cfg_modelo["variables"]
    if isinstance(v, list):
        return Pipeline([("escalado", StandardScaler()), ("clf", clf)])
    codificador = OneHotEncoder(handle_unknown="ignore", sparse_output=False)   # categorías nuevas → ceros
    if cfg_modelo.get("imputar", False):
        # Imputación con estadísticos del entrenamiento (mediana / moda) + indicador de imputación (3.3.5.5)
        num = Pipeline([("imputar", SimpleImputer(strategy="median", add_indicator=True)), ("escalado", StandardScaler())])
        cat = Pipeline([("imputar", SimpleImputer(strategy="most_frequent", add_indicator=True)), ("codificar", codificador)])
    else:
        num, cat = StandardScaler(), codificador
    pre = ColumnTransformer([
        ("num", num, v.get("numericas", [])),
        ("cat", cat, v.get("categoricas", [])),
    ], verbose_feature_names_out=False)
    return Pipeline([("preprocesado", pre), ("clf", clf)])
