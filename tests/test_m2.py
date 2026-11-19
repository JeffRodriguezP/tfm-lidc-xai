"""Pruebas de M2 (modelos solo de atributos semánticos): la malignidad nunca es entrada (RF-D07), la moda de los
atributos nominales no usa datos de prueba y la imputación se ajusta solo con el entrenamiento."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfm_lidc import config
from tfm_lidc.data import nodulos as dn
from tfm_lidc.models import tabular

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs" / "modelos").glob("m2_*.yaml"))


def test_hay_configuraciones_de_m2():
    assert len(CONFIGS) >= 5


@pytest.mark.parametrize("ruta", CONFIGS, ids=lambda p: p.stem)
def test_la_malignidad_no_es_entrada(ruta):
    m = config.cargar(ruta)["modelo"]
    cols = tabular.columnas(m)
    assert not [c for c in cols if "malign" in c.lower()], f"La malignidad aparece entre las entradas: {cols}"
    assert all(c.split("_")[0] in {*dn.ATRIBUTOS, "diametro", "volumen"} for c in cols), cols
    assert m["imputar"] is True


def test_moda_con_empate_devuelve_nan():
    assert dn._moda(pd.Series([6, 6, 6])) == 6
    assert dn._moda(pd.Series([1, 1, 2])) == 1
    assert np.isnan(dn._moda(pd.Series([1, 2])))
    assert np.isnan(dn._moda(pd.Series([1, 1, 2, 2])))


def test_imputacion_con_estadisticos_del_entrenamiento():
    m = dict(variables=dict(numericas=["a"], categoricas=["c"]), estimador="logistica", imputar=True,
             parametros=dict(max_iter=500))
    tr = pd.DataFrame(dict(a=[1.0, 2, 3, 4, np.nan, 6], c=[6.0, 6, 6, 1, np.nan, 6]))
    te = pd.DataFrame(dict(a=[100.0, np.nan], c=[np.nan, 2.0]))          # valores de prueba extremos y faltantes
    pipe = tabular.construir(m, 0).fit(tr, [0, 1, 0, 1, 0, 1])
    pre = pipe.named_steps["preprocesado"]
    assert pre.named_transformers_["cat"].named_steps["imputar"].statistics_[0] == 6      # moda del entrenamiento
    assert pre.named_transformers_["num"].named_steps["imputar"].statistics_[0] == 3       # mediana del entrenamiento
    assert pipe.predict_proba(te).shape == (2, 2)                                           # faltantes y categoría nueva no rompen


@pytest.mark.parametrize("est,par", [("bosque", dict(n_estimators=20)), ("mlp", dict(hidden_layer_sizes=[8, 4], max_iter=300))])
def test_bosque_y_mlp_se_construyen_y_predicen(est, par):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(dict(a=rng.normal(size=80), c=rng.integers(1, 4, 80).astype(float)))
    y = (X.a + rng.normal(0, .3, 80) > 0).astype(int)
    m = dict(variables=dict(numericas=["a"], categoricas=["c"]), estimador=est, imputar=True, parametros=par)
    p = tabular.construir(m, 0).fit(X, y).predict_proba(X)[:, 1]
    assert p.shape == (80,) and ((p >= 0) & (p <= 1)).all()
