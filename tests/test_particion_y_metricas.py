"""Pruebas con datos sintéticos (no necesitan LIDC-IDRI): se ejecutan en la integración continua."""
import numpy as np
import pandas as pd
import pytest

from tfm_lidc.data import particion as dp
from tfm_lidc.data.nodulos import tabla_nodulos
from tfm_lidc.evaluation import metricas


def _nodulos_sinteticos(n_pac=120, semilla=0):
    rng = np.random.default_rng(semilla)
    filas = []
    for i in range(n_pac):
        for j in range(rng.integers(1, 4)):
            y = rng.integers(0, 2)
            filas.append(dict(nodule_id=f"P{i:03d}_n{j}", patient_id=f"P{i:03d}", etiqueta_r1=float(y),
                              diametro_mm=rng.normal(8 + 8 * y, 3)))
    return pd.DataFrame(filas)


def test_particion_sin_fuga_de_pacientes():
    p = dp.generar(_nodulos_sinteticos(), "etiqueta_r1", n_pliegues=5, semilla=1)
    dp.comprobar(p, 5)  # lanza AssertionError si hay fuga
    assert set(p.pliegue) == set(range(5))
    for k in range(5):
        assert set(p[f"rol_{k}"]) == {"entrenamiento", "validacion", "prueba"}
        frac_val = (p[f"rol_{k}"] == "validacion").mean()
        assert 0.08 < frac_val < 0.25


def test_particion_detecta_fuga():
    p = dp.generar(_nodulos_sinteticos(), "etiqueta_r1", semilla=2)
    pac = p.patient_id.value_counts().index[0]          # un paciente con varios nódulos
    fila = p.index[p.patient_id == pac][0]
    p.loc[fila, "pliegue"] = (p.loc[fila, "pliegue"] + 1) % 5   # uno de sus nódulos cambia de pliegue
    with pytest.raises(AssertionError):
        dp.comprobar(p, 5)


def test_reglas_de_etiquetado():
    lect = pd.DataFrame(dict(ann_id=range(7), scan_id=1, patient_id="P", series_uid="S",
                             malignancy=[2, 2, 4, 5, 3, 3, 4], diametro_mm=[5, 6, 7, 8, 9, 9, 9], volumen_mm3=1.0,
                             publicacion_inicial=False, **{c: 1 for c in ["subtlety", "internalStructure", "calcification",
                             "sphericity", "margin", "lobulation", "spiculation", "texture"]}))
    asig = pd.DataFrame(dict(ann_id=range(7), nodule_id=["a"] * 4 + ["b"] * 3))
    nod = tabla_nodulos(lect, asig).set_index("nodule_id")
    # a: [2,2,4,5] mediana 3 → R1 excluido; media 3,25 → R2 alta
    assert np.isnan(nod.loc["a", "etiqueta_r1"]) and nod.loc["a", "etiqueta_r2"] == 1
    # b: [3,3,4] mediana 3 → R1 excluido; media 3,33 → R2 y R3 alta
    assert np.isnan(nod.loc["b", "etiqueta_r1"]) and nod.loc["b", "etiqueta_r3"] == 1
    assert nod.loc["a", "diametro_mm"] == 6.5


def test_bootstrap_contiene_valor_puntual():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 300), 0, 1)
    r = metricas.con_intervalos(y, p, (p > .5).astype(int), np.repeat(np.arange(100), 3), n_boot=300)
    assert r["auc"]["ic95"][0] <= r["auc"]["valor"] <= r["auc"]["ic95"][1]


def test_clusteres_con_mas_de_4_lecturas_quedan_sin_etiqueta():
    """Un nódulo no puede tener más lecturas que lectores (4): se marca y se excluye de R1, R2 y R3."""
    lect = pd.DataFrame(dict(ann_id=range(9), scan_id=1, patient_id="P", series_uid="s", malignancy=[5, 5, 5, 1, 1, 1, 1, 1, 5],
                             diametro_mm=10.0, volumen_mm3=100.0, publicacion_inicial=False,
                             **{c: 3 for c in ["subtlety", "internalStructure", "calcification", "sphericity", "margin",
                                               "lobulation", "spiculation", "texture"]}))
    asig = pd.DataFrame(dict(ann_id=range(9), nodule_id=["n1"] * 4 + ["n2"] * 5))   # n1: 4 lecturas; n2: 5 (anómalo)
    con = tabla_nodulos(lect, asig).set_index("nodule_id")
    assert list(con.cluster_anomalo) == [False, True]
    assert con.loc["n1", "etiqueta_r1"] == 1.0                     # mediana de [5, 5, 5, 1] = 5
    assert con.loc["n2", ["etiqueta_r1", "etiqueta_r2", "etiqueta_r3"]].isna().all()
    sin = tabla_nodulos(lect, asig, excluir_clusteres_mas_de_4=False).set_index("nodule_id")
    assert sin.loc["n2", "etiqueta_r1"] == 0.0                     # mediana de [1, 1, 1, 1, 5] = 1


def test_sin_publicacion_inicial_conserva_pliegues_y_no_hay_fuga():
    n = _nodulos_sinteticos()
    n["publicacion_inicial"] = n.patient_id.str[1:].astype(int) % 3 == 0   # un tercio de los pacientes
    p = dp.generar(n, "etiqueta_r1", n_pliegues=5, semilla=1)
    q = dp.sin_publicacion_inicial(p, n)
    quitados = set(n.loc[n.publicacion_inicial, "nodule_id"])
    assert set(q.nodule_id) == set(p.nodule_id) - quitados
    m = q.merge(p, on="nodule_id", suffixes=("", "_base"))
    assert (m.pliegue == m.pliegue_base).all()                      # mismo pliegue que en la partición común
    assert all((m[f"rol_{k}"] == m[f"rol_{k}_base"]).all() for k in range(5))
