"""Pruebas de no fuga con datos sintéticos (se ejecutan siempre, también en la integración continua).

1. Ningún paciente en dos papeles (entrenamiento, validación, prueba) ni en dos pliegues de prueba.
2. Los pacientes con varias exploraciones quedan enteros en un mismo pliegue.
3. Nada se ajusta con la prueba: ni el escalado, ni el modelo, ni el umbral leen las etiquetas o las
   variables de los nódulos de prueba.
4. Control negativo: con etiquetas permutadas el AUC de prueba ronda 0,5.
"""
import numpy as np
import pandas as pd
import pytest

from tfm_lidc.data import particion as dp
from tfm_lidc.evaluation import controles
from tfm_lidc.lineas_base import ajustar_pliegue, predecir_fuera_de_pliegue

M_UNA = dict(variables=["x1"], estimador="logistica", parametros=dict(C=1e6, max_iter=1000))
M_TRES = dict(variables=["x1", "x2", "x3"], estimador="logistica", parametros=dict(max_iter=1000))


def _datos(n_pac=150, semilla=0, frac_multi=0.1):
    """Pacientes con 1–3 nódulos; un 10 % tiene dos exploraciones. La etiqueta depende de x1–x3."""
    rng = np.random.default_rng(semilla)
    filas = []
    for i in range(n_pac):
        n_scans = 2 if rng.random() < frac_multi else 1
        for s in range(n_scans):
            for j in range(rng.integers(1, 3)):
                x = rng.normal(size=3)
                y = int(x @ [1.5, 1.0, -0.8] + rng.normal(0, 0.8) > 0)
                filas.append(dict(nodule_id=f"P{i:03d}_s{s}_n{j}", patient_id=f"P{i:03d}", scan_id=f"P{i:03d}_s{s}",
                                  etiqueta_r1=float(y), x1=x[0], x2=x[1], x3=x[2]))
    nod = pd.DataFrame(filas)
    part = dp.generar(nod, "etiqueta_r1", n_pliegues=5, semilla=semilla)
    return nod, part.merge(nod.drop(columns="etiqueta_r1"), on=["nodule_id", "patient_id"])


# --- 1 y 2 · partición --------------------------------------------------------------------------------
def test_ningun_paciente_en_dos_papeles():
    _, d = _datos()
    for k in range(5):
        rol = d[f"rol_{k}"]
        conj = {r: set(d.patient_id[rol == r]) for r in ("entrenamiento", "validacion", "prueba")}
        assert not conj["entrenamiento"] & conj["prueba"]
        assert not conj["validacion"] & conj["prueba"]
        assert not conj["entrenamiento"] & conj["validacion"]
    assert d.groupby("patient_id").pliegue.nunique().max() == 1


def test_pacientes_con_varias_exploraciones_no_se_separan():
    _, d = _datos(frac_multi=0.3)
    multi = d.groupby("patient_id").scan_id.nunique()
    multi = multi[multi > 1].index
    assert len(multi) > 0
    assert d[d.patient_id.isin(multi)].groupby("patient_id").pliegue.nunique().max() == 1
    for k in range(5):
        assert d[d.patient_id.isin(multi)].groupby("patient_id")[f"rol_{k}"].nunique().max() == 1


def test_comprobar_detecta_un_paciente_partido():
    _, d = _datos()
    pac = d.patient_id.value_counts().index[0]
    fila = d.index[d.patient_id == pac][0]
    d.loc[fila, "pliegue"] = (d.loc[fila, "pliegue"] + 1) % 5
    with pytest.raises(AssertionError):
        dp.comprobar(d, 5)


# --- 3 · nada se ajusta con la prueba -------------------------------------------------------------------
@pytest.mark.parametrize("cfg_modelo", [M_UNA, M_TRES])
def test_umbral_y_predicciones_no_dependen_de_las_etiquetas_de_prueba(cfg_modelo):
    _, d = _datos()
    X, y, rol = d[cfg_modelo["variables"]], d.etiqueta.values, d["rol_0"].values
    _, u1, p1 = ajustar_pliegue(cfg_modelo, 0, X, y, rol)
    y2 = y.copy()
    te = rol == "prueba"
    y2[te] = 1 - y2[te]                         # se invierten todas las etiquetas de prueba
    _, u2, p2 = ajustar_pliegue(cfg_modelo, 0, X, y2, rol)
    assert u1 == u2 and np.array_equal(p1, p2)


def test_escalado_y_modelo_no_dependen_de_las_variables_de_prueba():
    _, d = _datos()
    X, y, rol = d[M_TRES["variables"]].copy(), d.etiqueta.values, d["rol_0"].values
    te = rol == "prueba"
    modelo, u1, _ = ajustar_pliegue(M_TRES, 0, X, y, rol)
    assert np.allclose(modelo.named_steps["escalado"].mean_, X[rol == "entrenamiento"].mean().values)
    X2 = X.copy()
    X2.loc[te, :] = 1e6                          # valores absurdos en la prueba
    modelo2, u2, _ = ajustar_pliegue(M_TRES, 0, X2, y, rol)
    assert u1 == u2 and np.allclose(modelo.named_steps["clf"].coef_, modelo2.named_steps["clf"].coef_)


# --- 4 · control negativo ---------------------------------------------------------------------------------
def test_control_negativo_ronda_0_5_y_el_modelo_real_no():
    _, d = _datos(n_pac=250)
    y = d.etiqueta.values
    real = predecir_fuera_de_pliegue(d, M_TRES, 0, 5)["probabilidad"]
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, real) > 0.8
    c = controles.control_negativo(lambda rng: predecir_fuera_de_pliegue(d, M_TRES, 0, 5, rng_permutacion=rng)["probabilidad"],
                                   y, n_permutaciones=40, semilla=1, tolerancia=0.08)
    assert c["superado"], c


def test_control_negativo_detecta_una_fuga_simulada():
    """Si la 'predicción' usa la etiqueta de prueba, el control debe fallar."""
    _, d = _datos()
    y = d.etiqueta.values
    c = controles.control_negativo(lambda rng: y + rng.normal(0, 0.1, len(y)), y, n_permutaciones=20)
    assert not c["superado"]


# --- 5 · preprocesado con variables categóricas (M1) ---------------------------------------------------
M_CAT = dict(variables=dict(numericas=["x1", "x2"], categoricas=["equipo"]), estimador="logistica",
             parametros=dict(max_iter=1000))


def _datos_cat(semilla=0):
    _, d = _datos(semilla=semilla)
    rng = np.random.default_rng(semilla)
    d["equipo"] = rng.choice(["A", "B", "C"], len(d))
    return d


def test_codificacion_solo_con_entrenamiento_y_categorias_nuevas():
    d = _datos_cat()
    rol = d["rol_0"].values.copy()
    te = rol == "prueba"
    d.loc[te, "equipo"] = "Z"                    # categoría que solo existe en la prueba
    modelo, _, p = ajustar_pliegue(M_CAT, 0, d[["x1", "x2", "equipo"]], d.etiqueta.values, rol)
    cats = modelo.named_steps["preprocesado"].named_transformers_["cat"].categories_[0].tolist()
    assert "Z" not in cats and set(cats) <= {"A", "B", "C"}
    assert np.isfinite(p).all()                  # la categoría desconocida no rompe la predicción


def test_categoricas_no_dependen_de_las_etiquetas_de_prueba():
    d = _datos_cat(1)
    X, y, rol = d[["x1", "x2", "equipo"]], d.etiqueta.values, d["rol_0"].values
    _, u1, p1 = ajustar_pliegue(M_CAT, 0, X, y, rol)
    y2 = y.copy()
    y2[rol == "prueba"] = 1 - y2[rol == "prueba"]
    _, u2, p2 = ajustar_pliegue(M_CAT, 0, X, y2, rol)
    assert u1 == u2 and np.array_equal(p1, p2)


def test_p_valor_de_permutacion():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    senal = y + rng.normal(0, .5, 200)
    from sklearn.metrics import roc_auc_score
    c = controles.control_negativo(lambda r: r.permutation(senal), y, n_permutaciones=200, semilla=1,
                                   auc_observado=roc_auc_score(y, senal))
    assert c["p_valor"] < 0.01
    ruido = rng.normal(size=200)
    c2 = controles.control_negativo(lambda r: r.permutation(ruido), y, n_permutaciones=200, semilla=1,
                                    auc_observado=roc_auc_score(y, ruido))
    assert c2["p_valor"] > 0.01
