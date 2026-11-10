"""Las mismas comprobaciones sobre la partición y la tabla de nódulos reales.

Necesitan la descarga de LIDC-IDRI (o la caché de resultados/datos) y la partición versionada en particiones/.
En la integración continua no hay datos y se omiten: `pytest -m datos` las ejecuta solas en local.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfm_lidc import config
from tfm_lidc.data import nodulos as dn
from tfm_lidc.data import particion as dp
from tfm_lidc.lineas_base import ajustar_pliegue

RAIZ = Path(__file__).resolve().parents[1]
CFG = config.cargar(RAIZ / "configs/modelos/m0_diametro.yaml")
F_PART = RAIZ / CFG["particion"]["dir"] / f"particion_{CFG['datos']['regla']}_s{CFG['particion']['semilla']}_k{CFG['particion']['n_pliegues']}.csv"
CACHE = RAIZ / CFG["salidas"]["dir_resultados"] / "datos"

pytestmark = [pytest.mark.datos,
              pytest.mark.skipif(not (F_PART.exists() and (CACHE / "lecturas_v0.csv").exists()),
                                 reason="sin partición versionada o sin caché de lecturas (ejecutar antes M0)")]


@pytest.fixture(scope="module")
def d():
    nod = dn.construir_o_cargar(CFG, CACHE)
    part = dp.generar_o_cargar(nod, CFG, RAIZ)   # también comprueba que corresponde a la tabla de nódulos
    return part.merge(nod, on=["nodule_id", "patient_id"], validate="one_to_one")


def test_cifras_de_la_regla_r1(d):
    assert len(d) == 879 and d.patient_id.nunique() == 535 and int(d.etiqueta.sum()) == 430


def test_ningun_paciente_en_dos_papeles(d):
    for k in range(CFG["particion"]["n_pliegues"]):
        rol = d[f"rol_{k}"]
        conj = {r: set(d.patient_id[rol == r]) for r in ("entrenamiento", "validacion", "prueba")}
        assert not (conj["entrenamiento"] & conj["prueba"] or conj["validacion"] & conj["prueba"]
                    or conj["entrenamiento"] & conj["validacion"]), f"fuga en el pliegue {k}"


def test_pacientes_con_dos_tc_en_un_solo_pliegue(d):
    multi = d.groupby("patient_id").scan_id.nunique()
    multi = multi[multi > 1].index
    assert d[d.patient_id.isin(multi)].groupby("patient_id").pliegue.nunique().max() == 1


def test_m0_no_lee_la_prueba(d):
    m = CFG["modelo"]
    X, y, rol = d[m["variables"]], d.etiqueta.values, d["rol_0"].values
    _, u1, p1 = ajustar_pliegue(m, 42, X, y, rol)
    y2 = y.copy()
    y2[rol == "prueba"] = 1 - y2[rol == "prueba"]
    _, u2, p2 = ajustar_pliegue(m, 42, X, y2, rol)
    assert u1 == u2 and np.array_equal(p1, p2)


CFG_M1 = config.cargar(RAIZ / "configs/modelos/m1_metadatos.yaml")


@pytest.mark.skipif(not CFG_M1["datos"].get("metadata_csv"), reason="sin datos.metadata_csv en configs/local.yaml")
def test_metadatos_completos_para_m1():
    nod = dn.construir_o_cargar(CFG_M1, CACHE)
    d = nod.dropna(subset=["etiqueta_r1"])
    cols = ["grosor_mm", "px_mm", "espaciado_z_mm", "contraste", "fabricante"]
    assert not d[cols].isna().any().any(), "Hay nódulos de R1 sin metadatos de adquisición"
    assert set(d.fabricante) <= {"GE", "Siemens", "Philips", "Toshiba"}
