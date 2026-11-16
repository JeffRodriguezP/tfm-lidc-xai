"""Partición por paciente (Tabla 19): 5 pliegues externos agrupados y estratificados por etiqueta,
y dentro de cada pliegue de entrenamiento un subconjunto de validación agrupado (~16 % de los nódulos).

La partición es común a todos los modelos: se genera una vez, se guarda con su huella y se reutiliza.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def generar(nodulos: pd.DataFrame, etiqueta: str, n_pliegues: int = 5, frac_validacion: float = 0.2,
            semilla: int = 42) -> pd.DataFrame:
    """Devuelve una fila por nódulo etiquetado con su pliegue de prueba y, para cada pliegue k,
    el papel que desempeña cuando k es el pliegue de prueba (columna `rol_k`: entrenamiento/validacion/prueba)."""
    d = nodulos.dropna(subset=[etiqueta]).reset_index(drop=True)
    y, g = d[etiqueta].astype(int).values, d.patient_id.values
    externo = StratifiedGroupKFold(n_splits=n_pliegues, shuffle=True, random_state=semilla)
    out = d[["nodule_id", "patient_id"]].copy()
    out["etiqueta"] = y
    out["pliegue"] = -1
    n_int = max(2, round(1 / frac_validacion))
    for k, (tr, te) in enumerate(externo.split(d, y, g)):
        out.loc[te, "pliegue"] = k
        interno = StratifiedGroupKFold(n_splits=n_int, shuffle=True, random_state=semilla + k)
        _, va = next(interno.split(tr, y[tr], g[tr]))
        rol = np.full(len(d), "entrenamiento", dtype=object)
        rol[te] = "prueba"
        rol[tr[va]] = "validacion"
        out[f"rol_{k}"] = rol
    comprobar(out, n_pliegues)
    return out


def comprobar(p: pd.DataFrame, n_pliegues: int) -> None:
    """Pruebas de no fuga: ningún paciente en dos subconjuntos de un mismo pliegue ni en dos pliegues de prueba."""
    assert (p.pliegue >= 0).all(), "Hay nódulos sin pliegue"
    assert p.groupby("patient_id").pliegue.nunique().max() == 1, "Un paciente aparece en más de un pliegue de prueba"
    for k in range(n_pliegues):
        roles = p.groupby("patient_id")[f"rol_{k}"].nunique()
        assert roles.max() == 1, f"Fuga en el pliegue {k}: un paciente comparte roles"


def huella(p: pd.DataFrame) -> str:
    return hashlib.sha256(p.sort_values("nodule_id").to_csv(index=False).encode()).hexdigest()[:16]


def generar_o_cargar(nodulos: pd.DataFrame, cfg: dict, raiz_repo: Path) -> pd.DataFrame:
    """Carga la partición versionada en `particiones/`; si no existe, la genera y pide subirla al repositorio.

    La partición NO debe regenerarse en cada máquina: con la misma semilla, StratifiedGroupKFold reparte
    distinto según la versión de scikit-learn (comprobado: 1.7.2 frente a 1.9.1 solo coinciden en el 20 %
    de los nódulos). El archivo solo contiene identificadores, así que puede versionarse sin problema (RF-E03).
    """
    import json
    import time

    import sklearn

    regla, cp = cfg["datos"]["regla"], cfg["particion"]
    carpeta = raiz_repo / cfg["particion"].get("dir", "particiones")
    sufijo = "" if cfg["datos"].get("excluir_clusteres_mas_de_4", True) else "_sin-exclusion"   # análisis de sensibilidad
    f = carpeta / f"particion_{regla}_s{cp['semilla']}_k{cp['n_pliegues']}{sufijo}.csv"
    if f.exists():
        p = pd.read_csv(f)
        comprobar(p, cp["n_pliegues"])
        esperados = set(nodulos.dropna(subset=[f"etiqueta_{regla.lower()}"]).nodule_id)
        assert set(p.nodule_id) == esperados, "La partición versionada no corresponde a la tabla de nódulos actual"
        return p
    carpeta.mkdir(parents=True, exist_ok=True)
    p = generar(nodulos, f"etiqueta_{regla.lower()}", cp["n_pliegues"], cp["frac_validacion"], cp["semilla"])
    p.to_csv(f, index=False)
    f.with_suffix(".json").write_text(json.dumps(dict(
        archivo=f.name, huella=huella(p), regla=regla, semilla=cp["semilla"], n_pliegues=cp["n_pliegues"],
        nodulos=int(len(p)), pacientes=int(p.patient_id.nunique()), sklearn=sklearn.__version__,
        generada=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    print(f"⚠ Partición nueva en {f}: súbela al repositorio para que todos los modelos y máquinas usen la misma.")
    return p
