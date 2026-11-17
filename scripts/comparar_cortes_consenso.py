"""Cortes por regla de etiquetado: unión de lectores frente a distintas definiciones de consenso (S2).

Aclara la diferencia entre dos cifras de «cortes» del TFM:
  * unión      : cortes donde dibujó contorno ALGÚN radiólogo del nódulo (la cifra de `contar_cortes.py`).
  * consenso   : cortes con máscara de consenso (política S2, ≥ 50 % de acuerdo entre lectores).
Como «consenso» admite varias definiciones, se calculan todas con las mismas anotaciones de pylidc
(no necesita los DICOM) para que el equipo fije una y la documente.

Entrada : resultados/datos/lecturas_v0.csv y asignacion_nodulos_v0.csv (los genera el pipeline de datos).
Salida  : resultados/datos/cortes_consenso_por_nodulo.csv y tabla por regla en pantalla.

Uso:  python scripts/comparar_cortes_consenso.py [--datos resultados/datos]
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Definiciones de «corte de consenso» (columna → significado)
DEFINICIONES = {
    "union": "contorno de algún lector (unión)",
    "mask50": "máscara pylidc.utils.consensus con clevel = 0,5 (≥ 50 % de los lectores por vóxel)",
    "mask50_5px": "igual, exigiendo ≥ 5 píxeles de máscara en el corte",
    "lect_ge_mitad": "contorno de al menos la mitad de los lectores en el corte (n/2 redondeado hacia arriba)",
    "lect_mayoria": "contorno de más de la mitad de los lectores",
    "todos": "contorno de todos los lectores",
}


def _alias_numpy() -> None:
    for nombre, tipo in (("int", int), ("float", float), ("bool", bool)):
        if not hasattr(np, nombre):
            setattr(np, nombre, tipo)


def tabla_nodulos(datos: Path) -> pd.DataFrame:
    lect = pd.read_csv(datos / "lecturas_v0.csv")
    asig = pd.read_csv(datos / "asignacion_nodulos_v0.csv")
    d = lect.merge(asig, on="ann_id", validate="one_to_one")
    g = d.groupby("nodule_id")
    nod = pd.DataFrame({"patient_id": g.patient_id.first(), "n": g.size(),
                        "mediana": g.malignancy.median(), "media": g.malignancy.mean()})
    nod["ann_ids"] = d.groupby("nodule_id").ann_id.apply(list)
    nod = nod[nod.n <= 4].copy()                      # RF-D02: se excluyen los clústeres con > 4 lecturas
    ok = nod.n >= 3
    nod["R1"] = ok & (nod.mediana != 3)
    nod["R2"] = ok & (nod.media != 3)
    nod["R3"] = True
    return nod


def cortes_por_nodulo(nod: pd.DataFrame) -> pd.DataFrame:
    _alias_numpy()
    import pylidc as pl
    from pylidc.utils import consensus

    db = {a.id: a for a in pl.query(pl.Annotation).all()}
    filas = []
    for nid, fila in nod.iterrows():
        anns = [db[a] for a in fila.ann_ids]
        n = len(anns)
        c = Counter(int(k) for a in anns for k in set(a.contour_slice_indices))
        cm, _, _ = consensus(anns, clevel=0.5, pad=[(0, 0)] * 3)
        px = cm.sum(axis=(0, 1))
        filas.append({"nodule_id": nid, "union": len(c),
                      "mask50": int((px >= 1).sum()), "mask50_5px": int((px >= 5).sum()),
                      "lect_ge_mitad": sum(v >= math.ceil(n / 2) for v in c.values()),
                      "lect_mayoria": sum(v > n / 2 for v in c.values()),
                      "todos": sum(v == n for v in c.values())})
    return pd.DataFrame(filas).set_index("nodule_id")


def resumen(nod: pd.DataFrame, cortes: pd.DataFrame) -> pd.DataFrame:
    x = nod.join(cortes, how="inner")
    filas = []
    for regla in ("R1", "R2", "R3"):
        s = x[x[regla]]
        filas.append({"regla": regla, "nódulos": len(s), "pacientes": s.patient_id.nunique(),
                      **{k: int(s[k].sum()) for k in DEFINICIONES}})
    return pd.DataFrame(filas)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datos", default="resultados/datos")
    a = p.parse_args()
    datos = Path(a.datos)
    nod = tabla_nodulos(datos)
    cortes = cortes_por_nodulo(nod)
    cortes.to_csv(datos / "cortes_consenso_por_nodulo.csv")
    r = resumen(nod, cortes)
    print(r.to_string(index=False))
    for k, v in DEFINICIONES.items():
        print(f"  {k:14s} {v}")


if __name__ == "__main__":
    main()
