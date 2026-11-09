"""Tabla de lecturas y tabla de nódulos (contrato RD-02 v0) a partir de pylidc.

Unidad de análisis: el nódulo (agrupación de las lecturas de los radiólogos con `cluster_annotations`).
Etiquetas:
  R1 · ≥ min_lectores lecturas; mediana de malignidad < 3 → 0, > 3 → 1, = 3 → excluido.
  R2 · ≥ min_lectores lecturas; media < 3 → 0, > 3 → 1, = 3 → excluido.
  R3 · cualquier número de lecturas; media ≤ 3 → 0, > 3 → 1.
"""
from __future__ import annotations

import configparser
import os
from pathlib import Path

import numpy as np
import pandas as pd

# pylidc 0.2.3 usa los alias np.int / np.float / np.bool, eliminados en NumPy ≥ 1.24
for _n, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _n):
        setattr(np, _n, _t)

ATRIBUTOS = ["subtlety", "internalStructure", "calcification", "sphericity", "margin",
             "lobulation", "spiculation", "texture"]
COLUMNAS_LECTURA = ["ann_id", "scan_id", "patient_id", "series_uid", *ATRIBUTOS, "malignancy",
                    "diametro_mm", "volumen_mm3", "publicacion_inicial"]


def _configurar_pylidc(dicom_root: str) -> None:
    """Escribe ~/.pylidcrc solo si se indica una ruta (pylidc únicamente la usa para leer DICOM)."""
    if not dicom_root:
        return
    cfg = configparser.ConfigParser()
    cfg["dicom"] = {"path": str(dicom_root), "warn": "True"}
    nombre = "pylidc.conf" if os.name == "nt" else ".pylidcrc"
    with open(Path.home() / nombre, "w") as f:
        cfg.write(f)


def tabla_lecturas(dicom_root: str = "") -> pd.DataFrame:
    """Una fila por lectura de nódulo ≥ 3 mm: 9 valoraciones, diámetro y volumen estimados por pylidc."""
    _configurar_pylidc(dicom_root)
    import pylidc as pl

    scans = {s.id: s for s in pl.query(pl.Scan).all()}
    filas = []
    for a in pl.query(pl.Annotation).all():
        s = scans[a.scan_id]
        filas.append(dict(ann_id=a.id, scan_id=a.scan_id, patient_id=s.patient_id, series_uid=s.series_instance_uid,
                          **{c: int(getattr(a, c)) for c in ATRIBUTOS}, malignancy=int(a.malignancy),
                          diametro_mm=float(a.diameter), volumen_mm3=float(a.volume),
                          publicacion_inicial=bool(s.is_from_initial)))
    return pd.DataFrame(filas, columns=COLUMNAS_LECTURA)


def agrupar(dicom_root: str = "") -> pd.DataFrame:
    """Asignación lectura → nódulo con los parámetros por defecto de pylidc."""
    _configurar_pylidc(dicom_root)
    import pylidc as pl

    filas = []
    for s in pl.query(pl.Scan).order_by(pl.Scan.patient_id).all():
        if not s.annotations:
            continue
        for j, grupo in enumerate(s.cluster_annotations(verbose=False), 1):
            nid = f"{s.patient_id}_s{s.id}_n{j:02d}"
            filas += [dict(ann_id=a.id, nodule_id=nid) for a in grupo]
    return pd.DataFrame(filas)


def tabla_nodulos(lecturas: pd.DataFrame, asignacion: pd.DataFrame, agregacion_diametro: str = "mediana",
                  min_lectores: int = 3) -> pd.DataFrame:
    """Agrega las lecturas por nódulo y calcula las etiquetas R1, R2 y R3."""
    d = lecturas.merge(asignacion, on="ann_id", validate="one_to_one")
    agg_d = "median" if agregacion_diametro == "mediana" else "mean"
    g = d.groupby("nodule_id")
    nod = pd.DataFrame({
        "patient_id": g.patient_id.first(), "scan_id": g.scan_id.first(), "series_uid": g.series_uid.first(),
        "n_lectores": g.size(),
        "malignidad_mediana": g.malignancy.median(), "malignidad_media": g.malignancy.mean(),
        "diametro_mm": g.diametro_mm.agg(agg_d), "volumen_mm3": g.volumen_mm3.agg(agg_d),
        **{f"{c}_mediana": g[c].median() for c in ATRIBUTOS},
        "publicacion_inicial": g.publicacion_inicial.first(),
    }).reset_index()
    al = nod.n_lectores >= min_lectores
    nod["etiqueta_r1"] = np.where(al & (nod.malignidad_mediana < 3), 0.0, np.where(al & (nod.malignidad_mediana > 3), 1.0, np.nan))
    nod["etiqueta_r2"] = np.where(al & (nod.malignidad_media < 3), 0.0, np.where(al & (nod.malignidad_media > 3), 1.0, np.nan))
    nod["etiqueta_r3"] = (nod.malignidad_media > 3).astype(float)
    return nod


def construir_o_cargar(cfg: dict, cache_dir: Path) -> pd.DataFrame:
    """Tabla de nódulos con caché en resultados/datos (la primera vez tarda unos minutos)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    f_lect, f_asig = cache_dir / "lecturas_v0.csv", cache_dir / "asignacion_nodulos_v0.csv"
    root = cfg["datos"].get("dicom_root", "")
    if f_lect.exists():
        lect = pd.read_csv(f_lect)
    else:
        print("Leyendo lecturas de pylidc (solo la primera vez, unos 2–3 min)…")
        lect = tabla_lecturas(root); lect.to_csv(f_lect, index=False)
    if f_asig.exists():
        asig = pd.read_csv(f_asig)
    else:
        print("Agrupando lecturas en nódulos (solo la primera vez, ~1–2 min)…")
        asig = agrupar(root); asig.to_csv(f_asig, index=False)
    return tabla_nodulos(lect, asig, cfg["datos"]["agregacion_diametro"], cfg["datos"]["min_lectores"])
