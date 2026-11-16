"""Recuento de cortes con contorno por nódulo y cifras consolidadas por regla (R1, R2, R3).

Cortes = unión, entre los lectores del nódulo, de los índices de corte con contorno
(`Annotation.contour_slice_indices` de pylidc; no requiere DICOM). Escribe
`resultados/datos/cortes_por_nodulo_v0.csv` y muestra nódulos (baja / alta), pacientes y cortes
por regla. Con la configuración por defecto (excluye los clústeres con más de 4 lecturas) debe dar
R1 = 869 nódulos (448/421), 530 pacientes y 6.852 cortes.

Uso:
    python scripts/contar_cortes.py --config configs/base.yaml [--sin-exclusion]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tfm_lidc import config
from tfm_lidc.data import nodulos as dn


def cortes_por_nodulo(asignacion: pd.DataFrame, dicom_root: str = "") -> pd.DataFrame:
    dn._configurar_pylidc(dicom_root)
    import pylidc as pl

    nodulo_de = dict(zip(asignacion.ann_id, asignacion.nodule_id))
    cortes: dict[str, set] = {}
    for a in pl.query(pl.Annotation).all():
        nid = nodulo_de.get(a.id)
        if nid is not None:
            cortes.setdefault(nid, set()).update(int(i) for i in a.contour_slice_indices)
    return pd.DataFrame({"nodule_id": list(cortes), "n_cortes": [len(v) for v in cortes.values()]})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--sin-exclusion", action="store_true", help="análisis de sensibilidad (no excluye clústeres > 4 lecturas)")
    a = p.parse_args()
    cfg = config.cargar(a.config)
    if a.sin_exclusion:
        cfg["datos"]["excluir_clusteres_mas_de_4"] = False
    cache = Path(cfg.get("rutas", {}).get("resultados", "resultados")) / "datos"
    nod = dn.construir_o_cargar(cfg, cache)
    f = cache / "cortes_por_nodulo_v0.csv"
    if f.exists():
        cor = pd.read_csv(f)
    else:
        asig = pd.read_csv(cache / "asignacion_nodulos_v0.csv")
        cor = cortes_por_nodulo(asig, cfg["datos"].get("dicom_root", "")); cor.to_csv(f, index=False)
    nod = nod.merge(cor, on="nodule_id", how="left")
    filas = []
    for r in ("R1", "R2", "R3"):
        d = nod[nod[f"etiqueta_{r.lower()}"].notna()]
        e = d[f"etiqueta_{r.lower()}"]
        filas.append(dict(regla=r, nodulos=len(d), baja=int((e == 0).sum()), alta=int((e == 1).sum()),
                          pacientes=d.patient_id.nunique(), cortes=int(d.n_cortes.sum())))
    print(pd.DataFrame(filas).to_string(index=False))


if __name__ == "__main__":
    main()
