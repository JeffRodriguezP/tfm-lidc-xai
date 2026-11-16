"""Cifras consolidadas del TFM: nódulos, pacientes y cortes por regla de etiquetado (R1, R2, R3).

CIFRAS FINALES (configuración por defecto, tras excluir los 14 clústeres con más de 4 lecturas):

    R1 (principal)    869 nódulos (448 baja / 421 alta)    530 pacientes     6.852 cortes
    R2              1.180 nódulos (662 / 518)               633 pacientes     8.416 cortes
    R3              2.637 nódulos (2.004 / 633)             870 pacientes    15.571 cortes

Cómo se llega a ellas, a partir de las anotaciones XML de LIDC-IDRI leídas con pylidc:
  1. lecturas      : cada anotación de nódulo ≥ 3 mm de cada radiólogo (6.859).
  2. nódulos       : las lecturas de distintos radiólogos que describen la misma lesión se agrupan con
                     `cluster_annotations` de pylidc (los XML no las enlazan): 2.651 clústeres.
  3. exclusión     : un nódulo no puede tener más de 4 lecturas (hay 4 radiólogos): se excluyen 14 clústeres.
  4. etiquetado    : R1 exige ≥ 3 lectores, usa la mediana de malignidad y excluye la mediana = 3.
  5. cortes        : unión, entre los lectores del nódulo, de los cortes con contorno
                     (`Annotation.contour_slice_indices`; no necesita los DICOM).

Escribe `resultados/datos/cortes_por_nodulo_v0.csv` y muestra la tabla por regla.

Uso:
    python scripts/contar_cortes.py --config configs/base.yaml               # cifras finales
    python scripts/contar_cortes.py --config configs/base.yaml --verificar   # además comprueba que coinciden
    python scripts/contar_cortes.py --config configs/base.yaml --sin-exclusion   # sensibilidad: R1 = 879 / 535 / 6.973
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tfm_lidc import config
from tfm_lidc.data import nodulos as dn

# Cifras finales documentadas en el TFM (Tabla 17): (nódulos, baja, alta, pacientes, cortes)
CIFRAS_FINALES = {"R1": (869, 448, 421, 530, 6852),
                  "R2": (1180, 662, 518, 633, 8416),
                  "R3": (2637, 2004, 633, 870, 15571)}


def cortes_por_nodulo(asignacion: pd.DataFrame, dicom_root: str = "") -> pd.DataFrame:
    """Número de cortes con contorno de cada nódulo (unión de los cortes de todos sus lectores)."""
    dn._configurar_pylidc(dicom_root)
    import pylidc as pl

    nodulo_de = dict(zip(asignacion.ann_id, asignacion.nodule_id))   # lectura → nódulo
    cortes: dict[str, set] = {}
    for a in pl.query(pl.Annotation).all():
        nid = nodulo_de.get(a.id)
        if nid is not None:
            # conjunto: si dos lectores marcan el mismo corte, cuenta una sola vez
            cortes.setdefault(nid, set()).update(int(i) for i in a.contour_slice_indices)
    return pd.DataFrame({"nodule_id": list(cortes), "n_cortes": [len(v) for v in cortes.values()]})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--sin-exclusion", action="store_true",
                   help="análisis de sensibilidad: conserva los clústeres con más de 4 lecturas")
    p.add_argument("--verificar", action="store_true", help="falla si las cifras no coinciden con las finales del TFM")
    a = p.parse_args()
    cfg = config.cargar(a.config)
    if a.sin_exclusion:
        cfg["datos"]["excluir_clusteres_mas_de_4"] = False
    cache = Path(cfg.get("rutas", {}).get("resultados", "resultados")) / "datos"

    # Tabla de nódulos (pasos 1 a 4): aquí se aplica la exclusión de clústeres y se calculan R1, R2 y R3
    nod = dn.construir_o_cargar(cfg, cache)

    # Cortes por nódulo (paso 5); se guarda en caché porque recorrer pylidc tarda unos minutos
    f = cache / "cortes_por_nodulo_v0.csv"
    if f.exists():
        cor = pd.read_csv(f)
    else:
        asig = pd.read_csv(cache / "asignacion_nodulos_v0.csv")
        cor = cortes_por_nodulo(asig, cfg["datos"].get("dicom_root", "")); cor.to_csv(f, index=False)
    nod = nod.merge(cor, on="nodule_id", how="left")

    # Una fila por regla: solo cuentan los nódulos que reciben etiqueta (los excluidos tienen NaN)
    filas = []
    for r in ("R1", "R2", "R3"):
        d = nod[nod[f"etiqueta_{r.lower()}"].notna()]
        e = d[f"etiqueta_{r.lower()}"]
        filas.append(dict(regla=r, nodulos=len(d), baja=int((e == 0).sum()), alta=int((e == 1).sum()),
                          pacientes=d.patient_id.nunique(), cortes=int(d.n_cortes.sum())))
    tabla = pd.DataFrame(filas)
    print(tabla.to_string(index=False))

    if a.verificar:
        obtenido = {r.regla: (r.nodulos, r.baja, r.alta, r.pacientes, r.cortes) for r in tabla.itertuples()}
        assert not a.sin_exclusion, "--verificar compara con las cifras finales, que incluyen la exclusión"
        assert obtenido == CIFRAS_FINALES, f"Las cifras no coinciden con las del TFM:\n{obtenido}\n{CIFRAS_FINALES}"
        print("✓ Coinciden con las cifras finales del TFM (R1: 869 nódulos, 530 pacientes, 6.852 cortes).")


if __name__ == "__main__":
    main()
