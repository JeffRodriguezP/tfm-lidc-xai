"""Evidencia sobre el lanzamiento inicial de LIDC (campo `is_from_initial` de pylidc).

Cuenta, con los datos locales: exploraciones y pacientes marcados, exploraciones con anotaciones, lecturas y nódulos
(R1) afectados, y reproduce la cohorte de Hancock y Magnan (2016) al excluir ese bloque. Describe además la espiculación
y la lobulación dentro y fuera del bloque. NO excluye nada del conjunto del TFM: la marca solo se usa como indicador.

Uso:  python scripts/auditar_publicacion_inicial.py --config configs/base.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tfm_lidc import config
from tfm_lidc.data import nodulos as dn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/base.yaml")
    a = p.parse_args()
    cfg = config.cargar(a.config)
    dn._configurar_pylidc(cfg["datos"].get("dicom_root", ""))
    import pylidc as pl

    scans = pl.query(pl.Scan).all()
    ini = [s for s in scans if s.is_from_initial]
    cache = Path(cfg.get("rutas", {}).get("resultados", "resultados")) / "datos"
    nod = dn.construir_o_cargar(cfg, cache)
    lect = pd.read_csv(cache / "lecturas_v0.csv")
    pac_ini = {s.patient_id for s in ini}
    multi = sum(1 for p_ in pac_ini if sum(s.patient_id == p_ for s in ini) > 1)
    hm = lect[~lect.publicacion_inicial & (lect.malignancy != 3)]
    r1 = nod[nod.etiqueta_r1.notna()]
    filas = [
        ("Exploraciones en pylidc", len(scans)),
        ("Exploraciones marcadas is_from_initial", len(ini)),
        ("Pacientes con exploraciones marcadas", len(pac_ini)),
        ("Pacientes con más de una exploración marcada", multi),
        ("Exploraciones marcadas con anotaciones ≥ 3 mm", sum(1 for s in ini if s.annotations)),
        ("Lecturas (anotaciones) en exploraciones marcadas", int(lect.publicacion_inicial.sum())),
        ("Lecturas totales", len(lect)),
        ("Nódulos (clústeres) en exploraciones marcadas", int(nod.publicacion_inicial.sum())),
        ("Nódulos con etiqueta R1 en exploraciones marcadas", int(r1.publicacion_inicial.sum())),
        ("Nódulos con etiqueta R1 totales", len(r1)),
        ("Hancock y Magnan: lecturas sin lanzamiento inicial y malignidad ≠ 3 (publicadas: 2.817)", len(hm)),
        ("Hancock y Magnan: exploraciones de esa cohorte (publicadas: 530)", hm.scan_id.nunique()),
    ]
    tab = pd.DataFrame(filas, columns=["magnitud", "valor"])
    print(tab.to_string(index=False))
    tab.to_csv(cache / "auditoria_publicacion_inicial.csv", index=False)
    print("\nEspiculación y lobulación por lectura (media, desviación, % con valor 5):")
    d = lect.groupby("publicacion_inicial")[["spiculation", "lobulation"]].agg(["mean", "std", lambda s: (s == 5).mean()])
    d.columns = [f"{c}_{m}" for c, m in zip(["spic"] * 3 + ["lob"] * 3, ["media", "desv", "pct5"] * 2)]
    print(d.round(3).to_string())


if __name__ == "__main__":
    main()
