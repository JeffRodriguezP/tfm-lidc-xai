"""Réplica del experimento uno de Hancock y Magnan (2016, J. Med. Imag. 3(4) 044504, Tabla 4).

No es parte del protocolo del TFM: sirve para validar que la lectura de LIDC-IDRI con pylidc reproduce
la cohorte y los resultados publicados antes de aplicar el protocolo propio (nivel nódulo, partición por paciente).

Protocolo del artículo:
  * unidad: la anotación de cada lector (no el nódulo), cada una como muestra independiente;
  * se excluyen las exploraciones de la publicación inicial (posible inversión de escalas de espiculación y lobulación);
  * malignidad 1–2 → benigno, 4–5 → maligno, 3 → excluida  → 2.817 anotaciones de 530 exploraciones;
  * 75 % / 25 % aleatorio (sin agrupar por paciente), repetido N veces;
  * regresión logística y bosque aleatorio (100 árboles, profundidad 8), con 8 atributos o con 8 + diámetro + volumen;
  * exactitud y sensibilidad con umbral 0,5; AUC por la regla del trapecio con 101 umbrales.

Uso:
    python scripts/replica_hancock_magnan.py --config configs/base.yaml --repeticiones 1000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tfm_lidc import config
from tfm_lidc.data import nodulos as dn

A8 = dn.ATRIBUTOS
PUBLICADO = {("LR", "8"): (83.23, 0.8013, 0.9164), ("LR", "10"): (84.64, 0.7906, 0.9302),
             ("RF", "8"): (85.74, 0.8430, 0.9322), ("RF", "10"): (88.08, 0.8461, 0.9492)}


def auc_101(y, p):
    t = np.linspace(0, 1, 101)
    tpr = np.array([((p >= u) & (y == 1)).sum() for u in t]) / (y == 1).sum()
    fpr = np.array([((p >= u) & (y == 0)).sum() for u in t]) / (y == 0).sum()
    trap = getattr(np, "trapezoid", None) or np.trapz  # NumPy ≥ 2 / < 2
    return float(-trap(tpr, fpr))


def una_repeticion(d, y, i):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=i)
    out = {}
    for nombre, cols in (("8", A8), ("10", A8 + ["diametro_mm", "volumen_mm3"])):
        X = d[cols].values
        for mod, clf in (("LR", LogisticRegression(max_iter=5000)),
                         ("RF", RandomForestClassifier(100, max_depth=8, random_state=i, n_jobs=1))):
            p = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            yt = y[te]
            out[(mod, nombre)] = (((p >= .5) == yt).mean() * 100, ((p >= .5) & (yt == 1)).sum() / (yt == 1).sum(), auc_101(yt, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/base.yaml"))
    ap.add_argument("--repeticiones", type=int, default=1000)
    a = ap.parse_args()
    cfg = config.cargar(a.config)
    raiz = Path(__file__).resolve().parents[1]
    f = raiz / cfg["salidas"]["dir_resultados"] / "datos" / "lecturas_v0.csv"
    lect = pd.read_csv(f) if f.exists() else dn.tabla_lecturas(cfg["datos"].get("dicom_root", ""))
    d = lect[~lect.publicacion_inicial & (lect.malignancy != 3)].reset_index(drop=True)
    y = (d.malignancy > 3).astype(int).values
    print(f"Cohorte: {len(d)} anotaciones de {d.scan_id.nunique()} exploraciones; malignas {y.mean():.2%}  "
          f"(artículo: 2.817 de 530; 41,04 %)")
    R = Parallel(n_jobs=-1)(delayed(una_repeticion)(d, y, i) for i in range(a.repeticiones))
    filas = []
    for k, pub in PUBLICADO.items():
        v = np.array([r[k] for r in R])
        filas.append(dict(clasificador=k[0], entradas=k[1],
                          exactitud=f"{v[:, 0].mean():.2f} ({v[:, 0].std():.2f})", exactitud_articulo=pub[0],
                          TPR=round(v[:, 1].mean(), 4), TPR_articulo=pub[1],
                          AUC=f"{v[:, 2].mean():.4f} ({v[:, 2].std():.4f})", AUC_articulo=pub[2]))
    print(pd.DataFrame(filas).to_string(index=False))


if __name__ == "__main__":
    main()
