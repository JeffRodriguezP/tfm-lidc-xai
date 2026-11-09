"""Métricas de clasificación con IC 95 % por bootstrap de pacientes (apartado 3.3.7.1)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def umbral_youden(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def puntuales(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict:
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    return dict(auc=roc_auc_score(y, p), auprc=average_precision_score(y, p), brier=brier_score_loss(y, p),
                exactitud=(tp + tn) / len(y), sensibilidad=tp / (tp + fn), especificidad=tn / (tn + fp))


def con_intervalos(y, p, pred, grupos, n_boot: int = 2000, semilla: int = 42) -> dict:
    """Bootstrap por paciente: se remuestrean pacientes (con todos sus nódulos), no nódulos."""
    y, p, pred, grupos = map(np.asarray, (y, p, pred, grupos))
    base = puntuales(y, p, pred)
    pac = np.unique(grupos)
    idx = {g: np.flatnonzero(grupos == g) for g in pac}
    rng = np.random.default_rng(semilla)
    muestras = {k: [] for k in base}
    for _ in range(n_boot):
        s = np.concatenate([idx[g] for g in rng.choice(pac, len(pac))])
        if len(np.unique(y[s])) < 2:
            continue
        for k, v in puntuales(y[s], p[s], pred[s]).items():
            muestras[k].append(v)
    return {k: dict(valor=round(float(v), 4),
                    ic95=[round(float(np.percentile(muestras[k], 2.5)), 4), round(float(np.percentile(muestras[k], 97.5)), 4)])
            for k, v in base.items()}
