"""Entrenamiento y evaluación de las líneas base tabulares (M0, y en adelante M1–M2).

Uso:
    python -m tfm_lidc.lineas_base --config configs/modelos/m0_diametro.yaml

Protocolo (Tabla 19): en cada uno de los 5 pliegues externos el modelo se ajusta con el subconjunto de
entrenamiento, el umbral de decisión se elige en la validación interna (Youden) y se predicen los nódulos
del pliegue de prueba. Las métricas se calculan sobre las predicciones fuera de muestra de los 5 pliegues,
con IC 95 % por bootstrap de pacientes.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from . import config
from .data import nodulos as dn
from .data import particion as dp
from .evaluation import controles, metricas
from sklearn.metrics import roc_auc_score
from .models import tabular


def _commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        sucio = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return sha + ("-modificado" if sucio else "")
    except Exception:
        return "sin-git"


def ajustar_pliegue(cfg_modelo: dict, semilla: int, X: pd.DataFrame, y: np.ndarray, rol: np.ndarray):
    """Un pliegue: ajusta con 'entrenamiento', elige el umbral con 'validacion' y predice 'prueba'.

    Las etiquetas de 'prueba' no se leen en ningún momento (lo verifica tests/test_fugas.py)."""
    tr, va, te = rol == "entrenamiento", rol == "validacion", rol == "prueba"
    modelo = tabular.construir(cfg_modelo, semilla).fit(X[tr], y[tr])
    umbral = metricas.umbral_youden(y[va], modelo.predict_proba(X[va])[:, 1])
    return modelo, umbral, modelo.predict_proba(X[te])[:, 1]


def predecir_fuera_de_pliegue(d: pd.DataFrame, cfg_modelo: dict, semilla: int, n_pliegues: int,
                              rng_permutacion: np.random.Generator | None = None) -> dict:
    """Predicción de cada nódulo con el modelo del pliegue en el que es prueba.

    Con `rng_permutacion`, las etiquetas de entrenamiento y validación se permutan en cada pliegue
    (control negativo de la Tabla 19); las de prueba no se tocan."""
    X, y = d[cfg_modelo["variables"]], d.etiqueta.values
    p, pred, umbrales, coefs = np.full(len(d), np.nan), np.full(len(d), -1), [], []
    for k in range(n_pliegues):
        rol = d[f"rol_{k}"].values
        te = rol == "prueba"
        y_aj = y.copy()
        if rng_permutacion is not None:
            y_aj[~te] = rng_permutacion.permutation(y[~te])
        modelo, u, p_te = ajustar_pliegue(cfg_modelo, semilla, X, y_aj, rol)
        p[te], pred[te] = p_te, (p_te >= u).astype(int)
        umbrales.append(round(float(u), 4))
        clf = modelo.named_steps["clf"]
        if hasattr(clf, "coef_"):
            coefs.append(dict(zip(cfg_modelo["variables"], np.round(clf.coef_[0], 4).tolist()),
                              intercepto=round(float(clf.intercept_[0]), 4)))
    assert not np.isnan(p).any(), "Hay nódulos sin predicción fuera de muestra"
    return dict(probabilidad=p, prediccion=pred, umbrales=umbrales, coeficientes=coefs)


def ejecutar(ruta_cfg: str, n_control: int = 0) -> dict:
    """`ruta_cfg` es configs/modelos/<modelo>.yaml; la raíz del repositorio está dos niveles por encima."""
    t0 = time.time()
    cfg = config.cargar(ruta_cfg)
    m, cp, regla = cfg["modelo"], cfg["particion"], cfg["datos"]["regla"]
    raiz = Path(ruta_cfg).resolve().parents[2]          # raíz del repositorio
    res_dir = Path(cfg["salidas"]["dir_resultados"])
    res_dir = res_dir if res_dir.is_absolute() else raiz / res_dir   # igual desde la terminal o desde un notebook
    datos_dir = res_dir / "datos"

    nod = dn.construir_o_cargar(cfg, datos_dir)
    part = dp.generar_o_cargar(nod, cfg, raiz)
    d = part.merge(nod, on=["nodule_id", "patient_id"], validate="one_to_one")
    X, y = d[m["variables"]], d.etiqueta.values

    oof = predecir_fuera_de_pliegue(d, m, cp["semilla"], cp["n_pliegues"])
    p_oof, pred_oof, umbrales, coefs = oof["probabilidad"], oof["prediccion"], oof["umbrales"], oof["coeficientes"]
    auc_por_pliegue = [round(float(roc_auc_score(y[d.pliegue == k], p_oof[d.pliegue == k])), 4) for k in range(cp["n_pliegues"])]

    control = None
    if n_control:
        control = controles.control_negativo(
            lambda rng: predecir_fuera_de_pliegue(d, m, cp["semilla"], cp["n_pliegues"], rng_permutacion=rng)["probabilidad"],
            y, n_permutaciones=n_control, semilla=cp["semilla"])
        control["nota"] = ("Con una sola variable el modelo solo puede aprender el signo: el AUC con etiquetas "
                           "permutadas oscila entre valores altos y bajos y la prueba es poco informativa.") if len(m["variables"]) == 1 else ""

    met = metricas.con_intervalos(y, p_oof, pred_oof, d.patient_id.values, cfg["evaluacion"]["n_bootstrap"], cp["semilla"])
    commit = _commit()
    exp_id = f"{m['nombre']}__{regla}__s{cp['semilla']}__{commit}"
    out = res_dir / exp_id
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(nodule_id=d.nodule_id, patient_id=d.patient_id, pliegue=d.pliegue, etiqueta=y,
                      probabilidad=p_oof.round(6), prediccion=pred_oof)).to_csv(out / "predicciones_oof.csv", index=False)
    registro = dict(
        experimento=exp_id, modelo=m["id"], commit=commit, fecha=time.strftime("%Y-%m-%d %H:%M:%S"),
        configuracion=cfg, huella_configuracion=config.huella(cfg),
        datos=dict(regla=regla, nodulos=int(len(d)), pacientes=int(d.patient_id.nunique()),
                   alta_sospecha=int(y.sum()), huella_particion=dp.huella(part)),
        umbrales_por_pliegue=umbrales, coeficientes_por_pliegue=coefs, auc_prueba_por_pliegue=auc_por_pliegue,
        metricas=met, control_negativo=control,
        entorno=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                     sklearn=sklearn.__version__, maquina=platform.platform()),
        carpeta_resultados=str(out), duracion_s=round(time.time() - t0, 1))
    (out / "registro.json").write_text(json.dumps(registro, indent=1, ensure_ascii=False))
    return registro


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--control-negativo", type=int, default=0, metavar="N",
                    help="repite el entrenamiento N veces con etiquetas permutadas (el AUC debe rondar 0,5)")
    a = ap.parse_args()
    r = ejecutar(a.config, a.control_negativo)
    print(f"\n{r['modelo']} · {r['datos']['regla']} · {r['datos']['nodulos']} nódulos de {r['datos']['pacientes']} pacientes")
    for k, v in r["metricas"].items():
        print(f"  {k:<14} {v['valor']:.3f}  IC95 [{v['ic95'][0]:.3f}, {v['ic95'][1]:.3f}]")
    print(f"AUC de prueba por pliegue: {r['auc_prueba_por_pliegue']}")
    print(f"Umbrales por pliegue: {r['umbrales_por_pliegue']}")
    if r["control_negativo"]:
        c = r["control_negativo"]
        print(f"Control negativo ({c['n_permutaciones']} permutaciones): AUC media {c['auc_media']:.3f} "
              f"[{c['auc_p2_5']:.3f}, {c['auc_p97_5']:.3f}] · {'superado' if c['superado'] else 'NO superado: revisar fuga'}")
    print(f"Resultados en {r['carpeta_resultados']}")


if __name__ == "__main__":
    main()
