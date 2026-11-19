"""Compara variantes de M3 por AUC de VALIDACIÓN (criterio de selección), con la misma semilla.

La selección de ajustes no puede mirar la prueba (RNF03; búsqueda de hiperparámetros dentro de cada pliegue externo,
apartado 3.3.6.2): este script ordena las variantes por la media, entre los 5 pliegues, del AUC de nódulo en
validación de la mejor época. El AUC de prueba solo se muestra con --mostrar-prueba, y conviene mirarlo una sola
vez, para la variante elegida.

Lee los `pliegue.json` de cada carpeta `resultados/m3_cnn*__R1__*/s<semilla>/`, así que funciona también mientras
una ejecución está en curso (las variantes incompletas se marcan).

Uso:
    python scripts/comparar_variantes_m3.py                    # semilla 42
    python scripts/comparar_variantes_m3.py --semilla 42 --mostrar-prueba
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--regla", default="R1")
    ap.add_argument("--mostrar-prueba", action="store_true", help="añade el AUC de prueba (mirar solo al final)")
    a = ap.parse_args()

    part = pd.read_csv(RAIZ / "particiones" / f"particion_{a.regla}_s42_k5.csv")[["nodule_id", "etiqueta"]]
    filas = []
    for carpeta in sorted((RAIZ / "resultados").glob(f"m3_cnn*__{a.regla}__*")):
        pls = sorted((carpeta / f"s{a.semilla}").glob("pliegue*/pliegue.json"))
        if not pls:
            continue
        info = [json.loads(p.read_text()) for p in pls]
        reg = carpeta / "registro.json"
        if reg.exists() and json.loads(reg.read_text()).get("nodulos_excluidos"):
            continue                                                # ejecuciones de prueba con nódulos excluidos
        fila = dict(variante=carpeta.name.split("__")[0], pliegues=len(info),
                    auc_val_media=round(sum(i["auc_validacion"] for i in info) / len(info), 4),
                    auc_val_min=min(i["auc_validacion"] for i in info),
                    mejor_epoca_media=round(sum(i["mejor_epoca"] for i in info) / len(info), 1),
                    minutos=round(sum(i["segundos"] for i in info) / 60, 1), carpeta=carpeta.name)
        if a.mostrar_prueba and len(info) == 5:
            p = pd.concat(pd.read_csv(f) for f in (carpeta / f"s{a.semilla}").glob("pliegue*/predicciones_nodulo.csv"))
            p = p.merge(part, on="nodule_id")
            fila["auc_prueba"] = round(roc_auc_score(p.etiqueta, p.probabilidad), 4)
        filas.append(fila)
    if not filas:
        print("No hay variantes con resultados para esa semilla.")
        return
    t = pd.DataFrame(filas).sort_values(["pliegues", "auc_val_media"], ascending=False)
    t.loc[t.pliegues < 5, "variante"] += " (incompleta)"
    print(f"Variantes de M3 · {a.regla} · semilla {a.semilla} · ordenadas por AUC medio de VALIDACIÓN\n")
    print(t.drop(columns="carpeta").to_string(index=False))
    if not a.mostrar_prueba:
        print("\nEl AUC de prueba no se muestra: elige con la validación y míralo solo para la variante final "
              "(--mostrar-prueba).")


if __name__ == "__main__":
    main()
