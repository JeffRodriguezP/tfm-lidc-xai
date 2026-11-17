"""Cifras consolidadas del TFM: nódulos, pacientes y cortes por regla de etiquetado (R1, R2, R3).

Este script responde a una pregunta concreta: ¿cuántos nódulos, pacientes y CORTES (imágenes 2D) de TC
quedan para cada regla de etiquetado? Todo se calcula a partir de las anotaciones XML de LIDC-IDRI que
trae la librería pylidc; NO necesita los archivos DICOM.

Hay dos formas de contar cortes y el script muestra las dos, porque se confundieron en la memoria:

  * cortes_union     : cortes en los que dibujó contorno ALGÚN radiólogo del nódulo.
  * cortes_consenso  : cortes con máscara de CONSENSO no vacía (política S2 de la memoria, apartado 3.3.5.6).
                       Un píxel está en la máscara de consenso si lo incluye al menos el 50 % de los lectores
                       del nódulo, y el corte cuenta solo si esa máscara ocupa al menos `area_minima_corte_mm2`
                       (paso 6 del embudo, Tabla 16).  ← esta es la cifra que usan los modelos.

CIFRAS FINALES (configuración por defecto: consenso 50 %, área mínima 2,5 mm², sin los 14 clústeres con > 4 lecturas):

    regla    nódulos (baja / alta)   pacientes   cortes de consenso (S2)   cortes de unión
    R1         869 (448 / 421)          530              6.070                   6.852
    R2       1.180 (662 / 518)          633              7.445                   8.416
    R3       2.637 (2.004 / 633)        870             14.173                  15.571

Cómo se llega a ellas:
  1. lecturas      : cada anotación de nódulo ≥ 3 mm de cada radiólogo (6.859).
  2. nódulos       : las lecturas de distintos radiólogos que describen la misma lesión se agrupan con
                     `cluster_annotations` de pylidc (los XML no las enlazan): 2.651 clústeres.
  3. exclusión     : un nódulo no puede tener más de 4 lecturas (hay 4 radiólogos): se excluyen 14 clústeres.
  4. etiquetado    : R1 exige ≥ 3 lectores, usa la mediana de malignidad y excluye la mediana = 3.
  5. cortes        : para cada nódulo se calcula, corte a corte, la máscara de consenso y su área en mm²;
                     se cuentan los cortes con área ≥ `area_minima_corte_mm2` (y, aparte, la unión de lectores).

Escribe `resultados/datos/cortes_por_corte_v1.csv` (una fila por nódulo y corte) y muestra la tabla por regla.
El cálculo lento (recorrer pylidc) se hace una sola vez y queda en esa caché; cambiar el umbral de área
o el nivel de consenso NO obliga a repetirlo, salvo el nivel de consenso (ver `--recalcular`).

Uso:
    python scripts/contar_cortes.py --config configs/base.yaml                  # cifras finales
    python scripts/contar_cortes.py --config configs/base.yaml --verificar      # además comprueba que coinciden
    python scripts/contar_cortes.py --config configs/base.yaml --sin-exclusion  # sensibilidad: conserva los 14 clústeres
    python scripts/contar_cortes.py --config configs/base.yaml --area-minima 0  # consenso sin área mínima (cualquier píxel)
    python scripts/contar_cortes.py --config configs/base.yaml --recalcular     # ignora la caché y vuelve a leer pylidc
"""
# Esta línea hace que las anotaciones de tipo (`str | Path`, `dict[str, set]`) no se evalúen al importar;
# permite escribirlas con la sintaxis moderna aunque el entorno tenga una versión de Python más antigua.
from __future__ import annotations

import argparse                      # lectura de opciones de la línea de comandos (--config, --verificar, ...)
from collections import Counter      # contador: cuántos lectores dibujaron contorno en cada corte
from pathlib import Path             # rutas de archivos independientes del sistema operativo

import numpy as np                   # arreglos numéricos (las máscaras de consenso son arreglos 3D de numpy)
import pandas as pd                  # tablas (DataFrame) para nódulos, cortes y el resumen final

from tfm_lidc import config          # carga de los YAML con herencia (configs/base.yaml + configs/local.yaml)
from tfm_lidc.data import nodulos as dn   # tabla de nódulos y etiquetas R1, R2, R3 (pasos 1 a 4)

# ---------------------------------------------------------------------------------------------------------
# Cifras finales documentadas en la memoria (Tabla 17). Cada tupla es:
#   (nódulos, nódulos de sospecha baja, nódulos de sospecha alta, pacientes, cortes)
# Se usan solo con --verificar, para detectar que algo cambió sin que nadie se dé cuenta.
# ---------------------------------------------------------------------------------------------------------
CIFRAS_CONSENSO = {"R1": (869, 448, 421, 530, 6070),     # cortes de consenso (S2) con área mínima 2,5 mm²
                   "R2": (1180, 662, 518, 633, 7445),
                   "R3": (2637, 2004, 633, 870, 14173)}
CIFRAS_UNION = {"R1": 6852, "R2": 8416, "R3": 15571}     # cortes donde dibujó contorno algún lector

# Valores por defecto si el YAML no los declara (coinciden con configs/base.yaml)
NIVEL_CONSENSO = 0.5         # un píxel es de consenso si lo marca ≥ 50 % de los lectores del nódulo
AREA_MINIMA_MM2 = 2.5        # un corte cuenta si la máscara de consenso ocupa ≥ 2,5 mm²

# Columnas de la caché por corte (una fila = un corte de un nódulo)
COLUMNAS_CORTE = ["nodule_id", "k_corte", "n_lectores", "area_consenso_mm2"]


def _alias_numpy() -> None:
    """pylidc 0.2.3 usa `np.int`, `np.float` y `np.bool`, que NumPy ≥ 1.24 eliminó.

    Se recrean como alias de los tipos de Python antes de importar pylidc. Solo se añaden si faltan,
    así que con versiones antiguas de NumPy esta función no hace nada.
    """
    for nombre, tipo in (("int", int), ("float", float), ("bool", bool)):
        if not hasattr(np, nombre):
            setattr(np, nombre, tipo)


def cortes_por_corte(asignacion: pd.DataFrame, dicom_root: str = "", nivel: float = NIVEL_CONSENSO) -> pd.DataFrame:
    """Para cada nódulo, una fila por corte con: cuántos lectores dibujaron contorno y el área de consenso.

    Devuelve las columnas `COLUMNAS_CORTE`:
      nodule_id          identificador del nódulo (el de la tabla de asignación);
      k_corte            índice del corte dentro de la exploración (el mismo que usa pylidc);
      n_lectores         cuántos radiólogos del nódulo dibujaron contorno en ese corte (≥ 1 si está en la unión;
                         0 si el corte solo aparece por la máscara de consenso, caso que no debería ocurrir);
      area_consenso_mm2  área de la máscara de consenso en ese corte, en mm² (0 si el consenso es vacío).

    De aquí salen las dos cifras: la UNIÓN cuenta todas las filas; el CONSENSO cuenta las de área ≥ umbral.
    """
    dn._configurar_pylidc(dicom_root)     # escribe ~/.pylidcrc solo si hay una ruta DICOM (aquí no se leen DICOM)
    _alias_numpy()                        # parche de compatibilidad con NumPy moderno (ver arriba)
    import pylidc as pl                   # se importa aquí, no arriba, para que el alias ya exista
    from pylidc.utils import consensus    # función de pylidc que fusiona las máscaras de varios lectores

    # Todas las anotaciones de la base de datos de pylidc, indexadas por su identificador (ann_id)
    anotaciones = {a.id: a for a in pl.query(pl.Annotation).all()}

    # Para cada nódulo, la lista de las anotaciones (lecturas) que lo forman: {nodule_id: [ann_id, ...]}
    ids_por_nodulo = asignacion.groupby("nodule_id").ann_id.apply(list)

    filas = []
    for nid, ids in ids_por_nodulo.items():
        anns = [anotaciones[i] for i in ids]     # objetos Annotation de los lectores de este nódulo

        # --- Parte 1: UNIÓN. Cuántos lectores dibujaron contorno en cada corte ---------------------------
        # `contour_slice_indices` son los cortes con contorno de UN lector. `set(...)` evita contar dos veces
        # un corte repetido dentro del mismo lector; el Counter suma, por corte, cuántos lectores lo dibujaron.
        lectores_por_corte = Counter(int(k) for a in anns for k in set(a.contour_slice_indices))

        # --- Parte 2: CONSENSO. Máscara 3D donde ≥ `nivel` de los lectores incluyen el vóxel --------------
        # `pad=[(0,0)]*3` = sin margen alrededor de la caja envolvente (solo interesa el interior del nódulo).
        # Devuelve (máscara booleana 3D, caja envolvente, máscaras individuales); solo usamos las dos primeras.
        mascara, caja, _ = consensus(anns, clevel=nivel, pad=[(0, 0)] * 3)

        # La máscara tiene ejes (fila, columna, corte). Sumando sobre filas y columnas queda, para cada
        # corte, el número de píxeles de consenso.
        pixeles_por_corte = mascara.sum(axis=(0, 1))

        # Cada píxel mide `pixel_spacing` mm por lado; su área es el cuadrado de ese valor.
        # Es un dato de la exploración: todos los lectores del nódulo comparten exploración.
        mm_por_pixel = float(anns[0].scan.pixel_spacing)
        area_pixel_mm2 = mm_por_pixel ** 2

        # El eje de cortes de la máscara empieza en `caja[2].start`: el corte i de la máscara es el
        # corte `inicio + i` de la exploración. Así los índices de la máscara y de los contornos coinciden.
        inicio = caja[2].start
        area_por_corte = {inicio + i: float(px) * area_pixel_mm2 for i, px in enumerate(pixeles_por_corte)}

        # Se guarda una fila por cada corte que aparezca en la unión o en el consenso
        for k in sorted(set(lectores_por_corte) | {k for k, a in area_por_corte.items() if a > 0}):
            filas.append((nid, k, lectores_por_corte.get(k, 0), area_por_corte.get(k, 0.0)))

    return pd.DataFrame(filas, columns=COLUMNAS_CORTE)


def resumen_por_nodulo(cortes: pd.DataFrame, area_minima_mm2: float) -> pd.DataFrame:
    """Colapsa la tabla por corte a una fila por nódulo con sus dos recuentos de cortes.

    Devuelve: nodule_id, cortes_union (todos los cortes con contorno de algún lector) y
    cortes_consenso (los de área de consenso ≥ `area_minima_mm2`).
    """
    # Un corte es de UNIÓN si al menos un lector dibujó contorno en él (n_lectores ≥ 1).
    cortes = cortes.assign(en_union=cortes.n_lectores >= 1,
                           # Un corte es de CONSENSO si la máscara de consenso ocupa el área mínima o más.
                           # Con área mínima 0 se exige solo que la máscara no esté vacía (> 0).
                           en_consenso=(cortes.area_consenso_mm2 >= area_minima_mm2) & (cortes.area_consenso_mm2 > 0))
    return (cortes.groupby("nodule_id")[["en_union", "en_consenso"]].sum().astype(int)
            .rename(columns={"en_union": "cortes_union", "en_consenso": "cortes_consenso"}).reset_index())


def main() -> None:
    # ---- Opciones de línea de comandos ---------------------------------------------------------------
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])   # la descripción es la primera línea del docstring
    p.add_argument("--config", default="configs/base.yaml", help="YAML de configuración (hereda configs/local.yaml)")
    p.add_argument("--sin-exclusion", action="store_true",
                   help="análisis de sensibilidad: conserva los clústeres con más de 4 lecturas")
    p.add_argument("--verificar", action="store_true", help="falla si las cifras no coinciden con las finales del TFM")
    p.add_argument("--area-minima", type=float, default=None,
                   help="área mínima de consenso por corte, en mm² (por defecto datos.area_minima_corte_mm2 del YAML)")
    p.add_argument("--recalcular", action="store_true", help="ignora la caché por corte y vuelve a leer pylidc")
    a = p.parse_args()

    # ---- Configuración -------------------------------------------------------------------------------
    cfg = config.cargar(a.config)                       # base.yaml (+ local.yaml si existe)
    if a.sin_exclusion:                                 # sensibilidad: no excluir los 14 clústeres
        cfg["datos"]["excluir_clusteres_mas_de_4"] = False
    nivel = float(cfg["datos"].get("nivel_consenso", NIVEL_CONSENSO))     # 0,5 por defecto
    area_min = a.area_minima if a.area_minima is not None else float(
        cfg["datos"].get("area_minima_corte_mm2", AREA_MINIMA_MM2))       # 2,5 mm² por defecto
    cache = Path(cfg.get("rutas", {}).get("resultados", "resultados")) / "datos"   # carpeta de la caché

    # ---- Pasos 1 a 4: tabla de nódulos con etiquetas R1, R2, R3 (aplica la exclusión de clústeres) ---
    nod = dn.construir_o_cargar(cfg, cache)

    # ---- Paso 5: cortes por nódulo --------------------------------------------------------------------
    # La caché guarda una fila por nódulo y corte, no el recuento: así se puede cambiar el área mínima sin
    # volver a recorrer pylidc, que tarda unos minutos.
    f = cache / "cortes_por_corte_v1.csv"
    if f.exists() and not a.recalcular:
        por_corte = pd.read_csv(f)                                       # caché ya calculada
    else:
        print("Calculando máscaras de consenso con pylidc (solo la primera vez, unos 3 min)…")
        asig = pd.read_csv(cache / "asignacion_nodulos_v0.csv")          # qué lecturas forman cada nódulo
        por_corte = cortes_por_corte(asig, cfg["datos"].get("dicom_root", ""), nivel)
        por_corte.to_csv(f, index=False)                                 # se guarda para la próxima vez

    # Colapsa a una fila por nódulo y la une a la tabla de nódulos (left: conserva los nódulos sin cortes)
    por_nodulo = resumen_por_nodulo(por_corte, area_min)
    nod = nod.merge(por_nodulo, on="nodule_id", how="left")
    nod[["cortes_union", "cortes_consenso"]] = nod[["cortes_union", "cortes_consenso"]].fillna(0).astype(int)

    # ---- Una fila por regla ---------------------------------------------------------------------------
    # Solo cuentan los nódulos que reciben etiqueta: los excluidos por la regla (mediana = 3, menos de
    # 3 lectores en R1/R2) o por ser clúster anómalo tienen etiqueta NaN y se descartan aquí.
    filas = []
    for r in ("R1", "R2", "R3"):
        d = nod[nod[f"etiqueta_{r.lower()}"].notna()]      # nódulos con etiqueta en esta regla
        e = d[f"etiqueta_{r.lower()}"]                     # 0 = sospecha baja, 1 = sospecha alta
        filas.append(dict(regla=r, nodulos=len(d), baja=int((e == 0).sum()), alta=int((e == 1).sum()),
                          pacientes=d.patient_id.nunique(),
                          cortes_consenso=int(d.cortes_consenso.sum()),
                          cortes_union=int(d.cortes_union.sum()),
                          # nódulos que se quedan SIN ningún corte de consenso con ese umbral (idealmente 0)
                          nodulos_sin_corte=int((d.cortes_consenso == 0).sum())))
    tabla = pd.DataFrame(filas)
    print(f"\nConsenso ≥ {nivel:.0%} de los lectores; área mínima por corte = {area_min} mm²\n")
    print(tabla.to_string(index=False))

    # ---- Verificación opcional ------------------------------------------------------------------------
    if a.verificar:
        assert not a.sin_exclusion, "--verificar compara con las cifras finales, que incluyen la exclusión"
        assert area_min == AREA_MINIMA_MM2 and nivel == NIVEL_CONSENSO, \
            "--verificar solo es válido con las cifras por defecto (consenso 50 %, área mínima 2,5 mm²)"
        obtenido = {r.regla: (r.nodulos, r.baja, r.alta, r.pacientes, r.cortes_consenso) for r in tabla.itertuples()}
        union = {r.regla: r.cortes_union for r in tabla.itertuples()}
        assert obtenido == CIFRAS_CONSENSO, f"Las cifras no coinciden con las del TFM:\n{obtenido}\n{CIFRAS_CONSENSO}"
        assert union == CIFRAS_UNION, f"Los cortes de unión no coinciden:\n{union}\n{CIFRAS_UNION}"
        print("\n✓ Coinciden con las cifras finales del TFM "
              "(R1: 869 nódulos, 530 pacientes, 6.070 cortes de consenso y 6.852 de unión).")


if __name__ == "__main__":
    main()
