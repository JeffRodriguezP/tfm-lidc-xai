"""Recortes de imagen de cada nódulo a partir de los DICOM (RF-D06, RD-03; apartado 3.3.5.7 de la memoria).

Pasos (todos deterministas e idénticos para cualquier subconjunto, así que no pueden filtrar información):
  1. Lectura y orden: pylidc lee los DICOM con pydicom y los ordena por la coordenada z de
     ImagePositionPatient (no por el nombre de archivo, que es aleatorio). Se comprueba que z es estrictamente
     creciente y se registra si el espaciado entre cortes es irregular.
  2. Unidades Hounsfield: pendiente y ordenada de reescalado de cada imagen.
  3. Vínculo contorno-imagen: cada contorno de cada lector debe caer en un corte cuya z difiera menos de
     `TOLERANCIA_Z_MM` de la z del contorno; la diferencia máxima se registra en el control de calidad.
  4. Recorte de campo de visión físico fijo (64 mm) centrado en el centroide de la máscara de consenso del nódulo
     (el mismo centro para todos sus cortes), remuestreado a espaciado fijo en el plano (0,5 mm → 128 × 128 px).
  5. Ventana fija [−1000, 400] HU reescalada a [0, 1]. Fuera de la imagen se rellena con aire (−1000 HU → 0).
  6. Máscara de consenso (≥ 50 % de los lectores) con el mismo remuestreo, por vecino más próximo. Se guarda
     solo para evaluar la explicabilidad: NUNCA es entrada del modelo.

Se guardan todos los cortes con máscara de consenso no vacía y su área en mm² (medida en la resolución original,
igual que `scripts/contar_cortes.py`); la política S2 (área ≥ 2,5 mm²) se aplica al cargar, de modo que cambiar el
umbral no obliga a extraer de nuevo. Cada corte guarda también sus vecinos k−1 y k+1 (canales 0 y 2) para la
política S3 (2,5D); el canal 1 es el corte k.

Caché: `resultados/datos/recortes/<parámetros>/<nodule_id>.npz`, un archivo por nódulo, de modo que cualquier regla
de etiquetado reutiliza lo ya extraído. Los recortes no se suben al repositorio (RF-E03, .gitignore).

Uso:
    python -m tfm_lidc.data.recortes --config configs/modelos/m3_cnn.yaml          # extrae lo que falte
    python -m tfm_lidc.data.recortes --config configs/modelos/m3_cnn.yaml --n 3    # prueba con 3 exploraciones
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from . import nodulos as dn

VERSION = "v1"                 # cambia si cambia el procedimiento: la caché antigua deja de usarse
TOLERANCIA_Z_MM = 0.1          # diferencia máxima admitida entre la z del contorno y la de su corte
AIRE_HU = -1000.0


# --------------------------------------------------------------------------------------------------------
# Parámetros y rutas
# --------------------------------------------------------------------------------------------------------
def parametros(cfg: dict) -> dict:
    """Parámetros del recorte (configs/base.yaml, sección `recortes`) y nivel de consenso de la máscara."""
    r = cfg.get("recortes", {})
    p = dict(fov_mm=float(r.get("fov_mm", 64.0)), mm_por_pixel=float(r.get("mm_por_pixel", 0.5)),
             ventana_hu=[float(v) for v in r.get("ventana_hu", [-1000.0, 400.0])],
             nivel_consenso=float(cfg["datos"].get("nivel_consenso", 0.5)))
    p["tam_px"] = int(round(p["fov_mm"] / p["mm_por_pixel"]))
    return p


def carpeta_cache(cfg: dict, datos_dir: Path) -> Path:
    p = parametros(cfg)
    nombre = (f"fov{p['fov_mm']:g}_px{p['mm_por_pixel']:g}_w{p['ventana_hu'][0]:g}_{p['ventana_hu'][1]:g}"
              f"_c{p['nivel_consenso']:g}_{VERSION}")
    return datos_dir / "recortes" / nombre


# --------------------------------------------------------------------------------------------------------
# Geometría (funciones puras: se prueban con volúmenes sintéticos en tests/test_m3.py)
# --------------------------------------------------------------------------------------------------------
def ventana(hu: np.ndarray, ventana_hu: list[float]) -> np.ndarray:
    lo, hi = ventana_hu
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0)


def rejilla(centro_fila: float, centro_col: float, mm_por_pixel_original: float, fov_mm: float,
            mm_por_pixel: float) -> tuple[np.ndarray, np.ndarray]:
    """Coordenadas (en píxeles de la imagen original) del centro de cada píxel del recorte."""
    n = int(round(fov_mm / mm_por_pixel))
    desp_mm = (np.arange(n) - (n - 1) / 2) * mm_por_pixel          # desplazamiento físico de cada píxel
    filas = centro_fila + desp_mm / mm_por_pixel_original
    cols = centro_col + desp_mm / mm_por_pixel_original
    return np.meshgrid(filas, cols, indexing="ij")


def recortar(corte: np.ndarray, malla: tuple[np.ndarray, np.ndarray], orden: int, relleno: float) -> np.ndarray:
    """Remuestrea un corte 2D sobre la malla: orden 1 (bilineal) para HU, 0 (vecino más próximo) para máscaras."""
    return ndimage.map_coordinates(corte.astype(np.float32), np.stack(malla), order=orden, mode="constant",
                                   cval=relleno, prefilter=False)


def recortes_de_nodulo(vol_hu: np.ndarray, mascara: np.ndarray, mm_px: float, p: dict) -> dict:
    """Recortes de un nódulo.

    vol_hu  : volumen de la exploración en HU, ejes (fila, columna, corte).
    mascara : máscara de consenso del nódulo en el MISMO marco que vol_hu (booleana).
    Devuelve imagen (N, 3, H, W) float16 en [0, 1] (canales k−1, k, k+1), mascara (N, H, W) uint8,
    los índices de corte k, el área de consenso por corte en mm² y el centro del recorte.
    """
    ks = np.flatnonzero(mascara.any(axis=(0, 1)))
    cf, cc = ndimage.center_of_mass(mascara)[:2]                    # centroide 3D proyectado en el plano
    malla = rejilla(cf, cc, mm_px, p["fov_mm"], p["mm_por_pixel"])
    nz = vol_hu.shape[2]
    img = np.empty((len(ks), 3, p["tam_px"], p["tam_px"]), np.float16)
    msk = np.empty((len(ks), p["tam_px"], p["tam_px"]), np.uint8)
    cache: dict[int, np.ndarray] = {}

    def hu(k):
        k = min(max(k, 0), nz - 1)                                  # en los extremos se repite el corte
        if k not in cache:
            cache[k] = ventana(recortar(vol_hu[:, :, k], malla, 1, AIRE_HU), p["ventana_hu"])
        return cache[k]

    for i, k in enumerate(ks):
        img[i] = np.stack([hu(k - 1), hu(k), hu(k + 1)]).astype(np.float16)
        msk[i] = recortar(mascara[:, :, k], malla, 0, 0) > 0.5
    area = mascara[:, :, ks].sum(axis=(0, 1)) * mm_px ** 2
    return dict(imagen=img, mascara=msk, k_corte=ks.astype(np.int32), area_consenso_mm2=area.astype(np.float32),
                centro_fila=float(cf), centro_columna=float(cc))


# --------------------------------------------------------------------------------------------------------
# Lectura de una exploración con pylidc
# --------------------------------------------------------------------------------------------------------
def _imagenes(scan) -> tuple[list, int]:
    """Mismo criterio que `pylidc.Scan.load_all_dicom_images` (archivos .dcm de la serie, sin z repetidas, orden por
    la z de ImagePositionPatient), de modo que los índices de corte coinciden con los de los contornos de pylidc.
    Diferencia: si un archivo no tiene el preámbulo 'DICM' (DICOM antiguo sin metainformación), se lee con
    `force=True` en lugar de abortar; el número de archivos así leídos se registra en el control de calidad."""
    import os

    import pydicom

    ruta = scan.get_path_to_dicom_files()
    imgs, forzados = [], 0
    for f in sorted(os.listdir(ruta)):
        if not f.endswith(".dcm") or f.startswith("."):
            continue
        try:
            im = pydicom.dcmread(os.path.join(ruta, f))
        except pydicom.errors.InvalidDicomError:
            im, forzados = pydicom.dcmread(os.path.join(ruta, f), force=True), forzados + 1
            if "SeriesInstanceUID" not in im:
                # No se puede saltar el archivo: cambiaría los índices de corte respecto de los contornos de pylidc
                raise ValueError(f"archivo DICOM dañado (cabecera vacía): {os.path.join(ruta, f)}. "
                                 "Vuelve a descargar esa serie de TCIA.")
        if (str(im.SeriesInstanceUID).strip() == scan.series_instance_uid
                and str(im.StudyInstanceUID).strip() == scan.study_instance_uid):
            imgs.append(im)
    por_z: dict[float, object] = {}
    for im in imgs:                                                 # z repetida: pylidc conserva el menor InstanceNumber
        z = float(im.ImagePositionPatient[2])
        if z not in por_z or float(im.InstanceNumber) < float(por_z[z].InstanceNumber):
            por_z[z] = im
    return [por_z[z] for z in sorted(por_z)], forzados


def _volumen(scan) -> tuple[np.ndarray, np.ndarray, dict]:
    """Volumen en HU ordenado por z y su control de calidad (Tabla 20: cabeceras y rango de intensidades)."""
    imgs, forzados = _imagenes(scan)
    z = np.array([float(im.ImagePositionPatient[2]) for im in imgs])
    vol = np.stack([im.pixel_array.astype(np.float32) * float(im.RescaleSlope) + float(im.RescaleIntercept)
                    for im in imgs], axis=-1)
    dz = np.diff(z)
    qc = dict(n_cortes=len(imgs), archivos_sin_preambulo=forzados, z_monotona=bool((dz > 0).all()),
              dz_mediana_mm=round(float(np.median(dz)), 4) if len(dz) else float("nan"),
              dz_irregular=bool(len(dz) and np.ptp(dz) > 0.01),
              filas_columnas_homogeneas=len({im.pixel_array.shape for im in imgs}) == 1,
              representacion_pixel=sorted({int(getattr(im, "PixelRepresentation", -1)) for im in imgs}),
              hu_min=float(vol.min()), hu_max=float(vol.max()), mm_por_pixel=float(scan.pixel_spacing))
    return vol, z, qc


def procesar_exploracion(scan_id: int, nodulos: dict[str, list[int]], destino: Path, p: dict,
                         dicom_root: str = "") -> dict:
    """Extrae y guarda los recortes de los nódulos indicados ({nodule_id: [ann_id, ...]}) de una exploración."""
    dn._configurar_pylidc(dicom_root)
    import pylidc as pl
    from pylidc.utils import consensus

    scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()
    vol, z, qc = _volumen(scan)
    assert qc["z_monotona"], f"scan {scan_id}: z no es estrictamente creciente"
    anotaciones = {a.id: a for a in scan.annotations}
    dz_contorno = 0.0
    for nid, ids in nodulos.items():
        anns = [anotaciones[i] for i in ids]
        for a in anns:                                              # paso 3: vínculo contorno-imagen
            for c in a.contours:
                dz_contorno = max(dz_contorno, abs(z[c.image_k_position] - c.image_z_position))
        m_local, caja, _ = consensus(anns, clevel=p["nivel_consenso"], pad=[(0, 0)] * 3)
        mascara = np.zeros(vol.shape, bool)
        mascara[caja] = m_local
        r = recortes_de_nodulo(vol, mascara, float(scan.pixel_spacing), p)
        r["z_mm"] = z[r["k_corte"]].astype(np.float32)
        info = dict(nodule_id=nid, scan_id=int(scan_id), mm_por_pixel_original=float(scan.pixel_spacing),
                    centro_fila=r.pop("centro_fila"), centro_columna=r.pop("centro_columna"), parametros=p)
        np.savez_compressed(destino / f"{nid}.npz", info=json.dumps(info), **r)
    qc.update(scan_id=int(scan_id), n_nodulos=len(nodulos), dz_contorno_max_mm=round(float(dz_contorno), 4),
              vinculo_contorno_ok=bool(dz_contorno <= TOLERANCIA_Z_MM))
    assert qc["vinculo_contorno_ok"], f"scan {scan_id}: contorno a {dz_contorno:.3f} mm de su corte"
    return qc


def _procesar_seguro(scan_id: int, nodulos: dict, destino: Path, p: dict, dicom_root: str) -> dict:
    """Una exploración que falla no detiene el lote: el error queda en el control de calidad."""
    try:
        return dict(procesar_exploracion(scan_id, nodulos, destino, p, dicom_root), error=None)
    except Exception as e:                                          # noqa: BLE001
        return dict(scan_id=int(scan_id), n_nodulos=len(nodulos), error=f"{type(e).__name__}: {e}"[:300])


# --------------------------------------------------------------------------------------------------------
# Extracción por lotes y carga
# --------------------------------------------------------------------------------------------------------
def asegurar(cfg: dict, nodulos_ids: list[str], datos_dir: Path, n_procesos: int | None = None,
             max_exploraciones: int | None = None) -> Path:
    """Extrae los recortes que falten para `nodulos_ids` (en paralelo por exploración) y devuelve la carpeta."""
    from joblib import Parallel, delayed

    p = parametros(cfg)
    destino = carpeta_cache(cfg, datos_dir)
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "parametros.json").write_text(json.dumps(dict(p, version=VERSION), indent=1))
    asig = pd.read_csv(datos_dir / "asignacion_nodulos_v0.csv")
    lect = pd.read_csv(datos_dir / "lecturas_v0.csv", usecols=["ann_id", "scan_id"])
    asig = asig.merge(lect, on="ann_id")
    faltan = [n for n in nodulos_ids if not (destino / f"{n}.npz").exists()]
    if not faltan:
        return destino
    tareas = asig[asig.nodule_id.isin(faltan)].groupby("scan_id")
    tareas = [(int(s), g.groupby("nodule_id").ann_id.apply(list).to_dict()) for s, g in tareas]
    if max_exploraciones:
        tareas = tareas[:max_exploraciones]
    n_procesos = n_procesos or int(cfg.get("recortes", {}).get("n_procesos", 4))
    print(f"Extrayendo recortes de {sum(len(t[1]) for t in tareas)} nódulos en {len(tareas)} exploraciones "
          f"({n_procesos} procesos) → {destino}")
    t0 = time.time()
    root = cfg["datos"].get("dicom_root", "")
    qc = Parallel(n_jobs=n_procesos, verbose=5)(delayed(_procesar_seguro)(s, nods, destino, p, root)
                                                for s, nods in tareas)
    f_qc = destino / "control_calidad.csv"
    qc = pd.DataFrame(qc)
    if f_qc.exists():
        qc = pd.concat([pd.read_csv(f_qc), qc]).drop_duplicates("scan_id", keep="last")
    qc.to_csv(f_qc, index=False)
    print(f"Recortes listos en {time.time() - t0:.0f} s; control de calidad en {f_qc}")
    fallos = qc[qc.error.notna()] if "error" in qc else qc.iloc[:0]
    if len(fallos):
        raise RuntimeError(f"{len(fallos)} exploraciones fallaron (ver columna 'error' de {f_qc}):\n"
                           + fallos[["scan_id", "error"]].to_string(index=False))
    return destino


def cargar(carpeta: Path, nodulos_ids: list[str], politica: str = "S2", area_minima_mm2: float = 2.5
           ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Imágenes, máscaras y manifiesto (una fila por corte) de los nódulos pedidos, según la política:
    S1 = el corte de mayor área de consenso; S2 = todos los cortes con área ≥ `area_minima_mm2` (1 canal);
    S3 = como S2 con los cortes k−1, k, k+1 como 3 canales (2,5D)."""
    imgs, msks, filas = [], [], []
    for nid in nodulos_ids:
        z = np.load(carpeta / f"{nid}.npz")
        area = z["area_consenso_mm2"]
        sel = np.flatnonzero(area >= area_minima_mm2) if politica in ("S2", "S3") else np.array([int(area.argmax())])
        if len(sel) == 0:                                           # no debería ocurrir (contar_cortes.py lo vigila)
            continue
        im = z["imagen"][sel]
        imgs.append(im if politica == "S3" else im[:, 1:2])
        msks.append(z["mascara"][sel])
        filas.append(pd.DataFrame(dict(nodule_id=nid, k_corte=z["k_corte"][sel], area_consenso_mm2=area[sel])))
    return np.concatenate(imgs), np.concatenate(msks), pd.concat(filas, ignore_index=True)


def main() -> None:
    from .. import config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--n", type=int, default=None, help="solo las N primeras exploraciones (prueba rápida)")
    ap.add_argument("--procesos", type=int, default=None)
    a = ap.parse_args()
    cfg = config.cargar(a.config)
    raiz = Path(__file__).resolve().parents[3]                    # raíz del repositorio
    res = Path(cfg["salidas"]["dir_resultados"])
    datos_dir = (res if res.is_absolute() else raiz / res) / "datos"
    nod = dn.construir_o_cargar(cfg, datos_dir)
    ids = nod.dropna(subset=[f"etiqueta_{cfg['datos']['regla'].lower()}"]).nodule_id.tolist()
    asegurar(cfg, ids, datos_dir, a.procesos, a.n)


if __name__ == "__main__":
    main()
