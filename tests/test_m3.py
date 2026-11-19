"""Pruebas de M3 con datos sintéticos (no necesitan los DICOM; se ejecutan también en la integración continua).

1. Recortes (RF-D06): tamaño, centrado, espaciado físico, ventana y máscara por vecino más próximo.
2. Agregación de cortes en nódulos (RF-M05) y pesos por corte (RF-M02).
3. Arquitectura: salida y resolución de la última capa convolucional.
4. No fuga (RNF03): ni el umbral, ni la parada temprana, ni la normalización, ni las predicciones dependen de las
   etiquetas o de las imágenes de prueba.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tfm_lidc import config
from tfm_lidc.data import recortes as dr
from tfm_lidc.evaluation import agregacion
from tfm_lidc.imagen import aumentar, entrenar_pliegue, pesos_cortes
from tfm_lidc.models import cnn

P = dict(fov_mm=64.0, mm_por_pixel=0.5, ventana_hu=[-1000.0, 400.0], nivel_consenso=0.5, tam_px=128)
CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs" / "modelos").glob("m3_*.yaml"))


def _disco(forma, centro, radio):
    f, c = np.ogrid[:forma[0], :forma[1]]
    return (f - centro[0]) ** 2 + (c - centro[1]) ** 2 <= radio ** 2


# --- 1 · recortes -------------------------------------------------------------------------------------------
def test_ventana():
    v = dr.ventana(np.array([-2000.0, -1000, -300, 400, 3000]), [-1000, 400])
    assert np.allclose(v, [0, 0, 0.5, 1, 1])


def test_recorte_centrado_y_espaciado_fisico():
    mm_px = 0.7
    vol = np.full((512, 512, 5), -800.0)                      # parénquima
    disco = _disco((512, 512), (300, 200), 10)                # nódulo de 10 px × 0,7 mm = radio 7 mm
    vol[disco, 1:4] = 50.0
    masc = np.zeros(vol.shape, bool)
    masc[disco, 1:4] = True
    r = dr.recortes_de_nodulo(vol, masc, mm_px, P)
    assert r["imagen"].shape == (3, 3, 128, 128) and r["mascara"].shape == (3, 128, 128)
    assert list(r["k_corte"]) == [1, 2, 3]
    assert abs(r["centro_fila"] - 300) < 0.5 and abs(r["centro_columna"] - 200) < 0.5
    m = r["mascara"][1]
    assert m[63:65, 63:65].all()                              # el nódulo está en el centro del recorte
    radio_px = np.sqrt(m.sum() / np.pi)                       # 7 mm / 0,5 mm = 14 px
    assert abs(radio_px - 14) < 1
    assert np.isclose(r["area_consenso_mm2"][0], disco.sum() * mm_px ** 2)
    img = r["imagen"].astype(float)
    assert img.min() >= 0 and img.max() <= 1
    assert np.isclose(img[1, 1, 64, 64], (50 + 1000) / 1400, atol=1e-3)


def test_canales_25d_y_bordes():
    vol = np.stack([np.full((64, 64), v) for v in (-1000.0, -500, 0, 400)], axis=-1)
    masc = np.zeros(vol.shape, bool)
    masc[28:36, 28:36, 0] = masc[28:36, 28:36, 3] = True      # primer y último corte del volumen
    r = dr.recortes_de_nodulo(vol, masc, 0.5, dict(P, fov_mm=16.0, tam_px=32))
    c = r["imagen"][:, :, 16, 16].astype(float)
    assert np.allclose(c[0], dr.ventana(np.array([-1000, -1000, -500]), [-1000, 400]), atol=1e-3)  # k−1 repite k=0
    assert np.allclose(c[1], dr.ventana(np.array([0, 400, 400]), [-1000, 400]), atol=1e-3)         # k+1 repite k=3


def test_fuera_de_la_imagen_se_rellena_con_aire():
    vol = np.zeros((40, 40, 1))                               # 0 HU en toda la imagen
    masc = np.zeros(vol.shape, bool)
    masc[0:3, 0:3, 0] = True                                  # nódulo en la esquina: el recorte se sale
    r = dr.recortes_de_nodulo(vol, masc, 1.0, dict(P, fov_mm=32.0, mm_por_pixel=1.0, tam_px=32))
    assert r["imagen"][0, 1, 0, 0] == 0                       # aire
    assert np.isclose(float(r["imagen"][0, 1, -1, -1]), 1000 / 1400, atol=1e-3)


# --- 2 · agregación y pesos ------------------------------------------------------------------------------
def test_agregacion_por_nodulo():
    ids = ["a", "a", "a", "a", "a", "b"]
    p = [0.1, 0.2, 0.3, 0.4, 1.0, 0.7]
    assert np.allclose(agregacion.por_nodulo(ids, p, "media").values, [0.4, 0.7])
    assert np.allclose(agregacion.por_nodulo(ids, p, "maximo").values, [1.0, 0.7])
    assert np.allclose(agregacion.por_nodulo(ids, p, "media_recortada").values, [0.3, 0.7])
    with pytest.raises(ValueError):
        agregacion.por_nodulo(ids, p, "mediana")


def test_pesos_cada_nodulo_suma_uno_y_las_clases_se_equilibran():
    ids = np.array(["a"] * 4 + ["b"] * 2 + ["c"] * 1 + ["d"] * 3)
    y = np.array([0] * 4 + [0] * 2 + [0] * 1 + [1] * 3)       # 3 nódulos de clase 0, 1 de clase 1
    w = pesos_cortes(ids, y, por_cortes=True, por_clase=False)
    assert np.allclose(pd.Series(w).groupby(ids).sum().values, 1)
    w = pesos_cortes(ids, y)
    assert np.isclose(w[y == 0].sum(), w[y == 1].sum())


# --- 3 · arquitectura --------------------------------------------------------------------------------------
@pytest.mark.parametrize("canales", [1, 3])
def test_cnn_salida_y_mapa_de_16x16(canales):
    m = cnn.CNNCompacta(canales).eval()
    x = torch.rand(2, canales, 128, 128)
    assert m(x).shape == (2,)
    assert m.caracteristicas(x).shape == (2, 256, 16, 16)
    assert cnn.n_parametros(m) < 2_000_000


@pytest.mark.parametrize("extra", [{}, dict(ruido=0.05, desenfoque=0.5)])
def test_aumento_conserva_forma_rango_y_es_reproducible(extra):
    x = torch.rand(8, 3, 32, 32)
    a = dict(giros=True, rotacion_grados=15, desplazamiento_px=4, intensidad=0.1, **extra)
    x1 = aumentar(x, a, torch.Generator().manual_seed(0))
    x2 = aumentar(x, a, torch.Generator().manual_seed(0))
    assert x1.shape == x.shape and torch.equal(x1, x2) and x1.min() >= 0 and x1.max() <= 1


@pytest.mark.parametrize("ruta", CONFIGS, ids=lambda p: p.stem)
def test_configuraciones_m3(ruta):
    cfg = config.cargar(ruta)
    m = cfg["modelo"]
    assert m["id"] == "M3" and m["familia"] == "imagen"
    assert m["entrada"]["politica"] in {"S1", "S2", "S3"}
    assert dr.parametros(cfg)["tam_px"] == (96 if ruta.stem == "m3_cnn_fov48" else 128)
    assert m["nombre"] == ruta.stem                           # cada variante escribe en su propia carpeta


# --- 4 · no fuga -------------------------------------------------------------------------------------------
M_PEQ = dict(parametros=dict(anchos=[4, 8, 8], abandono=0.0),
             entrenamiento=dict(epocas=3, paciencia=5, tam_lote=16),
             aumento=dict(giros=True, rotacion_grados=10, desplazamiento_px=2, intensidad=0.05))


def _sinteticos(n_nod=60, cortes=3, semilla=0):
    """Nódulos con 3 cortes de 32 × 32; los de clase 1 son discos más grandes. Roles fijos por nódulo."""
    rng = np.random.default_rng(semilla)
    X, ids, y, rol = [], [], [], []
    roles = np.array(["entrenamiento"] * 30 + ["validacion"] * 14 + ["prueba"] * 16)
    for i in range(n_nod):
        yi = i % 2
        for _ in range(cortes):
            im = rng.normal(0.2, 0.05, (32, 32))
            im[_disco((32, 32), (16, 16), 4 + 4 * yi + rng.integers(0, 2))] += 0.5
            X.append(im[None])
            ids.append(f"n{i:02d}")
            y.append(yi)
            rol.append(roles[i])
    return torch.tensor(np.array(X), dtype=torch.float16).clamp(0, 1), np.array(ids), np.array(y), np.array(rol)


def test_umbral_y_predicciones_no_dependen_de_las_etiquetas_de_prueba():
    X, ids, y, rol = _sinteticos()
    r1 = entrenar_pliegue(M_PEQ, X, ids, y, rol, 0, torch.device("cpu"), verbose=False)
    y2 = y.copy()
    y2[rol == "prueba"] = 1 - y2[rol == "prueba"]
    r2 = entrenar_pliegue(M_PEQ, X, ids, y2, rol, 0, torch.device("cpu"), verbose=False)
    assert r1["umbral"] == r2["umbral"] and r1["mejor_epoca"] == r2["mejor_epoca"]
    assert np.array_equal(r1["prob_cortes_prueba"], r2["prob_cortes_prueba"])


def test_normalizacion_y_entrenamiento_no_dependen_de_las_imagenes_de_prueba():
    X, ids, y, rol = _sinteticos(semilla=1)
    r1 = entrenar_pliegue(M_PEQ, X, ids, y, rol, 0, torch.device("cpu"), verbose=False)
    X2 = X.clone()
    X2[torch.from_numpy(rol == "prueba")] = 1.0                # imágenes de prueba absurdas
    r2 = entrenar_pliegue(M_PEQ, X2, ids, y, rol, 0, torch.device("cpu"), verbose=False)
    assert r1["normalizacion"] == r2["normalizacion"] and r1["umbral"] == r2["umbral"]
    for (k, a), b in zip(r1["modelo"].state_dict().items(), r2["modelo"].state_dict().values()):
        assert torch.equal(a, b), k


def test_entrena_y_separa_un_problema_facil():
    X, ids, y, rol = _sinteticos(semilla=2)
    m = dict(M_PEQ, entrenamiento=dict(epocas=15, paciencia=15, tam_lote=16, tasa_aprendizaje=0.01))
    r = entrenar_pliegue(m, X, ids, y, rol, 0, torch.device("cpu"), verbose=False)
    te = r["indices_prueba"]
    p = agregacion.por_nodulo(ids[te], r["prob_cortes_prueba"])
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(pd.Series(y[te], index=ids[te]).groupby(level=0).first().loc[p.index], p) > 0.9


def test_desenfoque_suaviza_y_ruido_añade_varianza():
    g = lambda: torch.Generator().manual_seed(0)                # noqa: E731
    x = torch.zeros(4, 1, 32, 32)
    x[:, :, :, 16:] = 1.0                                     # borde vertical nítido
    xb = aumentar(x, dict(desenfoque=1.0), g())
    assert 0 < xb[0, 0, 16, 15] < 1                           # el borde se suaviza
    xr = aumentar(torch.full((4, 1, 32, 32), 0.5), dict(ruido=0.1), g())
    assert xr.std() > 0
    # sin las claves nuevas el aumento es idéntico al de la ejecución base (no consume aleatoriedad extra)
    base = dict(giros=True, rotacion_grados=15, desplazamiento_px=4, intensidad=0.1)
    assert torch.equal(aumentar(x, base, g()), aumentar(x, dict(base, ruido=0, desenfoque=0), g()))
