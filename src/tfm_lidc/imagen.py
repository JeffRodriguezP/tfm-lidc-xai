"""Entrenamiento y evaluación de los modelos de imagen (M3; la misma interfaz servirá para M4 y M5).

Uso:
    python -m tfm_lidc.imagen --config configs/modelos/m3_cnn.yaml                       # 5 pliegues × semillas del YAML
    python -m tfm_lidc.imagen --config configs/modelos/m3_cnn.yaml --pliegues 0 --semillas 42 --epocas 3   # prueba rápida

Protocolo (Tabla 19 y apartado 3.3.6.2), idéntico al de M0–M2 en todo lo que no es propio de la imagen:
  · Misma partición versionada por paciente. En cada pliegue externo k la red se entrena con los cortes de los nódulos
    de 'entrenamiento', la parada temprana y el umbral (Youden) se deciden con 'validacion' y se predice 'prueba'.
    Las etiquetas de prueba no se leen nunca (tests/test_m3.py lo comprueba).
  · Unidad de entrenamiento: el corte (política S2 por defecto). Unidad de evaluación: el nódulo, con la media de las
    probabilidades de sus cortes (RF-M05). Cada corte pesa 1 / nº de cortes de su nódulo (RF-M02) y cada clase según
    su frecuencia de nódulos en el entrenamiento.
  · Entropía cruzada binaria ponderada, AdamW, tasa de aprendizaje con planificación coseno, parada temprana por AUC
    de nódulo en validación. Aumento de datos solo en el entrenamiento: giros, rotación moderada, traslación pequeña
    y variación suave de intensidad (3.3.5.10).
  · Normalización (media y desviación típica) estimada solo con los cortes de entrenamiento del pliegue.
  · Reanudación (RNF04): cada (semilla, pliegue) terminado se guarda en su carpeta y no se repite.
Las métricas se calculan con las predicciones fuera de muestra de los 5 pliegues, con IC 95 % por bootstrap de
pacientes, para cada semilla; se informa además la media y la desviación típica entre semillas.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from . import config
from .data import nodulos as dn
from .data import particion as dp
from .data import recortes as dr
from .evaluation import agregacion, controles, metricas
from .lineas_base import _commit
from .models import cnn

ENTRENO_DEF = dict(epocas=60, paciencia=12, tam_lote=64, tasa_aprendizaje=1e-3, decaimiento_pesos=1e-4,
                   ponderar_clase=True, ponderar_por_cortes=True, normalizar=True, agregacion="media")
AUMENTO_DEF = dict(giros=True, rotacion_grados=15.0, desplazamiento_px=8.0, intensidad=0.1,
                   desenfoque=0.0, ruido=0.0)                       # desenfoque y ruido: desactivados por defecto


# --------------------------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------------------------
def dispositivo(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fijar_semillas(semilla: int) -> None:
    """Semillas de numpy y PyTorch. Las operaciones no deterministas que queden (p. ej. en MPS/CUDA) se avisan,
    no se bloquean (RNF02: se declaran y su efecto se mide repitiendo con otra semilla)."""
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.use_deterministic_algorithms(True, warn_only=True)


def pesos_cortes(nodule_id: np.ndarray, y: np.ndarray, por_cortes: bool = True, por_clase: bool = True) -> np.ndarray:
    """Peso de cada corte: 1 / nº de cortes de su nódulo (cada nódulo suma 1) × peso de su clase (cada clase suma
    lo mismo, contando nódulos). Se calcula solo con los cortes que recibe (los de entrenamiento)."""
    n = pd.Series(nodule_id).map(pd.Series(nodule_id).value_counts()).values.astype(float)
    w = 1.0 / n if por_cortes else np.ones(len(y))
    if por_clase:
        nod = pd.DataFrame(dict(n=nodule_id, y=y)).drop_duplicates("n")
        frec = nod.y.value_counts()
        w = w * np.array([len(nod) / (2.0 * frec[c]) for c in y])
    return w.astype(np.float32)


def aumentar(x: torch.Tensor, a: dict, g: torch.Generator) -> torch.Tensor:
    """Aumento en el espacio [0, 1] de la ventana (0 = aire, que es también el relleno de la rotación/traslación)."""
    n, dev = x.shape[0], x.device
    u = lambda *s: torch.rand(*s, generator=g).to(dev)            # noqa: E731  (aleatoriedad en CPU: reproducible)
    if a.get("giros", False):
        x = torch.where((u(n, 1, 1, 1) < 0.5), x.flip(-1), x)
        x = torch.where((u(n, 1, 1, 1) < 0.5), x.flip(-2), x)
    rot, desp = float(a.get("rotacion_grados", 0)), float(a.get("desplazamiento_px", 0))
    if rot or desp:
        th = (u(n) * 2 - 1) * math.radians(rot)
        t = (u(n, 2) * 2 - 1) * desp * 2 / x.shape[-1]               # en coordenadas normalizadas [-1, 1]
        cos, sin = torch.cos(th), torch.sin(th)
        A = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1), torch.stack([sin, cos, t[:, 1]], 1)], 1)
        malla = F.affine_grid(A, list(x.shape), align_corners=False)
        x = F.grid_sample(x, malla, mode="bilinear", padding_mode="zeros", align_corners=False)
    i = float(a.get("intensidad", 0))
    if i:
        x = x * (1 + (u(n, 1, 1, 1) * 2 - 1) * i) + (u(n, 1, 1, 1) * 2 - 1) * i / 2
    p_blur = float(a.get("desenfoque", 0))
    if p_blur:                                                      # desenfoque gaussiano (σ 0,5–1,5 px) con prob. p
        sigma = 0.5 + float(torch.rand(1, generator=g))
        r = int(math.ceil(3 * sigma))
        k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
        k = (k / k.sum()).to(dev)
        c = x.shape[1]
        xb = F.pad(x, (r, r, r, r), mode="replicate")
        xb = F.conv2d(xb, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
        xb = F.conv2d(xb, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
        x = torch.where(u(n, 1, 1, 1) < p_blur, xb, x)
    ruido = float(a.get("ruido", 0))
    if ruido:                                                       # ruido gaussiano, σ ~ U(0, ruido) por imagen
        x = x + torch.randn(x.shape, generator=g).to(dev) * (u(n, 1, 1, 1) * ruido)
    return x.clamp(0, 1)


@torch.no_grad()
def predecir(modelo: torch.nn.Module, X: torch.Tensor, media: float, dt: float, dev: torch.device,
             lote: int = 256) -> np.ndarray:
    modelo.eval()
    out = [torch.sigmoid(modelo((X[i:i + lote].to(dev).float() - media) / dt)).cpu()
           for i in range(0, len(X), lote)]
    return torch.cat(out).numpy().astype(float)


# --------------------------------------------------------------------------------------------------------
# Un pliegue
# --------------------------------------------------------------------------------------------------------
def entrenar_pliegue(cfg_modelo: dict, X: torch.Tensor, nodule_id: np.ndarray, y: np.ndarray, rol: np.ndarray,
                     semilla: int, dev: torch.device, epocas: int | None = None, verbose: bool = True) -> dict:
    """Entrena con 'entrenamiento', elige época y umbral con 'validacion' y predice 'prueba'.

    X: cortes (N, C, H, W) en [0, 1]; nodule_id, y, rol: uno por corte. Las etiquetas de los cortes de prueba
    se reciben (para no complicar la interfaz) pero no se leen: el resultado es idéntico si se cambian."""
    e = {**ENTRENO_DEF, **cfg_modelo.get("entrenamiento", {})}
    a = {**AUMENTO_DEF, **cfg_modelo.get("aumento", {})}
    epocas = int(epocas or e["epocas"])
    fijar_semillas(semilla)
    g = torch.Generator().manual_seed(semilla)
    tr, va, te = (np.flatnonzero(rol == r) for r in ("entrenamiento", "validacion", "prueba"))

    Xtr = X[tr]
    if e["normalizar"]:                                            # estadísticos SOLO del entrenamiento
        media = float(Xtr.float().mean())
        dt = float(Xtr.float().std()) or 1.0
    else:
        media, dt = 0.0, 1.0
    ytr = torch.tensor(y[tr], dtype=torch.float32)
    wtr = torch.from_numpy(pesos_cortes(nodule_id[tr], y[tr], e["ponderar_por_cortes"], e["ponderar_clase"]))

    modelo = cnn.construir(cfg_modelo, X.shape[1]).to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=float(e["tasa_aprendizaje"]),
                            weight_decay=float(e["decaimiento_pesos"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas)
    y_va_nod = pd.Series(y[va], index=nodule_id[va]).groupby(level=0, sort=False).first()

    mejor, mejor_ep, mejor_estado, hist = -np.inf, -1, None, []
    lote = int(e["tam_lote"])
    for ep in range(epocas):
        t0 = time.time()
        modelo.train()
        perm = torch.randperm(len(tr), generator=g)
        if len(perm) % lote == 1:                                  # BatchNorm no admite lotes de 1
            perm = perm[:-1]
        perdida = 0.0
        for i in range(0, len(perm), lote):
            idx = perm[i:i + lote]
            xb = aumentar(Xtr[idx].to(dev).float(), a, g)
            xb = (xb - media) / dt
            yb, wb = ytr[idx].to(dev), wtr[idx].to(dev)
            l = (F.binary_cross_entropy_with_logits(modelo(xb), yb, reduction="none") * wb).sum() / wb.sum()
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            perdida += float(l.detach()) * len(idx)
        sched.step()
        p_va = agregacion.por_nodulo(nodule_id[va], predecir(modelo, X[va], media, dt, dev), e["agregacion"])
        auc_va = float(roc_auc_score(y_va_nod.loc[p_va.index], p_va.values))
        hist.append(dict(epoca=ep, perdida=perdida / len(perm), auc_validacion=auc_va,
                         lr=sched.get_last_lr()[0], segundos=round(time.time() - t0, 1)))
        if verbose:
            print(f"    época {ep:3d}  pérdida {hist[-1]['perdida']:.4f}  AUC val {auc_va:.4f}  ({hist[-1]['segundos']} s)")
        if auc_va > mejor + 1e-4:
            mejor, mejor_ep = auc_va, ep
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        elif ep - mejor_ep >= int(e["paciencia"]):
            break
    modelo.load_state_dict(mejor_estado)

    p_va = agregacion.por_nodulo(nodule_id[va], predecir(modelo, X[va], media, dt, dev), e["agregacion"])
    umbral = metricas.umbral_youden(y_va_nod.loc[p_va.index].values, p_va.values)
    p_te_cortes = predecir(modelo, X[te], media, dt, dev)
    return dict(modelo=modelo, umbral=umbral, mejor_epoca=mejor_ep, auc_validacion=mejor, historial=hist,
                normalizacion=dict(media=media, dt=dt), indices_prueba=te, prob_cortes_prueba=p_te_cortes)


def _nodulos_de_prueba(nodule_id_te: np.ndarray, p_cortes: np.ndarray, umbral: float, agg: str) -> pd.DataFrame:
    out = pd.DataFrame({m: agregacion.por_nodulo(nodule_id_te, p_cortes, m) for m in agregacion.METODOS})
    out["probabilidad"] = out[agg]
    out["prediccion"] = (out.probabilidad >= umbral).astype(int)
    out["umbral"] = umbral
    return out.reset_index()


# --------------------------------------------------------------------------------------------------------
# Experimento completo
# --------------------------------------------------------------------------------------------------------
def _datos(cfg: dict, raiz: Path, permitir_faltantes: bool = False
           ) -> tuple[pd.DataFrame, pd.DataFrame, torch.Tensor, Path, list[str]]:
    res_dir = Path(cfg["salidas"]["dir_resultados"])
    res_dir = res_dir if res_dir.is_absolute() else raiz / res_dir
    datos_dir = res_dir / "datos"
    nod = dn.construir_o_cargar(cfg, datos_dir)
    part = dp.generar_o_cargar(nod, cfg, raiz)
    d = part.merge(nod, on=["nodule_id", "patient_id"], validate="one_to_one")
    excluidos: list[str] = []
    try:
        carpeta = dr.asegurar(cfg, d.nodule_id.tolist(), datos_dir)
    except RuntimeError as err:
        if not permitir_faltantes:
            raise
        carpeta = dr.carpeta_cache(cfg, datos_dir)
        excluidos = [n for n in d.nodule_id if not (carpeta / f"{n}.npz").exists()]
        print(f"⚠ {err}\n⚠ --permitir-faltantes: se excluyen {len(excluidos)} nódulos sin recorte. "
              "Esta ejecución NO vale para el TFM.")
        d = d[~d.nodule_id.isin(excluidos)].reset_index(drop=True)
    politica = cfg["modelo"].get("entrada", {}).get("politica", "S2")
    X, _, man = dr.cargar(carpeta, d.nodule_id.tolist(), politica, float(cfg["datos"]["area_minima_corte_mm2"]))
    cols = ["nodule_id", "patient_id", "pliegue", "etiqueta", *[c for c in d if c.startswith("rol_")]]
    man = man.merge(d[cols], on="nodule_id", how="left", validate="many_to_one")
    faltan = set(d.nodule_id) - set(man.nodule_id)
    assert not faltan, f"{len(faltan)} nódulos sin ningún corte con la política {politica}"
    return d, man, torch.from_numpy(X), res_dir, excluidos


def ejecutar(ruta_cfg: str, pliegues: list[int] | None = None, semillas: list[int] | None = None,
             epocas: int | None = None, n_control: int = 0, pref_dispositivo: str = "auto",
             permitir_faltantes: bool = False) -> dict:
    t0 = time.time()
    cfg = config.cargar(ruta_cfg)
    m, cp, regla = cfg["modelo"], cfg["particion"], cfg["datos"]["regla"]
    e = {**ENTRENO_DEF, **m.get("entrenamiento", {})}
    raiz = Path(__file__).resolve().parents[2]
    d, man, X, res_dir, excluidos = _datos(cfg, raiz, permitir_faltantes)
    dev = dispositivo(pref_dispositivo)
    pliegues = list(range(cp["n_pliegues"])) if pliegues is None else pliegues
    semillas = list(e.get("semillas", [cp["semilla"]])) if semillas is None else semillas
    completo = len(pliegues) == cp["n_pliegues"] and not excluidos

    cfg_huella = json.loads(json.dumps(cfg))
    if epocas:
        cfg_huella["modelo"].setdefault("entrenamiento", {})["epocas"] = epocas
    if excluidos:
        cfg_huella["nodulos_excluidos"] = sorted(excluidos)
    exp_id = f"{m['nombre']}__{regla}__{config.huella(cfg_huella)}"
    out = res_dir / exp_id
    out.mkdir(parents=True, exist_ok=True)
    print(f"{m['id']} · {regla} · {len(d)} nódulos · {len(man)} cortes ({m.get('entrada', {}).get('politica', 'S2')}) "
          f"· entrada {tuple(X.shape[1:])} · dispositivo {dev} · {out}")

    nid, y_c = man.nodule_id.values, man.etiqueta.values.astype(int)
    por_semilla, info_pliegues = {}, []
    for s in semillas:
        preds = []
        for k in pliegues:
            sub = out / f"s{s}" / f"pliegue{k}"
            f_pred = sub / "predicciones_nodulo.csv"
            if f_pred.exists():                                    # reanudación (RNF04)
                print(f"  semilla {s} · pliegue {k}: ya terminado, se reutiliza")
                preds.append(pd.read_csv(f_pred))
                info_pliegues.append(json.loads((sub / "pliegue.json").read_text()))
                continue
            print(f"  semilla {s} · pliegue {k}")
            sub.mkdir(parents=True, exist_ok=True)
            tp = time.time()
            r = entrenar_pliegue(m, X, nid, y_c, man[f"rol_{k}"].values, s, dev, epocas)
            te = r["indices_prueba"]
            pn = _nodulos_de_prueba(nid[te], r["prob_cortes_prueba"], r["umbral"], e["agregacion"])
            pn.insert(1, "pliegue", k)
            torch.save(dict(estado=r["modelo"].state_dict(), normalizacion=r["normalizacion"], umbral=r["umbral"],
                            configuracion_modelo=m), sub / "modelo.pt")
            pd.DataFrame(r["historial"]).to_csv(sub / "historial.csv", index=False)
            pd.DataFrame(dict(nodule_id=nid[te], k_corte=man.k_corte.values[te],
                              probabilidad=r["prob_cortes_prueba"].round(6))).to_csv(sub / "predicciones_cortes.csv", index=False)
            info = dict(semilla=s, pliegue=k, mejor_epoca=r["mejor_epoca"], epocas_ejecutadas=len(r["historial"]),
                        auc_validacion=round(r["auc_validacion"], 4), umbral=round(float(r["umbral"]), 4),
                        n_cortes_entrenamiento=int((man[f"rol_{k}"] == "entrenamiento").sum()),
                        segundos=round(time.time() - tp, 1))
            (sub / "pliegue.json").write_text(json.dumps(info, indent=1))
            pn.to_csv(f_pred, index=False)                         # se escribe al final: marca de pliegue terminado
            info_pliegues.append(info)
            preds.append(pn)
        por_semilla[s] = pd.concat(preds).merge(d[["nodule_id", "patient_id", "etiqueta"]], on="nodule_id")

    resumen = {}
    for s, p in por_semilla.items():
        yv, pv = p.etiqueta.values, p.probabilidad.values
        resumen[s] = dict(
            metricas=metricas.con_intervalos(yv, pv, p.prediccion.values, p.patient_id.values,
                                             cfg["evaluacion"]["n_bootstrap"], cp["semilla"]),
            auc_prueba_por_pliegue={int(k): round(float(roc_auc_score(g.etiqueta, g.probabilidad)), 4)
                                    for k, g in p.groupby("pliegue")},
            auc_por_agregacion={a: round(float(roc_auc_score(yv, p[a])), 4) for a in agregacion.METODOS})
        p.to_csv(out / f"predicciones_oof_s{s}.csv", index=False)
    aucs = [r["metricas"]["auc"]["valor"] for r in resumen.values()]
    entre_semillas = dict(auc_media=round(float(np.mean(aucs)), 4),
                          auc_dt=round(float(np.std(aucs, ddof=1)), 4) if len(aucs) > 1 else None)

    control = None
    if n_control:
        s0 = semillas[0]

        def permutado(rng):
            p = np.full(len(d), np.nan)
            pos = {n: i for i, n in enumerate(d.nodule_id)}
            for k in range(cp["n_pliegues"]):
                rol_nod = d[f"rol_{k}"].values
                y_nod = d.etiqueta.values.copy()
                no_te = rol_nod != "prueba"
                y_nod[no_te] = rng.permutation(y_nod[no_te])       # las etiquetas de prueba no se tocan
                y_perm = pd.Series(y_nod, index=d.nodule_id).loc[nid].values
                r = entrenar_pliegue(m, X, nid, y_perm, man[f"rol_{k}"].values, s0, dev, epocas, verbose=False)
                pk = agregacion.por_nodulo(nid[r["indices_prueba"]], r["prob_cortes_prueba"], e["agregacion"])
                p[[pos[n] for n in pk.index]] = pk.values
            return p

        control = controles.control_negativo(permutado, d.etiqueta.values, n_permutaciones=n_control,
                                             semilla=cp["semilla"],
                                             auc_observado=resumen[s0]["metricas"]["auc"]["valor"])

    commit = _commit()
    registro = dict(
        experimento=exp_id, modelo=m["id"], commit=commit, fecha=time.strftime("%Y-%m-%d %H:%M:%S"),
        completo=completo, pliegues=pliegues, semillas=semillas,
        nota=None if completo else "Ejecución parcial (pliegues o nódulos incompletos): las métricas no son las del TFM.",
        nodulos_excluidos=sorted(excluidos),
        configuracion=cfg, huella_configuracion=config.huella(cfg_huella),
        datos=dict(regla=regla, nodulos=int(len(d)), pacientes=int(d.patient_id.nunique()),
                   alta_sospecha=int(d.etiqueta.sum()), cortes=int(len(man)), forma_entrada=list(X.shape[1:]),
                   huella_particion=dp.huella(d[[c for c in d if c in ("nodule_id", "patient_id", "etiqueta", "pliegue")
                                                 or c.startswith("rol_")]]),
                   parametros_recorte=dr.parametros(cfg)),
        modelo_info=dict(arquitectura=m.get("arquitectura"), n_parametros=cnn.n_parametros(cnn.construir(m, X.shape[1]))),
        pliegues_detalle=info_pliegues, resultados_por_semilla={str(k): v for k, v in resumen.items()},
        entre_semillas=entre_semillas, control_negativo=control,
        entorno=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                     torch=torch.__version__, dispositivo=str(dev), maquina=platform.platform()),
        carpeta_resultados=str(out), duracion_s=round(time.time() - t0, 1))
    (out / "registro.json").write_text(json.dumps(registro, indent=1, ensure_ascii=False, default=float))
    return registro


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pliegues", type=int, nargs="*", default=None, help="por defecto, todos")
    ap.add_argument("--semillas", type=int, nargs="*", default=None, help="por defecto, las del YAML")
    ap.add_argument("--epocas", type=int, default=None, help="sobrescribe el máximo de épocas (pruebas rápidas)")
    ap.add_argument("--dispositivo", default="auto", help="auto, cuda, mps o cpu")
    ap.add_argument("--control-negativo", type=int, default=0, metavar="N",
                    help="repite la validación cruzada N veces con etiquetas permutadas (el AUC debe rondar 0,5)")
    ap.add_argument("--permitir-faltantes", action="store_true",
                    help="solo para pruebas: excluye los nódulos cuya exploración no se pudo leer (no vale para el TFM)")
    a = ap.parse_args()
    r = ejecutar(a.config, a.pliegues, a.semillas, a.epocas, a.control_negativo, a.dispositivo, a.permitir_faltantes)
    print(f"\n{r['modelo']} · {r['datos']['regla']} · {r['datos']['nodulos']} nódulos · {r['datos']['cortes']} cortes"
          f" · {r['modelo_info']['n_parametros']:,} parámetros" + ("" if r["completo"] else "  [PARCIAL]"))
    for s, v in r["resultados_por_semilla"].items():
        print(f" semilla {s}:")
        for k, x in v["metricas"].items():
            print(f"   {k:<14} {x['valor']:.3f}  IC95 [{x['ic95'][0]:.3f}, {x['ic95'][1]:.3f}]")
        print(f"   AUC por pliegue {v['auc_prueba_por_pliegue']} · por agregación {v['auc_por_agregacion']}")
    es = r["entre_semillas"]
    if es["auc_dt"] is not None:
        print(f" AUC entre semillas: {es['auc_media']:.3f} ± {es['auc_dt']:.3f}")
    if r["control_negativo"]:
        c = r["control_negativo"]
        print(f" Control negativo ({c['n_permutaciones']}): AUC media {c['auc_media']:.3f} "
              f"[{c['auc_p2_5']:.3f}, {c['auc_p97_5']:.3f}] · p = {c['p_valor']:.4f}")
    print(f"Resultados en {r['carpeta_resultados']}")


if __name__ == "__main__":
    main()
