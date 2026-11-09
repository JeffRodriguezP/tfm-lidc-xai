"""Carga de configuraciones YAML con herencia (`hereda:`) y sobrescritura local."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml


def _fusionar(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        out[k] = _fusionar(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def cargar(ruta: str | Path) -> dict:
    """Lee un YAML; si declara `hereda`, fusiona primero el archivo padre (ruta relativa al hijo).

    Si existe configs/local.yaml (no versionado), se aplica al final: ahí cada integrante pone su ruta a los datos.
    """
    ruta = Path(ruta).resolve()
    cfg = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
    padre = cfg.pop("hereda", None)
    if padre:
        cfg = _fusionar(cargar(ruta.parent / padre), cfg)
    local = _raiz_configs(ruta) / "local.yaml"
    if local.exists() and ruta != local:
        cfg = _fusionar(cfg, yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    return cfg


def _raiz_configs(ruta: Path) -> Path:
    for p in [ruta.parent, *ruta.parents]:
        if p.name == "configs":
            return p
    return ruta.parent


def huella(cfg: dict) -> str:
    """Huella corta y estable de una configuración (para nombrar y auditar experimentos)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
