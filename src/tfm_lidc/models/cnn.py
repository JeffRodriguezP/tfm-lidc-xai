"""M3 · CNN compacta entrenada desde cero (Tabla 21): establece si el preentrenamiento de M4/M5 aporta algo.

Arquitectura (por defecto, recorte de 128 × 128):

    entrada (C × 128 × 128)                      C = 1 (S1/S2) o 3 (S3, 2,5D)
    bloque 0: [conv 3×3 → BN → ReLU] × 2,  32 canales → 128 × 128 → max-pool → 64 × 64
    bloque 1: [conv 3×3 → BN → ReLU] × 2,  64 canales →  64 ×  64 → max-pool → 32 × 32
    bloque 2: [conv 3×3 → BN → ReLU] × 2, 128 canales →  32 ×  32 → max-pool → 16 × 16
    bloque 3: [conv 3×3 → BN → ReLU] × 2, 256 canales →  16 ×  16          (última capa convolucional)
    agrupación global media → abandono → lineal → 1 logit (alta sospecha)

≈ 1,2 M de parámetros. La última capa convolucional conserva un mapa de 16 × 16 (4 mm por celda con 0,5 mm/px),
frente a los 4 × 4 de una ResNet con la misma entrada: es la capa de Grad-CAM++; LayerCAM puede usar además
`bloques.2` (32 × 32). `capas_cam()` devuelve esos módulos para el módulo de explicabilidad.
"""
from __future__ import annotations

import torch
from torch import nn


def _bloque(c_in: int, c_out: int) -> nn.Sequential:
    capas = []
    for c in (c_in, c_out):
        capas += [nn.Conv2d(c, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*capas)


class CNNCompacta(nn.Module):
    def __init__(self, canales_entrada: int = 1, anchos: tuple[int, ...] = (32, 64, 128, 256), abandono: float = 0.3):
        super().__init__()
        self.bloques = nn.ModuleList()
        c = canales_entrada
        for a in anchos:
            self.bloques.append(_bloque(c, a))
            c = a
        self.pool = nn.MaxPool2d(2)
        self.cabeza = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(abandono), nn.Linear(c, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def caracteristicas(self, x: torch.Tensor) -> torch.Tensor:
        """Mapa de la última capa convolucional (lo reutilizan M6 y M7 como rama de imagen)."""
        for i, b in enumerate(self.bloques):
            x = b(x)
            if i < len(self.bloques) - 1:
                x = self.pool(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.caracteristicas(x)).squeeze(1)

    def capas_cam(self) -> dict[str, nn.Module]:
        return {"ultima": self.bloques[-1], "intermedia": self.bloques[-2]}


def construir(cfg_modelo: dict, canales_entrada: int) -> nn.Module:
    par = dict(cfg_modelo.get("parametros", {}))
    arq = cfg_modelo.get("arquitectura", "cnn_compacta")
    if arq != "cnn_compacta":
        raise ValueError(f"Arquitectura no soportada: {arq}")
    return CNNCompacta(canales_entrada, tuple(par.get("anchos", (32, 64, 128, 256))), float(par.get("abandono", 0.3)))


def n_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
