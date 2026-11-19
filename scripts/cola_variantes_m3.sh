#!/bin/bash
# Cola de experimentos de ajuste de M3 (semilla 42, 5 pliegues cada uno). Reanudable: relanzar repite solo lo que falte.
cd /Users/jefferson.rodriguez/Documents/UNIR_II_Cuatrimestre/TFM/tfm-lidc-xai || exit 1
PY=~/Documents/miniconda/2026/TFM/bin/python
for c in m3_cnn_25d m3_cnn_reg m3_cnn_ruido m3_cnn_fov48; do
  echo "================ $c ($(date '+%H:%M')) ================"
  "$PY" -u -W ignore -m tfm_lidc.imagen --config "configs/modelos/$c.yaml" --semillas 42 || echo "!!! $c terminó con error"
done
echo "================ COMPARACIÓN (validación) ================"
"$PY" scripts/comparar_variantes_m3.py
echo "COLA TERMINADA $(date '+%H:%M')"
