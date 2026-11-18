# Guía de M3 · CNN compacta desde cero

M3 es el primer modelo de imagen (Tabla 21 de la memoria V14): una red convolucional pequeña entrenada desde cero con
recortes de TC. Sirve para medir si el preentrenamiento de M4 (ResNet) y M5 (EfficientNet) aporta algo, y comparte con
ellos los recortes, el protocolo y la evaluación.

## Qué se ha implementado

| Pieza | Archivo | Requisito / apartado |
|---|---|---|
| Recortes DICOM → 128 × 128 (64 mm a 0,5 mm/px), HU, ventana [−1000, 400], máscara de consenso | `src/tfm_lidc/data/recortes.py` | RF-D06, RD-03, 3.3.5.7 |
| Políticas S1 / S2 / S3 (2,5D) al cargar | `recortes.cargar` | 3.3.5.6 |
| Arquitectura (4 bloques, ≈1,17 M parámetros, último mapa de 16 × 16) | `src/tfm_lidc/models/cnn.py` | Tabla 21 |
| Agregación de cortes en nódulo (media; máximo y media recortada) | `src/tfm_lidc/evaluation/agregacion.py` | RF-M05 |
| Entrenamiento y evaluación por pliegue, reanudación, registro | `src/tfm_lidc/imagen.py` | RF-M02, RF-M06, RF-M10, RNF04 |
| Configuraciones | `configs/modelos/m3_cnn.yaml`, `m3_cnn_25d.yaml`, `m3_cnn_s1.yaml`; sección `recortes` en `base.yaml` | |
| Pruebas sintéticas (geometría, ventana, pesos, agregación, no fuga) | `tests/test_m3.py` | RNF03 |

## Paso a paso

### 1. Entorno

```bash
conda activate TFM
pip install -e ".[dev]"
pytest -q            # 48 pruebas, incluidas las 15 de M3
```

PyTorch usa la GPU de Apple (MPS) automáticamente; en Colab, CUDA. `--dispositivo cpu` lo fuerza a CPU.

### 2. Rutas a los datos

`configs/local.yaml` (no versionado) debe tener `datos.dicom_root` apuntando a `.../TCIA_LIDC-IDRI_20200921/lidc_idri`
(o bien `~/.pylidcrc`).

### 3. Extraer los recortes (una vez, ~6 min con 6 procesos)

```bash
python -m tfm_lidc.data.recortes --config configs/base.yaml --procesos 6
```

Crea `resultados/datos/recortes/fov64_px0.5_w-1000_400_c0.5_v1/` con un `.npz` por nódulo y `control_calidad.csv`
(z monótona, espaciado, rango HU, vínculo contorno-corte: Tabla 20). Lo ya extraído no se repite. Cada corte guarda
también sus vecinos k−1 y k+1, así que la variante 2,5D no necesita extraer de nuevo.

Comprobaciones hechas: las áreas de consenso por corte coinciden con `cortes_por_corte_v1.csv` (las de
`contar_cortes.py`) y los recortes coinciden (r = 0,99999) con los recortes parciales previos de `recortes_S2/_partes`.

### 4. Prueba rápida (1 pliegue, 1 semilla, 3 épocas, ~1 min)

```bash
python -m tfm_lidc.imagen --config configs/modelos/m3_cnn.yaml --pliegues 0 --semillas 42 --epocas 3
```

### 5. Ejecución completa (5 pliegues × 3 semillas)

```bash
python -m tfm_lidc.imagen --config configs/modelos/m3_cnn.yaml
```

Unos 12,5 s por época en MPS; con parada temprana, del orden de 1,5–3 h en total. Si se interrumpe, al relanzar el
mismo comando se reutilizan los pliegues terminados (`resultados/m3_cnn__R1__<huella>/s<semilla>/pliegue<k>/`).

Variantes (factores de sensibilidad): `m3_cnn_25d.yaml` (S3) y `m3_cnn_s1.yaml` (S1).
Control con etiquetas permutadas (RF-M08; caro: N × 5 entrenamientos): `--control-negativo 20`.

### 6. Resultados

`registro.json` contiene configuración, huellas, versiones, dispositivo, métricas con IC 95 % por semilla, AUC por
pliegue y por método de agregación, y la media ± desviación típica del AUC entre semillas. `predicciones_oof_s<semilla>.csv`
tiene la predicción fuera de muestra de cada nódulo, con las mismas columnas que M0–M2 para las comparaciones pareadas
(H1: mejor de M3–M5 frente a M0).

## Protocolo (resumen)

- Misma partición versionada por paciente que M0–M2. Entrenamiento con 'entrenamiento'; parada temprana (AUC de
  nódulo) y umbral de Youden con 'validacion'; predicción de 'prueba'. Las pruebas comprueban que cambiar las
  etiquetas o las imágenes de prueba no cambia nada del entrenamiento.
- Unidad de entrenamiento: el corte (S2, 6.070 cortes con R1). Unidad de evaluación: el nódulo (media de sus cortes).
- Pérdida BCE ponderada: cada nódulo suma 1 y cada clase pesa lo mismo. AdamW, coseno, lote 64, lr 1e-3, wd 1e-4.
- Aumento solo en entrenamiento: giros, rotación ±15°, traslación ±4 mm, intensidad ±10 %.
- Normalización con media y desviación típica de los cortes de entrenamiento de cada pliegue.

## Pendiente

- **Archivos DICOM dañados en la copia local.** 15 archivos `.dcm` de 3 series tienen la cabecera a ceros
  (LIDC-IDRI-0334, -0572 y -0625), por lo que faltan 3 de los 869 nódulos de R1. Hay que volver a descargar esas
  tres series de TCIA. Mientras tanto, `--permitir-faltantes` permite probar excluyéndolos, y el registro marca la
  ejecución como no válida para el TFM.
- Decisión A6 (128 frente a 224): basta con cambiar `recortes.mm_por_pixel` (0,2857 para 224 px); la red se adapta.
- Desplazamiento máximo del recorte (A7): ahora es aumento en entrenamiento (`aumento.desplazamiento_px`); el recorte
  guardado está centrado.
- Búsqueda aleatoria de hiperparámetros (3.3.6.2): todavía no; los valores del YAML son razonables pero fijos.
