# Guía de M3 · CNN compacta desde cero

M3 es el primer modelo de imagen (Tabla 21 de la memoria V14): una red convolucional pequeña entrenada desde cero con
recortes de TC. Sirve para medir si el preentrenamiento de M4 (ResNet) y M5 (EfficientNet) aporta algo, y comparte con
ellos los recortes, el protocolo y la evaluación.

> **Configuración seleccionada: `configs/modelos/m3_cnn_fov48.yaml`** (recortes de 48 mm, 96 × 96 px a 0,5 mm).
> La configuración inicial de la memoria, `m3_cnn.yaml` (64 mm, 128 × 128 px), se conserva sin cambios como
> análisis de sensibilidad del campo de visión. Decisión y justificación:
> [`docs/decisiones/2026-11-19_m3_campo_vision_48mm.md`](decisiones/2026-11-19_m3_campo_vision_48mm.md).

## Qué se ha implementado

| Pieza | Archivo | Requisito / apartado |
|---|---|---|
| Recortes DICOM → campo de visión y espaciado configurables (48 mm → 96 × 96; 64 mm → 128 × 128; 0,5 mm/px), HU, ventana [−1000, 400], máscara de consenso | `src/tfm_lidc/data/recortes.py` | RF-D06, RD-03, 3.3.5.7 |
| Políticas S1 / S2 / S3 (2,5D) al cargar | `recortes.cargar` | 3.3.5.6 |
| Arquitectura (4 bloques, 1.172.897 parámetros; último mapa de 12 × 12 con 96 px o 16 × 16 con 128 px, siempre 4 mm por celda) | `src/tfm_lidc/models/cnn.py` | Tabla 21 |
| Agregación de cortes en nódulo (media; máximo y media recortada) | `src/tfm_lidc/evaluation/agregacion.py` | RF-M05 |
| Entrenamiento y evaluación por pliegue, reanudación, registro | `src/tfm_lidc/imagen.py` | RF-M02, RF-M06, RF-M10, RNF04 |
| Configuraciones | `m3_cnn_fov48.yaml` (seleccionada), `m3_cnn.yaml` (64 mm), variantes `m3_cnn_25d`, `_s1`, `_reg`, `_ruido`; sección `recortes` en `base.yaml` | |
| Comparación de variantes por validación | `scripts/comparar_variantes_m3.py`, `scripts/cola_variantes_m3.sh` | RNF03 |
| Pruebas sintéticas (geometría, ventana, pesos, agregación, aumento, no fuga) | `tests/test_m3.py` | RNF03 |
| Cuaderno de exploración y resultados | `notebooks/06_m3_cnn.ipynb` | |

## Paso a paso

### 1. Entorno

```bash
conda activate TFM
pip install -e ".[dev]"
pytest -q            # incluye las pruebas de M3
```

PyTorch usa la GPU de Apple (MPS) automáticamente; en Colab, CUDA. `--dispositivo cpu` lo fuerza a CPU.

### 2. Rutas a los datos

`configs/local.yaml` (no versionado) debe tener `datos.dicom_root` apuntando a `.../TCIA_LIDC-IDRI_20200921/lidc_idri`
(o bien `~/.pylidcrc`).

### 3. Extraer los recortes (una vez por campo de visión, ~6–8 min)

```bash
python -m tfm_lidc.data.recortes --config configs/modelos/m3_cnn_fov48.yaml --procesos 6
```

Crea `resultados/datos/recortes/fov48_px0.5_w-1000_400_c0.5_v1/` con un `.npz` por nódulo y `control_calidad.csv`
(z monótona, espaciado, rango HU, vínculo contorno-corte: Tabla 20). Lo ya extraído no se repite, y el entrenamiento
extrae lo que falte. Cada campo de visión tiene su carpeta (`fov64_…` para `m3_cnn.yaml`). Cada corte guarda también
sus vecinos k−1 y k+1, así que la variante 2,5D no necesita extraer de nuevo.

Comprobaciones hechas: las áreas de consenso por corte coinciden con `cortes_por_corte_v1.csv` (las de
`contar_cortes.py`; R1: 869 nódulos, 6.070 cortes S2) y los recortes de 64 mm coinciden (r = 0,99999) con los recortes
parciales previos de `recortes_S2/_partes`. Con 48 mm, 2 de los 869 nódulos (de 30 y 38 mm) tocan el borde del
recorte en 1 corte cada uno; con 64 mm, ninguno.

### 4. Prueba rápida (1 pliegue, 1 semilla, 3 épocas, ~1 min)

```bash
python -m tfm_lidc.imagen --config configs/modelos/m3_cnn_fov48.yaml --pliegues 0 --semillas 42 --epocas 3
```

### 5. Ejecución completa (5 pliegues × 3 semillas, ~1 h en MPS)

```bash
python -m tfm_lidc.imagen --config configs/modelos/m3_cnn_fov48.yaml
```

Unos 7,5 s por época en MPS (12,5 s con 64 mm). Si se interrumpe, al relanzar el mismo comando se reutilizan los
pliegues terminados (`resultados/m3_cnn_fov48__R1__<huella>/s<semilla>/pliegue<k>/`).
Control con etiquetas permutadas (RF-M08; caro: N × 5 entrenamientos): `--control-negativo 20`.

### 6. Resultados

`registro.json` contiene configuración, huellas, versiones, dispositivo, métricas con IC 95 % por semilla, AUC por
pliegue y por método de agregación, y la media ± desviación típica del AUC entre semillas. `predicciones_oof_s<semilla>.csv`
tiene la predicción fuera de muestra de cada nódulo, con las mismas columnas que M0–M2 para las comparaciones pareadas
(H1: mejor de M3–M5 frente a M0). El cuaderno `notebooks/06_m3_cnn.ipynb` los presenta.

## Resultados (R1, 869 nódulos, 5 pliegues × 3 semillas)

AUC de prueba fuera de pliegue por nódulo, IC 95 % por bootstrap de pacientes; ΔAUC pareado frente a M0 (0,936
[0,917; 0,952]) con el mismo remuestreo.

**M3 seleccionado · 48 mm** (`resultados/m3_cnn_fov48__R1__d9a71489d195`):

| Semilla | AUC [IC 95 %] | Exactitud | Sensibilidad | Especificidad | Brier | ΔAUC vs M0 [IC 95 %] |
|---|---|---|---|---|---|---|
| 42 | 0,923 [0,902; 0,942] | 0,849 | 0,877 | 0,824 | 0,112 | −0,013 [−0,031; +0,007] |
| 43 | 0,919 [0,900; 0,938] | 0,830 | 0,839 | 0,821 | 0,114 | −0,017 [−0,036; +0,004] |
| 44 | 0,920 [0,900; 0,939] | 0,835 | 0,827 | 0,844 | 0,111 | −0,016 [−0,037; +0,006] |
| **Media ± DT** | **0,921 ± 0,002** | | | | | |

**Sensibilidad · 64 mm** (`resultados/m3_cnn__R1__4a6c74d26875`, configuración inicial de la memoria):

| Semilla | AUC [IC 95 %] | Exactitud | Sensibilidad | Especificidad | Brier | ΔAUC vs M0 [IC 95 %] |
|---|---|---|---|---|---|---|
| 42 | 0,889 [0,867; 0,912] | 0,795 | 0,853 | 0,741 | 0,134 | −0,046 [−0,069; −0,022] |
| 43 | 0,868 [0,836; 0,897] | 0,817 | 0,839 | 0,797 | 0,145 | −0,068 [−0,097; −0,040] |
| 44 | 0,917 [0,892; 0,940] | 0,835 | 0,815 | 0,855 | 0,111 | −0,019 [−0,043; +0,005] |
| **Media ± DT** | **0,891 ± 0,025** | | | | | |

Lectura: con 48 mm, M3 mejora en +0,030 de AUC medio y es mucho más estable entre semillas (DT 0,002 frente a 0,025).
Queda a la altura de M0 (ΔAUC ≈ −0,015, con todos los IC conteniendo el 0), pero **no lo supera**: H1 sigue
pendiente de M4 y M5. El promedio de probabilidades entre semillas (ensamble) es otro modelo y no es la cifra que pide
la memoria (media ± DT entre semillas).

Nota de trazabilidad: los `registro.json` de las ejecuciones anteriores al 19/11 a las 14:30 citan los identificadores
de commit previos a la reescritura de mensajes de la rama (p. ej. `afde3e6` → `e2149ee`); el código es idéntico.

## Experimentos de ajuste (semilla 42, 5 pliegues; selección por validación)

| Variante | YAML | Cambio frente a `m3_cnn` | AUC medio de validación |
|---|---|---|---|
| **Campo de 48 mm** | `m3_cnn_fov48.yaml` | `recortes.fov_mm: 48` (96 × 96 px a 0,5 mm), desplazamiento 6 px | **0,927** (mejor en 5 de 5 pliegues) |
| Base (64 mm) | `m3_cnn.yaml` | — | 0,917 |
| Ruido y desenfoque | `m3_cnn_ruido.yaml` | `aumento.ruido: 0.03`, `aumento.desenfoque: 0.3` | 0,915 |
| Regularización | `m3_cnn_reg.yaml` | Abandono 0,5; decaimiento 1e-3; lr 3e-4; paciencia 15 | 0,914 |
| 2,5D | `m3_cnn_25d.yaml` | Política S3: cortes k−1, k, k+1 como 3 canales | 0,912 |

La selección se hizo solo con la validación; la prueba se miró después, para la variante elegida, y se confirmó con
las semillas 43 y 44. Para repetir la comparación:

```bash
python scripts/comparar_variantes_m3.py                  # ordena por validación
python scripts/comparar_variantes_m3.py --mostrar-prueba # solo al final, para la variante elegida
```

## Dónde cambiar cada cosa

Regla general: **no edites los YAML de ejecuciones ya hechas** (`m3_cnn.yaml`, `m3_cnn_fov48.yaml`, `base.yaml`): su
huella da nombre a la carpeta de resultados. Crea un YAML nuevo que herede de la configuración seleccionada y cambie
solo lo necesario:

```yaml
# configs/modelos/m3_cnn_fov48_lr3e4.yaml
hereda: m3_cnn_fov48.yaml
modelo:
  nombre: m3_cnn_fov48_lr3e4    # obligatorio: da nombre a la carpeta de resultados
  entrenamiento:
    tasa_aprendizaje: 0.0003
```

```bash
python -m tfm_lidc.imagen --config configs/modelos/m3_cnn_fov48_lr3e4.yaml --semillas 42   # prueba con una semilla
```

| Quiero cambiar… | Dónde | Clave / qué tocar | ¿Hay que volver a extraer recortes? |
|---|---|---|---|
| Tasa de aprendizaje, decaimiento, lote | YAML del modelo | `entrenamiento.tasa_aprendizaje`, `decaimiento_pesos`, `tam_lote` | No |
| Épocas máximas, paciencia | YAML del modelo | `entrenamiento.epocas`, `paciencia` | No |
| Número de semillas | YAML (`entrenamiento.semillas`) o `--semillas` | Lista de enteros | No |
| Pesos de la pérdida | YAML del modelo | `ponderar_por_cortes`, `ponderar_clase` | No |
| Agregación de cortes en nódulo | YAML del modelo | `entrenamiento.agregacion`: `media`, `maximo`, `media_recortada` | No |
| Aumento de datos | YAML del modelo | `aumento.giros`, `rotacion_grados`, `desplazamiento_px`, `intensidad`, `ruido`, `desenfoque` | No |
| Anchura / profundidad de la red, abandono | YAML del modelo | `parametros.anchos` (p. ej. `[16, 32, 64, 128]`; un elemento más = un bloque más), `abandono` | No |
| Política de cortes (S1 / S2 / S3) | YAML del modelo | `entrada.politica` | No |
| Área mínima de consenso por corte | YAML del modelo | `datos.area_minima_corte_mm2` | No (se filtra al cargar) |
| Campo de visión del recorte | YAML del modelo | `recortes.fov_mm` | **Sí** (carpeta nueva automática) |
| Resolución del recorte (p. ej. 224 px) | YAML del modelo | `recortes.mm_por_pixel` (48 mm / 224 px = 0,2143) | **Sí** |
| Ventana de intensidades | YAML del modelo | `recortes.ventana_hu` | **Sí** |
| Nivel de consenso de la máscara | YAML del modelo | `datos.nivel_consenso` | **Sí** (y cambia los cortes S2: revisar `contar_cortes.py`) |
| Regla de etiquetado (R2, R3) | YAML del modelo | `datos.regla` | Solo los nódulos nuevos |
| Sin el lanzamiento inicial | YAML del modelo | `datos.excluir_publicacion_inicial: true` | No |
| Otra arquitectura (otra capa, residual…) | `src/tfm_lidc/models/cnn.py` | Nueva clase + rama en `construir()` con `arquitectura: <nombre>` | No |
| Optimizador, planificador, pérdida focal | `src/tfm_lidc/imagen.py` | `entrenar_pliegue` (líneas `torch.optim…` y `binary_cross_entropy_with_logits`) | No |
| Criterio de umbral | `src/tfm_lidc/imagen.py` | `metricas.umbral_youden(...)` en `entrenar_pliegue` | No |
| Métricas | `src/tfm_lidc/evaluation/metricas.py` | `puntuales()` (afecta a todos los modelos) | No |

Antes de lanzar una ejecución completa: `pytest -q` (las pruebas de no fuga deben seguir pasando) y una prueba
rápida con `--pliegues 0 --semillas 42 --epocas 3`. Para comparar ajustes sin la terminal, usa la celda
«Laboratorio» del cuaderno 06 (un pliegue, pocas épocas; no son cifras del TFM). Elige siempre por AUC de validación.

## Protocolo (resumen)

- Misma partición versionada por paciente que M0–M2. Entrenamiento con 'entrenamiento'; parada temprana (AUC de
  nódulo) y umbral de Youden con 'validacion'; predicción de 'prueba'. Las pruebas comprueban que cambiar las
  etiquetas o las imágenes de prueba no cambia nada del entrenamiento.
- Unidad de entrenamiento: el corte (S2, 6.070 cortes con R1). Unidad de evaluación: el nódulo (media de sus cortes).
- Pérdida BCE ponderada: cada nódulo suma 1 y cada clase pesa lo mismo. AdamW, coseno, lote 64, lr 1e-3, wd 1e-4,
  abandono 0,3, hasta 60 épocas con paciencia 12.
- Aumento solo en entrenamiento: giros, rotación ±15°, traslación ±3 mm (6 px), intensidad ±10 %.
- Normalización con media y desviación típica de los cortes de entrenamiento de cada pliegue.

## Pendiente

- **Memoria V14:** reflejar la decisión del campo de visión (apartado 3.3.5.7, paso 4 y párrafo siguiente; criterio de
  aceptación de RF-D06, que cita 128 × 128) y presentar 64 mm como análisis de sensibilidad. Confirmar con la directora
  junto con las decisiones A6 y A7.
- **M4/M5:** la decisión A6 (224 frente a 128 px para redes preentrenadas) sigue abierta; 48 mm es el punto de partida
  a probar, no una decisión tomada para esos modelos.
- **Desplazamiento máximo del recorte (A7):** ahora es aumento en entrenamiento (`aumento.desplazamiento_px`); el
  recorte guardado está centrado.
- **Búsqueda aleatoria de hiperparámetros (3.3.6.2):** no implementada; los valores del YAML son razonables pero fijos.
- Resuelto el 18/11: los 15 archivos DICOM dañados de LIDC-IDRI-0334, -0572 y -0625 se volvieron a descargar del IDC;
  los 869 nódulos de R1 tienen recorte.
