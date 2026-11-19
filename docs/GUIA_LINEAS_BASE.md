# Guía de las líneas base tabulares · M0, M1 y M2

Las tres líneas base comparten código (`src/tfm_lidc/lineas_base.py`, `src/tfm_lidc/models/tabular.py`), partición y protocolo de evaluación. Cada una responde a una pregunta distinta y fija una cota que los modelos de imagen (M3 en adelante) deben tener en cuenta.

| Modelo | Entrada | Pregunta que responde | Tabla 21 |
|---|---|---|---|
| **M0** | Diámetro del nódulo | ¿Qué rendimiento da el tamaño por sí solo? Es el mínimo que debe superar el modelo de imagen (H1). | Línea base trivial |
| **M1** | Metadatos de adquisición (sin diámetro) | ¿Se puede predecir la etiqueta por cómo se adquirió la TC y no por cómo es el nódulo? Detector de atajos y de confusión. | Detector de atajos |
| **M2** | Ocho valoraciones semánticas del radiólogo (con y sin diámetro y volumen) | ¿Cuánto explican por sí solas las descripciones del radiólogo? Cota, no competidor de los modelos de imagen. | Atributos semánticos |

Ninguna usa imágenes. M0 y M2 solo necesitan la base de anotaciones de pylidc; M1 necesita además el `metadata.csv` de TCIA.

---

## 1. Datos y etiquetas (común a las tres)

- **Unidad de análisis:** el nódulo, es decir, las lecturas de los radiólogos agrupadas con `cluster_annotations` de pylidc (`data/nodulos.py`).
- **Regla R1 (principal):** al menos 3 lecturas. Mediana de malignidad < 3 → baja sospecha (0); > 3 → alta sospecha (1); = 3 → excluido.
- **Exclusión de clústeres anómalos:** un nódulo no puede tener más de 4 lecturas (hay 4 radiólogos). Los 14 clústeres que lo superan indican un fallo de agrupación y se excluyen: R1 pasa de 879 a **869 nódulos** (530 pacientes). `datos.excluir_clusteres_mas_de_4: false` los conserva.
- **Lanzamiento inicial de LIDC:** no se excluye, se marca (`publicacion_inicial`). `datos.excluir_publicacion_inicial: true` repite el análisis sin esos nódulos (584 nódulos, 339 pacientes), con la misma partición.
- **Agregación entre lectores:** diámetro y volumen = mediana de las lecturas; atributos ordinales = mediana; atributos nominales = moda.

## 2. Protocolo de evaluación (Tabla 19)

1. **Partición por paciente:** `StratifiedGroupKFold` con 5 pliegues, estratificada por etiqueta y agrupada por paciente, semilla 42. En cada pliegue de entrenamiento se reserva un 20 % de pacientes como validación (cerca del 16 % de los nódulos).
2. **Por pliegue:** el modelo se ajusta con el entrenamiento, el umbral se elige con la validación (índice de Youden) y se predice el pliegue de prueba. Las etiquetas de prueba no se leen nunca.
3. **Métricas:** AUC, AUPRC, Brier, exactitud, sensibilidad y especificidad sobre las predicciones fuera de muestra de los 5 pliegues, con IC 95 % por bootstrap de pacientes (2.000 remuestreos).
4. **Una sola partición para todos los modelos:** está versionada en `particiones/` (huella `0e6393241e7c69e0` para R1 con 869 nódulos). No se regenera: con otra versión de scikit-learn, el mismo código reparte distinto (1.7.2 y 1.9.1 solo coinciden en el 20 % de los nódulos).

### Por qué el umbral de Youden
Las métricas de clasificación (sensibilidad, especificidad, exactitud) necesitan un umbral; el AUC no. Youden maximiza J = sensibilidad + especificidad − 1, da el mismo peso a ambos errores y no depende de la prevalencia. Se elige en la validación para no ajustarlo con datos de evaluación. Supone el mismo coste para falsos positivos y falsos negativos, y con unos 140 nódulos de validación puede variar entre pliegues. No afecta al AUC, al AUPRC ni al Brier.

## 3. Configuración

Cada modelo tiene un YAML que hereda de `configs/base.yaml` y solo declara lo que cambia. `config.cargar` fusiona primero el padre y luego el hijo; al final aplica `configs/local.yaml` (no versionado) con las rutas de cada persona.

```
base.yaml                              regla R1, exclusión de clústeres, semilla 42, 5 pliegues, bootstrap, Youden
 ├── m0_diametro.yaml                  M0 principal
 │    ├── m0_diametro_sin_exclusion.yaml   hereda de m0_diametro.yaml (879 nódulos)
 │    └── m0_diametro_sin_inicial.yaml     hereda de m0_diametro.yaml (584 nódulos)
 ├── m1_metadatos.yaml                 M1 principal
 │    └── m1_metadatos_sin_inicial.yaml    hereda de m1_metadatos.yaml
 └── m2_logistica.yaml · m2_logistica_diam_vol.yaml
     m2_bosque.yaml · m2_bosque_diam_vol.yaml · m2_mlp.yaml
```

El archivo que se pasa en `--config` decide qué variante se ejecuta.

## 4. M0 · diámetro

- **Variable:** `diametro_mm`.
- **Estimador:** regresión logística con `C = 1e6` (sin regularización efectiva, porque hay una sola variable) y `max_iter = 1000`.
- **Preprocesado:** `StandardScaler` dentro de un `Pipeline`, ajustado solo con el entrenamiento de cada pliegue.
- **Control con etiquetas permutadas:** poco informativo en M0. Con una sola variable el modelo solo aprende el signo, y el AUC permutado oscila entre valores altos y bajos.

## 5. M1 · metadatos de adquisición

- **Variables:** grosor de corte, espaciado en el plano, espaciado entre cortes, contraste (numéricas) y fabricante (categórica, codificación one-hot). Sin diámetro: es la variante «solo metadatos».
- **Estimador:** regresión logística con `C = 1.0` (L2) y `max_iter = 2000`, por las variables correlacionadas y las categorías poco frecuentes.
- **Interpretación:** un AUC próximo a 0,5 descarta el atajo; uno claramente mayor obliga a controlarlo en M3–M7 (DeGrave et al., 2021; Geirhos et al., 2020).
- **Prueba de permutación:** `--control-negativo 1000` reentrena con las etiquetas de entrenamiento y validación permutadas y devuelve el p-valor del AUC observado (corrección +1 de Phipson y Smyth).
- **Dato adicional:** necesita `datos.metadata_csv` en `configs/local.yaml`.

## 6. M2 · atributos semánticos

- **Entrada:** ocho valoraciones del radiólogo. Seis ordinales (subtlety, sphericity, margin, lobulation, spiculation, texture; mediana entre lectores, estandarizadas) y dos nominales (internalStructure y calcification; moda entre lectores, one-hot). **La malignidad nunca es entrada**: lo comprueba `tests/test_m2.py`.
- **Imputación:** con estadísticos del entrenamiento (mediana o moda) más un indicador de imputación, ajustada solo con cada pliegue de entrenamiento.
- **Variantes de estimador:**

| Archivo | Estimador | Parámetros |
|---|---|---|
| `m2_logistica.yaml` | Regresión logística | `C = 1.0`, `max_iter = 2000` |
| `m2_bosque.yaml` | Bosque aleatorio | 500 árboles, `min_samples_leaf = 2` |
| `m2_mlp.yaml` | Perceptrón multicapa | capas `[32, 16]`, `alpha = 0.001`, `max_iter = 2000` |

- **Variantes con diámetro y volumen:** `m2_logistica_diam_vol` y `m2_bosque_diam_vol` añaden `diametro_mm` y `volumen_mm3` a las ocho valoraciones, para medir cuánto añade el tamaño.
- **Cómo leerlo:** las valoraciones y la etiqueta de malignidad las asignan los mismos radiólogos, en la misma sesión y sobre el mismo nódulo. Por eso M2 mide cuánto de la etiqueta explican esas valoraciones (escenario A): sirve como cota y referencia, no como predicción independiente de la malignidad.

## 7. Cómo ejecutar

```bash
# Entorno (una vez)
conda env create -f environment.yml && conda activate TFM && pip install -e ".[dev]"

# M0 y sus variantes
python -m tfm_lidc.lineas_base --config configs/modelos/m0_diametro.yaml
python -m tfm_lidc.lineas_base --config configs/modelos/m0_diametro_sin_exclusion.yaml
python -m tfm_lidc.lineas_base --config configs/modelos/m0_diametro_sin_inicial.yaml

# M1 (necesita datos.metadata_csv)
python -m tfm_lidc.lineas_base --config configs/modelos/m1_metadatos.yaml --control-negativo 1000

# M2
python -m tfm_lidc.lineas_base --config configs/modelos/m2_logistica.yaml
python -m tfm_lidc.lineas_base --config configs/modelos/m2_bosque.yaml
python -m tfm_lidc.lineas_base --config configs/modelos/m2_mlp.yaml
```

La primera ejecución lee las 6.859 lecturas y agrupa los nódulos (3–4 min); después queda en caché en `resultados/datos/`.

## 8. Salidas y trazabilidad

Cada ejecución crea `resultados/<modelo>__<regla>__s<semilla>__<commit>/` con:

| Archivo | Contenido |
|---|---|
| `predicciones_oof.csv` | Una fila por nódulo: pliegue, etiqueta, probabilidad fuera de muestra y predicción con el umbral del pliegue |
| `registro.json` | Configuración y su huella, commit, huella de la partición, umbrales y coeficientes por pliegue, AUC por pliegue, métricas con IC 95 %, control negativo y versiones del entorno |

Si el nombre de la carpeta acaba en `-modificado`, la ejecución se hizo con cambios sin commitear. **Las cifras que se citen en la memoria deben salir de una ejecución con el árbol limpio.**

## 9. Resultados (R1, 869 nódulos, 530 pacientes)

| Modelo | AUC [IC 95 %] | Sensibilidad / Especificidad |
|---|---|---|
| M0 · diámetro | 0,936 [0,917; 0,952] | 0,822 / 0,882 |
| M0 · sin exclusión de clústeres (879 nódulos) | 0,937 [0,920; 0,953] | 0,835 / 0,889 |
| M0 · sin lanzamiento inicial (584 nódulos) | 0,942 [0,920; 0,961] | 0,820 / 0,921 |
| M1 · metadatos | 0,541 [0,496; 0,588] · p = 0,047 por permutación | 0,601 / 0,469 |
| M2 · logística | 0,949 [0,933; 0,963] | 0,896 / 0,866 |
| M2 · bosque | 0,954 [0,940; 0,966] | 0,912 / 0,882 |
| M2 · perceptrón | 0,946 [0,931; 0,960] | 0,815 / 0,913 |
| M2 · logística + diámetro y volumen | 0,973 [0,962; 0,982] | 0,917 / 0,909 |
| M2 · bosque + diámetro y volumen | 0,969 [0,957; 0,979] | 0,922 / 0,909 |

Las cifras de M2 son **provisionales**: provienen de ejecuciones con el árbol modificado. Se confirman al repetirlas con el árbol limpio.

## 10. Pruebas de no fuga

| Prueba | Qué comprueba |
|---|---|
| `tests/test_fugas.py` | Ningún paciente en dos papeles ni en dos pliegues; el umbral, el modelo y el escalado no cambian si se alteran etiquetas o variables de prueba; el control negativo ronda 0,5 y detecta una fuga simulada; las categorías nuevas no filtran información |
| `tests/test_particion_y_metricas.py` | Partición sin fuga, reglas de etiquetado, cobertura del bootstrap, clústeres con más de 4 lecturas sin etiqueta, partición sin lanzamiento inicial |
| `tests/test_datos_reales.py` | Lo mismo sobre la partición versionada y la tabla real (`pytest -m datos`) |
| `tests/test_m2.py` | La malignidad no es entrada; los modelos de M2 se construyen y predicen |

## 11. Qué cambiar y dónde

| Quiero… | Cambio |
|---|---|
| Otra regla de etiquetado | `datos.regla` en `base.yaml` (R1, R2, R3) |
| Otra semilla o número de pliegues | `particion.semilla`, `particion.n_pliegues` (genera una partición nueva: súbela al repositorio) |
| Más remuestreos bootstrap | `evaluacion.n_bootstrap` |
| Otro hiperparámetro de un modelo | `modelo.parametros` en su YAML |
| Añadir o quitar una variable | `modelo.variables` en su YAML |
| Una variante nueva | Un YAML nuevo con `hereda:` del modelo base y solo el cambio |
| Mis rutas locales | `configs/local.yaml` (no se versiona) |
