# M0 · Línea base de diámetro

Regresión logística con el diámetro del nódulo (Tabla 21). Es la referencia mínima que debe superar el modelo de imagen (H1).

## Instalación (una vez)

```bash
pip install -e ".[dev]"          # desde la raíz del repositorio, con el entorno del proyecto activo
```

pylidc solo necesita la ruta a los DICOM para leer imágenes; M0 usa únicamente la base de anotaciones que trae pylidc. Si quieres fijar la ruta, crea `configs/local.yaml` (no se sube al repositorio):

```yaml
datos:
  dicom_root: /ruta/a/TCIA_LIDC-IDRI_20200921/lidc_idri
```

## Ejecución

```bash
pytest                                                    # todas las pruebas (las de datos reales se omiten si no hay datos)
pytest -m datos                                           # solo las comprobaciones sobre la partición real
python -m tfm_lidc.lineas_base --config configs/modelos/m0_diametro.yaml
python -m tfm_lidc.lineas_base --config configs/modelos/m0_diametro.yaml --control-negativo 200
python scripts/replica_hancock_magnan.py --repeticiones 1000   # opcional: valida la lectura de los datos
```

La primera ejecución lee las 6.859 lecturas y agrupa los nódulos (unos 3–4 min); queda en caché en `resultados/datos/`.

## Salidas (`resultados/m0_diametro__R1__s42__<commit>/`)

| Archivo | Contenido |
|---|---|
| `predicciones_oof.csv` | Una fila por nódulo: pliegue, etiqueta, probabilidad fuera de muestra y predicción con el umbral del pliegue |
| `registro.json` | Configuración y su huella, commit, huella de la partición, umbrales y coeficientes por pliegue, métricas con IC 95 % y entorno |

## La partición se genera una sola vez

La primera ejecución crea `particiones/particion_R1_s42_k5.csv` y su `.json` (huella, versión de scikit-learn, fecha). **Hay que subirlos al repositorio** y no volver a generarlos: con la misma semilla, `StratifiedGroupKFold` reparte distinto según la versión de scikit-learn (1.7.2 y 1.9.1 solo coinciden en el 20 % de los nódulos). Si cada integrante generara la suya, M0 y M4 no se evaluarían sobre los mismos nódulos y las comparaciones pareadas dejarían de ser válidas. El archivo solo contiene identificadores de nódulo y paciente.

## Pruebas de no fuga

| Prueba | Qué comprueba | Datos |
|---|---|---|
| `test_fugas.py` | Ningún paciente en dos papeles ni en dos pliegues; pacientes con varias TC enteros en un pliegue; umbral, modelo y escalado idénticos si se cambian las etiquetas o las variables de la prueba; control negativo ≈ 0,5 y detección de una fuga simulada | Sintéticos (integración continua) |
| `test_datos_reales.py` | Lo mismo sobre la partición versionada y la tabla de nódulos real (879 nódulos, 535 pacientes, 430 de alta sospecha) | LIDC-IDRI (`pytest -m datos`) |

Las pruebas se validaron introduciendo a propósito dos fugas (umbral elegido con la prueba; escalado ajustado con todos los datos): ambas hacen fallar la batería.

`evaluation/controles.control_negativo` recibe una función que entrena con etiquetas permutadas y predice fuera de pliegue, así que los modelos de imagen (M3–M7) lo usarán sin cambios. En M0 es poco informativo: con una sola variable el modelo solo aprende el signo.
