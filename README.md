# tfm-lidc-xai

**Clasificación de malignidad de nódulos pulmonares en tomografía computarizada con evaluación cuantitativa de su explicabilidad**

Código del Trabajo de Fin de Máster (UNIR) sobre el conjunto público [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/). El proyecto clasifica la sospecha de malignidad de nódulos pulmonares ya localizados por radiólogos y mide, de forma cuantitativa, hasta qué punto los mapas de explicación **Grad-CAM++** y **LayerCAM** coinciden con los contornos que dibujaron los propios radiólogos.

> **Estado:** trabajo en curso. Este repositorio es material de investigación y docencia. **No es un dispositivo médico** ni sirve para diagnóstico: trabaja sobre nódulos ya anotados y no detecta lesiones.

---

## Qué contiene y qué falta

| Componente | Descripción | Estado |
|---|---|---|
| Lectura y agrupación de anotaciones | Lecturas de pylidc agrupadas en nódulos; reglas de etiquetado R1, R2, R3 (con exclusión de los clústeres de más de 4 lecturas) | Implementado |
| Partición por paciente | 5 pliegues agrupados por paciente (semilla 42), versionada en `particiones/` con su huella | Implementado |
| **M0** · línea base de diámetro | Regresión logística con el diámetro del nódulo | Implementado |
| **M1** · línea base de metadatos | Regresión logística con los metadatos de adquisición (sin diámetro) y control con etiquetas permutadas | Implementado |
| **M2** · atributos semánticos | Logística, bosque aleatorio y perceptrón con las ocho valoraciones del radiólogo, y variantes con diámetro y volumen | Implementado (pendiente de integrar en `main`) |
| Extractor de recortes | DICOM → recortes de 64 mm a 0,5 mm/px (128 × 128), ventana fija, máscara de consenso; políticas S1/S2/S3 | Implementado |
| **M3** · CNN compacta | Red convolucional desde cero con recortes 2D (S2) o 2,5D (S3); ver [`docs/GUIA_M3.md`](docs/GUIA_M3.md) | Implementado y ejecutado (5 pliegues × 3 semillas) |
| M4–M7 (ResNet, EfficientNet, fusión, multitarea) | Modelos de imagen, fusión con atributos y multitarea | Diseñado |
| Evaluación de la explicabilidad | Grad-CAM++ y LayerCAM frente a máscaras de consenso, con referencias nulas | Diseñado |
| Aplicación de consulta | Interfaz para consultar resultados | Diseñado |

*Implementado*: existe y se ha ejecutado. *Parcial*: existe parte del código. *Diseñado*: descrito en la memoria, sin código todavía.

## Resultados disponibles (regla R1: 869 nódulos, 530 pacientes)

Validación cruzada de 5 pliegues agrupados por paciente; intervalos del 95 % por remuestreo de pacientes.

| Modelo | Entrada | AUC [IC 95 %] |
|---|---|---|
| M0 | Diámetro del nódulo | 0,936 [0,917; 0,952] |
| M1 | Metadatos de adquisición | 0,541 [0,496; 0,588] (p = 0,047 por permutación) |

M1 sirve como detector de atajos: un AUC próximo a 0,5 indica que la etiqueta no se predice por cómo se adquirió la TC. Las cifras de M2 se publicarán cuando se ejecuten en el entorno fijado de este repositorio.

## Instalación

Requiere [Conda](https://docs.conda.io/en/latest/miniconda.html) y Git.

```bash
git clone https://github.com/JeffRodriguezP/tfm-lidc-xai.git
cd tfm-lidc-xai
conda env create -f environment.yml
conda activate TFM
pip install -e ".[dev]"
git config core.hooksPath .githooks      # activa el freno contra envíos directos a main
pytest -q
```

El entorno fija Python 3.11 y las versiones que afectan a los resultados (por ejemplo, scikit-learn 1.7.2). La guía paso a paso, con las salidas esperadas y los problemas frecuentes, está en [`docs/GUIA_INICIO.md`](docs/GUIA_INICIO.md).

## Datos

Los datos **no** están en el repositorio y no deben subirse (el `.gitignore` los excluye).

- **M0 y M2** solo necesitan la base de anotaciones que trae pylidc.
- **M1** necesita además el `metadata.csv` de la descarga de LIDC-IDRI (fabricante del equipo).
- Los modelos de imagen (M3 en adelante) necesitan además los DICOM (`datos.dicom_root`): los recortes se generan una vez con `python -m tfm_lidc.data.recortes --config configs/base.yaml` (unos 6 min).

Cada persona indica sus rutas en `configs/local.yaml`, que no se versiona:

```yaml
datos:
  dicom_root: /ruta/a/TCIA_LIDC-IDRI_20200921/lidc_idri
  metadata_csv: /ruta/a/TCIA_LIDC-IDRI_20200921/metadata/metadata.csv
```

## Uso

Cada modelo se ejecuta con su archivo de configuración, que hereda de `configs/base.yaml`:

```bash
# Líneas base tabulares
python -m tfm_lidc.lineas_base --config configs/modelos/m0_diametro.yaml
python -m tfm_lidc.lineas_base --config configs/modelos/m1_metadatos.yaml --control-negativo 1000
python -m tfm_lidc.lineas_base --config configs/modelos/m2_logistica.yaml
python -m tfm_lidc.lineas_base --config configs/modelos/m2_bosque.yaml
python -m tfm_lidc.lineas_base --config configs/modelos/m2_mlp.yaml

# Verificar las cifras de datos citadas en la memoria
python scripts/contar_cortes.py --verificar

# Modelos de imagen: extraer recortes una vez y entrenar M3
python -m tfm_lidc.data.recortes --config configs/base.yaml --procesos 6
python -m tfm_lidc.imagen --config configs/modelos/m3_cnn.yaml
```

Cada ejecución crea `resultados/<modelo>__<regla>__s<semilla>__<commit>/` con `registro.json` (configuración, huella de la partición, versiones, métricas) y las predicciones fuera de muestra. Esa carpeta no se sube al repositorio.

## Estructura

```
configs/            base.yaml y un YAML por modelo (hereda de base.yaml); local.yaml no se versiona
docs/               guía de inicio
notebooks/          resultados de las líneas base y análisis de cortes
particiones/        partición por paciente versionada (solo identificadores)
scripts/            auditorías y verificación de cifras de datos
src/tfm_lidc/
  config.py         carga de configuraciones con herencia
  data/             nódulos y partición
  evaluation/       métricas con intervalos y controles de validez
  models/           modelos tabulares (M0–M2)
  lineas_base.py    entrenamiento y evaluación de M0–M2
tests/              pruebas de no fuga, partición, métricas y M2
resultados/         salidas de las ejecuciones (no versionado)
```

## Decisiones de diseño que conviene conocer

- **Partición por paciente.** Todos los nódulos y cortes de un paciente están en el mismo subconjunto. La partición se genera una vez, se versiona y **no se regenera**: con otra versión de scikit-learn, el mismo código reparte distinto.
- **Unidad de evaluación: el nódulo.** Los modelos de imagen se entrenan con cortes y se evalúan agregando las probabilidades de los cortes de cada nódulo.
- **Exclusión de clústeres anómalos.** Un nódulo no puede tener más de 4 lecturas (hay 4 radiólogos); los 14 clústeres que lo superan se excluyen (R1: 879 → 869 nódulos).
- **La malignidad nunca es entrada.** Es la etiqueta; los modelos de atributos usan solo las otras ocho valoraciones.
- **Estandarización e imputación solo con el entrenamiento** de cada pliegue, y umbral de decisión fijado en validación.
- **Cifras reproducibles.** Los recuentos de la memoria se generan con `scripts/contar_cortes.py --verificar`, no se copian a mano.

## Flujo de trabajo del equipo

No se trabaja directamente sobre `main`. Cada tarea va en una rama (`feat/<parte>-<RF>-<descripcion>`) y se integra con una solicitud de cambio que revisa otra persona del equipo. Formato de los mensajes de commit: `tipo(módulo): descripción [RF-xx]`. Más detalle en [`docs/GUIA_INICIO.md`](docs/GUIA_INICIO.md#reglas-del-repositorio-resumen).

## Autoría y contexto

TFM del Máster de la Universidad Internacional de La Rioja (UNIR), realizado en grupo por Jefferson Rodríguez, María Leonor Cubero y Yessica Descalzi, bajo la dirección de Elsa Fernández.

## Licencia y cómo citar

**Pendiente de definir.** Hasta que se añada un archivo `LICENSE`, no se concede licencia de uso. LIDC-IDRI tiene sus propias condiciones de uso en TCIA, que hay que respetar.

## Referencias de los datos

Armato, S. G. et al. (2011). The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans. *Medical Physics*, 38(2), 915-931.
