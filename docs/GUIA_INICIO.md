# Guía de inicio: de la invitación a reproducir M0 y M1

**Objetivo:** tener el repositorio y el entorno del TFM en tu ordenador y comprobar que obtienes exactamente los mismos resultados de las líneas base M0 y M1.
**Tiempo estimado:** 30–40 minutos (la mayor parte es la instalación del entorno).
**No necesitas las imágenes DICOM:** M0 usa solo la base de anotaciones que viene con pylidc, y M1 añade el `metadata.csv` de tu descarga de LIDC-IDRI.

> En **Mac** los comandos se escriben en la app **Terminal**. En **Windows**, en **Anaconda Prompt** (menú Inicio). Copia cada bloque y ejecútalo en orden.

---

## 1. Aceptar la invitación

1. Abre el correo de GitHub «JeffRodriguezP invited you to collaborate on tfm-lidc-xai», o entra en <https://github.com/JeffRodriguezP/tfm-lidc-xai/invitations>.
2. Pulsa **Accept invitation**.
3. Comprueba que ves el repositorio en <https://github.com/JeffRodriguezP/tfm-lidc-xai>.

Sin este paso, el `git clone` del punto 4 responde «repository not found».

## 2. Comprobar que tienes Git y Conda

```bash
git --version
conda --version
```

| Si falla… | Qué hacer |
|---|---|
| `git: command not found` (Mac) | Ejecuta `xcode-select --install` y acepta la instalación |
| `git` no se reconoce (Windows) | Instala Git desde <https://git-scm.com/download/win> con las opciones por defecto y abre de nuevo Anaconda Prompt |
| `conda` no se reconoce | Instala Miniconda desde <https://docs.conda.io/en/latest/miniconda.html> |

## 3. Conectar tu ordenador con GitHub (una sola vez)

```bash
conda install -n base -c conda-forge gh -y
gh auth login
```

Responde así:

1. *Where do you use GitHub?* → **GitHub.com**
2. *Preferred protocol?* → **HTTPS**
3. *Authenticate Git with your GitHub credentials?* → **Yes**
4. *How would you like to authenticate?* → **Login with a web browser**
5. Copia el código de 8 caracteres, pulsa Enter, inicia sesión en el navegador con tu usuario y pega el código.

Debe terminar con `✓ Logged in as <tu_usuario>`. Después, para que tus commits salgan a tu nombre:

```bash
git config --global user.name "Tu Nombre"
git config --global user.email "el_correo_de_tu_cuenta_de_github"
```

## 4. Clonar el repositorio

Elige la carpeta donde quieres el proyecto (por ejemplo, la del TFM) y:

```bash
cd <carpeta_del_TFM>
git clone https://github.com/JeffRodriguezP/tfm-lidc-xai.git
cd tfm-lidc-xai
git log --oneline
```

Deben aparecer los commits del proyecto, empezando por el más reciente (por ejemplo, `Merge pull request #1 …`).

Activa después el freno que impide subir cambios directamente a `main` (una sola vez por clon):

```bash
git config core.hooksPath .githooks
```

A partir de ahí, un `git push` dirigido a `main` se rechaza con un aviso: los cambios se suben siempre desde una rama y se integran con una solicitud de cambio (ver «Reglas del repositorio» al final).

## 5. Crear el entorno del TFM

Desde la carpeta `tfm-lidc-xai`:

```bash
conda env create -f environment.yml
conda activate TFM
python --version
```

- Tarda varios minutos: instala Python 3.11, pylidc, scikit-learn 1.7.2, PyTorch y el resto con las versiones fijadas.
- `python --version` debe mostrar **3.11.x** y el prompt debe empezar por `(TFM)`.
- Si ya tienes un entorno llamado `TFM`, usa otro nombre: `conda env create -n TFM_lidc -f environment.yml`.

## 6. Instalar el paquete del proyecto y el kernel de Jupyter

Con el entorno activo y dentro de `tfm-lidc-xai`:

```bash
pip install -e ".[dev]"
python -m ipykernel install --user --name TFM --display-name "Python (TFM)"
python -c "import tfm_lidc, sklearn; print('ok', sklearn.__version__)"
```

Debe imprimir **`ok 1.7.2`**. Si sale otra versión de scikit-learn, avisa: la partición y los resultados dependen de ella.

## 7. Ejecutar las pruebas (antes de M0)

```bash
pytest -q
```

Resultado esperado: **`15 passed, 5 skipped`**. Las 5 omitidas son las pruebas sobre los datos reales, que se activan cuando M0 haya construido su caché y cuando se configure el `metadata.csv` (punto 10).

## 8. Ejecutar M0

```bash
python -m tfm_lidc.lineas_base --config configs/modelos/m0_diametro.yaml
```

La primera vez tarda **2–3 minutos** (lee las 6.859 anotaciones de pylidc y agrupa los nódulos); las siguientes, segundos. Puede aparecer un aviso de pylidc sobre la ruta a los DICOM: es normal, M0 no los usa.

### Resultado que debes obtener (exacto)

```
M0 · R1 · 869 nódulos de 530 pacientes
  auc            0.936  IC95 [0.917, 0.952]
  auprc          0.926  IC95 [0.899, 0.950]
  brier          0.101  IC95 [0.087, 0.117]
  exactitud      0.853  IC95 [0.826, 0.877]
  sensibilidad   0.822  IC95 [0.776, 0.862]
  especificidad  0.882  IC95 [0.850, 0.911]
AUC de prueba por pliegue: [0.9483, 0.9455, 0.9337, 0.9046, 0.9519]
Umbrales por pliegue: [0.5573, 0.5241, 0.5159, 0.2681, 0.4369]
```

> **Cambio de partición (14/11/2026).** Desde esta versión se excluyen los 14 clústeres con más de 4 lecturas (un nódulo no puede tener más lecturas que lectores; Tabla 16 paso 3, RF-D02): R1 pasa de 879 a **869 nódulos (530 pacientes)** y la partición versionada cambia (huella `0e6393241e7c69e0`). La partición anterior se conserva como `particion_R1_s42_k5_sin-exclusion.csv` para el análisis de sensibilidad: `python -m tfm_lidc.lineas_base --config configs/modelos/m0_diametro_sin_exclusion.yaml` reproduce los resultados antiguos (879 nódulos, AUC 0,937 [0,920–0,953]).

Lo importante es que coincidan también **el AUC y el umbral de cada pliegue**: eso confirma que usas la misma partición (`particiones/particion_R1_s42_k5.csv`, huella `0e6393241e7c69e0`) y el mismo entorno.

Los resultados quedan en `resultados/m0_diametro__R1__s42__<commit>/` (`registro.json` y `predicciones_oof.csv`). Esa carpeta no se sube al repositorio.

## 9. Repetir las pruebas (ahora con datos reales)

```bash
pytest -q
```

Resultado esperado: **`19 passed, 1 skipped`**. Ahora se ejecutan también las comprobaciones de no fuga sobre la partición real. La que sigue omitida necesita el `metadata.csv` del punto 10.

## 10. Ejecutar M1 (metadatos de adquisición)

M1 comprueba si la etiqueta puede predecirse solo con cómo se adquirió la TC (fabricante, grosor, espaciados, contraste). El fabricante sale del `metadata.csv` de **tu descarga de LIDC-IDRI**, así que hay que indicar su ruta.

Crea el archivo `configs/local.yaml` (no se sube al repositorio: cada una pone su ruta) con este contenido, cambiando la ruta por la tuya:

```yaml
datos:
  metadata_csv: /ruta/a/TCIA_LIDC-IDRI_20200921/metadata/metadata.csv
```

En Windows la ruta se escribe con barras normales o entre comillas, por ejemplo `metadata_csv: "C:/Users/<usuario>/TFM/TCIA_LIDC-IDRI_20200921/metadata/metadata.csv"`.

Después:

```bash
python -m tfm_lidc.lineas_base --config configs/modelos/m1_metadatos.yaml --control-negativo 1000
```

Tarda alrededor de un minuto (las 1.000 permutaciones dan el p-valor). Resultado que debes obtener (exacto):

```
M1 · R1 · 869 nódulos de 530 pacientes
  auc            0.541  IC95 [0.496, 0.588]
  auprc          0.514  IC95 [0.463, 0.578]
  brier          0.252  IC95 [0.244, 0.258]
  exactitud      0.533  IC95 [0.491, 0.576]
  sensibilidad   0.601  IC95 [0.541, 0.662]
  especificidad  0.469  IC95 [0.402, 0.539]
AUC de prueba por pliegue: [0.5649, 0.4879, 0.6053, 0.5395, 0.5439]
Umbrales por pliegue: [0.3796, 0.3745, 0.4961, 0.4817, 0.5421]
Control negativo (1000 permutaciones): AUC media 0.491 [0.437, 0.550] · superado
P-valor por permutación del AUC observado: 0.0470
```

Por último, las pruebas completas:

```bash
pytest -q
```

Resultado esperado: **`20 passed`**.

## 11. Ver el notebook de resultados

```bash
cd ..
jupyter lab
```

1. En el explorador de JupyterLab abre `tfm-lidc-xai/notebooks/03_lineas_base_M0_M2.ipynb`.
2. Arriba a la derecha elige el kernel **Python (TFM)**.
3. Menú **Kernel → Restart Kernel and Run All Cells**.

Verás, para M0, las métricas, la tabla por pliegue, las curvas ROC y de calibración, la curva frente al diámetro y el análisis de errores; y en la parte 2, M1: comparación con M0, coeficientes por pliegue, asociación de cada metadato con la etiqueta, estratificación por tamaño y la prueba de permutación.

Ejecutar el notebook lo modifica. Para no generar cambios accidentales, al terminar deja el archivo como estaba:

```bash
cd tfm-lidc-xai
git restore notebooks/03_lineas_base_M0_M2.ipynb
```

## 12. Cuéntanos el resultado

Escribe en el grupo:

- sistema operativo y procesador (p. ej., «Windows 11, Intel» o «macOS, Apple M2»);
- la salida de los puntos 8 (M0) y 10 (M1), o capturas;
- el resultado del último `pytest -q` del punto 10.

Si las cifras coinciden, el entorno común queda validado en las tres máquinas (tarea T3 de I0).

---

## Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `repository not found` al clonar | No se aceptó la invitación (punto 1) o `gh auth login` se hizo con otra cuenta (`gh auth status` para comprobarlo) |
| `conda env create` falla en un paquete | Copia el mensaje de error al grupo; no cambies versiones por tu cuenta |
| `Faltan valores en ['fabricante']` o `No existe datos.metadata_csv` al ejecutar M1 | Falta `configs/local.yaml` o la ruta al `metadata.csv` es incorrecta (punto 10) |
| `ModuleNotFoundError: tfm_lidc` | El entorno no está activo (`conda activate TFM`) o faltó `pip install -e ".[dev]"` |
| El kernel «Python (TFM)» no aparece en JupyterLab | Repite el comando `ipykernel install` del punto 6 y recarga la página |
| Cifras de M0 distintas | Comprueba `sklearn` = 1.7.2 y que no has modificado `particiones/` (`git status` no debe listarla) |
| `✗ No se sube directamente a main` al hacer `git push` | Es el hook funcionando: crea una rama (`git switch -c feat/...`), haz el push de esa rama y abre una solicitud de cambio |
| JupyterLab no deja subir a otras carpetas | Solo muestra la carpeta desde la que se lanzó; ciérralo (Ctrl+C dos veces) y lánzalo desde una carpeta superior |

## Reglas del repositorio (resumen)

- **Nunca** se trabaja directamente sobre `main`: cada tarea en su rama (`feat/<parte>-<RF>-<descripcion>`) y se integra con una solicitud de cambio revisada por otra integrante. El hook de `.githooks/pre-push` (punto 4) bloquea los envíos directos a `main`.

  ```bash
  git switch main && git pull                  # partir siempre de main actualizado
  git switch -c feat/datos-RF-D08-particion    # una rama por tarea
  # … cambios, git add, git commit …
  git push -u origin feat/datos-RF-D08-particion
  gh pr create --fill                          # solicitud de cambio; la revisa otra integrante
  ```
- **Nunca** se suben imágenes, recortes ni la carpeta `resultados/` (el `.gitignore` ya lo impide).
- La partición de `particiones/` **no se regenera**: todos los modelos se evalúan sobre ella.
- Mensajes de commit con el formato `tipo(módulo): descripción [RF-xx]`.
