# Guía de inicio: de la invitación a reproducir M0

**Objetivo:** tener el repositorio y el entorno del TFM en tu ordenador y comprobar que obtienes exactamente los mismos resultados de la línea base M0.
**Tiempo estimado:** 30–40 minutos (la mayor parte es la instalación del entorno).
**No necesitas las imágenes DICOM:** M0 usa solo la base de anotaciones que viene con pylidc.

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

Resultado esperado: **`12 passed, 4 skipped`**. Las 4 omitidas son las pruebas sobre los datos reales, que se activan cuando M0 haya construido su caché.

## 8. Ejecutar M0

```bash
python -m tfm_lidc.lineas_base --config configs/modelos/m0_diametro.yaml
```

La primera vez tarda **2–3 minutos** (lee las 6.859 anotaciones de pylidc y agrupa los nódulos); las siguientes, segundos. Puede aparecer un aviso de pylidc sobre la ruta a los DICOM: es normal, M0 no los usa.

### Resultado que debes obtener (exacto)

```
M0 · R1 · 879 nódulos de 535 pacientes
  auc            0.937  IC95 [0.920, 0.953]
  auprc          0.931  IC95 [0.908, 0.952]
  brier          0.100  IC95 [0.087, 0.115]
  exactitud      0.862  IC95 [0.839, 0.885]
  sensibilidad   0.835  IC95 [0.798, 0.870]
  especificidad  0.889  IC95 [0.857, 0.919]
AUC de prueba por pliegue: [0.9336, 0.9197, 0.9264, 0.9244, 0.969]
Umbrales por pliegue: [0.6528, 0.4571, 0.3184, 0.3601, 0.6265]
```

Lo importante es que coincidan también **el AUC y el umbral de cada pliegue**: eso confirma que usas la misma partición (`particiones/particion_R1_s42_k5.csv`) y el mismo entorno.

Los resultados quedan en `resultados/m0_diametro__R1__s42__<commit>/` (`registro.json` y `predicciones_oof.csv`). Esa carpeta no se sube al repositorio.

## 9. Repetir las pruebas (ahora con datos reales)

```bash
pytest -q
```

Resultado esperado: **`16 passed`**. Ahora se ejecutan también las comprobaciones de no fuga sobre la partición real.

## 10. Ver el notebook de resultados

```bash
cd ..
jupyter lab
```

1. En el explorador de JupyterLab abre `tfm-lidc-xai/notebooks/03_lineas_base_M0_M2.ipynb`.
2. Arriba a la derecha elige el kernel **Python (TFM)**.
3. Menú **Kernel → Restart Kernel and Run All Cells**.

Verás las métricas, la tabla por pliegue, las curvas ROC y de calibración, la curva de M0 frente al diámetro y el análisis de errores.

Ejecutar el notebook lo modifica. Para no generar cambios accidentales, al terminar deja el archivo como estaba:

```bash
cd tfm-lidc-xai
git restore notebooks/03_lineas_base_M0_M2.ipynb
```

## 11. Cuéntanos el resultado

Escribe en el grupo:

- sistema operativo y procesador (p. ej., «Windows 11, Intel» o «macOS, Apple M2»);
- la salida del punto 8 (o una captura);
- el resultado de `pytest -q` del punto 9.

Si las cifras coinciden, el entorno común queda validado en las tres máquinas (tarea T3 de I0).

---

## Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `repository not found` al clonar | No se aceptó la invitación (punto 1) o `gh auth login` se hizo con otra cuenta (`gh auth status` para comprobarlo) |
| `conda env create` falla en un paquete | Copia el mensaje de error al grupo; no cambies versiones por tu cuenta |
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
