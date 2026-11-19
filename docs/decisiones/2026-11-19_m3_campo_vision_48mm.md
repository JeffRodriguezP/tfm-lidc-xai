# Decisión · Campo de visión de 48 mm para M3

- **Fecha:** 19/11/2026
- **Estado:** adoptada en el código; pendiente de reflejar en la memoria y de confirmar con la directora
- **Ámbito:** M3 (CNN compacta desde cero). No decide el recorte de M4/M5 (decisión A6, abierta).
- **Requisitos y apartados afectados:** 3.3.5.7 (paso 4), 3.3.6, 3.3.7 (factor de sensibilidad «campo de visión»),
  RF-D06, RF-M02

## Contexto

La memoria V14 fija valores **iniciales** para el recorte (64 mm a 0,5 mm/px, es decir, 128 × 128 px) y dice que «se
confirmarán o ajustarán» y que su efecto «se estudia como factor de sensibilidad». Con 64 mm, M3 obtuvo un AUC de
0,891 ± 0,025 entre semillas (R1, 5 pliegues × 3 semillas), con mucha variación entre semillas, y quedó por debajo de
M0 (0,936). Un nódulo típico ocupaba solo un 1,2 % del recorte.

## Opciones evaluadas

Cuatro variantes de un solo cambio frente a la base, con la semilla 42 y los 5 pliegues, elegidas **por AUC medio de
validación** (no de prueba; RNF03, 3.3.6.2):

| Variante | AUC medio de validación |
|---|---|
| **Campo de 48 mm** (96 × 96 px a 0,5 mm; desplazamiento de aumento 6 px = 3 mm) | **0,927** (mejor en 5 de 5 pliegues) |
| Base, 64 mm | 0,917 |
| Ruido y desenfoque en el aumento | 0,915 |
| Regularización más fuerte | 0,914 |
| 2,5D (S3) | 0,912 |

## Decisión

M3 usa recortes de **48 mm** con el mismo espaciado de **0,5 mm/px** (96 × 96 px): `configs/modelos/m3_cnn_fov48.yaml`.
La configuración de 64 mm (`m3_cnn.yaml`) se conserva sin cambios y se informa como análisis de sensibilidad.

## Evidencia de confirmación (prueba, tras la elección; semillas 42–44)

| | AUC medio ± DT entre semillas | ΔAUC frente a M0 por semilla [IC 95 %] |
|---|---|---|
| 48 mm | **0,921 ± 0,002** | −0,013 [−0,031; +0,007] · −0,017 [−0,036; +0,004] · −0,016 [−0,037; +0,006] |
| 64 mm | 0,891 ± 0,025 | −0,046 [−0,069; −0,022] · −0,068 [−0,097; −0,040] · −0,019 [−0,043; +0,005] |

## Consecuencias

- **A favor:** +0,030 de AUC medio, variación entre semillas 12 veces menor, entrenamiento ~40 % más rápido. El
  espaciado no cambia, así que el tamaño absoluto del nódulo sigue siendo comparable entre exploraciones. La última
  capa convolucional pasa a 12 × 12 pero conserva 4 mm por celda: los mapas Grad-CAM++ tienen la misma resolución física.
- **En contra:** menos contexto alrededor del nódulo. 2 de los 869 nódulos (30 y 38 mm, alta sospecha) tocan el borde
  del recorte en 1 corte cada uno (de 24 y 33); con 64 mm, ninguno. Irrelevante para clasificar; en la evaluación de
  la explicabilidad, esos dos cortes tienen la máscara ligeramente recortada.
- **Sobre H1:** con 48 mm, M3 queda a la altura de M0 (todos los IC del ΔAUC contienen el 0) pero no lo supera.
- **Proceso:** el campo de visión se ajustó por validación del modelo, no por el análisis exploratorio que menciona
  3.3.5.7; la memoria debe decir cuál se usó.

## Pendiente en la memoria

1. 3.3.5.7, paso 4: «campo de visión físico fijo de 48 mm (seleccionado en validación frente al valor inicial de
   64 mm) … 0,5 mm por píxel, es decir, 96 × 96 píxeles».
2. 3.3.5.7, párrafo siguiente: indicar que el campo de visión se ajustó con la validación de M3 y que 64 mm se
   informa como sensibilidad.
3. RF-D06, criterio de aceptación: «… al tamaño configurado (96 × 96 o 128 × 128 píxeles) a 0,5 mm …».
4. Capítulo 6: tabla de sensibilidad 48 frente a 64 mm.
5. Confirmar con la directora junto con A6 (224 frente a 128 en M4/M5) y A7 (desplazamiento del recorte).
