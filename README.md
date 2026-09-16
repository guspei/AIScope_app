<p align="center"><img src="docs/assets/aiscope-logo.svg" alt="AiScope" width="320"></p>

# AiScope · visión

Modelo de detección de parásitos de malaria en fotos de microscopio tomadas con el móvil. Cada caja es un parásito: contar es contar cajas y reconocer es la clase de cada caja. Aquí sale el modelo exportado a LiteRT int8 y el contrato de entrada y salida que usará la app Android. La app funciona sin conexión: el modelo va dentro del APK y toda la inferencia ocurre en el teléfono.

## Estructura

```
dataset/        zips crudos (ignorado en git)
data/           derivados: interim/ (índice), processed/ (splits, tiles) (ignorado en git)
notebooks/      01_exploracion, 02_limpieza, 03_particion; cada uno se ejecuta de arriba abajo
src/aiscope/
  data/         índice de los zips, máscaras → instancias, leyenda de clases, exportación a YOLO, visualización
  style.py      paleta y tipografía AiScope para figuras
  benchmark.py  rendimiento de entrenamiento (img/s) y tamaño/latencia LiteRT int8 por modelo
  train.py      entrenamiento (CLI, pensado para GPU remota)
  evaluate.py   mAP por clase + MAE de conteo por imagen sobre test
  export.py     exportación a LiteRT int8 y comparación antes/después de cuantizar
models/         pesos, exportaciones y resultados de benchmark (ignorado en git salvo .gitkeep)
docs/assets/    logos de AiScope (de GDD-app, licencia MIT)
```

## Entorno

Python 3.11. Las versiones directas están en `pyproject.toml` y el entorno completo resuelto en `requirements.lock`.

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.lock && .venv/bin/pip install --no-deps -e .
```

Registrar el entorno como kernel de Jupyter (los notebooks piden el kernel «AiScope (.venv)»):

```bash
.venv/bin/python -m ipykernel install --user --name aiscope --display-name "AiScope (.venv)"
```

Notas de compatibilidad:
- La exportación a LiteRT usa `litert-torch` (PyTorch → `.tflite` directo), que es la vía de Ultralytics desde la 8.4.83. No hace falta TensorFlow ni onnx2tf.
- `torch==2.13.0`, porque `litert-torch` 0.9 exige `torch<2.14`.
- `import aiscope` pone `YOLO_AUTOINSTALL=false`, así Ultralytics no instala paquetes por su cuenta y no rompe el entorno fijado.
- En MPS, Ultralytics fuerza `workers=0`. `benchmark.py train --force-workers` los restaura.
- Ultralytics es AGPL-3.0, así que la app que lo integre tiene que publicarse con licencia compatible.

## Datos

Los zips van en `dataset/`, o en la ruta que indique la variable `AISCOPE_RAW`. Cada carpeta UUID es una muestra con `image_N.jpg`, `mask_N.png` y `metadata.json`, tal como las sube la app de etiquetado [GDD-app](https://github.com/theaiscope/GDD-app).

Leyenda (`src/aiscope/data/classes.py`, tomada de GDD-app):

| Color de máscara | Estadio | ¿Parásito? |
|---|---|---|
| `#5CBFB0` | anillo (ring) | sí |
| `#BFBE52` | trofozoíto | sí |
| `#BF6B49` | esquizonte | sí |
| `#946FBF` | gametocito | sí |
| `#4F6FD0` | artefacto | no |

- `metadata.species`: 1 *P. falciparum*, 2 *P. vivax*, 3 *P. ovale*, 4 *P. malariae*, 5 *P. knowlesi*.
- `metadata.bloodType`: 1 extensión fina, 2 gota gruesa.
- `preparation.sampleAge`: `fresh` o `old` (muestra antigua de laboratorio).
- El pincel de anotación mide `80 px / zoom`, así que el tamaño del trazo depende del zoom del anotador y no del tamaño del parásito.

Flujo de datos, un notebook detrás de otro:

| Notebook | Qué hace | Salida |
|---|---|---|
| `01_exploracion` | índice de los zips, clases, tamaños, calidad | `data/interim/{samples,images,instances}.parquet` |
| `02_limpieza` | trazos → cajas ajustadas a la zona teñida; parte garabatos; excluye anotaciones masivas | `data/interim/{images_clean,boxes_clean}.parquet` |
| `03_particion` | train/val/test 70/15/15 por sesión (+ duplicados) y exportación YOLO | `data/processed/splits.parquet`, `data/processed/yolo/{campo_1280,mosaicos_640}` |

Clases del detector (id → estadio): 0 anillo, 1 trofozoíto, 2 esquizonte, 3 gametocito, 4 artefacto.

Ejecutar un notebook sin abrirlo:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_limpieza.ipynb
```

## Entrenamiento y evaluación

Variantes del experimento local: A = `yolo26n-p2` con el campo a 640; B = `yolo26n` con 4 mosaicos de 640; C = `yolo26n` con el campo a 1280.

```bash
.venv/bin/python -m aiscope.train --variant C --epochs 30 --device mps
```

```bash
.venv/bin/python -m aiscope.evaluate --weights models/runs/<run>/weights/best.pt --variant C --split val
```

`evaluate.py` mide siempre sobre el campo a 1280 (mismas cajas para A, B y C) y da mAP por clase junto con el error absoluto medio del conteo de parásitos por imagen (sin artefactos). El umbral de conteo se elige en val y se reutiliza en test.

## Modelo

Base: YOLO26 de Ultralytics, que no necesita NMS y exporta directo a LiteRT. La arquitectura es un parámetro (`yolo26n`, `yolo26n-p2`, `yolo26s`…), así que se puede cambiar sin tocar datos ni evaluación. Salir de la familia Ultralytics solo exige adaptar `train.py` y `export.py`; los datos siguen en formato YOLO.

Dónde se entrena: experimentos cortos en local (MPS) y entrenamientos completos en GPU CUDA en la nube, con el mismo código.

## Contrato del modelo para la app

Provisional, a falta de medir la latencia en un teléfono. La referencia ejecutable es [infer.py](src/aiscope/infer.py), que reproduce exactamente la salida de Ultralytics (48 cajas comparadas, IoU 1,000).

**Modelo**: `models/exported/<run>_w8a32.tflite`, 2,99 MB. Pesos en int8 y activaciones en float (`w8a32`). Cuantizar también las activaciones (`int8`) hunde la precisión: el mAP50 cae de 0,270 a 0,132 y el error de conteo sube de 1,53 a 3,55.

**Entrada**
1. Detectar el campo del ocular sobre la foto completa: mayor región con gris > 40 en una miniatura de 512 px, rellenar huecos y tomar el cuadrado que la envuelve ([field.py](src/aiscope/field.py)).
2. Recortar ese cuadrado y llevarlo a 1280 × 1280 con interpolación bilineal.
3. Tensor `[1, 3, 1280, 1280]` (NCHW), float32, valores en 0-1 (píxel / 255).

**Salida**: un tensor `[1, 9, 33600]`, con 4 coordenadas `xywh` normalizadas en 0-1 y 5 puntuaciones de clase. Postprocesado:
1. Clase y confianza = máximo de las 5 puntuaciones.
2. Multiplicar las coordenadas por 1280 y pasar a esquinas.
3. NMS por clase con IoU 0,7, máximo 300 detecciones. **Hace falta**: la cabeza exportada es la de una a muchas, no la variante sin NMS de YOLO26.
4. Contar las detecciones con confianza ≥ **0,25** (umbral elegido en val), excluyendo los artefactos.

**Clases**: 0 anillo, 1 trofozoíto, 2 esquizonte, 3 gametocito, 4 artefacto. El artefacto se detecta pero no suma en el conteo.

**Ejecución en el teléfono**: delegado de GPU, con CPU (XNNPACK, 4 hilos) como respaldo. Medido con `benchmark_model` en un OPPO Find X5 Pro (Snapdragon 8 Gen 1, Android 16): 72 ms por imagen en GPU y 235 ms en CPU. La primera carga en GPU tarda unos 2 s. Resultados completos en `models/benchmarks/telefono_CPH2305.json`. Es un teléfono de gama alta; en un móvil medio hay que volver a medir.

**Sin conexión**: el `.tflite` va dentro del APK y el runtime de LiteRT se empaqueta en la app, no el de Google Play services.

**Casos de prueba**: `models/exported/golden/` tiene 12 fotos completas con su salida esperada en `esperado.json`. La app debe reproducir esas detecciones y esos conteos.
