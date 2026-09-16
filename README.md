<p align="center"><img src="docs/assets/aiscope-logo.svg" alt="AiScope" width="320"></p>

# AiScope vision

Object detection model for malaria parasites in blood smear photos taken with a phone held against a microscope eyepiece. Every box is one parasite, so counting parasites means counting boxes, and the class of each box is the parasite's life stage.

This repository covers everything up to the model that runs on the phone: data cleaning and splitting, training, evaluation, export to LiteRT and a reference implementation of the pre- and post-processing. The Android app that uses the model lives in [AIScope_android](https://github.com/guspei/AIScope_android).

This is research code. The model is not a diagnostic tool.

## Status

The model exported today comes from a short local run: YOLO26n trained for 15 epochs on half of the training set. It is good enough to build and test the app against, not to draw conclusions about accuracy. A full training run on a cloud GPU is the next step, after which the model will be evaluated on the test split for the first time.

## Data

The images were captured and annotated with [GDD-app](https://github.com/theaiscope/GDD-app), AiScope's data collection app. Each sample is a folder with `image_N.jpg`, a painted mask `mask_N.png` per image and a `metadata.json` (species, thin or thick smear, preparation details).

The dataset is not included in this repository and is not public. The zips go in `dataset/`, or wherever the `AISCOPE_RAW` environment variable points.

After cleaning there are about 6,800 images:

| | Images |
|---|---|
| Thick smear | 3,942 |
| Thin smear | 2,864 |
| *P. falciparum* | 2,526 |
| *P. ovale* | 1,937 |
| *P. malariae* | 1,393 |
| *P. vivax* | 950 |

and about 21,400 annotated objects:

| Class id | Stage | Mask color | Boxes | Counted as parasite |
|---|---|---|---|---|
| 0 | ring | `#5CBFB0` | 10,133 | yes |
| 1 | trophozoite | `#BFBE52` | 8,974 | yes |
| 2 | schizont | `#BF6B49` | 919 | yes |
| 3 | gametocyte | `#946FBF` | 1,138 | yes |
| 4 | artefact | `#4F6FD0` | 213 | no |

The colors, species codes and smear codes come from GDD-app (`src/aiscope/data/classes.py`). Species is a label of the whole sample, not of each box.

The annotations are brush strokes, not boxes, and the brush size depends on the annotator's zoom level (`80 px / zoom`). Turning them into boxes is most of the cleaning work:

| Notebook | What it does | Output |
|---|---|---|
| `01_exploracion` | Indexes the zips and looks at classes, object sizes, image quality and duplicates | `data/interim/{samples,images,instances}.parquet` |
| `02_limpieza` | Tightens each stroke's box to the stained area inside it, splits scribbles that cover several parasites, drops images with mass annotations | `data/interim/{images_clean,boxes_clean}.parquet` |
| `03_particion` | 70/15/15 train/val/test split and export to YOLO format | `data/processed/splits.parquet`, `data/processed/yolo/{campo_1280,mosaicos_640}` |

The split is never done per image. Images are grouped into sessions (same health facility, microscopist and day) and whole sessions go to one split, so near-identical photos of the same slide cannot end up in both train and test. Exact duplicates always fall in the same session.

The notebooks are committed without outputs because their figures show sample photos from the dataset. Run them locally to see the figures:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_limpieza.ipynb
```

## Model

The detector is [YOLO26](https://docs.ultralytics.com/) nano from Ultralytics. The architecture is just a parameter (`yolo26n`, `yolo26n-p2`, `yolo26s`...), so it can be changed without touching the data or the evaluation.

The phone photo is mostly black with a bright circle where the eyepiece field is. The input to the model is a square crop around that circle resized to 1280 x 1280. Three ways of feeding the image were compared in a short local run (15 epochs, 50 % of train, evaluated on val):

| | A | B | C |
|---|---|---|---|
| Model | yolo26n-p2 | yolo26n | yolo26n |
| Input | field at 640 | 4 tiles of 640 | field at 1280 |
| mAP50 | 0.158 | 0.259 | 0.270 |
| mAP50-95 | 0.083 | 0.148 | 0.165 |
| Count MAE per image | 2.02 | 1.65 | 1.53 |

C was kept. Artefacts get an AP of 0 in all three; there are too few of them.

Evaluation always runs on the 1280 px field, so the three variants are scored on the same boxes. It reports mAP per class together with the mean absolute error of the parasite count per image (artefacts excluded). The confidence threshold for counting is chosen on val and reused on test.

### Quantization

Full int8 quantization (weights and activations) breaks this model. Quantizing only the weights (`w8a32`) keeps the accuracy and the file size small:

| C on val | PyTorch | w8a32 | int8 |
|---|---|---|---|
| mAP50 | 0.270 | 0.270 | 0.132 |
| Count MAE | 1.53 | 1.54 | 3.55 |

The `w8a32` file is 2.99 MB. On an OPPO Find X5 Pro (Snapdragon 8 Gen 1) it runs in 72 ms per image on the GPU delegate and 235 ms on the CPU with 4 threads, measured with `benchmark_model`.

## Setup

Python 3.11. Direct dependencies are in `pyproject.toml` and the fully resolved environment in `requirements.lock`.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
```

The notebooks expect a Jupyter kernel called `aiscope`:

```bash
.venv/bin/python -m ipykernel install --user --name aiscope --display-name "AiScope (.venv)"
```

A few things worth knowing:

- Export to LiteRT goes through `litert-torch` (PyTorch straight to `.tflite`), which Ultralytics uses since 8.4.83. TensorFlow and onnx2tf are not needed.
- `torch` is pinned to 2.13.0 because `litert-torch` 0.9 requires `torch<2.14`.
- Importing `aiscope` sets `YOLO_AUTOINSTALL=false` so Ultralytics does not install packages on its own and break the pinned environment.
- On Apple silicon (MPS) Ultralytics forces `workers=0`. `benchmark.py train --force-workers` overrides that.

## Usage

Short local experiment comparing the variants:

```bash
.venv/bin/python -m aiscope.experiment --variants A B C --epochs 15 --fraction 0.5 --device mps
```

Training a single variant (on a CUDA machine, use `--device 0`):

```bash
.venv/bin/python -m aiscope.train --variant C --epochs 100 --device 0
```

Evaluation:

```bash
.venv/bin/python -m aiscope.evaluate --weights models/runs/<run>/weights/best.pt --variant C --split val
```

Export to LiteRT and compare the metrics before and after quantization. Calibration uses training images only:

```bash
.venv/bin/python -m aiscope.export --weights models/runs/<run>/weights/best.pt --variant C --quantize w8a32
```

This writes `models/exported/<run>_w8a32.tflite` and `<run>_w8a32_contrato.json`, the file the app reads to know the input size, the classes and the threshold.

Reference inference without Ultralytics, and generation of the golden test cases for the app (12 test images with their expected output):

```bash
.venv/bin/python -m aiscope.infer --model models/exported/<run>_w8a32.tflite --images photo.jpg
.venv/bin/python -m aiscope.infer --model models/exported/<run>_w8a32.tflite --golden 12
```

## What the app has to do

`src/aiscope/infer.py` is the executable reference. It gives the same boxes as Ultralytics (48 boxes compared, IoU 1.000).

Input:

1. Find the eyepiece field on the full photo: largest region with gray level above 40 on a 512 px thumbnail, holes filled, and the square that encloses it (`src/aiscope/field.py`).
2. Crop that square (anything outside the photo is black) and resize it to 1280 x 1280 with bilinear interpolation.
3. Tensor `[1, 3, 1280, 1280]`, NCHW, float32, pixel / 255.

Output is a `[1, 9, 33600]` tensor: normalized `cx, cy, w, h` followed by one score per class. Post-processing:

1. Class and confidence are the maximum of the 5 scores.
2. Multiply the coordinates by 1280 and convert to corners.
3. Per-class NMS with IoU 0.7, at most 300 detections. This is required: the exported head is the one-to-many branch, not YOLO26's NMS-free one.
4. Count detections with confidence at or above 0.25, leaving artefacts out.

The app bundles the `.tflite` and the LiteRT runtime in the APK, so nothing needs a network connection.

## Repository layout

```
dataset/        raw zips (not in git)
data/           derived tables, splits and YOLO exports (not in git)
notebooks/      01_exploracion, 02_limpieza, 03_particion
src/aiscope/
  data/         zip index, masks to instances, class legend, YOLO export, plotting helpers
  field.py      eyepiece field detection and input preprocessing
  train.py      training CLI
  evaluate.py   mAP per class and count MAE
  export.py     LiteRT export and before/after comparison
  infer.py      reference inference and golden cases
  benchmark.py  training throughput and LiteRT size/latency per model
  experiment.py the A/B/C comparison
  style.py      AiScope colors and fonts for figures
models/         weights, exports and benchmark results (not in git)
docs/assets/    AiScope logos, from GDD-app
```

## License

Ultralytics YOLO is AGPL-3.0, so anything built on this code, including the app, has to be released under a compatible license. The AiScope logos in `docs/assets/` come from GDD-app, which is MIT licensed.
