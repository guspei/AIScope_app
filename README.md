<p align="center"><img src="docs/assets/aiscope-logo.svg" alt="AiScope" width="320"></p>

# AiScope vision

Object detection model for malaria parasites in blood smear photos taken with a phone held against a microscope eyepiece. Every box is one parasite, so counting parasites means counting boxes.

This repository covers everything up to the model that runs on the phone: data cleaning and splitting, training, evaluation, export to LiteRT and a reference implementation of the pre- and post-processing. The Android app that uses the model lives in [AIScope_android](https://github.com/guspei/AIScope_android).

This is research code. The model is not a diagnostic tool.

## Status

The current model is `r2_E1_20260916_w8a32`: YOLO26 nano with a single class, `parasite`, on the eyepiece field resized to 1280 px. It is the one bundled in the Android app. On the test split it counts with a mean absolute error of 1.33 parasites per image (37 % of the parasites present) and gets the exact count in 44.5 % of images. It does not tell the life stage; [Experiments and decisions](#experiments-and-decisions) explains why.

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

| Stage | Mask color | Boxes | Parasite |
|---|---|---|---|
| ring | `#5CBFB0` | 10,133 | yes |
| trophozoite | `#BFBE52` | 8,974 | yes |
| schizont | `#BF6B49` | 919 | yes |
| gametocyte | `#946FBF` | 1,138 | yes |
| artefact | `#4F6FD0` | 213 | no |

The colors, species codes and smear codes come from GDD-app (`src/aiscope/data/classes.py`). Species is a label of the whole sample, not of each box. The first models used the five annotated classes; the current one merges the four stages into `parasite` and drops artefacts.

Only 18 images have no parasite at all, so the data says little about empty fields.

The annotations are brush strokes, not boxes, and the brush size depends on the annotator's zoom level (`80 px / zoom`). Turning them into boxes is most of the cleaning work:

| Notebook | What it does | Output |
|---|---|---|
| `01_exploracion` | Indexes the zips and looks at classes, object sizes, image quality and duplicates | `data/interim/{samples,images,instances}.parquet` |
| `02_limpieza` | Tightens each stroke's box to the stained area inside it, splits scribbles that cover several parasites, drops images with mass annotations | `data/interim/{images_clean,boxes_clean}.parquet` |
| `03_particion` | 70/15/15 train/val/test split and export to YOLO format, with stages and single-class | `data/processed/splits.parquet`, `data/processed/yolo/{campo_1280,mosaicos_640,parasito_1280,parasito_1920}` |

The split is never done per image. Images are grouped into sessions (same health facility, microscopist and day) and whole sessions go to one split, so near-identical photos of the same slide cannot end up in both train and test. Exact duplicates always fall in the same session.

The notebooks are committed without outputs because their figures show sample photos from the dataset. Run them locally to see the figures:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_limpieza.ipynb
```

## Experiments and decisions

In the order they were done. Every choice, including count thresholds, was made on val. Test numbers are only reported, never used to pick anything.

### 1. Detection, not segmentation or image classification

The app has to count parasites and show where they are. A detector gives both directly: one box per parasite. The chosen family is YOLO26 from Ultralytics because it trains quickly, has nano and small sizes that fit a phone, and exports straight to LiteRT. The architecture is a parameter, so it can be swapped without touching data or evaluation.

### 2. From brush strokes to boxes

The annotations are painted strokes whose size depends on the annotator's zoom, so a box around the stroke is often several times bigger than the parasite. Notebook `02_limpieza` tightens each box to the stained area inside the stroke (darkness plus magenta), splits strokes that contain holes or scribbles covering several parasites, and falls back to the stroke box when no stained area stands out. Two images where whole regions were painted at once (104 parasites) were dropped.

### 3. Split by session, never by image

There is no patient or slide id. Photos from the same session (health facility, microscopist and day) are often near-duplicates of the same slide, so images are grouped by session, groups that share a near-duplicate photo (perceptual hash) are merged, and whole groups go to train, val or test (70/15/15), balancing species, smear type and stage counts.

### 4. Input: the eyepiece field at 1280 px

Parasites are small: on the field resized to 640 px, 58 % of the thin-smear boxes are under 12 px; at 1280 px, 19 %. Three ways of feeding the image were compared in a short local run (15 epochs, half of train, val):

| | A | B | C |
|---|---|---|---|
| Model | yolo26n-p2 | yolo26n | yolo26n |
| Input | field at 640 | 4 tiles of 640 from the field at 1200 | field at 1280 |
| mAP50 | 0.158 | 0.259 | 0.270 |
| mAP50-95 | 0.083 | 0.148 | 0.165 |
| Count MAE per image | 2.02 | 1.65 | 1.53 |

A loses the small parasites even with the extra high-resolution head. B and C are close, but C is slightly better and runs once per photo instead of four times with a merge step, so the app stays simpler. C was kept.

### 5. Quantization: weights only (`w8a32`)

| C on val | PyTorch | int8 | w8a16 | w8a32 |
|---|---|---|---|---|
| mAP50 | 0.270 | 0.153 | 0.271 | 0.270 |
| Count MAE | 1.53 | 3.58 | 1.50 | 1.54 |
| Mac CPU, ms per image | | 148 | 2,238 | 88 |

Full int8 (weights and activations) breaks the model, and tripling the calibration images barely helped (mAP50 0.132 → 0.153). `w8a16` keeps the accuracy but runs 25 times slower on CPU. `w8a32` keeps the accuracy, is fast and weighs 3 MB, so it is the export format.

Measured on an OPPO Find X5 Pro (Snapdragon 8 Gen 1) with `benchmark_model`: 72 ms per image on the GPU delegate and 235 ms on the CPU with 4 threads. B would need about 60 ms per tile on CPU, 4 tiles per photo, so it would not be faster either. A mid-range phone still has to be measured.

### 6. Where to train

A full run of C on the Mac (M3 Pro, MPS) takes about 11 minutes per epoch. Full runs were done on RunPod instead: RTX 4090, Secure Cloud, EU data center, about 50 seconds per epoch at 0.74 $/h, with the same code and `--device 0`. The three cloud rounds cost 1.75 $, 1.74 $ and 0.62 $.

### 7. Full training with stages

C trained on the whole training set (up to 100 epochs, early stopping after 20 epochs without improvement) stopped at epoch 48, with the best weights at epoch 28. Val mAP50 went from 0.270 to 0.310 and the count MAE from 1.53 to 1.35. On test: mAP50 0.432, count MAE 1.56.

### 8. Error analysis of that model

On val, at the 0.25 count threshold:

- **Telling stages apart was half of the problem.** Ignoring the stage, "is there a parasite here" reaches AP50 0.60; with stages, 0.31. In thin smears it found 64 % of the parasites and 58 % with the right stage; in thick smears, 62 % and only 40 %.
- **Boxes were in the right place.** Accepting much looser boxes barely changed the score (mAP at IoU 0.1: 0.355, at 0.5: 0.307).
- **Small parasites were missed.** Recall was 50 % for boxes under 16 px and 76 % for boxes of 48 px or more.
- **Dense images carried the count error.** Images with 4 or more parasites were 23 % of val and 48 % of the error; the model under-counted them (−2.9 on images with 7 to 10 parasites) and over-counted images with a single parasite (+0.4).
- **Metrics also penalise annotations.** Of the 913 boxes that did not match an annotated parasite, 238 were a second box on a parasite already found (with another stage), 206 overlapped an annotated parasite only partly, and 461 were away from any annotation. Some of those 461 are likely parasites nobody marked, but there is nobody to review them.
- **Recall depends on the threshold.** At 0.05 the model found 87 % of the parasites with 52 % precision; at 0.25, 62 % with 76 %. The count threshold is chosen on val as the one with the lowest count MAE.
- **Longer training would not help.** Val mAP50 was already around 0.30 by epoch 15, and after epoch 28 the classification loss on val rose while the training loss kept falling.

A cheaper fix was also tested without retraining: counting with the sum of the four stage scores and NMS that ignores the stage (threshold 0.30) lowered the count MAE from 1.35 to 1.23 on val and from 1.56 to 1.44 on test. It was not shipped because the next step made it unnecessary.

### 9. Dropping stages and artefacts

No new photos can be taken and nobody is available to review labels. Given the analysis, the model was simplified to what matters most for the app, counting parasites:

- the four stages are merged into one class, `parasite`, so the model no longer has to split its confidence between ring and trophozoite;
- artefacts become background: 213 boxes were never enough to learn them (AP ≈ 0 in every run).

### 10. Second round: resolution and model size

Three single-class variants on the same data, 40 fixed epochs so the learning rate decays to the end, compared on val:

| | Reference: stage model counted without stages | E1 | E2 | E3 |
|---|---|---|---|---|
| Model | yolo26n | yolo26n | yolo26n | yolo26s |
| Input | field at 1280 | field at 1280 | field at 1920 | field at 1280 |
| AP50 | 0.60 | 0.665 | 0.689 | stopped, see [12](#12-third-round-a-bigger-model) |
| Count MAE per image | 1.23 | 1.12 | 1.04 | |
| Recall, boxes under 16 px | 50 % | 64 % | 65 % | |
| Phone, GPU | 72 ms | 71 ms | fails | |
| Phone, CPU | 235 ms | 233 ms | 625 ms | |

- **Single class helps.** E1 beats the reference on every metric, most of all on small parasites.
- **More resolution helps little.** The field is about 2,700 px in the original photo, so 1920 px keeps more detail, but E2 gains only 0.08 in count MAE and nothing meaningful on small parasites. On the phone it is 2.7 times slower on CPU, uses 677 MB, and does not run on the GPU delegate at all (one tensor exceeds the maximum OpenCL texture size). On a mid-range phone it would likely go over one second per photo.
- **E3 was stopped.** The three runs were launched together on one RTX 4090; the GPU was saturated, E2 ran out of memory and E3 needed about 9 minutes per epoch, too slow to finish within the spending limit. It was relaunched on its own later.

E1 was chosen.

### 11. Result on test

E1 (`w8a32`, threshold 0.30 from val) against the stage model that was in the app, on the same 1,020 test images:

| | Stage model | E1 |
|---|---|---|
| Count MAE per image | 1.56 | 1.33 |
| Relative count error | 44 % | 37 % |
| Exact counts | 36 % | 44.5 % |
| Images off by 3 or more | 15.6 % | 12.1 % |
| AP50 | 0.43 (with stages) | 0.635 |

The count MAE is better on thin smears (1.28) than on thick ones (1.37). Compared image by image, E1 is better on 339 images, equal on 494 and worse on 187; a paired bootstrap puts the MAE improvement between 0.12 and 0.34 parasites per image (95 %).

### 12. Third round: a bigger model

Before spending on it, an untrained `yolo26s` with one class was exported and timed on the phone: 165 ms on GPU and 477 ms on CPU, slower than E1 but usable. `yolo26n-p2` (an extra high-resolution head for small objects) was also timed and ruled out: it does not run on the GPU delegate, for the same reason as E2.

E3 (`yolo26s`, field at 1280, single class) was then trained alone on the RTX 4090 with the same recipe as E1 (40 epochs, about 65 seconds each). On val, both `w8a32`, both at their best count threshold (0.30):

| | E1 | E3 |
|---|---|---|
| AP50 | 0.665 | 0.681 |
| Count MAE per image | 1.12 | 1.05 |
| Exact counts | 44.9 % | 49.9 % |
| Count MAE, thin / thick smears | 0.92 / 1.26 | 0.68 / 1.30 |
| Recall / precision at the threshold (PyTorch weights) | 68.7 % / 73.6 % | 68.3 % / 74.9 % |
| Recall, boxes under 16 px (PyTorch weights) | 64 % | 63 % |
| Size | 3 MB | 9.9 MB |
| Phone, GPU | 71 ms | 165 ms |
| Phone, CPU | 233 ms | 475 ms |

- **The gain is small and uncertain.** Image by image, E3 is better on 267 images, equal on 553 and worse on 200; a paired bootstrap puts the MAE improvement between 0.00 and 0.14 parasites per image (95 %). At a threshold of 0.35 the two are the same.
- **It does not find more parasites.** Recall, including small parasites, is unchanged; the bigger model is only slightly more precise. The improvement is concentrated in thin smears, while thick smears are no better and E3 under-counts them more (bias −0.58 against −0.18).
- **It costs 2.3 times the latency**, which matters for the live video mode and for slower phones.

E1 stays in the app. E3 was not evaluated on test. It is the candidate if the app ever offers a slower, more precise mode for thin smears.

### What has not been tried

- Retraining E1 on train and val together, keeping the 0.30 threshold.
- Reviewing labels: high-confidence detections with no annotation, and ring against trophozoite in thick smears, if the stage is needed again.
- Photos of fields without parasites, to measure false positives on negative slides.
- Measuring latency on a mid-range phone.

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

Variants, defined in `src/aiscope/train.py`:

| Variant | Model | Data | Input |
|---|---|---|---|
| A | yolo26n-p2 | stages, `campo_1280` | 640 |
| B | yolo26n | stages, `mosaicos_640` | 640 tiles |
| C | yolo26n | stages, `campo_1280` | 1280 |
| E1 | yolo26n | single class, `parasito_1280` | 1280 |
| E2 | yolo26n | single class, `parasito_1920` | 1920 |
| E3 | yolo26s | single class, `parasito_1280` | 1280 |

Short local experiment comparing variants:

```bash
.venv/bin/python -m aiscope.experiment --variants A B C --epochs 15 --fraction 0.5 --device mps
```

Training a single variant (on a CUDA machine, use `--device 0`). The second round used 40 fixed epochs:

```bash
.venv/bin/python -m aiscope.train --variant E1 --epochs 40 --patience 1000 --device 0
```

On a cloud GPU the environment is the same `requirements.lock` with the CUDA build of `torch==2.13.0` and `torchvision==0.28.0` from the PyTorch index, and the absolute `path` in each `data.yaml` rewritten for the machine.

Evaluation. It always scores against the 1280 px field boxes, maps the annotated classes to the model's classes by name, and chooses the count threshold on val and reuses it on test:

```bash
.venv/bin/python -m aiscope.evaluate --weights models/runs/<run>/weights/best.pt --variant E1 --split val
```

Export to LiteRT and compare the metrics before and after quantization:

```bash
.venv/bin/python -m aiscope.export --weights models/runs/<run>/weights/best.pt --variant E1 --quantize w8a32
```

This writes `models/exported/<run>_w8a32.tflite` and `<run>_w8a32_contrato.json`, the file the app reads to know the input size, the classes, which classes are not counted and the threshold.

Reference inference without Ultralytics, and generation of the golden test cases for the app (12 test images with their expected output):

```bash
.venv/bin/python -m aiscope.infer --model models/exported/<run>_w8a32.tflite --images photo.jpg
.venv/bin/python -m aiscope.infer --model models/exported/<run>_w8a32.tflite --golden 12
```

## What the app has to do

`src/aiscope/infer.py` is the executable reference. It gives the same boxes as Ultralytics (48 boxes compared, IoU 1.000). Everything below is read from the contract, so the app works with the stage model and the single-class one.

Input:

1. Find the eyepiece field on the full photo: largest region with gray level above 40 on a 512 px thumbnail, holes filled, and the square that encloses it (`src/aiscope/field.py`).
2. Crop that square (anything outside the photo is black) and resize it to 1280 x 1280 with bilinear interpolation.
3. Tensor `[1, 3, 1280, 1280]`, NCHW, float32, pixel / 255.

Output is a `[1, 4 + classes, 33600]` tensor, `[1, 5, 33600]` for the current model: normalized `cx, cy, w, h` followed by one score per class. Post-processing:

1. Class and confidence are the maximum of the class scores.
2. Multiply the coordinates by 1280 and convert to corners.
3. Per-class NMS with IoU 0.7, at most 300 detections. This is required: the exported head is the one-to-many branch, not YOLO26's NMS-free one.
4. Count detections with confidence at or above the contract threshold (0.30 for the current model), leaving out the classes listed as not counted (none for the current model; artefacts for the stage model).

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
  experiment.py runs several variants and writes a summary
  style.py      AiScope colors and fonts for figures
models/         weights, exports and benchmark results (not in git)
docs/assets/    AiScope logos, from GDD-app
```

## License

Ultralytics YOLO is AGPL-3.0, so anything built on this code, including the app, has to be released under a compatible license. The AiScope logos in `docs/assets/` come from GDD-app, which is MIT licensed.
