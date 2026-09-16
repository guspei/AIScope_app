"""Entrenamiento del detector (CLI, igual en local que en una GPU remota).

    python -m aiscope.train --variant C --epochs 30 --device mps
    python -m aiscope.train --data data/processed/yolo/campo_1280/data.yaml --model yolo26n --imgsz 1280 --device 0

Variantes del experimento (datos exportados en notebooks/03_particion):
    A  yolo26n-p2  campo recortado a 640 (desde campo_1280)
    B  yolo26n     mosaicos de 640 (campo a 1200, 4 mosaicos con solape)
    C  yolo26n     campo recortado a 1280
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import aiscope  # noqa: F401  (desactiva el autoinstalado de Ultralytics)
from aiscope.paths import MODELS_DIR, PROCESSED_DIR

PRETRAINED_DIR = MODELS_DIR / "pretrained"
RUNS_DIR = MODELS_DIR / "runs"

VARIANTS = {
    "A": {"model": "yolo26n-p2", "data": PROCESSED_DIR / "yolo/campo_1280/data.yaml", "imgsz": 640, "batch": 16},
    "B": {"model": "yolo26n", "data": PROCESSED_DIR / "yolo/mosaicos_640/data.yaml", "imgsz": 640, "batch": 16},
    "C": {"model": "yolo26n", "data": PROCESSED_DIR / "yolo/campo_1280/data.yaml", "imgsz": 1280, "batch": 8},
}

# Aumentos: el frotis no tiene orientación (volteos en ambos ejes); la tinción varía entre centros (HSV por defecto).
AUGMENT = {"fliplr": 0.5, "flipud": 0.5, "degrees": 0.0, "mosaic": 1.0, "close_mosaic": 5}


def pretrained_weights(model: str) -> Path:
    """Pesos COCO de la escala base (yolo26n-p2 → yolo26n.pt), descargados a models/pretrained/."""
    from ultralytics.utils.downloads import attempt_download_asset

    base = model.split("-")[0]  # yolo26n-p2 → yolo26n
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    return Path(attempt_download_asset(str(PRETRAINED_DIR / f"{base}.pt")))


def build_model(model: str):
    from ultralytics import YOLO

    weights = pretrained_weights(model)
    if "-" in model:  # arquitectura distinta de la preentrenada: se construye y se cargan los pesos compatibles
        return YOLO(f"{model}.yaml").load(str(weights))
    return YOLO(str(weights))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    ap.add_argument("--model")
    ap.add_argument("--data")
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--fraction", type=float, default=1.0, help="fracción de train (experimentos cortos)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name")
    a = ap.parse_args()

    cfg = dict(VARIANTS[a.variant]) if a.variant else {}
    for k in ("model", "data", "imgsz", "batch"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    missing = [k for k in ("model", "data", "imgsz", "batch") if k not in cfg]
    if missing:
        ap.error(f"faltan {missing} (usa --variant o pásalos)")

    from ultralytics.models.yolo.detect import DetectionTrainer

    workers = a.workers

    class Trainer(DetectionTrainer):
        """Ultralytics fuerza workers=0 en cpu/mps; con 8 workers MPS entrena ~25 % más rápido (benchmark)."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.args.workers = workers

    model = build_model(cfg["model"])
    name = a.name or f"{a.variant or 'custom'}_{cfg['model']}_{cfg['imgsz']}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"variant": a.variant, **{k: str(v) for k, v in cfg.items()}, "epochs": a.epochs, "fraction": a.fraction,
            "device": a.device, "seed": a.seed, "augment": AUGMENT, "host": platform.node(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))

    model.train(
        trainer=Trainer if a.device in ("mps", "cpu") else None,
        data=str(cfg["data"]), imgsz=cfg["imgsz"], batch=cfg["batch"], epochs=a.epochs, patience=a.patience,
        fraction=a.fraction, device=a.device, workers=workers, seed=a.seed, deterministic=False,
        project=str(RUNS_DIR), name=name, exist_ok=True, plots=True, **AUGMENT,
    )
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    print(run_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()
