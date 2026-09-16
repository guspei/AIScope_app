"""Mediciones para decidir modelo y dónde entrenar.

    python -m aiscope.benchmark subset --n-train 400 --n-val 80 --size 1280
    python -m aiscope.benchmark train --models yolo26n yolo26s --imgsz 640 1280 --device mps
    python -m aiscope.benchmark tflite --models yolo26n yolo26s --imgsz 640 1280

`train` mide imágenes/s de entrenamiento con aumentos por defecto y pesos aleatorios.
El rendimiento no depende de los pesos, así que no se descarga nada.
`tflite` exporta a LiteRT int8 (litert-torch, sin TensorFlow) y mide tamaño y latencia en la CPU de esta máquina.
"""
from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

from aiscope.paths import INTERIM_DIR, PROCESSED_DIR, RAW_DIR, REPO

BENCH_DIR = PROCESSED_DIR / "bench_1280"
RESULTS = REPO / "models" / "benchmarks"


class _Stop(Exception):
    pass


def cmd_subset(a):
    from aiscope.data.classes import CLASSES, MASK_COLORS
    from aiscope.data.index import build_index, to_image_coords
    from aiscope.data.yolo import export_dataset

    samples, images, instances = build_index(RAW_DIR, INTERIM_DIR)
    images = images[images["image_file"].notna() & images["mask_file"].notna()]
    instances = to_image_coords(instances, images)
    instances["class_id"] = instances["color"].map(MASK_COLORS).map(CLASSES.index)
    # Agrupación anti-fuga: misma sesión de captura (centro, microscopista, día)
    ses = (samples["health_facility"] + "|" + samples["microscopist"] + "|" + samples["created_on"].str[:10]).set_axis(samples["folder"])
    images = images.assign(session=images["folder"].map(ses))
    sess = np.random.default_rng(a.seed).permutation(images["session"].unique())
    val_s = set(sess[: len(sess) // 5])
    tr = images[~images["session"].isin(val_s)].sample(a.n_train, random_state=a.seed)["image_id"].tolist()
    va = images[images["session"].isin(val_s)].sample(a.n_val, random_state=a.seed)["image_id"].tolist()
    out = PROCESSED_DIR / f"bench_{a.size}"
    y = export_dataset(RAW_DIR, images, instances, {"train": tr, "val": va}, CLASSES, out, size=a.size)
    print(y)


def _model_yaml(name: str, nc: int, tmp: Path) -> str:
    """yaml de la arquitectura con nc ajustado; la escala (n/s/m) la toma Ultralytics del nombre."""
    import yaml
    from ultralytics.nn.tasks import yaml_model_load

    cfg = yaml_model_load(f"{name}.yaml")
    cfg["nc"] = nc
    p = tmp / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(p)


def _nc(data_yaml: Path) -> int:
    import yaml

    return len(yaml.safe_load(data_yaml.read_text())["names"])


def cmd_train(a):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    class ForcedWorkersTrainer(DetectionTrainer):
        """Ultralytics fuerza workers=0 en cpu/mps; aquí se restauran para cargar datos en paralelo."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.args.workers = a.workers

    trainer_cls = ForcedWorkersTrainer if a.force_workers else None
    data = Path(a.data)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in a.models:
            for imgsz in a.imgsz:
                batch = a.batch or (16 if imgsz <= 640 else 8)
                model = YOLO(_model_yaml(name, _nc(data), Path(tmp)))
                stamps = []

                def on_batch_end(trainer, stamps=stamps):
                    stamps.append(time.perf_counter())
                    if len(stamps) >= a.warmup + a.batches:
                        raise _Stop

                model.add_callback("on_train_batch_end", on_batch_end)
                err = None
                try:
                    model.train(trainer=trainer_cls, data=str(data), imgsz=imgsz, batch=batch, epochs=1000, device=a.device,
                                workers=a.workers, val=False, plots=False, amp=a.device.startswith("cuda") or a.device.isdigit(),
                                project=tmp, name=f"{name}_{imgsz}", exist_ok=True, verbose=False, pretrained=False)
                except _Stop:
                    pass
                except Exception as e:  # p. ej. memoria insuficiente
                    err = f"{type(e).__name__}: {e}"[:200]
                t = np.diff(stamps[a.warmup - 1 :]) if len(stamps) > a.warmup else np.array([])
                mem = None
                if a.device == "mps":
                    mem = round(torch.mps.driver_allocated_memory() / 2**30, 2)
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    mem = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                    torch.cuda.reset_peak_memory_stats()
                row = {
                    "model": name, "imgsz": imgsz, "batch": batch, "device": a.device,
                    "workers": a.workers if (a.force_workers or a.device not in {"cpu", "mps"}) else 0,
                    "img_per_s": round(batch / float(np.median(t)), 1) if t.size else None,
                    "s_per_batch_p50": round(float(np.median(t)), 3) if t.size else None,
                    "mem_gib": mem, "error": err,
                }
                print(json.dumps(row))
                rows.append(row)
    _save("train", rows)


def cmd_tflite(a):
    from ai_edge_litert.interpreter import Interpreter
    from ultralytics import YOLO

    data = Path(a.data)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in a.models:
            for imgsz in a.imgsz:
                model = YOLO(_model_yaml(name, _nc(data), Path(tmp)))
                f = model.export(format="litert", quantize=8, data=str(data), imgsz=imgsz, fraction=a.calib_fraction)
                tfl = Path(f)
                if tfl.is_dir():
                    tfl = next(tfl.rglob("*_int8.tflite"))
                interp = Interpreter(model_path=str(tfl), num_threads=a.threads)
                interp.allocate_tensors()
                inp = interp.get_input_details()[0]
                x = np.random.default_rng(0).random(inp["shape"], dtype=np.float32)
                if inp["dtype"] != np.float32:
                    x = (x * 255).astype(inp["dtype"])
                for _ in range(3):
                    interp.set_tensor(inp["index"], x)
                    interp.invoke()
                lat = []
                for _ in range(a.runs):
                    interp.set_tensor(inp["index"], x)
                    t0 = time.perf_counter()
                    interp.invoke()
                    lat.append(time.perf_counter() - t0)
                row = {
                    "model": name, "imgsz": imgsz, "file": tfl.name, "size_mb": round(tfl.stat().st_size / 2**20, 2),
                    "input": [int(s) for s in inp["shape"]], "input_dtype": str(np.dtype(inp["dtype"])),
                    "threads": a.threads, "lat_ms_p50": round(1000 * float(np.median(lat)), 1),
                    "lat_ms_p90": round(1000 * float(np.percentile(lat, 90)), 1), "host": platform.processor() or platform.machine(),
                }
                print(json.dumps(row))
                rows.append(row)
    _save("tflite", rows)


def _save(kind, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"{kind}_{platform.node()}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(rows, indent=2))
    print(p)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--n-train", type=int, default=400)
    s.add_argument("--n-val", type=int, default=80)
    s.add_argument("--size", type=int, default=1280)
    s.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--data", default=str(BENCH_DIR / "data.yaml"))
    t.add_argument("--models", nargs="+", default=["yolo26n", "yolo26s"])
    t.add_argument("--imgsz", nargs="+", type=int, default=[640, 1280])
    t.add_argument("--batch", type=int, default=None)
    t.add_argument("--device", default="mps")
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--force-workers", action="store_true", help="usar --workers también en cpu/mps")
    t.add_argument("--warmup", type=int, default=5)
    t.add_argument("--batches", type=int, default=25)
    x = sub.add_parser("tflite")
    x.add_argument("--data", default=str(BENCH_DIR / "data.yaml"))
    x.add_argument("--models", nargs="+", default=["yolo26n", "yolo26s"])
    x.add_argument("--imgsz", nargs="+", type=int, default=[640, 1280])
    x.add_argument("--threads", type=int, default=4)
    x.add_argument("--runs", type=int, default=30)
    x.add_argument("--calib-fraction", type=float, default=1.0)
    a = ap.parse_args()
    {"subset": cmd_subset, "train": cmd_train, "tflite": cmd_tflite}[a.cmd](a)


if __name__ == "__main__":
    main()
