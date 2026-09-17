"""Evaluación: mAP por clase y error absoluto medio del conteo de parásitos por imagen, siempre juntos.

    python -m aiscope.evaluate --weights models/runs/<run>/weights/best.pt --variant C --split val
    python -m aiscope.evaluate --weights models/runs/<run>/weights/best.pt --variant C --split test   # usa el umbral de val
    python -m aiscope.evaluate --weights modelo_int8.tflite --mode campo --imgsz 1280 --split test --conf 0.3

La referencia es siempre el campo recortado a 1280 px (data/processed/yolo/campo_1280), así todas las variantes se
miden sobre las mismas cajas. En modo mosaicos el campo se lleva a 1200 px, se parte en 4 mosaicos de 640 y las
detecciones se unen con NMS por clase antes de medir. Un modelo con entrada mayor (E2, 1920) predice sobre su propio
export y sus cajas se llevan a la escala de 1280.

Las clases salen del modelo (o del contrato, para un .tflite) y las cajas de referencia se reasignan por nombre: para
el detector de una clase, los cuatro estadios son «parasite» y los artefactos se quitan.

Conteo: parásitos = detecciones de estadios de parásito (no artefactos) con confianza ≥ umbral. El umbral se elige
en val minimizando el MAE y se aplica tal cual en test.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import aiscope  # noqa: F401
from aiscope.data.classes import CLASSES, NOT_PARASITE, class_map
from aiscope.data.yolo import tile_offsets
from aiscope.paths import INTERIM_DIR, PROCESSED_DIR

REF_DIR = PROCESSED_DIR / "yolo" / "campo_1280"
REF_SIZE = 1280
CONF_MIN = 0.001
MAX_DET = 300
PARASITE_IDS = [i for i, c in enumerate(CLASSES) if c not in NOT_PARASITE]
MODES = {"A": ("campo", 640), "B": ("mosaicos", 640), "C": ("campo", 1280),
         "E1": ("campo", 1280), "E2": ("campo", 1920), "E3": ("campo", 1280)}
SOURCES = {"E2": PROCESSED_DIR / "yolo" / "parasito_1920"}  # export del que se predice si no es la referencia


def model_names(model, weights: Path) -> list[str]:
    """Clases del modelo; un .tflite no las lleva dentro y se leen de su contrato."""
    contrato = weights.with_name(f"{weights.stem}_contrato.json")
    if weights.suffix == ".tflite" and contrato.exists():
        clases = json.loads(contrato.read_text())["salida"]["clases"]
        return [clases[k] for k in sorted(clases, key=int)]
    return [model.names[i] for i in sorted(model.names)]


def remap_gt(gt: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    keep = np.array([int(c) in mapping for c in gt[:, 0]], dtype=bool)
    gt = gt[keep].copy()
    gt[:, 0] = [mapping[int(c)] for c in gt[:, 0]]
    return gt


def load_gt(lbl_path: Path, size: int = REF_SIZE) -> np.ndarray:
    """Etiquetas YOLO → (M, 5) [cls, x0, y0, x1, y1] en píxeles del recorte."""
    rows = [line.split() for line in lbl_path.read_text().splitlines() if line.strip()]
    if not rows:
        return np.zeros((0, 5))
    a = np.asarray(rows, dtype=float)
    x, y, w, h = (a[:, 1:] * size).T
    return np.column_stack([a[:, 0], x - w / 2, y - h / 2, x + w / 2, y + h / 2])


def _to_array(result) -> np.ndarray:
    b = result.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 6))
    return np.column_stack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()])


def predict(model, paths: list[Path], mode: str, imgsz: int, device: str, batch: int = 8):
    """Genera (path, preds (N, 6) [x0, y0, x1, y1, conf, cls]) en píxeles del recorte de 1280.

    El .tflite exportado tiene el lote fijado a 1, así que se procesa de una en una.
    """
    if mode == "campo":
        for i in range(0, len(paths), batch):
            chunk = [str(p) for p in paths[i:i + batch]]
            for p, r in zip(chunk, model.predict(chunk, imgsz=imgsz, conf=CONF_MIN, max_det=MAX_DET, device=device, verbose=False)):
                yield Path(p), _to_array(r)
        return

    import torch
    from PIL import Image
    from torchvision.ops import batched_nms

    big, tile = 1200, imgsz
    offs = [(ox, oy) for oy in tile_offsets(big, tile) for ox in tile_offsets(big, tile)]
    for p in paths:
        field = Image.open(p).convert("RGB").resize((big, big), Image.BILINEAR)
        tiles = [field.crop((ox, oy, ox + tile, oy + tile)) for ox, oy in offs]
        parts = []
        results = []
        for i in range(0, len(tiles), batch):
            results += list(model.predict(tiles[i:i + batch], imgsz=tile, conf=CONF_MIN, max_det=MAX_DET, device=device, verbose=False))
        for (ox, oy), r in zip(offs, results):
            a = _to_array(r)
            a[:, [0, 2]] += ox
            a[:, [1, 3]] += oy
            parts.append(a)
        a = np.concatenate(parts)
        if len(a):
            keep = batched_nms(torch.from_numpy(a[:, :4]).float(), torch.from_numpy(a[:, 4]).float(),
                               torch.from_numpy(a[:, 5]).long(), 0.5).numpy()
            a = a[keep][:MAX_DET]
            a[:, :4] *= REF_SIZE / big
        yield p, a


def match(preds: np.ndarray, gt: np.ndarray, iouv: np.ndarray) -> np.ndarray:
    """(N, len(iouv)) aciertos por umbral de IoU, con la misma asignación voraz que el validador de Ultralytics."""
    correct = np.zeros((len(preds), len(iouv)), dtype=bool)
    if not len(preds) or not len(gt):
        return correct
    x0 = np.maximum(gt[:, None, 1], preds[None, :, 0])
    y0 = np.maximum(gt[:, None, 2], preds[None, :, 1])
    x1 = np.minimum(gt[:, None, 3], preds[None, :, 2])
    y1 = np.minimum(gt[:, None, 4], preds[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_g = (gt[:, 3] - gt[:, 1]) * (gt[:, 4] - gt[:, 2])
    area_p = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
    iou = inter / (area_g[:, None] + area_p[None, :] - inter + 1e-9)
    iou = iou * (gt[:, None, 0] == preds[None, :, 5])
    for i, t in enumerate(iouv):
        m = np.argwhere(iou >= t)
        if len(m):
            if len(m) > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1], i] = True
    return correct


def count_metrics(df: pd.DataFrame, conf: float) -> dict:
    err = df["pred_" + f"{conf:.2f}"] - df["gt"]
    return {"mae": float(err.abs().mean()), "bias": float(err.mean()), "mae_rel_%": float(100 * err.abs().sum() / max(df["gt"].sum(), 1)),
            "exactas_%": float(100 * (err == 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--variant", choices=sorted(MODES))
    ap.add_argument("--mode", choices=["campo", "mosaicos"])
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--conf", type=float, help="umbral de conteo; por defecto el elegido en val (o se elige si split=val)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, help="evaluar solo N imágenes (prueba rápida)")
    ap.add_argument("--src", help="export YOLO del que se predice (por defecto, el de la variante o campo_1280)")
    ap.add_argument("--out")
    a = ap.parse_args()

    mode, imgsz = MODES[a.variant] if a.variant else (a.mode, a.imgsz)
    mode, imgsz = a.mode or mode, a.imgsz or imgsz
    if not mode or not imgsz:
        ap.error("usa --variant o --mode e --imgsz")

    from ultralytics import YOLO
    from ultralytics.utils.metrics import ap_per_class

    weights = Path(a.weights)
    out_dir = Path(a.out) if a.out else (weights.parent.parent if weights.parent.name == "weights" else weights.parent)
    model = YOLO(str(weights), task="detect")
    names = model_names(model, weights)
    mapping = class_map(CLASSES, names)
    par_ids = [i for i, c in enumerate(names) if c not in NOT_PARASITE]
    src = Path(a.src) if a.src else SOURCES.get(a.variant, REF_DIR)
    paths = sorted((src / "images" / a.split).glob("*.jpg"))[: a.limit]
    src_side = Image.open(paths[0]).width if paths else REF_SIZE
    images = pd.read_parquet(INTERIM_DIR / "images_clean.parquet")[["image_id", "preparacion", "especie"]]
    images["stem"] = images["image_id"].str.replace("/", "_", regex=False)
    iouv = np.linspace(0.5, 0.95, 10)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)

    es_tflite = weights.suffix == ".tflite"
    stats, rows, t0 = [], [], time.perf_counter()
    for p, preds in predict(model, paths, mode, imgsz, "cpu" if es_tflite else a.device, batch=1 if es_tflite else 8):
        if mode == "campo" and src_side != REF_SIZE:
            preds[:, :4] *= REF_SIZE / src_side
        gt = remap_gt(load_gt(REF_DIR / "labels" / a.split / f"{p.stem}.txt"), mapping)
        stats.append((match(preds, gt, iouv), preds[:, 4], preds[:, 5], gt[:, 0]))
        par = np.isin(preds[:, 5], par_ids)
        row = {"stem": p.stem, "gt": int(np.isin(gt[:, 0], par_ids).sum())}
        row.update({f"pred_{t:.2f}": int((par & (preds[:, 4] >= t)).sum()) for t in grid})
        rows.append(row)
    elapsed = time.perf_counter() - t0

    tp, conf, pcls, tcls = (np.concatenate(x, 0) for x in zip(*stats))
    res = ap_per_class(tp, conf, pcls, tcls, names=dict(enumerate(names)))
    p_, r_, ap, uniq = res[2], res[3], res[5], res[6].astype(int)
    per_class = pd.DataFrame({
        "clase": [names[c] for c in uniq], "cajas": [int((tcls == c).sum()) for c in uniq],
        "P": p_.round(3), "R": r_.round(3), "mAP50": ap[:, 0].round(3), "mAP50-95": ap.mean(1).round(3),
    })

    counts = pd.DataFrame(rows).merge(images, on="stem", how="left")
    val_json = out_dir / f"eval_val_{mode}{imgsz}.json"
    if a.conf is not None:
        thr, origen = a.conf, "argumento"
    elif a.split == "test" and val_json.exists():
        thr, origen = json.loads(val_json.read_text())["conteo"]["umbral"], "val"
    else:
        thr = float(min(grid, key=lambda t: count_metrics(counts, t)["mae"]))
        origen = "elegido en este split" + (" (¡en test!)" if a.split == "test" else "")
    thr = float(min(grid, key=lambda t: abs(t - thr)))

    report = {
        "weights": str(weights), "split": a.split, "modo": mode, "imgsz": imgsz, "imagenes": len(paths), "clases": names,
        "ms_por_imagen": round(1000 * elapsed / max(len(paths), 1), 1),
        "deteccion": {"mAP50": float(ap[:, 0].mean()), "mAP50-95": float(ap.mean()), "por_clase": per_class.to_dict("records")},
        "conteo": {"umbral": thr, "origen_umbral": origen, **count_metrics(counts, thr),
                   "por_preparacion": {k: count_metrics(g, thr) for k, g in counts.groupby("preparacion")},
                   "mae_por_umbral": {f"{t:.2f}": round(count_metrics(counts, t)["mae"], 3) for t in grid}},
    }
    out = out_dir / f"eval_{a.split}_{mode}{imgsz}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    pd.set_option("display.width", 160)
    print(f"\n{a.split} · {mode} {imgsz} · {len(paths)} imágenes · {report['ms_por_imagen']} ms/imagen")
    print(per_class.to_string(index=False))
    print(f"mAP50 {report['deteccion']['mAP50']:.3f} · mAP50-95 {report['deteccion']['mAP50-95']:.3f}")
    c = report["conteo"]
    print(f"conteo (umbral {thr:.2f}, {origen}): MAE {c['mae']:.2f} · sesgo {c['bias']:+.2f} · error relativo {c['mae_rel_%']:.1f} % · exactas {c['exactas_%']:.1f} %")
    for k, v in c["por_preparacion"].items():
        print(f"  {k}: MAE {v['mae']:.2f} · sesgo {v['bias']:+.2f}")
    print(out)


if __name__ == "__main__":
    main()
