"""Exportación a formato YOLO: recorte cuadrado del campo del ocular, redimensionado y, opcionalmente, mosaicos."""
from __future__ import annotations

import io
import os
import zipfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

MIN_VISIBLE = 0.5  # fracción mínima de la caja que debe quedar dentro del recorte o mosaico


def field_square(row) -> tuple[int, int, int]:
    """Centro y lado del cuadrado que contiene el campo circular detectado."""
    if pd.isna(row["field_x0"]):
        return int(row["img_w"]) // 2, int(row["img_h"]) // 2, int(min(row["img_w"], row["img_h"]))
    cx = (row["field_x0"] + row["field_x1"]) // 2
    cy = (row["field_y0"] + row["field_y1"]) // 2
    side = int(max(row["field_x1"] - row["field_x0"], row["field_y1"] - row["field_y0"]))
    return int(cx), int(cy), side


def tile_offsets(size: int, tile: int) -> list[int]:
    """Offsets para cubrir `size` con mosaicos de `tile` (el mínimo número, con solape repartido)."""
    if tile >= size:
        return [0]
    n = -(-(size - tile) // tile) + 1  # ceil((size - tile) / tile) + 1
    step = (size - tile) / (n - 1)
    return [round(i * step) for i in range(n)]


def _labels(boxes, x0, y0, side, scale, window) -> list[str]:
    """Cajas (coordenadas de imagen) → líneas YOLO normalizadas a la ventana (wx, wy, w) en el espacio escalado."""
    wx, wy, w = window
    lines = []
    for b in boxes:
        bx0, by0 = (b["x0"] - x0) * scale - wx, (b["y0"] - y0) * scale - wy
        bx1, by1 = (b["x1"] - x0) * scale - wx, (b["y1"] - y0) * scale - wy
        cx0, cy0, cx1, cy1 = max(bx0, 0), max(by0, 0), min(bx1, w), min(by1, w)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if (cx1 - cx0) * (cy1 - cy0) < MIN_VISIBLE * (bx1 - bx0) * (by1 - by0):
            continue
        lines.append(f"{b['class_id']} {(cx0 + cx1) / 2 / w:.6f} {(cy0 + cy1) / 2 / w:.6f} {(cx1 - cx0) / w:.6f} {(cy1 - cy0) / w:.6f}")
    return lines


def _export_one(args) -> int:
    raw_dir, row, boxes, size, tile, img_dir, lbl_dir = args
    cx, cy, side = field_square(row)
    x0, y0 = cx - side // 2, cy - side // 2
    with zipfile.ZipFile(Path(raw_dir) / row["zip"]) as zf:
        img = Image.open(io.BytesIO(zf.read(f"{row['folder']}/{row['image_file']}")))
        img.draft("RGB", (int(row["img_w"]) * size // side, int(row["img_h"]) * size // side))
        d = img.size[0] / row["img_w"]
        img = img.convert("RGB")
    s = int(round(side * d))
    field = img.crop((int(x0 * d), int(y0 * d), int(x0 * d) + s, int(y0 * d) + s)).resize((size, size), Image.BILINEAR)
    scale = size / side
    stem = row["image_id"].replace("/", "_")
    n = 0
    windows = [(0, 0, size)] if not tile else [(ox, oy, tile) for oy in tile_offsets(size, tile) for ox in tile_offsets(size, tile)]
    for wx, wy, w in windows:
        name = stem if not tile else f"{stem}_t{wx}_{wy}"
        im = field if not tile else field.crop((wx, wy, wx + w, wy + w))
        im.save(Path(img_dir) / f"{name}.jpg", quality=95)
        lines = _labels(boxes, x0, y0, side, scale, (wx, wy, w))
        (Path(lbl_dir) / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n += len(lines)
    return n


def export_dataset(raw_dir: Path, images: pd.DataFrame, boxes: pd.DataFrame, splits: dict[str, list[str]],
                   class_names: list[str], out_dir: Path, size: int = 1280, tile: int | None = None,
                   workers: int | None = None) -> Path:
    """Escribe out_dir/{images,labels}/{split}/ y out_dir/data.yaml.

    `boxes`: coordenadas de imagen con `class_id`. `splits`: split → lista de image_id.
    Con `tile`, el campo se lleva a `size` y se parte en mosaicos de `tile` con solape.
    """
    out_dir = Path(out_dir)
    by_img = {k: v[["class_id", "x0", "y0", "x1", "y1"]].to_dict("records") for k, v in boxes.groupby("image_id")}
    img_idx = images.set_index("image_id", drop=False)
    tasks = []
    for split, ids in splits.items():
        img_dir, lbl_dir = out_dir / "images" / split, out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        cols = ["image_id", "folder", "zip", "image_file", "img_w", "img_h", "field_x0", "field_y0", "field_x1", "field_y1"]
        for image_id in ids:
            tasks.append((str(raw_dir), img_idx.loc[image_id, cols].to_dict(), by_img.get(image_id, []), size, tile, img_dir, lbl_dir))
    with Pool(workers or max(1, (os.cpu_count() or 2) - 2)) as pool:
        pool.map(_export_one, tasks, chunksize=8)
    data = {"path": str(out_dir.resolve()), **{k: f"images/{k}" for k in splits}, "names": dict(enumerate(class_names))}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return out_dir / "data.yaml"


def relabel_dataset(src_dir: Path, out_dir: Path, class_names: list[str]) -> Path:
    """Copia un export YOLO con otras clases: mismas imágenes (enlaces duros, sin duplicar disco) y etiquetas
    reasignadas por nombre con `classes.class_map`; las cajas de clases que no existen en `class_names` se quitan.

    Enlaces duros y no simbólicos: Ultralytics resuelve las rutas y leería las etiquetas del export original.
    """
    from aiscope.data.classes import class_map

    src_dir, out_dir = Path(src_dir), Path(out_dir)
    src = yaml.safe_load((src_dir / "data.yaml").read_text())
    mapping = class_map([src["names"][i] for i in sorted(src["names"])], class_names)
    splits = [k for k in ("train", "val", "test") if k in src]
    for split in splits:
        for kind in ("images", "labels"):
            (out_dir / kind / split).mkdir(parents=True, exist_ok=True)
        for img in (src_dir / "images" / split).glob("*.jpg"):
            dst = out_dir / "images" / split / img.name
            if not dst.exists():
                os.link(img, dst)
        for lbl in (src_dir / "labels" / split).glob("*.txt"):
            lines = []
            for line in lbl.read_text().splitlines():
                c, *rest = line.split()
                if int(c) in mapping:
                    lines.append(" ".join([str(mapping[int(c)]), *rest]))
            (out_dir / "labels" / split / lbl.name).write_text("\n".join(lines) + ("\n" if lines else ""))
    data = {"path": str(out_dir.resolve()), **{k: f"images/{k}" for k in splits}, "names": dict(enumerate(class_names))}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return out_dir / "data.yaml"
