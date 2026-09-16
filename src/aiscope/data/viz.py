"""Utilidades para ver recortes y miniaturas directamente desde los zips."""
from __future__ import annotations

import io
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@lru_cache(maxsize=16)
def _zip(path: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def read_member(raw_dir: Path, zip_name: str, member: str) -> bytes:
    return _zip(str(Path(raw_dir) / zip_name)).read(member)


def thumbnail(raw_dir: Path, row, max_side: int = 384) -> Image.Image:
    img = Image.open(io.BytesIO(read_member(raw_dir, row["zip"], f"{row['folder']}/{row['image_file']}")))
    img.draft("RGB", (max_side, max_side))
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def instance_crop(raw_dir: Path, img_row, inst_row, out: int = 224, context: float = 1.6) -> Image.Image:
    """Recorte cuadrado centrado en la instancia con el contorno de su máscara en rojo."""
    img = Image.open(io.BytesIO(read_member(raw_dir, img_row["zip"], f"{img_row['folder']}/{img_row['image_file']}")))
    mask = Image.open(io.BytesIO(read_member(raw_dir, img_row["zip"], f"{img_row['folder']}/{img_row['mask_file']}")))
    side = int(max(inst_row["x1"] - inst_row["x0"], inst_row["y1"] - inst_row["y0"]) * context / 2) + 20
    cx, cy = int(inst_row["cx"]), int(inst_row["cy"])
    box = (cx - side, cy - side, cx + side, cy + side)
    crop = img.convert("RGB").crop(box)
    if mask.size != img.size:  # máscaras guardadas a otra resolución
        mask = mask.convert("RGBA").resize(img.size, Image.NEAREST)
    m = np.asarray(mask.convert("RGBA").crop(box))
    rgb = tuple(int(inst_row["color"][i : i + 2], 16) for i in (1, 3, 5))
    sel = (m[..., 3] == 255) & np.all(m[..., :3] == rgb, axis=-1)
    edge = Image.fromarray((sel * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(5))
    crop.paste((255, 0, 0), mask=edge)
    return crop.resize((out, out))


def boxes_crop(raw_dir: Path, img_row, boxes: list[tuple], stroke: tuple, out: int = 224, label: str = "") -> Image.Image:
    """Recorte alrededor de un trazo con su caja (rojo) y las cajas finales (verde)."""
    img = Image.open(io.BytesIO(read_member(raw_dir, img_row["zip"], f"{img_row['folder']}/{img_row['image_file']}")))
    x0, y0, x1, y1 = stroke
    s = int(max(x1 - x0, y1 - y0) * 0.7) + 20
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    crop = img.convert("RGB").crop((cx - s, cy - s, cx + s, cy + s))
    d = ImageDraw.Draw(crop)
    w = max(2, s // 70)
    d.rectangle([x0 - cx + s, y0 - cy + s, x1 - cx + s, y1 - cy + s], outline=(230, 0, 0), width=w)
    for b in boxes:
        d.rectangle([b[0] - cx + s, b[1] - cy + s, b[2] - cx + s, b[3] - cy + s], outline=(0, 190, 0), width=w)
    crop = crop.resize((out, out))
    if label:
        ImageDraw.Draw(crop).rectangle([0, 0, out, 13], fill=(255, 255, 255))
        ImageDraw.Draw(crop).text((2, 1), label, fill=(0, 0, 0))
    return crop


def grid(images: list[Image.Image], cols: int, pad: int = 4, bg=(255, 255, 255)) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), bg)
    w = max(i.width for i in images)
    h = max(i.height for i in images)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (w + pad), rows * (h + pad)), bg)
    for k, im in enumerate(images):
        sheet.paste(im, ((k % cols) * (w + pad), (k // cols) * (h + pad)))
    return sheet
