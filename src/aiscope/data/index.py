"""Índice del dataset crudo leído directamente de los zips.

Genera tres tablas en data/interim/:
- samples.parquet: una fila por carpeta (metadatos aplanados e integridad de ficheros).
- images.parquet: una fila por par imagen/máscara (tamaños, EXIF, calidad y hash perceptual).
- instances.parquet: una fila por componente conexa de color opaco en la máscara.

No se interpreta qué significa cada color: se guarda el RGB tal cual. La leyenda
color -> clase se fija fuera de este módulo.
"""
from __future__ import annotations

import io
import json
import os
import re
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

IMG_RE = re.compile(r"^image_(\d+)\.(jpe?g|png)$", re.I)
MASK_RE = re.compile(r"^mask_(\d+)\.png$", re.I)
MIN_COMPONENT_PX = 30  # por debajo se cuenta como píxel suelto, no como instancia
EIGHT_CONN = np.ones((3, 3), dtype=bool)


def list_samples(raw_dir: Path) -> list[dict]:
    """Recorre los zips y agrupa ficheros por carpeta."""
    samples = []
    for zpath in sorted(Path(raw_dir).glob("*.zip")):
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
        dirs, files = set(), defaultdict(set)
        for name in names:
            if "/" not in name:
                continue
            folder, fname = name.split("/", 1)
            dirs.add(folder)
            if fname:
                files[folder].add(fname)
        for folder in sorted(dirs):
            fs = files.get(folder, set())
            images = {int(m.group(1)): n for n in fs if (m := IMG_RE.match(n))}
            masks = {int(m.group(1)): n for n in fs if (m := MASK_RE.match(n))}
            known = set(images.values()) | set(masks.values()) | {"metadata.json"}
            samples.append(
                {
                    "zip": zpath.name,
                    "folder": folder,
                    "images": images,
                    "masks": masks,
                    "has_metadata": "metadata.json" in fs,
                    "other_files": sorted(fs - known),
                }
            )
    return samples


def flatten_metadata(meta: dict) -> dict:
    prep = meta.get("preparation") or {}
    micro = meta.get("microscopeQuality") or {}
    md = meta.get("metadata") or {}
    return {
        "meta_id": meta.get("id"),
        "health_facility": meta.get("healthFacility"),
        "microscopist": meta.get("microscopist"),
        "disease": meta.get("disease"),
        "water_type": prep.get("waterType"),
        "uses_giemsa": prep.get("usesGiemsa"),
        "giemsa_fp": prep.get("giemsaFP"),
        "uses_pbs": prep.get("usesPbs"),
        "reuses_slides": prep.get("reusesSlides"),
        "sample_age": prep.get("sampleAge"),
        "microscope_damaged": micro.get("isDamaged"),
        "magnification": micro.get("magnification"),
        "blood_type": md.get("bloodType"),
        "species": md.get("species"),
        "comments": md.get("comments") or "",
        "app_version": meta.get("appVersion"),
        "device": meta.get("device"),
        "created_on": meta.get("createdOn"),
        "last_modified": meta.get("lastModified"),
    }


def mask_instances(mask: Image.Image) -> tuple[dict, list[dict]]:
    """Componentes conexas (8-conectividad) por color RGB exacto sobre píxeles con alpha=255."""
    a = np.asarray(mask.convert("RGBA"))
    alpha = a[..., 3]
    stats = {
        "mask_w": a.shape[1],
        "mask_h": a.shape[0],
        "partial_alpha_px": int(((alpha > 0) & (alpha < 255)).sum()),
    }
    opaque = alpha == 255
    ys, xs = np.nonzero(opaque)
    if ys.size == 0:
        stats.update(opaque_px=0, stray_px=0, n_colors=0)
        return stats, []
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = a[y0:y1, x0:x1].astype(np.uint32)
    code = (crop[..., 0] << 16) | (crop[..., 1] << 8) | crop[..., 2]
    op = opaque[y0:y1, x0:x1]
    codes, counts = np.unique(code[op], return_counts=True)

    instances, stray = [], 0
    for c, n in zip(codes.tolist(), counts.tolist()):
        if n < MIN_COMPONENT_PX:
            stray += n
            continue
        m = op & (code == c)
        lab, k = ndimage.label(m, structure=EIGHT_CONN)
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            comp = lab[sl] == i
            area = int(comp.sum())
            if area < MIN_COMPONENT_PX:
                stray += area
                continue
            cy, cx = ndimage.center_of_mass(comp)
            instances.append(
                {
                    "color": f"#{c:06x}",
                    "x0": int(sl[1].start + x0),
                    "y0": int(sl[0].start + y0),
                    "x1": int(sl[1].stop + x0),
                    "y1": int(sl[0].stop + y0),
                    "cx": float(cx + sl[1].start + x0),
                    "cy": float(cy + sl[0].start + y0),
                    "area": area,
                    "hole_px": int(ndimage.binary_fill_holes(comp).sum() - area),
                }
            )
    stats.update(opaque_px=int(op.sum()), stray_px=int(stray), n_colors=int((counts >= MIN_COMPONENT_PX).sum()))
    return stats, instances


def image_quality(img: Image.Image) -> dict:
    """Métricas baratas sobre una decodificación JPEG a 1/4 de resolución."""
    w, h = img.size
    exif_orientation = img.getexif().get(0x0112, 1)
    img.draft("RGB", (w // 4, h // 4))
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = rgb.mean(axis=-1)
    scale = w / gray.shape[1]

    bright = gray > 40  # fuera del campo circular del ocular queda casi negro
    lab, k = ndimage.label(bright)
    field = np.zeros_like(bright)
    fx0 = fy0 = fx1 = fy1 = None
    if k:
        largest = int(np.argmax(ndimage.sum(bright, lab, range(1, k + 1)))) + 1
        field = ndimage.binary_fill_holes(lab == largest)
        sl = ndimage.find_objects(field.astype(np.int32))[0]
        fy0, fy1 = int(sl[0].start * scale), int(sl[0].stop * scale)
        fx0, fx1 = int(sl[1].start * scale), int(sl[1].stop * scale)
    inner = ndimage.binary_erosion(field, iterations=8)
    lap = ndimage.laplace(gray)
    inside = rgb[inner] if inner.any() else rgb.reshape(-1, 3)

    # Hash perceptual (dHash 16x16) del recorte central del campo
    ch, cw = gray.shape
    center = gray[ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4]
    small = np.asarray(Image.fromarray(center).resize((17, 16), Image.BILINEAR))
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    dhash = np.packbits(bits).tobytes().hex()

    return {
        "img_w": w,
        "img_h": h,
        "img_mode": img.mode,
        "exif_orientation": int(exif_orientation),
        "field_frac": float(field.mean()),
        "field_x0": fx0,
        "field_y0": fy0,
        "field_x1": fx1,
        "field_y1": fy1,
        "brightness": float(inside.mean()),
        "mean_r": float(inside[:, 0].mean()),
        "mean_g": float(inside[:, 1].mean()),
        "mean_b": float(inside[:, 2].mean()),
        "contrast": float(inside.mean(axis=1).std()),
        "sharpness": float(lap[inner].var()) if inner.any() else float(lap.var()),
        "dhash": dhash,
    }


def scan_sample(args: tuple[str, dict]) -> tuple[dict, list[dict], list[dict]]:
    raw_dir, s = args
    row = {k: s[k] for k in ("zip", "folder", "has_metadata")}
    row["n_image_files"] = len(s["images"])
    row["n_mask_files"] = len(s["masks"])
    row["images_without_mask"] = sorted(set(s["images"]) - set(s["masks"]))
    row["masks_without_image"] = sorted(set(s["masks"]) - set(s["images"]))
    row["other_files"] = s["other_files"]
    image_rows, instance_rows = [], []
    with zipfile.ZipFile(Path(raw_dir) / s["zip"]) as zf:
        if s["has_metadata"]:
            row.update(flatten_metadata(json.loads(zf.read(f"{s['folder']}/metadata.json"))))
        for idx in sorted(set(s["images"]) | set(s["masks"])):
            image_id = f"{s['folder']}/{idx}"
            irow = {"image_id": image_id, "folder": s["folder"], "idx": idx, "zip": s["zip"]}
            irow["image_file"] = s["images"].get(idx)
            irow["mask_file"] = s["masks"].get(idx)
            try:
                if irow["image_file"]:
                    data = zf.read(f"{s['folder']}/{irow['image_file']}")
                    irow["image_bytes"] = len(data)
                    irow.update(image_quality(Image.open(io.BytesIO(data))))
                if irow["mask_file"]:
                    mstats, inst = mask_instances(Image.open(io.BytesIO(zf.read(f"{s['folder']}/{irow['mask_file']}"))))
                    irow.update(mstats)
                    irow["n_instances"] = len(inst)
                    for j, r in enumerate(inst):
                        instance_rows.append({"image_id": image_id, "folder": s["folder"], "inst": j, **r})
            except Exception as e:  # se registra y se revisa en limpieza
                irow["read_error"] = f"{type(e).__name__}: {e}"
            image_rows.append(irow)
    return row, image_rows, instance_rows


def to_image_coords(instances: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Pasa cajas, centroides y áreas de coordenadas de máscara a coordenadas de imagen.

    Algunas máscaras están guardadas a otra resolución que su imagen (p. ej. a la mitad).
    """
    sc = images.set_index("image_id")
    sx = instances["image_id"].map(sc["img_w"] / sc["mask_w"])
    sy = instances["image_id"].map(sc["img_h"] / sc["mask_h"])
    out = instances.copy()
    out["mask_scale"] = sx
    for c in ("x0", "x1", "cx"):
        out[c] = instances[c] * sx
    for c in ("y0", "y1", "cy"):
        out[c] = instances[c] * sy
    out["area"] = instances["area"] * sx * sy
    out["hole_px"] = instances["hole_px"] * sx * sy
    return out


def build_index(raw_dir: Path, out_dir: Path, workers: int | None = None, force: bool = False):
    out_dir = Path(out_dir)
    paths = {k: out_dir / f"{k}.parquet" for k in ("samples", "images", "instances")}
    if not force and all(p.exists() for p in paths.values()):
        return tuple(pd.read_parquet(p) for p in paths.values())

    samples = list_samples(raw_dir)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    rows, images, instances = [], [], []
    with Pool(workers) as pool:
        for i, (r, im, ins) in enumerate(pool.imap_unordered(scan_sample, [(str(raw_dir), s) for s in samples])):
            rows.append(r)
            images.extend(im)
            instances.extend(ins)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(samples)} carpetas")

    out_dir.mkdir(parents=True, exist_ok=True)
    dfs = (
        pd.DataFrame(rows).sort_values(["zip", "folder"], ignore_index=True),
        pd.DataFrame(images).sort_values(["folder", "idx"], ignore_index=True),
        pd.DataFrame(instances).sort_values(["folder", "image_id", "inst"], ignore_index=True),
    )
    for df, p in zip(dfs, paths.values()):
        df.to_parquet(p, index=False)
    return dfs
