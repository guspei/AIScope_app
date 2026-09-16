"""Limpieza de anotaciones: máscara RGBA → instancias con estadio y caja ajustada a la zona teñida.

En GDD-app el pincel mide 80 px / zoom. Tipos de trazo vistos en la revisión visual (notebook 02):
- Toque o arrastre macizo sobre un parásito: una instancia.
- Círculo dibujado alrededor del parásito (un hueco grande): se rellena, una instancia.
- Varios círculos que se tocan (varios huecos grandes): cada hueco es un parásito.
- Garabato que tapa varios parásitos (varios huecos pequeños): cada grupo teñido separado es un parásito,
  salvo esquizontes y gametocitos, que se mantienen como un objeto.
La caja final se ajusta a los píxeles teñidos (oscuros y magenta) dentro del trazo; si no se encuentra
nada fiable se usa la caja del trazo.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from aiscope.data.classes import MASK_COLORS

EIGHT = np.ones((3, 3), dtype=bool)
REFERENCE_FIELD_PX = 2700  # lado típico del campo del ocular en las fotos (notebook 01)

PARAMS = {
    "min_component_px": 30,       # igual que el índice
    "split_min_hole_px": 100,     # hueco mínimo para contar como un círculo dibujado
    "split_min_hole_frac": 0.15,  # y al menos 15 % del hueco mayor de la componente
    "scribble_min_holes": 2,      # componente sin partir con ≥ 2 huecos pequeños = garabato
    "no_cluster_split": ("schizont", "gametocyte"),  # objetos formados por varias piezas teñidas
    "stain_threshold": 6.0,       # z combinado (oscuridad + magenta) respecto a la referencia
    "stain_low_threshold": 2.5,   # umbral de crecimiento desde el núcleo teñido
    "stain_min_px": 6,
    "tight_inside_max_frac": 0.5,  # con referencia interior, lo teñido debe ser minoría dentro del trazo
    "tight_max_area_frac": 0.8,   # con referencia exterior, si lo teñido llena el trazo no aporta
    "grow_max_area_frac": 0.6,    # si al crecer ocupa más del 60 % del trazo, se queda el núcleo
    "salience_keep_frac": 0.5,    # piezas con saliencia ≥ 50 % de la máxima (un solo objeto)
    "cluster_gap_px": 10,         # separación (px, a escala de REFERENCE_FIELD_PX) que une núcleos del mismo parásito
    "core_peak_frac": 0.5,        # en garabatos, núcleo = puntuación ≥ 50 % del pico del trazo
    "cluster_peak_frac": 0.5,     # grupo adicional si su pico ≥ 50 % del principal
    "cluster_size_frac": 0.15,    # y su tamaño ≥ 15 % del principal
    "tight_pad_frac": 0.1,
    "tight_min_pad_px": 2,
    "tight_min_side_px": 16,
}


def _rgba_code(mask: Image.Image, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    m = mask.convert("RGBA")
    if m.size != size:  # máscaras guardadas a menor resolución que la imagen
        m = m.resize(size, Image.NEAREST)
    a = np.asarray(m)
    code = (a[..., 0].astype(np.uint32) << 16) | (a[..., 1].astype(np.uint32) << 8) | a[..., 2]
    return code, a[..., 3] == 255


def _solidity(comp: np.ndarray) -> float:
    ys, xs = np.nonzero(comp)
    if len(ys) < 5:
        return 1.0
    try:
        hull = ConvexHull(np.column_stack([xs, ys]))
    except QhullError:
        return 1.0
    return float(comp.sum() / max(hull.volume, 1.0))


def split_component(comp: np.ndarray, p=PARAMS) -> tuple[list[np.ndarray], dict]:
    """Regiones (en coordenadas del recorte) y diagnóstico. Parte solo círculos dibujados que se tocan."""
    filled = ndimage.binary_fill_holes(comp)
    holes = filled & ~comp
    lab_h, n_h = ndimage.label(holes, structure=EIGHT)
    diag = {"hole_px": int(holes.sum()), "n_holes": int(n_h), "solidity": round(_solidity(filled), 3), "split": "no"}
    if n_h < 2:
        return [filled], diag
    areas = np.asarray(ndimage.sum(holes, lab_h, range(1, n_h + 1)))
    big = [i + 1 for i, a in enumerate(areas) if a >= max(p["split_min_hole_px"], p["split_min_hole_frac"] * areas.max())]
    if len(big) < 2:
        if n_h >= p["scribble_min_holes"]:
            diag["split"] = "garabato"
        return [filled], diag
    seeds = np.where(np.isin(lab_h, big), lab_h, 0)
    _, (iy, ix) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    owner = seeds[iy, ix]
    diag["split"] = "circulos"
    return [(owner == k) & filled for k in big], diag


def stained_boxes(rgb: np.ndarray, region: np.ndarray, offset: tuple[int, int], stage: str, scribble: bool,
                  scale: float, p=PARAMS) -> tuple[list[tuple[int, int, int, int]], dict]:
    """Cajas en coordenadas de imagen ajustadas a lo teñido dentro de `region` (una, o varias en garabatos).

    `region`: máscara del trazo en su recorte; `offset`: (x0, y0) del recorte en la imagen;
    `scale`: lado del campo / REFERENCE_FIELD_PX.
    """
    H, W = rgb.shape[:2]
    ys, xs = np.nonzero(region)
    ry0, ry1, rx0, rx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(ry1 - ry0, rx1 - rx0)
    ox, oy = offset
    m = max(int(0.5 * side), 20)
    cx0, cy0 = max(ox + rx0 - m, 0), max(oy + ry0 - m, 0)
    cx1, cy1 = min(ox + rx1 + m, W), min(oy + ry1 + m, H)
    crop = rgb[cy0:cy1, cx0:cx1].astype(np.float32)
    reg = np.zeros(crop.shape[:2], dtype=bool)
    ry, rx = oy + ry0 - cy0, ox + rx0 - cx0
    reg[ry:ry + (ry1 - ry0), rx:rx + (rx1 - rx0)] = region[ry0:ry1, rx0:rx1]
    stroke_box = (ox + rx0, oy + ry0, ox + rx1, oy + ry1)

    gray = crop.mean(axis=-1)
    magenta = (crop[..., 0] + crop[..., 2]) / 2 - crop[..., 1]
    in_field = gray > 40
    outside = ~ndimage.binary_dilation(reg, iterations=max(2, int(0.1 * side))) & in_field

    def score_vs(ref):
        def robust(v):
            med = float(np.median(v[ref]))
            return med, max(float(np.median(np.abs(v[ref] - med))) * 1.4826, 2.0)
        (mg, sg), (mm, sm) = robust(gray), robust(magenta)
        return (mg - gray) / sg + (magenta - mm) / sm

    # 1) Referencia = interior del trazo: en extensión fina el glóbulo rojo hace de fondo y el parásito destaca.
    # 2) Referencia = fondo exterior: parásitos grandes que ocupan buena parte del trazo.
    score, stained, ref_name = None, None, None
    for ref, max_frac, name in ((reg & in_field, p["tight_inside_max_frac"], "interior"),
                                (outside, p["tight_max_area_frac"], "exterior")):
        if ref.sum() < 50:
            continue
        s = score_vs(ref)
        st = ndimage.binary_opening(reg & (s > p["stain_threshold"]), structure=np.ones((2, 2), bool))
        if p["stain_min_px"] <= st.sum() <= max_frac * reg.sum():
            score, stained, ref_name = s, st, name
            break
    if stained is None:
        return [], {"reason": "sin_tincion_fiable"}

    multi = scribble and stage not in p["no_cluster_split"]
    if multi:
        # En garabatos lo teñido suele formar un bloque (citoplasma y restos unen parásitos): se agrupan solo
        # los núcleos intensos (cromatina), uno por parásito
        pieces = stained & (score > max(p["stain_threshold"], p["core_peak_frac"] * float(score[stained].max())))
        gap = max(2, int(round(p["cluster_gap_px"] * scale)))
    else:
        pieces, gap = stained, 2
    lab, n = ndimage.label(ndimage.binary_dilation(pieces, iterations=gap), structure=EIGHT)
    lab = lab * pieces
    idx = range(1, n + 1)
    sizes = np.asarray(ndimage.sum(pieces, lab, idx))
    peaks = np.asarray(ndimage.maximum(score, lab, idx))
    cents = np.asarray(ndimage.center_of_mass(pieces, lab, idx)).reshape(-1, 2)
    ryc, rxc = ndimage.center_of_mass(reg)
    d = np.hypot(cents[:, 0] - ryc, cents[:, 1] - rxc) / np.sqrt(reg.sum() / np.pi)
    salience = np.where(sizes >= p["stain_min_px"], peaks * np.exp(-2 * d**2), 0)
    if salience.max() <= 0:
        return [], {"reason": "sin_tincion_fiable"}

    main = int(np.argmax(salience))
    if multi:  # un grupo por parásito; los demás grupos deben ser comparables al principal
        groups = [[main + 1]] + [[i + 1] for i in range(n) if i != main and sizes[i] >= p["stain_min_px"]
                                 and peaks[i] >= p["cluster_peak_frac"] * peaks[main]
                                 and sizes[i] >= p["cluster_size_frac"] * sizes[main]]
    else:  # un solo objeto: se unen las piezas salientes
        groups = [[i + 1 for i in np.flatnonzero(salience >= p["salience_keep_frac"] * salience.max())]]

    low_lab, _ = ndimage.label(reg & (score > p["stain_low_threshold"]), structure=EIGHT)
    owner = None
    if len(groups) > 1:  # cada grupo solo crece en su zona (píxeles más cercanos a él que a otro grupo)
        seeds = np.zeros(lab.shape, dtype=np.int32)
        for gi, g in enumerate(groups, start=1):
            seeds[np.isin(lab, g)] = gi
        _, (iy, ix) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
        owner = seeds[iy, ix]
    boxes = []
    for gi, g in enumerate(groups, start=1):
        sel = np.isin(lab, g)
        # Histéresis: del núcleo teñido (cromatina) se crece al citoplasma conectado
        grown = np.isin(low_lab, np.unique(low_lab[sel & (low_lab > 0)]))
        if owner is not None:
            grown &= owner == gi
        if sel.sum() < grown.sum() <= p["grow_max_area_frac"] * reg.sum() / len(groups):
            sel = grown
        boxes.append(_box(sel, cx0, cy0, stroke_box, W, H, p))
    boxes = _merge_overlapping(boxes)
    diag = {"reason": f"ajustada_{ref_name}", "stain_px": int(stained.sum()), "n_clusters": int((sizes >= p["stain_min_px"]).sum()),
            "n_groups": len(groups)}
    return boxes, diag


def _box(sel, cx0, cy0, stroke_box, W, H, p):
    sy, sx = np.nonzero(sel)
    tx0, tx1, ty0, ty1 = sx.min(), sx.max() + 1, sy.min(), sy.max() + 1
    pad = max(p["tight_min_pad_px"], int(p["tight_pad_frac"] * max(tx1 - tx0, ty1 - ty0)))
    box = [tx0 - pad + cx0, ty0 - pad + cy0, tx1 + pad + cx0, ty1 + pad + cy0]
    for a, b in ((0, 2), (1, 3)):  # lado mínimo
        if box[b] - box[a] < p["tight_min_side_px"]:
            c = (box[a] + box[b]) / 2
            box[a], box[b] = int(c - p["tight_min_side_px"] / 2), int(c + p["tight_min_side_px"] / 2)
    # Nunca fuera del trazo: el parásito está dentro de lo que marcó el anotador
    sx0, sy0, sx1, sy1 = stroke_box
    return (int(max(box[0], sx0, 0)), int(max(box[1], sy0, 0)), int(min(box[2], sx1, W)), int(min(box[3], sy1, H)))


def _merge_overlapping(boxes):
    boxes = list(boxes)
    merged = True
    while merged and len(boxes) > 1:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                iw = min(a[2], b[2]) - max(a[0], b[0])
                ih = min(a[3], b[3]) - max(a[1], b[1])
                if iw > 0 and ih > 0 and iw * ih > 0.3 * min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])):
                    boxes[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    boxes.pop(j)
                    merged = True
                    break
            if merged:
                break
    return boxes


def clean_image(args) -> list[dict]:
    raw_dir, row, p = args
    with zipfile.ZipFile(Path(raw_dir) / row["zip"]) as zf:
        img = Image.open(io.BytesIO(zf.read(f"{row['folder']}/{row['image_file']}"))).convert("RGB")
        mask = Image.open(io.BytesIO(zf.read(f"{row['folder']}/{row['mask_file']}")))
    rgb = np.asarray(img)
    field_side = max(row["field_x1"] - row["field_x0"], row["field_y1"] - row["field_y0"]) if pd.notna(row.get("field_x0")) else REFERENCE_FIELD_PX
    scale = float(field_side) / REFERENCE_FIELD_PX
    code, opaque = _rgba_code(mask, img.size)
    ys, xs = np.nonzero(opaque)
    if ys.size == 0:
        return []
    y0, x0 = ys.min(), xs.min()
    code_c, op_c = code[y0:ys.max() + 1, x0:xs.max() + 1], opaque[y0:ys.max() + 1, x0:xs.max() + 1]
    out = []
    for hexcol, stage in MASK_COLORS.items():
        m = op_c & (code_c == int(hexcol[1:], 16))
        if m.sum() < p["min_component_px"]:
            continue
        lab, _ = ndimage.label(m, structure=EIGHT)
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            comp = lab[sl] == k
            if comp.sum() < p["min_component_px"]:
                continue
            regions, diag = split_component(comp, p)
            offset = (int(sl[1].start + x0), int(sl[0].start + y0))
            for j, reg in enumerate(regions):
                ry, rx = np.nonzero(reg)
                stroke = (offset[0] + int(rx.min()), offset[1] + int(ry.min()), offset[0] + int(rx.max()) + 1, offset[1] + int(ry.max()) + 1)
                boxes, tdiag = stained_boxes(rgb, reg, offset, stage, diag["split"] == "garabato", scale, p)
                method = "ajustada" if boxes else "trazo"
                if not boxes:
                    boxes = [stroke]
                for b_i, bx in enumerate(boxes):
                    out.append({
                        "image_id": row["image_id"], "folder": row["folder"], "stage": stage, "color": hexcol,
                        "comp": f"{hexcol}:{k}", "region": j, "n_regions": len(regions), "box_i": b_i, "n_boxes": len(boxes),
                        "stroke_x0": stroke[0], "stroke_y0": stroke[1], "stroke_x1": stroke[2], "stroke_y1": stroke[3],
                        "stroke_area": int(reg.sum()), "comp_area": int(comp.sum()), **diag,
                        "box_method": method, "tight_reason": tdiag.get("reason"), "n_clusters": tdiag.get("n_clusters"),
                        "n_groups": tdiag.get("n_groups"),
                        "x0": bx[0], "y0": bx[1], "x1": bx[2], "y1": bx[3],
                    })
    return out


def clean_all(raw_dir: Path, images: pd.DataFrame, out_path: Path | None = None, workers: int | None = None,
              params: dict | None = None, force: bool = False) -> pd.DataFrame:
    p = {**PARAMS, **(params or {})}
    params_path = Path(str(out_path) + ".params.json") if out_path is not None else None
    if out_path is not None and Path(out_path).exists() and not force and params_path.exists():
        if json.loads(params_path.read_text()) == json.loads(json.dumps(p)):
            return pd.read_parquet(out_path)
        print("Parámetros distintos a los de la caché: se recalcula.")
    cols = ["image_id", "folder", "zip", "image_file", "mask_file", "field_x0", "field_y0", "field_x1", "field_y1"]
    tasks = [(str(raw_dir), r, p) for r in images[cols].to_dict("records")]
    rows = []
    with Pool(workers or max(1, (os.cpu_count() or 2) - 2)) as pool:
        for i, recs in enumerate(pool.imap_unordered(clean_image, tasks, chunksize=2)):
            rows.extend(recs)
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(tasks)} imágenes")
    df = pd.DataFrame(rows)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        params_path.write_text(json.dumps(p, indent=2))
    return df
