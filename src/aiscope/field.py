"""Detección del campo del ocular y preprocesado de la entrada.

Es la parte que la app Android tiene que reproducir tal cual: de la foto completa del móvil sale un recorte
cuadrado centrado en el círculo del ocular, redimensionado al lado de entrada del modelo y normalizado a 0-1.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

DARK_THRESHOLD = 40  # fuera del círculo del ocular la foto queda casi negra
ANALYSIS_MAX_SIDE = 512  # la detección del campo se hace sobre una miniatura


def detect_field(img: Image.Image) -> tuple[int, int, int]:
    """(cx, cy, lado) del cuadrado que contiene el campo circular, en píxeles de la imagen original.

    Toma la mayor región clara de una miniatura, rellena sus huecos y devuelve el cuadrado que la envuelve.
    Si no encuentra región clara, devuelve el cuadrado centrado más grande que cabe en la imagen.
    """
    w, h = img.size
    thumb = img.convert("L")
    thumb.thumbnail((ANALYSIS_MAX_SIDE, ANALYSIS_MAX_SIDE))
    gray = np.asarray(thumb, dtype=np.float32)
    scale = w / gray.shape[1]

    bright = gray > DARK_THRESHOLD
    lab, n = ndimage.label(bright)
    if not n:
        return w // 2, h // 2, min(w, h)
    largest = int(np.argmax(ndimage.sum(bright, lab, range(1, n + 1)))) + 1
    field = ndimage.binary_fill_holes(lab == largest)
    sl = ndimage.find_objects(field.astype(np.int32))[0]
    y0, y1 = sl[0].start * scale, sl[0].stop * scale
    x0, x1 = sl[1].start * scale, sl[1].stop * scale
    return int((x0 + x1) / 2), int((y0 + y1) / 2), int(max(x1 - x0, y1 - y0))


def crop_field(img: Image.Image, size: int) -> tuple[Image.Image, tuple[int, int, int]]:
    """Recorte cuadrado del campo llevado a `size` px. Devuelve (recorte, (x0, y0, lado original))."""
    cx, cy, side = detect_field(img)
    x0, y0 = cx - side // 2, cy - side // 2
    crop = img.convert("RGB").crop((x0, y0, x0 + side, y0 + side)).resize((size, size), Image.BILINEAR)
    return crop, (x0, y0, side)


def to_tensor(img: Image.Image) -> np.ndarray:
    """PIL RGB → NCHW float32 en 0-1, tal como espera el .tflite exportado."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])
