"""Inferencia de referencia: lo que la app Android tiene que reproducir, paso a paso y sin Ultralytics.

    python -m aiscope.infer --model models/exported/<run>_int8.tflite --images foto1.jpg foto2.jpg
    python -m aiscope.infer --model models/exported/<run>_int8.tflite --golden 12

Pasos: detectar el campo del ocular → recorte cuadrado → redimensionar al lado de entrada → normalizar a 0-1
(NCHW float32) → interpretar el .tflite → decodificar la salida densa (4 coordenadas + una puntuación por clase,
en 0-1) → NMS por clase → filtrar por umbral → contar parásitos (los artefactos no cuentan).

Con `--golden` genera en models/exported/golden/ las imágenes de entrada y su salida esperada, para comprobar
que la app da lo mismo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from aiscope.data.classes import CLASSES, NOT_PARASITE
from aiscope.data.yolo import tile_offsets
from aiscope.field import crop_field, to_tensor
from aiscope.paths import MODELS_DIR, PROCESSED_DIR

NMS_IOU = 0.7      # igual que el postproceso de Ultralytics con el que se midieron las métricas
MERGE_IOU = 0.5    # al unir mosaicos
MAX_DET = 300
TILE_SIZE = 640
TILE_CANVAS = 1200


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """NMS voraz sobre cajas xyxy; devuelve los índices que se quedan."""
    order = scores.argsort()[::-1]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        x0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def nms_por_clase(det: np.ndarray, iou_thr: float) -> np.ndarray:
    """det: (N, 6) [x0, y0, x1, y1, conf, cls]."""
    if not len(det):
        return det
    keep = []
    for c in np.unique(det[:, 5]):
        idx = np.flatnonzero(det[:, 5] == c)
        keep += [idx[k] for k in nms(det[idx, :4], det[idx, 4], iou_thr)]
    det = det[sorted(keep, key=lambda k: -det[k, 4])]
    return det[:MAX_DET]


def decodificar(salida: np.ndarray, lado: int, conf_min: float) -> np.ndarray:
    """(1, 4 + nc, A) con xywh y puntuaciones en 0-1 → (N, 6) [x0, y0, x1, y1, conf, cls] en píxeles."""
    a = salida[0]
    xywh, scores = a[:4], a[4:]
    cls = scores.argmax(0)
    conf = scores.max(0)
    sel = conf >= conf_min
    if not sel.any():
        return np.zeros((0, 6))
    cx, cy, w, h = (xywh[:, sel] * lado)
    return np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf[sel], cls[sel]])


class Detector:
    """Modelo LiteRT int8 con su preproceso y postproceso."""

    def __init__(self, model: Path, contrato: dict | None = None, threads: int = 4):
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=str(model), num_threads=threads)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        self.lado = int(self.inp["shape"][2])
        c = contrato or {}
        self.modo = c.get("entrada", {}).get("modo", "campo")
        self.umbral = c.get("umbral_conteo") or 0.25

    def _invoke(self, img: Image.Image) -> np.ndarray:
        self.interpreter.set_tensor(self.inp["index"], to_tensor(img))
        self.interpreter.invoke()
        return decodificar(self.interpreter.get_tensor(self.out["index"]), self.lado, min(self.umbral, 0.05))

    def detectar(self, foto: Image.Image) -> dict:
        if self.modo == "mosaicos":
            campo, (x0, y0, lado) = crop_field(foto, TILE_CANVAS)
            offs = [(ox, oy) for oy in tile_offsets(TILE_CANVAS, TILE_SIZE) for ox in tile_offsets(TILE_CANVAS, TILE_SIZE)]
            partes = []
            for ox, oy in offs:
                d = nms_por_clase(self._invoke(campo.crop((ox, oy, ox + TILE_SIZE, oy + TILE_SIZE))), NMS_IOU)
                d[:, [0, 2]] += ox
                d[:, [1, 3]] += oy
                partes.append(d)
            det = nms_por_clase(np.concatenate(partes) if partes else np.zeros((0, 6)), MERGE_IOU)
            escala_campo = TILE_CANVAS
        else:
            campo, (x0, y0, lado) = crop_field(foto, self.lado)
            det = nms_por_clase(self._invoke(campo), NMS_IOU)
            escala_campo = self.lado

        det = det[det[:, 4] >= self.umbral]
        conteo = {c: int((det[:, 5] == i).sum()) for i, c in enumerate(CLASSES)}
        return {
            "campo": {"x0": x0, "y0": y0, "lado": lado, "lado_entrada": escala_campo},
            "umbral": self.umbral,
            "detecciones": [{"clase": CLASSES[int(d[5])], "conf": round(float(d[4]), 4),
                             "caja": [round(float(v), 1) for v in d[:4]]} for d in det],
            "conteo_por_clase": conteo,
            "parasitos": int(sum(v for k, v in conteo.items() if k not in NOT_PARASITE)),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--contract")
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--golden", type=int, help="genera N casos de prueba a partir del split de test")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()

    model = Path(a.model)
    contrato_path = Path(a.contract) if a.contract else model.with_name(f"{model.stem}_contrato.json")
    contrato = json.loads(contrato_path.read_text()) if contrato_path.exists() else {}
    det = Detector(model, contrato, a.threads)

    if a.golden:
        import pandas as pd

        from aiscope.paths import INTERIM_DIR, RAW_DIR
        from aiscope.data.viz import read_member

        splits = pd.read_parquet(PROCESSED_DIR / "splits.parquet")
        imgs = pd.read_parquet(INTERIM_DIR / "images_clean.parquet").merge(splits[["image_id", "split"]], on="image_id")
        test = imgs[imgs["split"] == "test"]
        sel = (test.sample(frac=1, random_state=0).groupby(["preparacion", "especie"]).head(2).head(a.golden))
        out_dir = MODELS_DIR / "exported" / "golden"
        out_dir.mkdir(parents=True, exist_ok=True)
        casos = []
        for _, r in sel.iterrows():
            import io

            # Se guardan los bytes originales: re-comprimir cambia los píxeles y la app no podría reproducir la salida
            original = read_member(RAW_DIR, r["zip"], f"{r['folder']}/{r['image_file']}")
            nombre = r["image_id"].replace("/", "_")
            (out_dir / f"{nombre}.jpg").write_bytes(original)
            res = det.detectar(Image.open(out_dir / f"{nombre}.jpg"))
            casos.append({"imagen": f"{nombre}.jpg", "image_id": r["image_id"], "especie": r["especie"],
                          "preparacion": r["preparacion"], "parasitos_anotados": int(r["n_parasitos"]), **res})
            print(f"{nombre}: {res['parasitos']} parásitos (anotados {int(r['n_parasitos'])})")
        (out_dir / "esperado.json").write_text(json.dumps({"modelo": model.name, "casos": casos}, indent=2, ensure_ascii=False))
        print(out_dir / "esperado.json")
        return

    salida = {}
    for p in a.images:
        salida[p] = det.detectar(Image.open(p))
        print(f"{p}: {salida[p]['parasitos']} parásitos · {salida[p]['conteo_por_clase']}")
    if a.out:
        Path(a.out).write_text(json.dumps(salida, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
