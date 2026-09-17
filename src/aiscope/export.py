"""Exportación a LiteRT int8 y comparación de métricas antes y después de cuantizar.

    python -m aiscope.export --weights models/runs/<run>/weights/best.pt --variant C

Calibra con imágenes de train (nunca de val ni de test), copia el .tflite a models/exported/ junto a un
contrato.json con lo que necesita la app, y vuelve a evaluar en val con el mismo umbral de conteo que el
modelo PyTorch, para que la comparación sea directa.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import aiscope  # noqa: F401
from aiscope.data.classes import NOT_PARASITE
from aiscope.evaluate import MODES
from aiscope.paths import MODELS_DIR

EXPORT_DIR = MODELS_DIR / "exported"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--variant", required=True, choices=sorted(MODES))
    ap.add_argument("--data", help="data.yaml de calibración (por defecto, el de la variante)")
    ap.add_argument("--fraction", type=float, help="fracción de train para calibrar (por defecto ~400 imágenes)")
    ap.add_argument("--quantize", default="8", choices=["8", "w8a16", "w8a32", "float32"],
                    help="8: int8 estático; w8a16: pesos int8 y activaciones int16; w8a32: solo pesos (sin calibrar); "
                         "float32: sin cuantizar (en Android el delegado de GPU lo ejecuta a media precisión)")
    ap.add_argument("--device", default="mps", help="dispositivo para la evaluación del modelo PyTorch")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO
    from aiscope.train import VARIANTS

    mode, imgsz = MODES[a.variant]
    data = Path(a.data) if a.data else Path(VARIANTS[a.variant]["data"])
    fraction = a.fraction if a.fraction is not None else (0.02 if mode == "mosaicos" else 0.1)
    weights = Path(a.weights)
    run_dir = weights.parent.parent

    quantize = {"8": 8, "float32": None}.get(a.quantize, a.quantize)
    sufijo = {"8": "int8", "w8a16": "w8a16", "w8a32": "w8a32", "float32": "float32"}[a.quantize]
    model = YOLO(str(weights))
    names = [model.names[i] for i in sorted(model.names)]
    f = model.export(format="litert", quantize=quantize, data=str(data), split="train", fraction=fraction, imgsz=imgsz)
    tfl = Path(f)
    if tfl.is_dir():
        tfl = next(tfl.rglob("*.tflite" if quantize is None else f"*_{sufijo}.tflite"))
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    dest = EXPORT_DIR / f"{run_dir.name}_{sufijo}.tflite"
    shutil.copy2(tfl, dest)
    print(f"exportado {dest} ({dest.stat().st_size / 2**20:.2f} MB)")

    pt_report = run_dir / f"eval_val_{mode}{imgsz}.json"
    conf = json.loads(pt_report.read_text())["conteo"]["umbral"] if pt_report.exists() else None

    contrato = {
        "modelo": str(dest.name), "variante": a.variant, "arquitectura": VARIANTS[a.variant]["model"],
        "entrada": {"modo": mode, "lado_px": imgsz, "orden": "NCHW", "tipo": "float32", "rango": "0-1",
                    "preproceso": ("recorte cuadrado del campo del ocular; se lleva a 1200 px y se parte en 4 mosaicos de 640 con 80 px de solape"
                                   if mode == "mosaicos" else "recorte cuadrado del campo del ocular redimensionado al lado de entrada")},
        "salida": {"tensor": "(1, 4 + nº de clases, anclajes): xywh normalizado 0-1 y una puntuación por clase",
                   "clases": dict(enumerate(names)), "no_cuentan": [n for n in names if n in NOT_PARASITE],
                   "postproceso": "NMS por clase con IoU 0.7 (la cabeza exportada es la de una a muchas)" +
                                  (" y, al unir los mosaicos, NMS por clase con IoU 0.5" if mode == "mosaicos" else "")},
        "umbral_conteo": conf,
        "cuantizacion": {"esquema": a.quantize, "calibracion": f"{data.parent.name}, split train, fracción {fraction}"},
    }
    (EXPORT_DIR / f"{run_dir.name}_{sufijo}_contrato.json").write_text(json.dumps(contrato, indent=2, ensure_ascii=False))

    if a.skip_eval:
        return
    cmd = [sys.executable, "-m", "aiscope.evaluate", "--weights", str(dest), "--variant", a.variant, "--split", "val"]
    if conf is not None:
        cmd += ["--conf", str(conf)]
    subprocess.run(cmd, check=False)

    int8_report = EXPORT_DIR / f"eval_val_{mode}{imgsz}.json"
    if pt_report.exists() and int8_report.exists():
        pt, q8 = (json.loads(p.read_text()) for p in (pt_report, int8_report))
        shutil.move(int8_report, EXPORT_DIR / f"{run_dir.name}_{sufijo}_eval_val.json")
        filas = [("mAP50", pt["deteccion"]["mAP50"], q8["deteccion"]["mAP50"]),
                 ("mAP50-95", pt["deteccion"]["mAP50-95"], q8["deteccion"]["mAP50-95"]),
                 ("MAE conteo", pt["conteo"]["mae"], q8["conteo"]["mae"]),
                 ("ms/imagen", pt["ms_por_imagen"], q8["ms_por_imagen"])]
        print(f"\n{a.variant} · {a.quantize} · antes y después de cuantizar (val, umbral {conf})")
        for k, v_pt, v_q8 in filas:
            print(f"  {k:<12} PyTorch {v_pt:>8.3f}   {sufijo:<7} {v_q8:>8.3f}   {v_q8 - v_pt:+.3f}")


if __name__ == "__main__":
    main()
