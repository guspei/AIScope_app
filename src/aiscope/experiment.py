"""Experimento local A/B/C: entrena cada variante en serie (un proceso por variante) y la evalúa en val.

    python -m aiscope.experiment --variants A B C --epochs 15 --fraction 0.5 --device mps

Escribe models/runs/<tag>_resumen.md y .json, actualizados al terminar cada variante.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from aiscope.evaluate import MODES
from aiscope.paths import MODELS_DIR
from aiscope.train import VARIANTS

RUNS_DIR = MODELS_DIR / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C"], choices=sorted(VARIANTS))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default=time.strftime("exp%Y%m%d-%H%M"))
    a = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in a.variants:
        name = f"{a.tag}_{v}"
        print(f"== {v} · inicio {time.strftime('%H:%M')}", flush=True)
        t0 = time.time()
        train = subprocess.run([sys.executable, "-m", "aiscope.train", "--variant", v, "--epochs", str(a.epochs),
                                "--fraction", str(a.fraction), "--device", a.device, "--workers", str(a.workers), "--name", name])
        row = {"variante": v, "modelo": VARIANTS[v]["model"], "entrada": f"{MODES[v][0]} {MODES[v][1]}",
               "min_entreno": round((time.time() - t0) / 60, 1)}
        weights = RUNS_DIR / name / "weights" / "best.pt"
        if train.returncode or not weights.exists():
            row["error"] = f"entrenamiento terminó con código {train.returncode}"
            print(f"== {v} · ERROR {row['error']}", flush=True)
        else:
            ev = subprocess.run([sys.executable, "-m", "aiscope.evaluate", "--weights", str(weights), "--variant", v,
                                 "--split", "val", "--device", a.device])
            rep_path = RUNS_DIR / name / f"eval_val_{MODES[v][0]}{MODES[v][1]}.json"
            if ev.returncode or not rep_path.exists():
                row["error"] = f"evaluación terminó con código {ev.returncode}"
            else:
                rep = json.loads(rep_path.read_text())
                c = rep["conteo"]
                row.update({"mAP50": round(rep["deteccion"]["mAP50"], 3), "mAP50-95": round(rep["deteccion"]["mAP50-95"], 3),
                            **{f"AP50 {r['clase']}": r["mAP50"] for r in rep["deteccion"]["por_clase"]},
                            "MAE conteo": round(c["mae"], 2), "sesgo": round(c["bias"], 2), "umbral": c["umbral"],
                            "MAE fina": round(c["por_preparacion"].get("fina", {}).get("mae", float("nan")), 2),
                            "MAE gruesa": round(c["por_preparacion"].get("gruesa", {}).get("mae", float("nan")), 2),
                            "ms/imagen (MPS)": rep["ms_por_imagen"]})
            print(f"== {v} · fin {time.strftime('%H:%M')} · {json.dumps(row, ensure_ascii=False)}", flush=True)
        rows.append(row)
        _write_summary(a, rows)
    print(f"== FIN {time.strftime('%H:%M')}", flush=True)


def _write_summary(a, rows):
    import pandas as pd

    df = pd.DataFrame(rows).set_index("variante")
    head = f"# Experimento {a.tag}\n\n{a.epochs} épocas · {int(a.fraction * 100)} % de train · {a.device} · evaluación en val\n\n"
    (RUNS_DIR / f"{a.tag}_resumen.md").write_text(head + df.T.to_markdown() + "\n")
    (RUNS_DIR / f"{a.tag}_resumen.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
