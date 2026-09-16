"""Partición train/val/test por grupos, sin que un grupo caiga en dos splits.

Grupo = sesión de captura (centro, microscopista, día), unida con cualquier otra sesión que comparta una imagen
casi duplicada. Entre los repartos aleatorios de grupos se elige el que mejor iguala, en cada split, la fracción
objetivo de imágenes, cajas por clase, imágenes por especie e imágenes por preparación.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def near_duplicate_pairs(images: pd.DataFrame, max_hamming: int = 12) -> pd.DataFrame:
    """Pares de imágenes con dHash (256 bits) a distancia de Hamming ≤ max_hamming."""
    H = np.stack([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in images["dhash"]])
    B = np.unpackbits(H, axis=1).astype(np.float32)
    pc = B.sum(1)
    D = pc[:, None] + pc[None, :] - 2 * (B @ B.T)
    i, j = np.where(np.triu(D <= max_hamming, 1))
    ids = images["image_id"].to_numpy()
    return pd.DataFrame({"a": ids[i], "b": ids[j], "hamming": D[i, j].astype(int)})


def build_groups(images: pd.DataFrame, dup_pairs: pd.DataFrame, key: str = "sesion") -> pd.Series:
    """Grupo por imagen: componentes conexas de (misma sesión) ∪ (par casi duplicado). Devuelve Series image_id → grupo."""
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ses = images.set_index("image_id")[key]
    for s in ses.unique():
        find(f"s:{s}")
    for a, b in dup_pairs[["a", "b"]].itertuples(index=False):
        union(f"s:{ses[a]}", f"s:{ses[b]}")
    roots = {s: find(f"s:{s}") for s in ses.unique()}
    codes = {r: f"g{k:04d}" for k, r in enumerate(sorted(set(roots.values())))}
    return ses.map(lambda s: codes[roots[s]])


def group_features(images: pd.DataFrame, boxes: pd.DataFrame, groups: pd.Series, classes: list[str]) -> pd.DataFrame:
    img = images.assign(grupo=images["image_id"].map(groups))
    feats = {"imagenes": img.groupby("grupo").size()}
    bx = boxes.assign(grupo=boxes["image_id"].map(groups))
    for c in classes:
        feats[f"cajas_{c}"] = bx[bx["stage"] == c].groupby("grupo").size()
    for sp, g in img.groupby("especie"):
        feats[f"img_{sp}"] = g.groupby("grupo").size()
    for pr, g in img.groupby("preparacion"):
        feats[f"img_{pr}"] = g.groupby("grupo").size()
    return pd.DataFrame(feats).fillna(0).astype(int).sort_index()


def split_groups(features: pd.DataFrame, fractions: dict[str, float], trials: int = 3000, seed: int = 0) -> tuple[pd.Series, float]:
    """Reparto aleatorio-voraz de grupos; devuelve (grupo → split, error del mejor reparto).

    Error = media de |fracción obtenida - objetivo| relativa, sobre todas las columnas y splits, ponderando
    más las columnas con pocos ejemplos (clases raras), que son las que se descuadran.
    """
    rng = np.random.default_rng(seed)
    names = list(fractions)
    target = np.array([fractions[s] for s in names])
    X = features.to_numpy(dtype=float)
    tot = X.sum(0)
    tot[tot == 0] = 1
    w = 1 / np.sqrt(tot)
    w = w / w.sum()
    best, best_err = None, np.inf
    for _ in range(trials):
        # orden de asignación: grupos grandes primero con ruido (o aleatorio puro en 1 de cada 5 intentos)
        noise = rng.random(len(X)) * 0.7
        order = np.argsort(-(X[:, 0] / X[:, 0].max()) - noise) if rng.random() < 0.8 else rng.permutation(len(X))
        acc = np.zeros((len(names), X.shape[1]))
        assign = np.empty(len(X), dtype=int)
        for gi in order:
            # split más por debajo de su objetivo en imágenes, con desempate aleatorio
            deficit = target * acc.sum(0)[0] + target * X[gi, 0] - acc[:, 0]
            deficit = deficit + rng.random(len(names)) * 1e-6
            s = int(np.argmax(deficit))
            assign[gi] = s
            acc[s] += X[gi]
        frac = acc / tot
        err = float((np.abs(frac - target[:, None]) / target[:, None] * w).sum() / len(names))
        if err < best_err:
            best, best_err = assign.copy(), err
    return pd.Series([names[k] for k in best], index=features.index, name="split"), best_err
