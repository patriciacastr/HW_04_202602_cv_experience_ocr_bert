"""Utilidades compartidas: semillas, rutas y helpers de lectura/escritura."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

SEED = 42

# Raíz del repo = carpeta que contiene src/
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMG_DIR = RAW_DIR / "images"
OCR_DIR = DATA_DIR / "ocr"
PROC_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "output"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
MET_DIR = OUT_DIR / "metrics"

# Ocupaciones (nombre del shard en HF -> etiqueta legible)
OCCUPATIONS = {
    "registered_nurses": "Registered nurses",
    "elementary_middle_school_teachers": "Elementary/middle school teachers",
    "software_developers": "Software developers",
    "accountants_auditors": "Accountants & auditors",
    "construction_laborers": "Construction laborers",
}
# OJO: verifica los nombres exactos de los shards en la página del dataset
# (data/images/<occupation>-NNNN.parquet) y ajusta las claves si difieren.

CAP = 20  # tope de años de experiencia (misma cota para TODOS los modelos)
BANDS = [(0, 2), (3, 5), (6, 10), (11, 20)]


def set_seed(seed: int = SEED) -> None:
    """Fija todas las semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs() -> None:
    for d in (IMG_DIR, OCR_DIR, PROC_DIR, SPLIT_DIR, FIG_DIR, TAB_DIR, MET_DIR,
              MODELS_DIR / "baselines", MODELS_DIR / "bert"):
        d.mkdir(parents=True, exist_ok=True)


def clip_years(y, cap: int = CAP):
    """Recorta años de experiencia al rango [0, cap]."""
    return np.clip(y, 0, cap)


def to_band(y) -> np.ndarray:
    """Convierte años a banda: 0 = 0-2, 1 = 3-5, 2 = 6-10, 3 = 11-20."""
    y = np.asarray(y, dtype=float)
    return np.digitize(y, [2.5, 5.5, 10.5])


BAND_LABELS = ["0-2", "3-5", "6-10", "11-20"]


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    """Permite serializar tipos numpy."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"No serializable: {type(o)}")
