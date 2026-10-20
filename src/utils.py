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


def load_json_many(paths, workers: int = 32) -> dict:
    """Lee muchos JSON en paralelo (útil en Google Drive, donde cada lectura tiene latencia).
    Devuelve {Path: contenido}; los archivos que no existen se omiten."""
    from concurrent.futures import ThreadPoolExecutor

    paths = [Path(p) for p in paths if Path(p).exists()]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(zip(paths, ex.map(load_json, paths)))


def _json_default(o):
    """Permite serializar tipos numpy."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"No serializable: {type(o)}")


# --------------------------------------------------------------------------- #
# Modelos: nombres, etiquetas y colores fijos (el color sigue a la entidad)
# Paleta categórica validada (slots 1-4): azul, naranja, aqua, amarillo.
# --------------------------------------------------------------------------- #
MODEL_ORDER = ["median", "tfidf", "embeddings", "bert"]
MODEL_LABELS = {
    "median": "Median",
    "tfidf": "TF-IDF + Ridge",
    "embeddings": "Embeddings + Ridge",
    "bert": "Fine-tuned BERT",
}
MODEL_COLORS = {
    "bert": "#2a78d6",
    "tfidf": "#eb6834",
    "embeddings": "#1baf7a",
    "median": "#eda100",
}


# --------------------------------------------------------------------------- #
# Carga de datos compartida por todos los modelos (mismo split para todos)
# --------------------------------------------------------------------------- #
def load_split(name: str, text_col: str = "text_clean"):
    """Devuelve el split ('train'|'val'|'test') con columnas id, occupation, exp_year,
    band y text. NO incluye edad, sexo, raza, nombres ni ubicación."""
    import pandas as pd

    sp = pd.read_csv(SPLIT_DIR / f"{name}.csv", dtype={"id": str})
    ds = pd.read_csv(PROC_DIR / "dataset.csv", dtype={"id": str},
                     usecols=["id", "text_raw", "text_clean", "text_clean_lowconf"])
    df = sp.merge(ds[["id", text_col]].rename(columns={text_col: "text"}), on="id", how="left")
    df["text"] = df["text"].fillna("")
    return df


def load_reference_clean(ids) -> list[str]:
    """Texto de REFERENCIA limpiado igual que el OCR. Solo para evaluar (Pregunta 2),
    nunca para entrenar."""
    import pandas as pd
    from preprocessing import clean_reference_text

    ref = pd.read_csv(RAW_DIR / "reference_text.csv", dtype={"id": str}).set_index("id")["text"]
    pii_path = RAW_DIR / "pii_for_scrubbing.csv"
    pii = pd.read_csv(pii_path, dtype={"id": str}).set_index("id") if pii_path.exists() else None
    out = []
    for i in ids:
        names = None
        if pii is not None and i in pii.index:
            names = [str(pii.loc[i, "first_name"]), str(pii.loc[i, "last_name"])]
        out.append(clean_reference_text(ref.get(i, ""), names))
    return out


def save_predictions(name: str, frames: dict, info: dict) -> None:
    """Guarda predicciones de un modelo y su info (tiempos, hiperparámetros).

    frames: {'val': (df, y_pred, y_pred_ref|None), 'test': (df, y_pred, y_pred_ref|None)}
    Archivo: output/metrics/predictions_<name>.csv con id, split, y_true, y_pred, y_pred_ref.
    """
    import pandas as pd

    parts = []
    for split, (df, y_pred, y_ref) in frames.items():
        parts.append(pd.DataFrame({
            "id": df["id"].values, "split": split, "y_true": df["exp_year"].values,
            "y_pred": clip_years(np.asarray(y_pred, dtype=float)),
            "y_pred_ref": (clip_years(np.asarray(y_ref, dtype=float))
                           if y_ref is not None else np.nan),
        }))
    MET_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(parts).to_csv(MET_DIR / f"predictions_{name}.csv", index=False)
    save_json(info, MET_DIR / f"info_{name}.json")
    print(f"Guardado predictions_{name}.csv e info_{name}.json")


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))
