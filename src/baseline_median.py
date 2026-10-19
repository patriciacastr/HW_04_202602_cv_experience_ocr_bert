"""Baseline 0: predictor de la mediana.

Siempre predice la mediana de los años de experiencia del conjunto de entrenamiento.
Es el piso: cualquier otro modelo debe superarlo; si no, no aprendió nada.

Uso:
    python src/baseline_median.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import MODELS_DIR, SEED, ensure_dirs, load_split, mae, save_json, save_predictions, set_seed  # noqa: E402


def main() -> None:
    ensure_dirs()
    set_seed(SEED)
    train, val, test = (load_split(n) for n in ("train", "val", "test"))

    t0 = time.perf_counter()
    median = float(np.median(train["exp_year"]))
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    p_val, p_test = np.full(len(val), median), np.full(len(test), median)
    infer_time = (time.perf_counter() - t0) / max(len(test), 1)

    info = {
        "model": "median", "median_years": median,
        "val_mae": mae(val["exp_year"], p_val), "test_mae": mae(test["exp_year"], p_test),
        "train_time_s": train_time, "inference_time_per_cv_s": infer_time,
    }
    # No usa texto: la predicción con texto de referencia es idéntica.
    save_predictions("median", {"val": (val, p_val, p_val), "test": (test, p_test, p_test)}, info)
    save_json({"median_years": median}, MODELS_DIR / "baselines" / "median.json")
    print(f"Mediana de train = {median}  |  MAE val = {info['val_mae']:.3f}  |  MAE test = {info['test_mae']:.3f}")


if __name__ == "__main__":
    main()
