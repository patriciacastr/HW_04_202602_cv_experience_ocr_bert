"""Referencia basada en REGLAS (no es uno de los 4 modelos): suma las duraciones de los cargos.

Lee los rangos "año.mes-año.mes" del historial laboral y suma los intervalos sin contar dos veces
los solapes. No aprende nada. Sirve para responder: ¿cuánto de la experiencia está escrito
literalmente en las fechas?, y ¿cuánto se acerca BERT a "hacer la cuenta"? Si no encuentra
ningún rango, predice la mediana de train.

Uso:
    python src/baseline_rules.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dates import years_from_text  # noqa: E402
from utils import CAP, SEED, clip_years, ensure_dirs, load_reference_clean, load_split, mae, save_predictions, set_seed  # noqa: E402


def predict(texts, fallback: float) -> np.ndarray:
    out = []
    for t in texts:
        y = years_from_text(t)
        out.append(fallback if y is None else y)
    return clip_years(np.asarray(out, dtype=float), CAP)


def main() -> None:
    ensure_dirs()
    set_seed(SEED)
    train, val, test = (load_split(n) for n in ("train", "val", "test"))
    fallback = float(np.median(train["exp_year"]))

    p_val = predict(val["text"], fallback)
    t0 = time.perf_counter()
    p_test = predict(test["text"], fallback)
    infer = (time.perf_counter() - t0) / len(test)
    p_ref = predict(load_reference_clean(test["id"]), fallback)

    found = float(np.mean([years_from_text(t) is not None for t in test["text"]]))
    info = {"model": "rules", "description": "suma de duraciones de cargos (sin aprendizaje)",
            "fallback_median": fallback, "share_cvs_with_dates": found,
            "val_mae": mae(val["exp_year"], p_val), "test_mae": mae(test["exp_year"], p_test),
            "test_mae_reference": mae(test["exp_year"], p_ref),
            "train_time_s": 0.0, "inference_time_per_cv_s": infer}
    save_predictions("rules", {"val": (val, p_val, None), "test": (test, p_test, p_ref)}, info)
    print(f"[rules] CVs con fechas leídas: {100 * found:.1f}% | MAE val={info['val_mae']:.3f} "
          f"MAE test={info['test_mae']:.3f} (con texto de referencia: {info['test_mae_reference']:.3f})")


if __name__ == "__main__":
    main()
