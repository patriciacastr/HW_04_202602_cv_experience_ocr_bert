"""Baseline 1: TF-IDF + regresión Ridge.

- Palabras en n-gramas (1-2 por defecto), TF-IDF con tf sublineal.
- Números y años: por defecto SE CONSERVAN como tokens ("2014", "2019"), porque son la
  evidencia de la experiencia. Con --keep-digits 0 se eliminan (experimento de la
  Pregunta 3: ¿el modelo usa los años o solo el vocabulario?).
- Un solo carácter se descarta (token_pattern por defecto), así que "5" suelto no cuenta.
- El vectorizador se ajusta SOLO con train. La fuerza de regularización (alpha) se elige
  con el MAE de validación. El test no se toca.
- Predice también con el texto de referencia del test (Pregunta 2).

Uso:
    python src/baseline_tfidf.py
    python src/baseline_tfidf.py --keep-digits 0                 # -> tfidf_nodigits
    python src/baseline_tfidf.py --text-col text_clean_lowconf --name tfidf_lowconf
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (CAP, MODELS_DIR, SEED, clip_years, ensure_dirs, load_reference_clean,  # noqa: E402
                   load_split, mae, save_predictions, set_seed)

DIGITS = re.compile(r"\d+")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-col", default="text_clean",
                    choices=["text_raw", "text_clean", "text_clean_lowconf"])
    ap.add_argument("--keep-digits", type=int, default=1)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.3, 1, 3, 10, 30])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or ("tfidf" if args.keep_digits else "tfidf_nodigits")
    ensure_dirs()
    set_seed(SEED)

    def prep(texts):
        return list(texts) if args.keep_digits else [DIGITS.sub(" ", t) for t in texts]

    train, val, test = (load_split(n, args.text_col) for n in ("train", "val", "test"))
    ref_test = load_reference_clean(test["id"])

    t0 = time.perf_counter()
    vec = TfidfVectorizer(ngram_range=(1, args.ngram_max), min_df=args.min_df,
                          sublinear_tf=True, lowercase=True)
    X_tr = vec.fit_transform(prep(train["text"]))
    X_val = vec.transform(prep(val["text"]))
    fit_vec_time = time.perf_counter() - t0

    # Ajuste de alpha SOLO con validación
    t0 = time.perf_counter()
    best = (None, np.inf)
    for a in args.alphas:
        m = Ridge(alpha=a, random_state=SEED).fit(X_tr, train["exp_year"])
        v = mae(val["exp_year"], clip_years(m.predict(X_val), CAP))
        print(f"alpha={a:<6} MAE val = {v:.3f}")
        if v < best[1]:
            best = (a, v)
    tune_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = Ridge(alpha=best[0], random_state=SEED).fit(X_tr, train["exp_year"])
    fit_time = time.perf_counter() - t0

    p_val = clip_years(model.predict(X_val), CAP)
    t0 = time.perf_counter()
    p_test = clip_years(model.predict(vec.transform(prep(test["text"]))), CAP)
    infer_time = (time.perf_counter() - t0) / len(test)
    p_ref = clip_years(model.predict(vec.transform(prep(ref_test))), CAP)

    info = {
        "model": name, "text_col": args.text_col, "keep_digits": bool(args.keep_digits),
        "ngram_range": [1, args.ngram_max], "min_df": args.min_df, "best_alpha": best[0],
        "n_features": int(X_tr.shape[1]), "val_mae": best[1],
        "test_mae": mae(test["exp_year"], p_test), "test_mae_reference": mae(test["exp_year"], p_ref),
        "train_time_s": fit_vec_time + fit_time, "tuning_time_s": tune_time,
        "inference_time_per_cv_s": infer_time,
    }
    save_predictions(name, {"val": (val, p_val, None), "test": (test, p_test, p_ref)}, info)
    joblib.dump({"vectorizer": vec, "model": model, "keep_digits": bool(args.keep_digits)},
                MODELS_DIR / "baselines" / f"{name}.joblib")
    print(f"[{name}] alpha={best[0]}  MAE val={best[1]:.3f}  MAE test={info['test_mae']:.3f}  "
          f"(con texto de referencia: {info['test_mae_reference']:.3f})")


if __name__ == "__main__":
    main()
