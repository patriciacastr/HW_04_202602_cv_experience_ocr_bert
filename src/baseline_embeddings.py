"""Baseline 2: embeddings de oraciones congelados + regresión Ridge.

- Codificador preentrenado y CONGELADO (no se ajusta): por defecto all-MiniLM-L6-v2.
- CVs largos: un encoder solo lee max_seq_length tokens (256 en MiniLM), y un CV puede
  tener más. Se parte el texto en trozos de ese tamaño (usando el tokenizador del
  propio modelo), se codifica cada trozo y se PROMEDIAN los vectores. Así no se pierde
  el final del CV, donde suelen estar los trabajos más antiguos y la educación.
- Ridge se ajusta con train; alpha se elige con el MAE de validación.
- Predice también con el texto de referencia del test (Pregunta 2).

Uso:
    python src/baseline_embeddings.py
    python src/baseline_embeddings.py --encoder BAAI/bge-small-en-v1.5 --name embeddings_bge
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (CAP, MODELS_DIR, SEED, clip_years, ensure_dirs, load_reference_clean,  # noqa: E402
                   load_split, mae, save_predictions, set_seed)


def chunk_text(tokenizer, text: str, size: int) -> list[str]:
    """Parte el texto en trozos de `size` tokens (decodificados de vuelta a texto)."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return [""]
    return [tokenizer.decode(ids[i:i + size]) for i in range(0, len(ids), size)]


def embed_texts(model, texts: list[str], size: int, batch_size: int = 32) -> np.ndarray:
    """Un vector por CV = promedio de los vectores (normalizados) de sus trozos."""
    chunks, owner = [], []
    for k, t in enumerate(texts):
        cs = chunk_text(model.tokenizer, t, size)
        chunks.extend(cs)
        owner.extend([k] * len(cs))
    E = np.asarray(model.encode(chunks, batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=False))
    out = np.zeros((len(texts), E.shape[1]))
    np.add.at(out, np.asarray(owner), E)
    counts = np.bincount(owner, minlength=len(texts)).reshape(-1, 1)
    return out / np.maximum(counts, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--text-col", default="text_clean",
                    choices=["text_raw", "text_clean", "text_clean_lowconf"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1, 10, 100])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--name", default="embeddings")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    ensure_dirs()
    set_seed(SEED)
    enc = SentenceTransformer(args.encoder)
    size = int((enc.max_seq_length or 256) - 2)   # descuenta tokens especiales
    print(f"Encoder: {args.encoder} | trozos de {size} tokens")

    train, val, test = (load_split(n, args.text_col) for n in ("train", "val", "test"))

    t0 = time.perf_counter()
    X_tr = embed_texts(enc, list(train["text"]), size, args.batch_size)
    enc_train_time = time.perf_counter() - t0
    X_val = embed_texts(enc, list(val["text"]), size, args.batch_size)

    best = (None, np.inf)
    for a in args.alphas:
        m = Ridge(alpha=a, random_state=SEED).fit(X_tr, train["exp_year"])
        v = mae(val["exp_year"], clip_years(m.predict(X_val), CAP))
        print(f"alpha={a:<6} MAE val = {v:.3f}")
        if v < best[1]:
            best = (a, v)

    t0 = time.perf_counter()
    model = Ridge(alpha=best[0], random_state=SEED).fit(X_tr, train["exp_year"])
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_te = embed_texts(enc, list(test["text"]), size, args.batch_size)
    p_test = clip_years(model.predict(X_te), CAP)
    infer_time = (time.perf_counter() - t0) / len(test)
    p_val = clip_years(model.predict(X_val), CAP)
    X_ref = embed_texts(enc, load_reference_clean(test["id"]), size, args.batch_size)
    p_ref = clip_years(model.predict(X_ref), CAP)

    info = {
        "model": args.name, "encoder": args.encoder, "chunk_tokens": size,
        "text_col": args.text_col, "best_alpha": best[0], "val_mae": best[1],
        "test_mae": mae(test["exp_year"], p_test), "test_mae_reference": mae(test["exp_year"], p_ref),
        "train_time_s": enc_train_time + fit_time, "inference_time_per_cv_s": infer_time,
    }
    save_predictions(args.name, {"val": (val, p_val, None), "test": (test, p_test, p_ref)}, info)
    joblib.dump({"ridge": model, "encoder": args.encoder, "chunk_tokens": size},
                MODELS_DIR / "baselines" / f"{args.name}.joblib")
    print(f"[{args.name}] alpha={best[0]}  MAE val={best[1]:.3f}  MAE test={info['test_mae']:.3f}  "
          f"(con texto de referencia: {info['test_mae_reference']:.3f})")


if __name__ == "__main__":
    main()
