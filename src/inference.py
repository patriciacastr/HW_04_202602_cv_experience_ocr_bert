"""Inferencia para la app: OCR de una imagen de CV + predicción con los 4 modelos.

Todo se carga de forma perezosa (lazy) y cada modelo es opcional: si faltan sus pesos
(no se suben al repo) o su librería, ese modelo se marca como "no disponible" y la app
sigue funcionando con los demás.
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CAP, MET_DIR, MODEL_ORDER, MODELS_DIR, clip_years, load_json  # noqa: E402

CONF_THRESHOLD = 0.6  # mismo valor por defecto que preprocessing.py


# --------------------------------------------------------------------------- #
# Entrada: imagen o PDF -> imagen PIL
# --------------------------------------------------------------------------- #
def to_pil(file_bytes: bytes, filename: str):
    """Convierte lo que sube la persona (png/jpg o PDF) en una imagen PIL RGB.
    De un PDF se usa la primera página."""
    from PIL import Image

    if filename.lower().endswith(".pdf"):
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(file_bytes)
        return pdf[0].render(scale=2.0).to_pil().convert("RGB")
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
def load_ocr():
    """Crea el motor PaddleOCR (mismo modelo que el pipeline). Lanza ImportError si falta."""
    from ocr_extraction import build_ocr

    return build_ocr("mobile")


def run_ocr(ocr, pil_image) -> dict:
    """OCR de una imagen PIL con el mismo código del pipeline (orden de lectura incluido)."""
    from ocr_extraction import run_one

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "uploaded.png"
        pil_image.save(p)
        return run_one(ocr, p)


def text_variants(ocr_result: dict) -> dict:
    """Aplica la misma limpieza, borrado de datos personales y enmascarado que el pipeline."""
    from preprocessing import build_variants

    return build_variants(ocr_result, None, CONF_THRESHOLD)


def text_from_pasted(text: str) -> dict:
    """Variantes de texto a partir de texto pegado a mano (sin OCR)."""
    lines = [{"text": t, "score": 1.0, "box": [0, i * 10, 10, i * 10 + 8]}
             for i, t in enumerate(str(text).splitlines()) if t.strip()]
    d = {"lines_ordered": lines, "text_ordered": "\n".join(l["text"] for l in lines)}
    return text_variants(d)


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
def _info(name: str) -> dict:
    p = MET_DIR / f"info_{name}.json"
    return load_json(p) if p.exists() else {}


def _load_median():
    p = MODELS_DIR / "baselines" / "median.json"
    if not p.exists():
        raise FileNotFoundError("falta models/baselines/median.json")
    value = float(load_json(p)["median_years"])
    return lambda texts: np.full(len(texts), value)


def _load_tfidf():
    import joblib

    p = MODELS_DIR / "baselines" / "tfidf.joblib"
    if not p.exists():
        raise FileNotFoundError("falta models/baselines/tfidf.joblib")
    bundle = joblib.load(p)
    import re

    digits = re.compile(r"\d+")

    def predict(texts):
        t = list(texts) if bundle.get("keep_digits", True) else [digits.sub(" ", x) for x in texts]
        return clip_years(bundle["model"].predict(bundle["vectorizer"].transform(t)), CAP)

    return predict


def _load_embeddings():
    import joblib

    p = MODELS_DIR / "baselines" / "embeddings.joblib"
    if not p.exists():
        raise FileNotFoundError("falta models/baselines/embeddings.joblib")
    bundle = joblib.load(p)
    from baseline_embeddings import embed_texts
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(bundle["encoder"])

    def predict(texts):
        X = embed_texts(enc, list(texts), bundle["chunk_tokens"], 16)
        return clip_years(bundle["ridge"].predict(X), CAP)

    return predict


def _load_bert():
    d = MODELS_DIR / "bert" / "bert"
    if not (d / "config.json").exists():
        raise FileNotFoundError("falta models/bert/bert (pesos del BERT ajustado)")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from train_bert import build_items, predict as bert_predict

    info = _info("bert")
    args = SimpleNamespace(max_len=info.get("max_len", 512), strategy=info.get("strategy", "headtail"),
                           head_tokens=info.get("head_tokens", 128))
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSequenceClassification.from_pretrained(d)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    def predict(texts):
        items, owner = build_items(tok, list(texts), args)
        return bert_predict(torch, model, items, owner, len(texts), tok.pad_token_id, 8, device,
                            device == "cuda")

    return predict


_LOADERS = {"median": _load_median, "tfidf": _load_tfidf,
            "embeddings": _load_embeddings, "bert": _load_bert}


def load_predictors() -> tuple[dict, dict]:
    """Devuelve ({modelo: función(textos)->años}, {modelo: motivo de no disponible})."""
    ok, missing = {}, {}
    for name in MODEL_ORDER:
        try:
            ok[name] = _LOADERS[name]()
        except Exception as e:  # noqa: BLE001  (modelo opcional)
            missing[name] = f"{type(e).__name__}: {e}"
    return ok, missing


def predict_all(predictors: dict, variants: dict) -> dict:
    """Predice con cada modelo usando la variante de texto con la que fue entrenado."""
    out = {}
    for name, fn in predictors.items():
        col = _info(name).get("text_col", "text_clean")
        text = variants.get(col, variants["text_clean"])
        out[name] = float(np.asarray(fn([text]))[0])
    return out
