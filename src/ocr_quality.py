"""Etapa 2a: calidad del OCR frente al texto de referencia.

Métricas por CV:
  - CER y WER (jiwer), tras normalizar ambos textos igual (minúsculas, unicode, guiones,
    espacios).
  - bow_f1: F1 de bolsa de palabras. NO depende del orden, así que no se castiga si el
    orden de lectura (p. ej. dos columnas) difiere del texto de referencia.
  - years_recall: % de años (19xx/20xx) del texto de referencia que aparecen en el OCR.
  - ranges_recall: % de rangos de fechas ("2014 - 2019", "2020 - Present") leídos intactos.

Salidas:
  output/tables/ocr_quality_per_cv.csv
  output/tables/ocr_quality_by_occupation.csv
  output/metrics/ocr_quality_summary.json
  output/figures/ocr_cer_wer_hist.png

Usa SOLO para medir el OCR. El texto de referencia nunca es entrada de ningún modelo.

Uso:
    python src/ocr_quality.py
"""
from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (FIG_DIR, MET_DIR, OCR_DIR, RAW_DIR, TAB_DIR, ensure_dirs, load_json_many,  # noqa: E402
                   save_json)

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
# Rangos de fechas: "2018.05-2020.05" (formato de PopResume) o "2018 - 2020" / "2018 - present".
# Se compara el rango COMPLETO (inicio y fin con mes), no solo el año.
RANGE = re.compile(r"\b((?:19|20)\d{2}(?:[./]\d{1,2})?)\s*-\s*"
                   r"((?:19|20)\d{2}(?:[./]\d{1,2})?|present|current)\b")
TOKEN = re.compile(r"\w+")


def norm(text: str) -> str:
    """Misma normalización para referencia y OCR."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[‐-―−]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def recall(ref_items: Counter, hyp_items: Counter) -> float:
    total = sum(ref_items.values())
    if total == 0:
        return float("nan")
    hit = sum(min(c, hyp_items.get(k, 0)) for k, c in ref_items.items())
    return hit / total


def bow_f1(ref: str, hyp: str) -> float:
    r, h = Counter(TOKEN.findall(ref)), Counter(TOKEN.findall(hyp))
    common = sum((r & h).values())
    if common == 0:
        return 0.0
    p, rc = common / max(sum(h.values()), 1), common / max(sum(r.values()), 1)
    return 2 * p * rc / (p + rc)


def evaluate_pair(ref_raw: str, hyp_raw: str) -> dict:
    import jiwer

    ref, hyp = norm(ref_raw), norm(hyp_raw)
    if not ref:
        return {}
    return {
        "cer": jiwer.cer(ref, hyp) if hyp else 1.0,
        "wer": jiwer.wer(ref, hyp) if hyp else 1.0,
        "bow_f1": bow_f1(ref, hyp),
        "years_recall": recall(Counter(YEAR.findall(ref)), Counter(YEAR.findall(hyp))),
        "ranges_recall": recall(Counter(RANGE.findall(ref)), Counter(RANGE.findall(hyp))),
    }


def token_diff(ref_raw: str, hyp_raw: str) -> tuple[Counter, Counter]:
    """Tokens (palabras/números) de la referencia que NO aparecen en el OCR, y al revés."""
    r, h = Counter(TOKEN.findall(norm(ref_raw))), Counter(TOKEN.findall(norm(hyp_raw)))
    return r - h, h - r


def main() -> None:
    ensure_dirs()
    ref = pd.read_csv(RAW_DIR / "reference_text.csv", dtype={"id": str}).set_index("id")["text"]
    meta = pd.read_csv(RAW_DIR / "metadata.csv", dtype={"id": str}).set_index("id")

    print(f"Leyendo {len(ref)} archivos de OCR (en paralelo)...", flush=True)
    ocr = load_json_many([OCR_DIR / f"{i}.json" for i in ref.index])
    print(f"Leídos: {len(ocr)}. Calculando CER/WER...", flush=True)

    rows = []
    missing_tok, extra_tok = Counter(), Counter()
    for cv_id, ref_text in ref.items():
        d = ocr.get(OCR_DIR / f"{cv_id}.json")
        if d is None:
            continue
        miss, extra = token_diff(ref_text, d["text_ordered"])
        missing_tok.update(miss)
        extra_tok.update(extra)
        m = evaluate_pair(ref_text, d["text_ordered"])
        if m:
            rows.append({"id": cv_id, "occupation": meta.loc[cv_id, "occupation"],
                         "mean_conf": d["mean_conf"], **m})
    df = pd.DataFrame(rows)
    if df.empty:
        print("No hay CVs con OCR y texto de referencia. ¿Corriste ocr_extraction.py?")
        return
    df.to_csv(TAB_DIR / "ocr_quality_per_cv.csv", index=False)

    # Dónde falla el OCR: tokens más frecuentes que faltan en el OCR y que sobran (Pregunta 2)
    modes = pd.concat([
        pd.DataFrame(missing_tok.most_common(30), columns=["token", "count"]).assign(kind="missing in OCR"),
        pd.DataFrame(extra_tok.most_common(30), columns=["token", "count"]).assign(kind="extra in OCR"),
    ])[["kind", "token", "count"]]
    modes.to_csv(TAB_DIR / "ocr_failure_modes.csv", index=False)
    worst = df.sort_values("cer", ascending=False).head(10)[["id", "occupation", "cer", "wer", "mean_conf"]]
    worst.to_csv(TAB_DIR / "ocr_worst_cvs.csv", index=False)

    metrics = ["cer", "wer", "bow_f1", "years_recall", "ranges_recall", "mean_conf"]
    by_occ = df.groupby("occupation")[metrics].mean().round(4)
    by_occ.loc["TODAS"] = df[metrics].mean().round(4)
    by_occ.to_csv(TAB_DIR / "ocr_quality_by_occupation.csv")

    summary = {
        "n_cvs": int(len(df)),
        **{f"{m}_mean": round(float(df[m].mean()), 4) for m in metrics},
        **{f"{m}_median": round(float(df[m].median()), 4) for m in ("cer", "wer")},
        **{f"{m}_p95": round(float(df[m].quantile(0.95)), 4) for m in ("cer", "wer")},
    }
    save_json(summary, MET_DIR / "ocr_quality_summary.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, title in zip(axes, ["cer", "wer"], ["CER por CV", "WER por CV"]):
        ax.hist(df[col].clip(upper=1.5), bins=40, color="#3b6ea5")
        ax.axvline(df[col].mean(), color="#c0392b", ls="--", label=f"media {df[col].mean():.3f}")
        ax.set_title(title)
        ax.set_xlabel(col.upper() + " (recortado a 1.5)")
        ax.legend()
    axes[0].set_ylabel("N.º de CVs")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ocr_cer_wer_hist.png", dpi=150)

    print(by_occ.to_string())
    print("\nTokens que más FALTAN en el OCR:", missing_tok.most_common(12))
    print("Tokens que más SOBRAN en el OCR:", extra_tok.most_common(12))
    print("10 CVs con peor CER:", ", ".join(worst["id"]))
    print("\nResumen:", summary)
    print("\nNota: si CER/WER salen altos pero bow_f1 es alto, el problema es el ORDEN de lectura "
          "(p. ej. dos columnas), no la lectura de caracteres.")


if __name__ == "__main__":
    main()
