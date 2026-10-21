"""Análisis de errores de BERT: ¿en qué tipo de CV falla y por qué?

Para cada CV del test se calculan rasgos de su historial laboral (a partir del texto limpio):
  n_jobs        número de cargos con fechas
  gap_months    meses sin trabajo entre cargos consecutivos
  overlap       si dos cargos se solapan en el tiempo
  date_check    |suma de duraciones de las fechas leídas - etiqueta| (años)
                > 0.25  => la fecha se leyó mal o falta (error de OCR / fechas ausentes)
Con eso:
  1) tabla de error de BERT según estructura (output/tables/bert_error_by_structure.csv)
  2) causa de cada uno de los 10 mayores errores, agrupados como pide la tarea
     (output/tables/bert_top10_errors.csv, columna "cause")

Causas posibles: OCR misread a date / missing dates, wrong reading order*, career gap,
overlapping jobs, truncation*, single short job (other), other.
(*) Se comprueban con otros datos: orden de lectura (CVs de una sola columna) y truncamiento
(% de CVs sobre el límite de tokens, que reporta train_bert.py).

Uso:   python src/error_analysis.py --model bert
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dates import parse_jobs, years_from_jobs  # noqa: E402
from utils import MET_DIR, TAB_DIR, load_split  # noqa: E402


def structure(text: str) -> dict:
    jobs = sorted(parse_jobs(text), key=lambda j: j["start"])
    gap = overlap = 0
    for a, b in zip(jobs, jobs[1:]):
        d = b["start"] - a["end"]
        if d > 0:
            gap += d
        elif d < 0:
            overlap += 1
    return {"n_jobs": len(jobs), "gap_months": gap, "overlap": overlap > 0,
            "date_years": years_from_jobs(jobs)}


def cause_of(r) -> str:
    if r["date_years"] is None or np.isnan(r["date_years"]):
        return "missing dates (no readable date range)"
    if abs(r["date_years"] - r["true"]) > 0.25:
        return "OCR misread a date (dates do not add up to the label)"
    if r["overlap"]:
        return "overlapping jobs"
    if r["gap_months"] > 0:
        return "career gap between jobs"
    if r["n_jobs"] == 1:
        return "single short job (other)"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    test = load_split("test", "text_clean")
    pred = pd.read_csv(MET_DIR / f"predictions_{args.model}.csv", dtype={"id": str})
    pred = pred[pred["split"] == "test"][["id", "y_pred"]]
    df = test.merge(pred, on="id")
    feats = pd.DataFrame([structure(t) for t in df["text"]])
    df = pd.concat([df.reset_index(drop=True), feats], axis=1)
    df["true"] = df["exp_year"]
    df["error"] = df["y_pred"] - df["true"]
    df["abs_error"] = df["error"].abs()
    df["has_gap"] = df["gap_months"] > 0
    df["jobs_group"] = df["n_jobs"].clip(upper=4).map(lambda n: f"{n}" if n < 4 else "4+")
    df["date_check_ok"] = (df["date_years"] - df["true"]).abs() <= 0.25

    rows = []
    for name, col in (("number of jobs", "jobs_group"), ("career gap between jobs", "has_gap"),
                      ("overlapping jobs", "overlap"), ("dates add up to the label", "date_check_ok")):
        for val, g in df.groupby(col):
            rows.append({"structure": name, "group": str(val), "n CVs": len(g),
                         "MAE (years)": g["abs_error"].mean(),
                         "mean error (pred - true)": g["error"].mean()})
    out = pd.DataFrame(rows)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TAB_DIR / "bert_error_by_structure.csv", index=False)

    top = df.sort_values("abs_error", ascending=False).head(args.top).copy()
    top["cause"] = top.apply(cause_of, axis=1)
    top["text_start"] = top["text"].str.replace("\n", " | ").str[:420]
    top[["id", "occupation", "band", "true", "y_pred", "error", "abs_error", "n_jobs", "gap_months",
         "cause", "text_start"]].rename(columns={"y_pred": "pred"}).to_csv(
        TAB_DIR / f"{args.model}_top10_errors.csv", index=False)

    pd.set_option("display.width", 220)
    print("== Error de BERT según estructura del CV (test) ==")
    print(out.round(3).to_string(index=False))
    print(f"\n== Causa de los {args.top} mayores errores ==")
    print(top[["id", "true", "y_pred", "error", "n_jobs", "gap_months", "cause"]].round(2).to_string(index=False))
    print("\nResumen de causas:\n", top["cause"].value_counts().to_string())


if __name__ == "__main__":
    main()
