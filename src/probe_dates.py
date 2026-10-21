"""Pregunta 1 - prueba de sondeo: ¿el modelo lee las fechas o solo el vocabulario?

Se toman 20 CVs del test (con al menos 2 cargos, semilla fija) y se les hacen dos cambios:
  A) Mover TODAS las fechas +5 años. Quien lee DURACIONES no debería cambiar su predicción
     (la persona trabajó lo mismo, solo "en otra época"). Quien usa años ABSOLUTOS como si
     fueran palabras (p. ej. "2006" => mucha experiencia) la bajará.
  B) Borrar un cargo (con sus viñetas). Un reclutador esperaría que la experiencia baje en la
     duración de ese cargo. Un modelo que solo ve vocabulario apenas se moverá.

Cada modelo se evalúa con el texto limpio del OCR (text_clean). Se incluye la referencia basada
en reglas como punto de comparación.

Salida: output/tables/probe_dates.csv (resumen) y output/tables/probe_dates_detail.csv (por CV).

Uso:
    python src/probe_dates.py --n 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dates import delete_job, parse_jobs, shift_years, years_from_jobs  # noqa: E402
from utils import CAP, MODEL_LABELS, SEED, TAB_DIR, clip_years, ensure_dirs, load_split, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    ensure_dirs()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    test = load_split("test", "text_clean")
    test["jobs"] = test["text"].map(parse_jobs)
    pool = test[test["jobs"].map(len) >= 2].reset_index(drop=True)
    if len(pool) < args.n:
        print(f"AVISO: solo {len(pool)} CVs de test con >= 2 cargos legibles.")
    sel = pool.iloc[rng.choice(len(pool), size=min(args.n, len(pool)), replace=False)].reset_index(drop=True)

    # --- construir los textos modificados ---
    rows = []
    for _, r in sel.iterrows():
        jobs = r["jobs"]
        k = int(rng.integers(len(jobs)))                 # cargo a borrar (al azar)
        dur = (jobs[k]["end"] - jobs[k]["start"]) / 12.0
        rows.append({"id": r["id"], "occupation": r["occupation"], "true": r["exp_year"],
                     "orig": r["text"], "shifted": shift_years(r["text"], 5),
                     "deleted": delete_job(r["text"], jobs[k]["line"]), "expected_drop": dur})
    d = pd.DataFrame(rows)

    # --- predictores: los 4 modelos (los que existan) + reglas ---
    from inference import load_predictors

    predictors, missing = load_predictors()
    for m, why in missing.items():
        print(f"AVISO: modelo '{MODEL_LABELS[m]}' no disponible ({why[:80]})")

    def rules(texts):
        out = []
        for t in texts:
            y = years_from_jobs(parse_jobs(t))
            out.append(0.0 if y is None else y)
        return clip_years(np.asarray(out), CAP)

    predictors = {**predictors, "rules": rules}
    labels = {**MODEL_LABELS, "rules": "Rule-based (sum of job dates)"}

    detail, summary = [], []
    for name, fn in predictors.items():
        p0, p1, p2 = (np.asarray(fn(list(d[c])), dtype=float) for c in ("orig", "shifted", "deleted"))
        shift_change, del_change = p1 - p0, p2 - p0
        for i in range(len(d)):
            detail.append({"model": labels[name], "id": d.loc[i, "id"], "true": d.loc[i, "true"],
                           "pred_original": p0[i], "pred_dates_plus5": p1[i], "pred_job_deleted": p2[i],
                           "expected_drop_if_job_deleted": d.loc[i, "expected_drop"]})
        exp = d["expected_drop"].to_numpy()
        corr = float(np.corrcoef(-del_change, exp)[0, 1]) if np.std(del_change) > 0 else float("nan")
        summary.append({
            "Model": labels[name],
            "Mean change, all dates +5y (years)": shift_change.mean(),
            "Mean |change|, dates +5y (years)": np.abs(shift_change).mean(),
            "CVs unchanged (<0.5y), dates +5y": float((np.abs(shift_change) < 0.5).mean()),
            "Mean change, 1 job deleted (years)": del_change.mean(),
            "Expected change (years)": -exp.mean(),
            "Moved down when job deleted": float((del_change < -0.25).mean()),
            "Correlation drop vs job length": corr,
        })

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(summary)
    out.to_csv(TAB_DIR / "probe_dates.csv", index=False)
    pd.DataFrame(detail).to_csv(TAB_DIR / "probe_dates_detail.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(f"\nPrueba con {len(d)} CVs del test.\n")
    print(out.round(2).to_string(index=False))
    print("\nLectura: con +5 años en todas las fechas lo ideal es cambio ~0; al borrar un cargo lo "
          "ideal es bajar la duración de ese cargo (columna 'Expected change').")


if __name__ == "__main__":
    main()
