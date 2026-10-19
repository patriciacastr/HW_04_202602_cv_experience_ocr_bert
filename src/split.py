"""Etapa 3: split train / val / test (70 / 15 / 15) por ID.

Estratificado por ocupación y banda de experiencia, con semilla fija. Todos los modelos
usan estos mismos archivos. El test no se toca hasta la evaluación final.

Entrada: data/processed/dataset.csv (salida de preprocessing.py); si no existe, usa
         data/raw/metadata.csv.
Salida : data/splits/train.csv, val.csv, test.csv  (id, occupation, exp_year, band)
         output/tables/split_summary.csv

Uso:
    python src/split.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (BAND_LABELS, CAP, PROC_DIR, RAW_DIR, SEED, SPLIT_DIR, TAB_DIR,  # noqa: E402
                   clip_years, ensure_dirs, to_band)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--cap", type=int, default=CAP)
    args = ap.parse_args()
    ensure_dirs()

    src = PROC_DIR / "dataset.csv"
    if not src.exists():
        src = RAW_DIR / "metadata.csv"
        print(f"AVISO: no existe dataset.csv; usando {src.name}")
    df = pd.read_csv(src, dtype={"id": str})
    col = "exp_year_raw" if "exp_year_raw" in df.columns else "exp_year"
    df["exp_year"] = clip_years(pd.to_numeric(df[col], errors="coerce"), args.cap)
    df = df.dropna(subset=["exp_year"]).copy()
    df["band"] = [BAND_LABELS[b] for b in to_band(df["exp_year"])]
    df = df[["id", "occupation", "exp_year", "band"]]

    strat = df["occupation"] + "|" + df["band"]
    if strat.value_counts().min() < 3:
        print("AVISO: hay estratos con menos de 3 CVs; se estratifica solo por ocupación.")
        strat = df["occupation"]

    trainval, test = train_test_split(df, test_size=0.15, stratify=strat, random_state=args.seed)
    strat_tv = strat.loc[trainval.index]
    train, val = train_test_split(trainval, test_size=0.15 / 0.85, stratify=strat_tv,
                                  random_state=args.seed)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.sort_values("id").to_csv(SPLIT_DIR / f"{name}.csv", index=False)

    summ = pd.concat({"train": train, "val": val, "test": test}, names=["split"]).reset_index(level=0)
    table = (summ.groupby(["split", "occupation"]).size().unstack(0)
             .assign(total=lambda t: t.sum(axis=1)))
    table.to_csv(TAB_DIR / "split_summary.csv")
    print(f"train={len(train)} val={len(val)} test={len(test)}  (total {len(df)})")
    print(table.to_string())
    print("\nBandas por split (%):")
    print((pd.crosstab(summ["split"], summ["band"], normalize="index") * 100).round(1).to_string())


if __name__ == "__main__":
    main()
