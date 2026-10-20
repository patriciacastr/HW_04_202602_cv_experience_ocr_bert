"""Etapa 1a: muestreo estratificado y descarga de CVs desde PopResume.

- Solo imágenes SIN foto (has_photo == False).
- 800 CVs por ocupación (4000 en total); usa --n-per-occ 400 si no alcanza el cómputo.
- Semilla fija; guarda imágenes en data/raw/images/<id>.png
- Guarda metadata (id, ocupación, exp_year, sex, race) en data/raw/metadata.csv
  (sex y race SOLO para el chequeo de equidad de la Pregunta 4, nunca como features).
- Guarda el texto de referencia (id, text) en data/raw/reference_text.csv
  (SOLO para medir CER/WER; jamás como feature).
- Guarda la lista de IDs en data/raw/sampled_ids.txt.

Requisitos previos: aceptar las condiciones del dataset en Hugging Face y
ejecutar `huggingface-cli login`.

Uso:
    python src/download_data.py --n-per-occ 800 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import IMG_DIR, OCCUPATIONS, RAW_DIR, SEED, ensure_dirs, set_seed  # noqa: E402

REPO = "sumin-yu/PopResume"


def sample_occupation(occ: str, n: int, seed: int, buffer_size: int = 5000):
    """Hace streaming de una ocupación y toma n CVs sin foto, con semilla fija."""
    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{REPO}/data/images/{occ}-*.parquet",
        split="train",
        streaming=True,
    )
    ds = ds.filter(lambda r: not r["has_photo"]).shuffle(seed=seed, buffer_size=buffer_size)
    return list(ds.take(n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-occ", type=int, default=800)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--buffer-size", type=int, default=5000,
                    help="Tamaño del buffer de mezcla. Usa 50 para la prueba pequeña (es mucho más rápido).")
    args = ap.parse_args()

    ensure_dirs()
    set_seed(args.seed)

    rows = []
    pii_rows = []  # nombres/estado SOLO para borrarlos del texto OCR (nunca como feature)
    for occ in OCCUPATIONS:
        print(f"\n== {occ}: muestreando {args.n_per_occ} CVs ==")
        sample = sample_occupation(occ, args.n_per_occ, args.seed, args.buffer_size)
        for r in tqdm(sample, desc=occ):
            cv_id = str(r["id"])
            img_path = IMG_DIR / f"{cv_id}.png"
            if not img_path.exists():
                r["image"].save(img_path)  # r["image"] es un PIL.Image
            rows.append(
                {
                    "id": cv_id,
                    "occupation": occ,
                    "exp_year": r.get("exp_year"),
                    # Solo para equidad (Pregunta 4). Si el campo no existe quedará vacío.
                    "sex": r.get("sex"),
                    "race": r.get("race"),
                }
            )
            pii_rows.append(
                {
                    "id": cv_id,
                    "first_name": r.get("first_name"),
                    "last_name": r.get("last_name"),
                    "state": r.get("state"),
                }
            )

    meta = pd.DataFrame(rows)
    meta.to_csv(RAW_DIR / "metadata.csv", index=False)
    (RAW_DIR / "sampled_ids.txt").write_text("\n".join(meta["id"]), encoding="utf-8")
    print(f"\nImágenes guardadas: {len(meta)}")

    # --- Texto de referencia (solo para medir calidad del OCR) ---
    print("\nCargando config de texto para el texto de referencia...")
    text_ds = load_dataset(REPO, "text")
    split = list(text_ds.keys())[0] if "train" not in text_ds else "train"
    tdf = text_ds[split].to_pandas()
    tdf["id"] = tdf["id"].astype(str)
    ref = tdf[tdf["id"].isin(set(meta["id"]))][["id", "text"]]
    ref.to_csv(RAW_DIR / "reference_text.csv", index=False)
    print(f"Textos de referencia guardados: {len(ref)} / {len(meta)}")

    # Si exp_year / sex / race no venían en las imágenes, completarlos desde el config de texto
    for col in ("exp_year", "sex", "race"):
        if meta[col].isna().all() and col in tdf.columns:
            meta[col] = meta["id"].map(tdf.set_index("id")[col])
    meta.to_csv(RAW_DIR / "metadata.csv", index=False)

    # Nombres/estado: solo para que preprocessing.py los borre del texto. Jamás features.
    pii = pd.DataFrame(pii_rows)
    for col in ("first_name", "last_name", "state"):
        if pii[col].isna().all() and col in tdf.columns:
            pii[col] = pii["id"].map(tdf.set_index("id")[col])
    pii.to_csv(RAW_DIR / "pii_for_scrubbing.csv", index=False)

    # --- Reportes pedidos: distribución y tope ---
    meta["exp_year"] = pd.to_numeric(meta["exp_year"], errors="coerce")
    n_cap = int((meta["exp_year"] >= 20).sum())
    print(f"\nCVs con exp_year >= 20 (se recortarán a 20): {n_cap} "
          f"({100 * n_cap / len(meta):.1f}%)")
    if n_cap / len(meta) > 0.15:
        print("AVISO: más del 15% está en el tope. Puedes subir el cap a 25 o 30 "
              "(justifícalo y usa el mismo cap en todos los modelos).")
    print(meta.groupby("occupation")["exp_year"].describe())


if __name__ == "__main__":
    main()
