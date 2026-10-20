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
import json
import resource
import signal
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import IMG_DIR, OCCUPATIONS, RAW_DIR, SEED, ensure_dirs, set_seed  # noqa: E402

REPO = "sumin-yu/PopResume"


def _announce(signum, frame):  # deja huella si el sistema mata el proceso con una señal
    print(f"\n[download_data] recibí la señal {signum}; el proceso se detiene.", flush=True)
    sys.exit(1)


for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    try:
        signal.signal(_sig, _announce)
    except (ValueError, OSError):
        pass


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def stream_occupation(occ: str, seed: int, buffer_size: int):
    """Streaming de una ocupación: SOLO CVs sin foto, mezclados con semilla fija (iterador, no lista)."""
    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{REPO}/data/images/{occ}-*.parquet",
        split="train",
        streaming=True,
    )
    return ds.filter(lambda r: not r["has_photo"]).shuffle(seed=seed, buffer_size=buffer_size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-occ", type=int, default=800)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--buffer-size", type=int, default=5000,
                    help="Tamaño del buffer de mezcla. Usa 50 para la prueba pequeña (es mucho más rápido).")
    args = ap.parse_args()

    ensure_dirs()
    set_seed(args.seed)

    parts_dir = RAW_DIR / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    rows, pii_rows = [], []  # pii: nombres/estado SOLO para borrarlos del texto OCR (nunca features)
    for occ in OCCUPATIONS:
        part = parts_dir / f"{occ}.json"
        if part.exists():  # ocupación ya completa en una corrida anterior: se reutiliza
            saved = json.loads(part.read_text(encoding="utf-8"))
            if len(saved["rows"]) >= args.n_per_occ:
                print(f"== {occ}: ya completa ({len(saved['rows'])} CVs), se reutiliza ==", flush=True)
                rows += saved["rows"][: args.n_per_occ]
                pii_rows += saved["pii"][: args.n_per_occ]
                continue
        print(f"\n== {occ}: muestreando {args.n_per_occ} CVs (buffer {args.buffer_size}) ==", flush=True)
        occ_rows, occ_pii = [], []
        bar = tqdm(total=args.n_per_occ, desc=occ, mininterval=5)
        for r in stream_occupation(occ, args.seed, args.buffer_size):
            cv_id = str(r["id"])
            img_path = IMG_DIR / f"{cv_id}.png"
            if not img_path.exists():
                r["image"].save(img_path)  # r["image"] es un PIL.Image
            occ_rows.append({"id": cv_id, "occupation": occ, "exp_year": r.get("exp_year"),
                             # Solo para equidad (Pregunta 4)
                             "sex": r.get("sex"), "race": r.get("race")})
            occ_pii.append({"id": cv_id, "first_name": r.get("first_name"),
                            "last_name": r.get("last_name"), "state": r.get("state")})
            bar.update(1)
            if len(occ_rows) % 100 == 0:
                print(f"  {occ}: {len(occ_rows)}/{args.n_per_occ} | RAM pico {rss_mb():.0f} MB", flush=True)
            if len(occ_rows) >= args.n_per_occ:
                break
        bar.close()
        part.write_text(json.dumps({"rows": occ_rows, "pii": occ_pii}, default=str), encoding="utf-8")
        rows += occ_rows
        pii_rows += occ_pii

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
