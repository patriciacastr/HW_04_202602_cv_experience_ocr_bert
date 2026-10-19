"""Etapa 2b: limpieza del texto OCR, eliminación de datos personales y enmascarado
de respuestas directas.

Qué hace, por cada CV (JSON de data/ocr/):
  1. Toma las líneas en orden de lectura (ya reconstruido en ocr_extraction.py).
  2. Normaliza unicode y guiones; arregla guiones de fin de línea y líneas partidas.
  3. Elimina datos personales: nombre y apellido (usando la lista guardada SOLO para
     borrarlos), correo, teléfono, URLs, dirección y ciudad/estado/código postal.
  4. Enmascara frases que dan la respuesta: "12 years of experience", edad, fecha de
     nacimiento. Las fechas de empleos y de graduación NO se tocan.
  5. Genera tres variantes de texto (útiles para la Pregunta 3):
       - text_raw           : OCR en orden de lectura, SIN limpieza de formato, pero con lo obligatorio
                             (sin datos personales y sin respuestas directas)
       - text_clean         : limpieza completa (pasos 2-4)
       - text_clean_lowconf : como text_clean pero sin las líneas con confianza < umbral

Uso:
    python src/preprocessing.py --review 50      # imprime 50 textos OCR al azar para leerlos
    python src/preprocessing.py                  # procesa todo y escribe data/processed/dataset.csv
    python src/preprocessing.py --conf-threshold 0.6
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (CAP, MET_DIR, OCR_DIR, PROC_DIR, RAW_DIR, SEED, clip_years,  # noqa: E402
                   ensure_dirs, load_json, save_json)

# --------------------------------------------------------------------------- #
# Patrones de respuestas directas (se deben revisar leyendo >= 50 textos reales)
# --------------------------------------------------------------------------- #
NUM_WORDS = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
             "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty")
NUM = rf"(?:\d{{1,2}}|{NUM_WORDS})"

EXP_PATTERNS = [
    # "12 years of experience", "10+ years of professional experience", "over 8 years' experience"
    re.compile(rf"\b(?:(?:over|more than|nearly|almost|approximately|about)\s+)?{NUM}\s*\+?\s*"
               rf"(?:-\s*{NUM}\s*)?(?:years?|yrs?)\b['’]?[^.\n]{{0,45}}?\bexperience\b", re.I),
    # "experience: 12 years", "experience of 12+ years"
    re.compile(rf"\bexperience\b[^.\n]{{0,25}}?\b{NUM}\s*\+?\s*(?:years?|yrs?)\b", re.I),
    # "12 years in the field / industry / profession / career"
    re.compile(rf"\b{NUM}\s*\+?\s*(?:years?|yrs?)\b[^.\n]{{0,20}}"
               rf"\b(?:in the (?:field|industry|profession)|career|of service|in practice)\b", re.I),
]
AGE_PATTERNS = [
    re.compile(r"\bage[d]?\s*[:\-]?\s*\d{2}\b", re.I),
    re.compile(r"\b\d{2}\s*[- ]?\s*(?:years?|yrs?)[- ]\s*old\b", re.I),
    re.compile(r"\b(?:date of birth|d\.?o\.?b\.?|born)\b\s*[:\-]?\s*(?:on|in)?\s*"
               r"(?:[A-Za-z]{3,9}\.?\s+)?(?:\d{1,2}(?:st|nd|rd|th)?[,/\- ]+)?"
               r"(?:\d{1,2}[/\- ])?(?:19|20)\d{2}", re.I),
]

# --------------------------------------------------------------------------- #
# Patrones de datos personales
# --------------------------------------------------------------------------- #
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?:\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
URL = re.compile(r"(?:https?://|www\.|linkedin\.com/|github\.com/)\S+", re.I)
STREET = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-z0-9.'-]+\s+){1,4}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Court|Ct|"
    r"Place|Pl|Circle|Cir|Terrace|Trail)\b\.?", re.I)
CITY_STATE_ZIP = re.compile(r"\b[A-Z][A-Za-z.' -]+,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?\b")


# --------------------------------------------------------------------------- #
# Funciones de limpieza
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    """Unicode NFKC y guiones unificados (para que '2014 – 2019' y '2014 - 2019' coincidan)."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[‐-―−]", "-", text)
    text = text.replace(" ", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def fix_lines(lines: list[str]) -> list[str]:
    """Une guiones de fin de línea y continuaciones evidentes (conservador, para no
    pegar un cargo con su fecha)."""
    out: list[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if out and out[-1].endswith("-") and not out[-1].endswith(" -") and ln[:1].islower():
            out[-1] = out[-1][:-1] + ln                      # palabra partida con guion
        elif out and ln[:1].islower() and len(out[-1]) > 40 and not re.search(r"[.:;]$", out[-1]):
            out[-1] += " " + ln                              # continuación de oración
        else:
            out.append(ln)
    return out


def mask_direct_answers(text: str) -> tuple[str, int, int]:
    """Enmascara experiencia total, edad y fecha de nacimiento. Devuelve (texto, n_exp, n_age)."""
    n_exp = n_age = 0
    for p in EXP_PATTERNS:
        text, k = p.subn("[MASKED_EXP]", text)
        n_exp += k
    for p in AGE_PATTERNS:
        text, k = p.subn("[MASKED_AGE]", text)
        n_age += k
    return text, n_exp, n_age


def scrub_personal(text: str, names: list[str] | None) -> tuple[str, dict]:
    """Elimina nombre, correo, teléfono, URLs y dirección. Devuelve (texto, conteos)."""
    c = {"name": 0, "email": 0, "phone": 0, "url": 0, "address": 0}
    text, c["email"] = EMAIL.subn("[EMAIL]", text)
    text, c["url"] = URL.subn("[URL]", text)
    text, c["phone"] = PHONE.subn("[PHONE]", text)
    text, k1 = STREET.subn("[ADDRESS]", text)
    text, k2 = CITY_STATE_ZIP.subn("[ADDRESS]", text)
    c["address"] = k1 + k2
    for nm in names or []:
        if nm and len(nm) > 1:
            text, k = re.subn(rf"\b{re.escape(nm)}\b", "[NAME]", text, flags=re.I)
            c["name"] += k
    return text, c


def build_variants(d: dict, names: list[str] | None, conf_thr: float) -> dict:
    lines = d["lines_ordered"]

    def full_clean(ls):
        texts = fix_lines([normalize(l["text"]) for l in ls])
        t = "\n".join(texts)
        t, pii = scrub_personal(t, names)
        t, n_exp, n_age = mask_direct_answers(t)
        return t, pii, n_exp, n_age

    clean, pii, n_exp, n_age = full_clean(lines)
    clean_lc, _, _, _ = full_clean([l for l in lines if l["score"] >= conf_thr])
    # "raw" = sin limpieza de formato, pero con los pasos OBLIGATORIOS: sin datos personales
    # (nombres no pueden ser entrada) y sin respuestas directas.
    raw_scrubbed, _ = scrub_personal(d["text_ordered"], names)
    raw_masked, _, _ = mask_direct_answers(raw_scrubbed)
    return {
        "text_raw": raw_masked,
        "text_clean": clean,
        "text_clean_lowconf": clean_lc,
        "n_masked_exp": n_exp,
        "n_masked_age": n_age,
        "n_name": pii["name"], "n_email": pii["email"], "n_phone": pii["phone"],
        "n_url": pii["url"], "n_address": pii["address"],
        "n_lines_dropped_lowconf": sum(1 for l in lines if l["score"] < conf_thr),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def review(n: int, seed: int) -> None:
    """Imprime n textos OCR al azar para leerlos y ajustar los patrones de enmascarado."""
    files = sorted(OCR_DIR.glob("*.json"))
    random.Random(seed).shuffle(files)
    out = []
    for f in files[:n]:
        d = load_json(f)
        out.append(f"===== {d['id']} (confianza {d['mean_conf']:.2f}) =====\n{d['text_ordered']}\n")
    txt = "\n".join(out)
    (MET_DIR).mkdir(parents=True, exist_ok=True)
    (MET_DIR / "review_sample.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\n(Guardado en {MET_DIR / 'review_sample.txt'}). Busca frases que den la respuesta "
          "(años de experiencia, edad, fecha de nacimiento) y añade patrones si hace falta.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=int, default=None, help="imprime N textos OCR al azar")
    ap.add_argument("--conf-threshold", type=float, default=0.6)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    ensure_dirs()

    if args.review:
        review(args.review, args.seed)
        return

    meta = pd.read_csv(RAW_DIR / "metadata.csv", dtype={"id": str})
    pii_path = RAW_DIR / "pii_for_scrubbing.csv"
    if pii_path.exists():
        pii = pd.read_csv(pii_path, dtype={"id": str}).set_index("id")
    else:
        pii = None
        print("AVISO: no existe pii_for_scrubbing.csv; los nombres NO se podrán borrar por lista.")

    rows = []
    for _, m in meta.iterrows():
        f = OCR_DIR / f"{m['id']}.json"
        if not f.exists():
            continue
        d = load_json(f)
        names = None
        if pii is not None and m["id"] in pii.index:
            names = [str(pii.loc[m["id"], "first_name"]), str(pii.loc[m["id"], "last_name"])]
        v = build_variants(d, names, args.conf_threshold)
        rows.append({"id": m["id"], "occupation": m["occupation"],
                     "exp_year_raw": m["exp_year"], "n_lines": d["n_lines"],
                     "mean_conf": d["mean_conf"], **v})

    df = pd.DataFrame(rows)
    df["exp_year"] = clip_years(pd.to_numeric(df["exp_year_raw"], errors="coerce"), args.cap)
    df = df.dropna(subset=["exp_year"])
    df.to_csv(PROC_DIR / "dataset.csv", index=False)

    # --- Reporte (para el README) ---
    n = len(df)
    report = {
        "n_cvs": n,
        "cap": args.cap,
        "n_at_cap": int((pd.to_numeric(df["exp_year_raw"]) >= args.cap).sum()),
        "pct_at_cap": round(100 * float((pd.to_numeric(df["exp_year_raw"]) >= args.cap).mean()), 2),
        "cvs_with_masked_exp": int((df["n_masked_exp"] > 0).sum()),
        "cvs_with_masked_age": int((df["n_masked_age"] > 0).sum()),
        "cvs_with_any_masking": int(((df["n_masked_exp"] + df["n_masked_age"]) > 0).sum()),
        "cvs_name_found": int((df["n_name"] > 0).sum()),
        "cvs_email_found": int((df["n_email"] > 0).sum()),
        "cvs_phone_found": int((df["n_phone"] > 0).sum()),
        "cvs_address_found": int((df["n_address"] > 0).sum()),
        "conf_threshold": args.conf_threshold,
        "mean_pct_lines_dropped_lowconf": round(
            100 * float((df["n_lines_dropped_lowconf"] / df["n_lines"].clip(lower=1)).mean()), 2),
    }
    save_json(report, MET_DIR / "preprocessing_report.json")
    print(f"Guardado data/processed/dataset.csv ({n} CVs)")
    for k, v in report.items():
        print(f"  {k}: {v}")
    if report["pct_at_cap"] > 15:
        print("AVISO: más del 15% está en el tope; considera subir el cap a 25 o 30 (justifícalo).")


if __name__ == "__main__":
    main()
