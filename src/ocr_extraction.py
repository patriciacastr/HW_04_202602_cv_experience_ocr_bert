"""Etapa 1b: PaddleOCR sobre cada imagen, con caché (un JSON por CV).

Por cada línea de texto se guarda: texto, confianza y caja (bounding box).
También guarda el texto en orden de lectura reconstruido (ver reading_order).

Orden de lectura:
  1. Se calcula el centro vertical de cada caja y se agrupan en "filas" (cajas cuyo
     centro y está dentro de una fracción de la altura mediana de línea).
  2. Se detecta si hay dos columnas: si existe un hueco vertical amplio en el eje x
     que ninguna caja cruza, se leen primero todas las líneas de la columna izquierda
     (arriba->abajo) y luego las de la derecha. Así un cargo no se separa de sus fechas.
  3. Si hay una sola columna: arriba->abajo, izquierda->derecha.

Uso:
    python src/ocr_extraction.py                 # todas las imágenes sin JSON en caché
    python src/ocr_extraction.py --limit 20      # prueba rápida con 20 imágenes
    python src/ocr_extraction.py --model server  # PP-OCRv5_server_rec (más preciso)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import IMG_DIR, OCR_DIR, ensure_dirs, load_json, save_json  # noqa: E402


# --------------------------------------------------------------------------- #
# Orden de lectura
# --------------------------------------------------------------------------- #
def _group_rows(items: list[dict], tol_factor: float = 0.6) -> list[list[dict]]:
    """Agrupa líneas en filas por cercanía del centro vertical."""
    if not items:
        return []
    heights = [it["box"][3] - it["box"][1] for it in items]
    tol = tol_factor * float(np.median(heights))
    items = sorted(items, key=lambda it: (it["box"][1] + it["box"][3]) / 2)
    rows, current = [], [items[0]]
    for it in items[1:]:
        cy = (it["box"][1] + it["box"][3]) / 2
        cy_prev = np.mean([(c["box"][1] + c["box"][3]) / 2 for c in current])
        if abs(cy - cy_prev) <= tol:
            current.append(it)
        else:
            rows.append(current)
            current = [it]
    rows.append(current)
    return [sorted(r, key=lambda it: it["box"][0]) for r in rows]


def _find_column_gap(items: list[dict], img_width: float | None):
    """Devuelve la coordenada x de un hueco que separe dos columnas, o None."""
    if len(items) < 8:
        return None
    x_min = min(it["box"][0] for it in items)
    x_max = max(it["box"][2] for it in items)
    width = x_max - x_min
    if width <= 0:
        return None
    bins = 200
    coverage = np.zeros(bins)
    for it in items:
        a = int((it["box"][0] - x_min) / width * (bins - 1))
        b = int((it["box"][2] - x_min) / width * (bins - 1))
        coverage[a : b + 1] += 1
    # Hueco: tramo contiguo sin cajas, en el 25%-75% central, de al menos 3% del ancho
    lo, hi = int(bins * 0.25), int(bins * 0.75)
    best, start = None, None
    for i in range(lo, hi + 1):
        if coverage[i] == 0 and start is None:
            start = i
        if (coverage[i] > 0 or i == hi) and start is not None:
            end = i
            if (end - start) >= 0.03 * bins:
                best = (start + end) / 2
                break
            start = None
    if best is None:
        return None
    return x_min + best / (bins - 1) * width


def reading_order(lines: list[dict]) -> list[dict]:
    """Reordena las líneas: columnas de izquierda a derecha, filas de arriba a abajo."""
    if not lines:
        return []
    gap_x = _find_column_gap(lines, None)
    if gap_x is None:
        groups = [lines]
    else:
        left = [l for l in lines if (l["box"][0] + l["box"][2]) / 2 < gap_x]
        right = [l for l in lines if (l["box"][0] + l["box"][2]) / 2 >= gap_x]
        groups = [left, right]
    ordered = []
    for g in groups:
        for row in _group_rows(g):
            ordered.extend(row)
    return ordered


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
def build_ocr(model: str = "mobile"):
    from paddleocr import PaddleOCR

    kwargs = dict(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    # Modelo de reconocimiento: mobile (rápido) o server (más preciso, más lento)
    kwargs["text_recognition_model_name"] = (
        "PP-OCRv5_server_rec" if model == "server" else "PP-OCRv5_mobile_rec"
    )
    return PaddleOCR(**kwargs)


def run_one(ocr, img_path: Path) -> dict:
    t0 = time.perf_counter()
    result = ocr.predict(str(img_path))[0]
    elapsed = time.perf_counter() - t0
    texts = list(result["rec_texts"])
    scores = [float(s) for s in result["rec_scores"]]
    boxes = [list(map(int, b)) for b in result["rec_boxes"]]  # [x1, y1, x2, y2]
    lines = [{"text": t, "score": s, "box": b} for t, s, b in zip(texts, scores, boxes)]
    ordered = reading_order(lines)
    return {
        "id": img_path.stem,
        "n_lines": len(lines),
        "mean_conf": float(np.mean(scores)) if scores else 0.0,
        "ocr_seconds": elapsed,
        "lines": lines,  # orden original de PaddleOCR
        "lines_ordered": ordered,  # orden de lectura reconstruido
        "text_ordered": "\n".join(l["text"] for l in ordered),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", choices=["mobile", "server"], default="mobile")
    args = ap.parse_args()

    ensure_dirs()
    imgs = sorted(IMG_DIR.glob("*.png"))
    if args.limit:
        imgs = imgs[: args.limit]
    pending = [p for p in imgs if not (OCR_DIR / f"{p.stem}.json").exists()]
    print(f"Imágenes totales: {len(imgs)} | pendientes (sin caché): {len(pending)}")
    if not pending:
        return

    ocr = build_ocr(args.model)
    failed = []
    for p in tqdm(pending, desc="PaddleOCR"):
        try:
            save_json(run_one(ocr, p), OCR_DIR / f"{p.stem}.json")
        except Exception as e:  # noqa: BLE001  (registrar y seguir)
            failed.append((p.stem, repr(e)))

    # --- Reporte: tiempo por CV, confianza media, fallos ---
    done = [load_json(f) for f in OCR_DIR.glob("*.json")]
    secs = [d["ocr_seconds"] for d in done]
    confs = [d["mean_conf"] for d in done]
    print(f"\nCVs procesados: {len(done)}")
    print(f"Tiempo medio por CV: {np.mean(secs):.2f}s | confianza media: {np.mean(confs):.3f}")
    print(f"Modelo de reconocimiento: {'PP-OCRv5_server_rec' if args.model == 'server' else 'PP-OCRv5_mobile_rec'}")
    try:
        import paddle

        print(f"Dispositivo: {paddle.device.get_device()}")
    except Exception:  # noqa: BLE001
        pass
    print(f"Fallidos: {len(failed)}")
    if failed:
        save_json(failed, OCR_DIR.parent / "ocr_failed.json")
        for f in failed[:10]:
            print("  ", f)


if __name__ == "__main__":
    main()
