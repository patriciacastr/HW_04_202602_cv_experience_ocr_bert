"""Etapa 6: evaluación de todos los modelos sobre el MISMO test.

Lee output/metrics/predictions_<modelo>.csv (los escriben baseline_*.py y train_bert.py)
y genera, en output/tables, output/figures y output/metrics:

  tables/model_comparison.csv      MAE, RMSE, Acc ±1/2/3, banda, macro-F1, R², tiempos
  tables/bert_gain.csv             ganancia de MAE de BERT vs cada baseline + IC 95 % bootstrap
  tables/mae_by_band.csv           MAE por banda real de experiencia
  tables/mae_by_occupation.csv     MAE por ocupación
  tables/ocr_vs_reference.csv      Pregunta 2: MAE con texto OCR vs texto de referencia
  tables/sensitivity.csv           Pregunta 3: MAE de todas las variantes (dígitos, pérdida, ...)
  tables/fairness_<modelo>.csv     Pregunta 4: MAE y error medio por sexo y raza
  tables/<modelo>_top10_errors.csv los 10 mayores errores (para clasificarlos a mano)
  figures/pred_vs_true.png, mae_by_band.png, mae_by_occupation.png, confusion_matrices.png

Colores: cada modelo tiene SIEMPRE el mismo color (paleta validada en utils.py).

Uso:
    python src/evaluation.py
    python src/evaluation.py --focus bert --bootstrap 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (BAND_LABELS, CAP, FIG_DIR, MET_DIR, MODEL_COLORS, MODEL_LABELS,  # noqa: E402
                   MODEL_ORDER, PROC_DIR, RAW_DIR, SEED, SPLIT_DIR, TAB_DIR, ensure_dirs,
                   load_json, to_band)

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    bt, bp = to_band(y), to_band(p)
    return {
        "MAE (years)": float(np.mean(np.abs(err))),
        "RMSE (years)": float(np.sqrt(np.mean(err ** 2))),
        "Acc ±1": float(np.mean(np.abs(err) <= 1)),
        "Acc ±2": float(np.mean(np.abs(err) <= 2)),
        "Acc ±3": float(np.mean(np.abs(err) <= 3)),
        "Band accuracy": float(np.mean(bt == bp)),
        "Band macro-F1": float(f1_score(bt, bp, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "R²": float(r2_score(y, p)),
    }


def load_test_preds(name: str, test_ids: pd.Index) -> pd.DataFrame | None:
    path = MET_DIR / f"predictions_{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"id": str})
    df = df[df["split"] == "test"].set_index("id")
    missing = test_ids.difference(df.index)
    if len(missing):
        print(f"AVISO [{name}]: faltan {len(missing)} CVs del test; se omite este modelo.")
        return None
    return df.loc[test_ids]


def paired_bootstrap_gain(e_base: np.ndarray, e_new: np.ndarray, n_boot: int, seed: int):
    """Ganancia = MAE(base) - MAE(nuevo); IC 95 % con remuestreo pareado del test."""
    rng = np.random.default_rng(seed)
    n = len(e_base)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = e_base[idx].mean(axis=1) - e_new[idx].mean(axis=1)
    return float(e_base.mean() - e_new.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# --------------------------------------------------------------------------- #
# Estilo de gráficos
# --------------------------------------------------------------------------- #
def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    })
    return plt


def grouped_bars(plt, table: pd.DataFrame, models: list[str], title: str, xlabel: str, path: Path):
    """Barras agrupadas: una serie por modelo, color fijo, leyenda siempre presente."""
    n_m = len(models)
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(table)), 4))
    w = 0.8 / n_m
    x = np.arange(len(table))
    for k, m in enumerate(models):
        ax.bar(x + (k - (n_m - 1) / 2) * w, table[m].values, width=w * 0.92,
               color=MODEL_COLORS[m], label=MODEL_LABELS[m], edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in table.index], rotation=0 if len(table) <= 5 else 20, ha="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAE (años)")
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=n_m, loc="upper center", bbox_to_anchor=(0.5, -0.22), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--focus", default="bert", help="modelo para equidad y errores (def. bert)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    ensure_dirs()

    test = pd.read_csv(SPLIT_DIR / "test.csv", dtype={"id": str}).set_index("id")
    y = test["exp_year"].values.astype(float)

    preds = {m: load_test_preds(m, test.index) for m in MODEL_ORDER}
    models = [m for m in MODEL_ORDER if preds[m] is not None]
    print("Modelos con predicciones:", models)
    if not models:
        print("No hay predicciones. Corre primero baseline_*.py y train_bert.py.")
        return

    # ---- Tabla principal ---------------------------------------------------
    rows = []
    for m in models:
        info_path = MET_DIR / f"info_{m}.json"
        info = load_json(info_path) if info_path.exists() else {}
        rows.append({"Model": MODEL_LABELS[m], **compute_metrics(y, preds[m]["y_pred"].values),
                     "Training time": info.get("train_time_s", np.nan),
                     "Inference time per CV": info.get("inference_time_per_cv_s", np.nan)})
    comp = pd.DataFrame(rows)
    comp.to_csv(TAB_DIR / "model_comparison.csv", index=False)
    print("\n== Comparación de modelos (test) ==")
    print(comp.round(3).to_string(index=False))

    # ---- Ganancia de BERT con IC bootstrap ----------------------------------
    focus = args.focus if args.focus in models else None
    if focus is None:
        cand = [m for m in models if m != "median"]
        focus = min(cand, key=lambda m: np.abs(preds[m]["y_pred"].values - y).mean()) if cand else models[0]
        print(f"AVISO: no hay '{args.focus}'; el análisis de equidad y errores usará '{focus}'.")
    if "bert" in models:
        e_bert = np.abs(preds["bert"]["y_pred"].values - y)
        g = []
        for m in models:
            if m == "bert":
                continue
            e_b = np.abs(preds[m]["y_pred"].values - y)
            gain, lo, hi = paired_bootstrap_gain(e_b, e_bert, args.bootstrap, args.seed)
            g.append({"baseline": MODEL_LABELS[m], "MAE baseline": e_b.mean(), "MAE BERT": e_bert.mean(),
                      "gain (years)": gain, "CI95 low": lo, "CI95 high": hi,
                      "relative gain (%)": 100 * gain / e_b.mean(),
                      "statistically meaningful": bool(lo > 0)})
        pd.DataFrame(g).to_csv(TAB_DIR / "bert_gain.csv", index=False)
        print("\n== Ganancia de BERT (MAE baseline - MAE BERT, IC 95 % bootstrap) ==")
        print(pd.DataFrame(g).round(3).to_string(index=False))

    # ---- MAE por banda y por ocupación --------------------------------------
    band_true = np.array(BAND_LABELS)[to_band(y)]
    by_band = pd.DataFrame({m: pd.Series(np.abs(preds[m]["y_pred"].values - y)).groupby(band_true).mean()
                            for m in models}).reindex(BAND_LABELS)
    by_band.insert(0, "n", pd.Series(band_true).value_counts().reindex(BAND_LABELS).values)
    by_band.rename(columns=MODEL_LABELS).to_csv(TAB_DIR / "mae_by_band.csv")
    occ = test["occupation"].values
    by_occ = pd.DataFrame({m: pd.Series(np.abs(preds[m]["y_pred"].values - y)).groupby(occ).mean()
                           for m in models})
    by_occ.insert(0, "n", pd.Series(occ).value_counts().reindex(by_occ.index).values)
    by_occ.rename(columns=MODEL_LABELS).to_csv(TAB_DIR / "mae_by_occupation.csv")
    print("\n== MAE por banda ==\n", by_band.round(2).to_string())

    # ---- Pregunta 2: OCR vs texto de referencia -----------------------------
    q2 = []
    for m in models:
        ref = preds[m]["y_pred_ref"].values
        if np.isnan(ref).all():
            continue
        m_ocr, m_ref = np.abs(preds[m]["y_pred"].values - y).mean(), np.abs(ref - y).mean()
        q2.append({"Model": MODEL_LABELS[m], "MAE OCR text": m_ocr, "MAE reference text": m_ref,
                   "gap (OCR - reference)": m_ocr - m_ref,
                   "gap (%)": 100 * (m_ocr - m_ref) / m_ref if m_ref > 0 else np.nan})
    if q2:
        pd.DataFrame(q2).to_csv(TAB_DIR / "ocr_vs_reference.csv", index=False)
        print("\n== Pregunta 2: costo del ruido del OCR ==\n", pd.DataFrame(q2).round(3).to_string(index=False))

    # ---- Pregunta 3: sensibilidad (todas las variantes disponibles) ---------
    sens = []
    for f in sorted(MET_DIR.glob("predictions_*.csv")):
        name = f.stem.replace("predictions_", "")
        p = load_test_preds(name, test.index)
        if p is not None:
            sens.append({"variant": name, "MAE test (OCR text)": np.abs(p["y_pred"].values - y).mean()})
    pd.DataFrame(sens).to_csv(TAB_DIR / "sensitivity.csv", index=False)
    print("\n== Pregunta 3: sensibilidad ==\n", pd.DataFrame(sens).round(3).to_string(index=False))

    # ---- Pregunta 4: equidad (sexo y raza solo aquí) ------------------------
    meta_path = RAW_DIR / "metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path, dtype={"id": str}).set_index("id").reindex(test.index)
        err = preds[focus]["y_pred"].values - y
        fair = []
        for attr in ("sex", "race"):
            if attr not in meta or meta[attr].isna().all():
                continue
            g = pd.DataFrame({"attr": meta[attr].values, "err": err}).groupby("attr")["err"]
            for grp, s in g:
                fair.append({"attribute": attr, "group": grp, "n": len(s),
                             "MAE": float(np.abs(s).mean()), "mean error (pred - true)": float(s.mean())})
        if fair:
            pd.DataFrame(fair).to_csv(TAB_DIR / f"fairness_{focus}.csv", index=False)
            print(f"\n== Pregunta 4: equidad ({MODEL_LABELS[focus]}) ==\n",
                  pd.DataFrame(fair).round(3).to_string(index=False))
    else:
        print("AVISO: falta data/raw/metadata.csv; se omite el análisis de equidad.")

    # ---- 10 mayores errores del modelo focal --------------------------------
    pf = preds[focus]["y_pred"].values
    top = (test.assign(true=y, pred=pf, error=pf - y, abs_error=np.abs(pf - y))
           .sort_values("abs_error", ascending=False).head(10).reset_index())
    ds_path = PROC_DIR / "dataset.csv"
    if ds_path.exists():
        ds = pd.read_csv(ds_path, dtype={"id": str}, usecols=["id", "text_clean"]).set_index("id")
        top["text_start"] = top["id"].map(lambda i: str(ds.loc[i, "text_clean"])[:400].replace("\n", " | "))
    top["cause (rellenar a mano)"] = ""
    top.to_csv(TAB_DIR / f"{focus}_top10_errors.csv", index=False)

    # ---- Gráficos -------------------------------------------------------------
    plt = setup_plt()

    # 1) Predicho vs real (una sola escala, un panel por modelo)
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.9), sharex=True, sharey=True, squeeze=False)
    rng = np.random.default_rng(args.seed)
    for ax, m in zip(axes[0], models):
        p = preds[m]["y_pred"].values
        ax.scatter(y + rng.uniform(-0.25, 0.25, len(y)), p, s=12, alpha=0.35,
                   color=MODEL_COLORS[m], edgecolor="none")
        ax.plot([0, CAP], [0, CAP], color=INK2, lw=1, ls="--", label="y = x")
        bins = np.arange(0, CAP + 2, 2)
        c = np.digitize(y, bins) - 1
        xs = [y[c == b].mean() for b in range(len(bins)) if (c == b).sum() >= 3]
        ys = [p[c == b].mean() for b in range(len(bins)) if (c == b).sum() >= 3]
        ax.plot(xs, ys, color=INK, lw=2, label="media predicha")
        ax.set_title(f"{MODEL_LABELS[m]}  (MAE {np.abs(p - y).mean():.2f})", loc="left", fontsize=10)
        ax.set_xlabel("Años reales")
        ax.set_xlim(-0.5, CAP + 0.5)
        ax.set_ylim(-0.5, CAP + 0.5)
    axes[0][0].set_ylabel("Años predichos")
    axes[0][0].legend(frameon=False, loc="upper left", labelcolor=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pred_vs_true.png", dpi=150)
    plt.close(fig)

    # 2) MAE por banda y por ocupación
    grouped_bars(plt, by_band[models], models, "MAE por banda real de experiencia",
                 "Banda de experiencia (años)", FIG_DIR / "mae_by_band.png")
    grouped_bars(plt, by_occ[models], models, "MAE por ocupación", "Ocupación",
                 FIG_DIR / "mae_by_occupation.png")

    # 3) Matrices de confusión por banda (rampa secuencial de un solo tono)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#3987e5", "#0d366b"])
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.7), squeeze=False)
    bt = to_band(y)
    for ax, m in zip(axes[0], models):
        cm = confusion_matrix(bt, to_band(preds[m]["y_pred"].values), labels=[0, 1, 2, 3])
        ax.imshow(cm, cmap=cmap)
        ax.grid(False)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() * 0.55 else INK)
        ax.set_xticks(range(4))
        ax.set_xticklabels(BAND_LABELS)
        ax.set_yticks(range(4))
        ax.set_yticklabels(BAND_LABELS)
        ax.set_xlabel("Banda predicha")
        ax.set_title(MODEL_LABELS[m], loc="left", fontsize=10)
    axes[0][0].set_ylabel("Banda real")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    print(f"\nTablas en {TAB_DIR}\nFiguras en {FIG_DIR}")


if __name__ == "__main__":
    main()
