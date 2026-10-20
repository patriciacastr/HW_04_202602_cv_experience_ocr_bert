"""Streamlit app: estimating years of work experience from CV images (PaddleOCR + 4 models).

Run:  streamlit run app.py

The app reads the light result files produced by the pipeline (output/tables, output/figures,
output/metrics, data/splits). Heavy files (CV images, OCR cache, model weights) are NOT in
the repo: when they are missing, the sections that need them say so and the rest keeps working.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from utils import (BAND_LABELS, CAP, FIG_DIR, IMG_DIR, MET_DIR, MODEL_COLORS, MODEL_LABELS,  # noqa: E402
                   MODEL_ORDER, OCCUPATIONS, OCR_DIR, PROC_DIR, SPLIT_DIR, TAB_DIR, to_band)

NEUTRAL = "#5b6b82"      # data that is not a model (distributions, OCR quality)
INK_MUTED = "#8a8a82"
LABEL_COLORS = {MODEL_LABELS[m]: MODEL_COLORS[m] for m in MODEL_ORDER}
OCC_LABEL = OCCUPATIONS
OCC_SHORT = {"registered_nurses": "Nurses", "elementary_middle_school_teachers": "Teachers",
             "software_developers": "Developers", "accountants_auditors": "Accountants",
             "construction_laborers": "Laborers"}
DOCS_ANSWERS = ROOT / "docs" / "answers.md"

st.set_page_config(page_title="CV experience estimator", page_icon=None, layout="wide")


# --------------------------------------------------------------------------- #
# Loading helpers (everything is optional)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def read_csv(path: str, **kw):
    p = Path(path)
    return pd.read_csv(p, **kw) if p.exists() else None


@st.cache_data(show_spinner=False)
def read_json(path: str):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_splits() -> pd.DataFrame | None:
    parts = []
    for name in ("train", "val", "test"):
        d = read_csv(str(SPLIT_DIR / f"{name}.csv"), dtype={"id": str})
        if d is not None:
            d = d.copy()
            d["split"] = name
            parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else None


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame | None:
    """One row per test CV with the prediction of every available model."""
    base = None
    for m in MODEL_ORDER:
        d = read_csv(str(MET_DIR / f"predictions_{m}.csv"), dtype={"id": str})
        if d is None:
            continue
        d = d[d["split"] == "test"][["id", "y_true", "y_pred"]].rename(columns={"y_pred": m})
        base = d if base is None else base.merge(d.drop(columns="y_true"), on="id", how="outer")
    if base is None:
        return None
    sp = load_splits()
    if sp is not None:
        base = base.merge(sp[["id", "occupation"]], on="id", how="left")
    return base


@st.cache_data(show_spinner=False)
def load_local_texts() -> pd.DataFrame | None:
    d = read_csv(str(PROC_DIR / "dataset.csv"), dtype={"id": str},
                 usecols=["id", "text_clean", "n_lines", "mean_conf"])
    return d


def missing(what: str):
    st.info(f"{what} not found yet. It is created when the pipeline stage that produces it is run "
            "(see the README, section *How to reproduce*).")


def occ_name(x: str) -> str:
    return OCC_LABEL.get(x, x)


def chart_note(text: str):
    st.caption(f"Why this chart: {text}")


def style(fig, height: int = 380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.2, title_text=""))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    return fig


def answers_section(tag: str) -> str | None:
    """Narrative answer for one analytical question, from docs/answers.md ('## Q1 ...' headings)."""
    if not DOCS_ANSWERS.exists():
        return None
    txt = DOCS_ANSWERS.read_text(encoding="utf-8")
    blocks = {}
    cur = None
    for line in txt.splitlines():
        if line.startswith("## Q"):
            cur = line[3:5]
            blocks[cur] = []
        elif cur:
            blocks[cur].append(line)
    return "\n".join(blocks[tag]).strip() if tag in blocks else None


def show_answer(tag: str):
    a = answers_section(tag)
    if a:
        st.markdown(a)
    else:
        st.warning("The written answer for this question is added to docs/answers.md "
                   "once the final numbers are available.")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Estimating years of work experience from CV images")
st.caption("PaddleOCR + four models (median, TF-IDF, embeddings, fine-tuned BERT) on the synthetic "
           "PopResume dataset. Course project HW_04.")

tab1, tab2, tab3, tab4 = st.tabs(["Problem & Methodology", "OCR & Exploratory Analysis",
                                  "Model Results", "Interactive Experience Estimator"])

# =========================================================================== #
# TAB 1: Problem & Methodology
# =========================================================================== #
with tab1:
    sp = load_splits()
    prep = read_json(str(MET_DIR / "preprocessing_report.json")) or {}
    info_bert = read_json(str(MET_DIR / "info_bert.json")) or {}

    st.header("Objective")
    st.markdown(
        "Predict a candidate's **total years of work experience** (a number from 0 to "
        f"{CAP}) starting from the **image** of a CV, not from clean text. A recruiter, a job board "
        "or a public employment service that receives thousands of CVs could use the estimate to "
        "pre-fill a profile or to sort a queue for human review."
    )
    st.subheader("Why is experience hard to read from a CV?")
    st.markdown(
        "- **It is rarely written down.** The number has to be *inferred* from job dates, "
        "which are spread over the page in different formats (`2018.05–2020.05`, `2014 - 2018`).\n"
        "- **It is not a sum.** Jobs overlap, there are career gaps, some entries have no dates, "
        "and education years must not be counted as work.\n"
        "- **OCR adds noise.** A misread digit in a year changes the answer.\n"
        "- **Vocabulary is a shortcut.** Words such as *senior* or *manager* correlate with "
        "experience, so a model can look good without actually reading the dates."
    )

    st.header("Dataset, sampling and target")
    n_txt = f"{len(sp):,}" if sp is not None else "up to 4,000"
    st.markdown(
        f"- **Dataset:** PopResume (CC BY 4.0), synthetic CVs rendered as images. Only CVs **without a "
        f"photo** are used. **{n_txt} CVs**, sampled with a fixed seed (42) from 5 occupations: "
        + ", ".join(occ_name(o) for o in OCCUPATIONS) + ".\n"
        f"- **Target:** `exp_year`, clipped to [0, {CAP}] and the same cap for every model"
        + (f" ({prep['n_at_cap']} CVs, {prep['pct_at_cap']:.1f}%, were at or above the cap)."
           if "n_at_cap" in prep else ".") + "\n"
        "- **Masking rule:** phrases that give the answer away (*\"10 years of experience\"*, age, date "
        "of birth) are replaced by `[MASKED_EXP]` / `[MASKED_AGE]`; names, e-mail, phone, address and "
        "URL are removed. Sex, race, age and location are never model inputs."
    )
    if prep:
        st.caption(
            f"Masking affected {prep.get('cvs_with_any_masking', '?')} of {prep.get('n_cvs', '?')} CVs "
            f"(experience phrases: {prep.get('cvs_with_masked_exp', '?')}, age/birth date: "
            f"{prep.get('cvs_with_masked_age', '?')})."
        )

    st.header("OCR setup and reading order")
    st.markdown(
        "- **PaddleOCR (PP-OCRv5)** extracts, for every text line, the text, a confidence score and a "
        "bounding box. The result is cached as one JSON per CV, so OCR runs only once.\n"
        "- **Reading order is rebuilt from the boxes.** PaddleOCR returns lines in detection order, "
        "which scrambles two-column layouts. Lines are grouped into rows by vertical position and, "
        "if the page has two columns (a clear horizontal gap), each column is read top to bottom "
        "before the next.\n"
        "- **Cleaning:** Unicode normalization, unified dashes, repaired line breaks, personal-data "
        "removal and masking. Three text versions are kept to test the effect of cleaning: raw, "
        "cleaned, and cleaned without low-confidence lines."
    )

    st.header("The four models and the split")
    st.markdown(
        "**Split:** 70% train / 15% validation / 15% test, stratified by occupation × experience band "
        "(0-2, 3-5, 6-10, 11-20 years), seed 42. The test set is used once for the final numbers."
    )
    if sp is not None:
        tab = sp.pivot_table(index="occupation", columns="split", values="id", aggfunc="count",
                             fill_value=0)[["train", "val", "test"]]
        tab.index = [occ_name(i) for i in tab.index]
        st.dataframe(tab, width="stretch")
    c = st.columns(4)
    cards = [
        ("median", "Always predicts the **median** of the training CVs. Floor any real model must beat."),
        ("tfidf", "Counts weighted **words and word pairs** (keeping years as tokens), then a linear "
                  "Ridge regression. Sees vocabulary, not structure."),
        ("embeddings", "A **frozen** sentence encoder turns the CV into a vector (long CVs are cut in "
                       "chunks and averaged), then Ridge. Sees meaning, not exact dates."),
        ("bert", "A lightweight **BERT fine-tuned as a regressor** (one numeric output). Reads the "
                 "whole sequence, so it can in principle relate dates to each other."),
    ]
    for col, (m, desc) in zip(c, cards):
        with col:
            st.markdown(f"<span style='color:{MODEL_COLORS[m]}'>■</span> **{MODEL_LABELS[m]}**",
                        unsafe_allow_html=True)
            st.caption(desc)
    if info_bert.get("base_model"):
        st.caption(f"BERT setup: `{info_bert['base_model']}`, long-text strategy `{info_bert.get('strategy')}`, "
                   f"loss `{info_bert.get('loss')}`, max length {info_bert.get('max_len')} tokens.")

    st.header("Limitations")
    st.markdown(
        "- **Synthetic CVs:** generated from templates, cleaner and more regular than real CVs.\n"
        "- **Only 5 occupations**, **U.S.-based statistics** and **English only**: results do not transfer "
        "to other jobs, countries or languages.\n"
        "- **Experience is capped** at 20 years and the labels come from the generator, not from "
        "verified work histories.\n"
        "- **Never a hiring decision.** The estimate may pre-fill a profile or support a human "
        "reviewer; it must not decide who gets an interview (see the fairness check in Tab 3)."
    )

# =========================================================================== #
# TAB 2: OCR & EDA
# =========================================================================== #
with tab2:
    sp = load_splits()
    st.header("Years of experience")
    if sp is None:
        missing("The split files (data/splits/*.csv)")
    else:
        n_cap = int((sp["exp_year"] >= CAP).sum())
        k = st.columns(3)
        k[0].metric("CVs", f"{len(sp):,}")
        k[1].metric("Mean years", f"{sp['exp_year'].mean():.1f}")
        k[2].metric(f"At the cap ({CAP} years)", f"{n_cap} ({100 * n_cap / len(sp):.1f}%)")

        fig = px.histogram(sp, x="exp_year", nbins=CAP + 1, color_discrete_sequence=[NEUTRAL])
        fig.update_traces(xbins=dict(start=-0.5, end=CAP + 0.5, size=1))
        fig.update_layout(xaxis_title="Years of experience", yaxis_title="CVs", bargap=0.08)
        st.plotly_chart(style(fig, 330), width="stretch")
        chart_note("a histogram shows the shape of the target; the pile-up at the right end "
                   "shows how many CVs are clipped at the cap.")

        sp2 = sp.assign(Occupation=sp["occupation"].map(occ_name))
        fig = px.histogram(sp2, x="exp_year", facet_col="Occupation", facet_col_wrap=3,
                           nbins=CAP + 1, color_discrete_sequence=[NEUTRAL])
        fig.update_traces(xbins=dict(start=-0.5, end=CAP + 0.5, size=1))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(bargap=0.08, showlegend=False)
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="")
        st.plotly_chart(style(fig, 520), width="stretch")
        chart_note("one small panel per occupation with a shared scale makes it easy to see that "
                   "occupations differ in how much experience they typically have.")

    st.header("OCR quality")
    q = read_csv(str(TAB_DIR / "ocr_quality_per_cv.csv"), dtype={"id": str})
    qs = read_json(str(MET_DIR / "ocr_quality_summary.json"))
    if q is None:
        missing("The OCR quality table")
    else:
        if qs:
            k = st.columns(4)
            k[0].metric("Mean CER", f"{qs['cer_mean']:.3f}")
            k[1].metric("Mean WER", f"{qs['wer_mean']:.3f}")
            k[2].metric("Years read correctly", f"{100 * qs['years_recall_mean']:.1f}%")
            k[3].metric("Mean OCR confidence", f"{qs['mean_conf_mean']:.3f}")
        c1, c2 = st.columns(2)
        for col, metric, title in ((c1, "cer", "Character error rate (CER)"),
                                   (c2, "wer", "Word error rate (WER)")):
            fig = px.histogram(q, x=metric, nbins=40, color_discrete_sequence=[NEUTRAL])
            fig.update_layout(title=title, xaxis_title="Error rate (lower is better)", yaxis_title="CVs")
            col.plotly_chart(style(fig, 300), width="stretch")
        chart_note("the distributions show whether the average hides a long tail of badly read CVs; "
                   "most mass near 0 means the OCR is reliable, a long right tail means some CVs fail.")

        by = q.assign(Occupation=q["occupation"].map(lambda x: OCC_SHORT.get(x, x))).groupby("Occupation")[
            ["years_recall", "ranges_recall"]].mean().reset_index()
        by = by.melt("Occupation", var_name="measure", value_name="share")
        by["measure"] = by["measure"].map({"years_recall": "Single years read correctly",
                                           "ranges_recall": "Date ranges read correctly"})
        fig = px.bar(by, x="Occupation", y="share", color="measure", barmode="group",
                     color_discrete_sequence=[NEUTRAL, "#9aa7ba"])
        fig.update_layout(yaxis_title="Share of the reference", yaxis_tickformat=".0%", xaxis_title="")
        st.plotly_chart(style(fig, 340), width="stretch")
        chart_note("years and date ranges are what the models actually need, so this is the OCR "
                   "metric that matters most; CER alone can look good while a digit in a year is wrong.")

    st.header("Example: CV image, OCR boxes and extracted text")
    ocr_files = sorted(OCR_DIR.glob("*.json")) if OCR_DIR.exists() else []
    if not ocr_files or not IMG_DIR.exists():
        st.info("The CV images and the OCR cache are not stored in the repo (they are regenerated by the "
                "pipeline). Run `python src/download_data.py` and `python src/ocr_extraction.py` locally to "
                "see a side-by-side example here, or use Tab 4 to upload a CV.")
    else:
        ids = [p.stem for p in ocr_files]
        meta = sp.set_index("id") if sp is not None else None
        cid = st.selectbox("CV", ids[:500], index=0)
        data = json.loads((OCR_DIR / f"{cid}.json").read_text(encoding="utf-8"))
        img_path = IMG_DIR / f"{cid}.png"
        left, right = st.columns(2)
        with left:
            if img_path.exists():
                from PIL import Image, ImageDraw

                img = Image.open(img_path).convert("RGB")
                dr = ImageDraw.Draw(img)
                for n, ln in enumerate(data["lines_ordered"], 1):
                    x1, y1, x2, y2 = ln["box"]
                    col = (235, 104, 52) if ln["score"] < 0.6 else (42, 120, 214)
                    dr.rectangle([x1, y1, x2, y2], outline=col, width=2)
                    dr.text((x1 + 2, max(y1 - 11, 0)), str(n), fill=col)
                st.image(img, caption="Blue: confident lines. Orange: confidence below 0.6. "
                                      "Numbers show the rebuilt reading order.", width="stretch")
            else:
                st.warning("Image not available locally.")
        with right:
            st.metric("Lines / mean confidence", f"{data['n_lines']} / {data['mean_conf']:.3f}")
            st.text_area("Extracted text (reading order)", data["text_ordered"], height=420)
        chart_note("putting the boxes next to the text lets you check by eye that the reading order "
                   "follows the layout and that dates were captured.")

# =========================================================================== #
# TAB 3: Model results
# =========================================================================== #
with tab3:
    comp = read_csv(str(TAB_DIR / "model_comparison.csv"))
    preds = load_test_predictions()
    avail = [m for m in MODEL_ORDER if (MET_DIR / f"predictions_{m}.csv").exists()]

    st.header("Model comparison (test set)")
    if comp is None:
        missing("The model comparison table")
    else:
        show = comp.copy()
        pct = ["Acc ±1", "Acc ±2", "Acc ±3", "Band accuracy"]
        fmt = {c: "{:.2f}" for c in ("MAE (years)", "RMSE (years)", "Band macro-F1", "R²")}
        fmt.update({c: "{:.1%}" for c in pct if c in show})
        for c in ("Training time", "Inference time per CV"):
            if c in show:
                fmt[c] = "{:.3g} s"
        st.dataframe(show.style.format(fmt, na_rep="-"), hide_index=True, width="stretch")
        st.caption("MAE and RMSE in years; Acc ±k = share of CVs predicted within k years; "
                   "bands are 0-2, 3-5, 6-10 and 11-20 years.")

    st.header("Predicted vs true")
    if not avail:
        missing("The prediction files")
    else:
        m = st.radio("Model", avail, format_func=lambda x: MODEL_LABELS[x], horizontal=True, key="sc_model")
        d = preds.dropna(subset=[m])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, CAP], y=[0, CAP], mode="lines", name="Perfect prediction",
                                 line=dict(color=INK_MUTED, dash="dash", width=1.5)))
        fig.add_trace(go.Scatter(x=d["y_true"], y=d[m], mode="markers", name=MODEL_LABELS[m],
                                 marker=dict(color=MODEL_COLORS[m], size=7, opacity=0.55,
                                             line=dict(color="rgba(255,255,255,0.8)", width=0.6)),
                                 customdata=d["id"], hovertemplate="id %{customdata}<br>true %{x:.1f}<br>"
                                                                   "pred %{y:.1f}<extra></extra>"))
        fig.update_layout(xaxis_title="True years", yaxis_title="Predicted years")
        fig.update_xaxes(range=[-0.5, CAP + 0.5], constrain="domain")
        fig.update_yaxes(range=[-0.5, CAP + 0.5])
        st.plotly_chart(style(fig, 460), width="stretch")
        chart_note("points on the dashed line are perfect; a flat cloud means the model mostly predicts "
                   "the average, and a tilt away from the line shows over- or under-estimation at one end.")

    st.header("Error by experience band and by occupation")
    c1, c2 = st.columns(2)
    for col, fname, xt in ((c1, "mae_by_band.csv", "Experience band (years)"),
                           (c2, "mae_by_occupation.csv", "Occupation")):
        t = read_csv(str(TAB_DIR / fname), index_col=0)
        if t is None:
            with col:
                missing(fname)
            continue
        models = [c for c in t.columns if c in LABEL_COLORS]
        long = t[models].reset_index().rename(columns={t.index.name or "index": "group"})
        long = long.rename(columns={long.columns[0]: "group"}).melt("group", var_name="Model", value_name="MAE")
        long["group"] = long["group"].map(lambda x: OCC_SHORT.get(x, x))
        fig = px.bar(long, x="group", y="MAE", color="Model", barmode="group",
                     color_discrete_map=LABEL_COLORS,
                     category_orders={"Model": [MODEL_LABELS[x] for x in MODEL_ORDER]})
        fig.update_layout(xaxis_title=xt, yaxis_title="MAE (years)")
        col.plotly_chart(style(fig, 380), width="stretch")
    chart_note("grouped bars put the four models side by side in each group, so you can see where a "
               "model gains (for example, juniors vs seniors) rather than only its overall average.")

    st.header("Band confusion matrix")
    if avail:
        m2 = st.radio("Model", avail, format_func=lambda x: MODEL_LABELS[x], horizontal=True, key="cm_model")
        d = preds.dropna(subset=[m2])
        tb, pb = to_band(d["y_true"]), to_band(d[m2])
        cm = np.zeros((4, 4), dtype=int)
        np.add.at(cm, (tb, pb), 1)
        rows = cm.sum(1, keepdims=True)
        share = np.divide(cm, rows, out=np.zeros(cm.shape), where=rows > 0)
        text = [[f"{cm[i, j]}<br>{share[i, j]:.0%}" for j in range(4)] for i in range(4)]
        color = MODEL_COLORS[m2]
        fig = go.Figure(go.Heatmap(z=share, x=BAND_LABELS, y=BAND_LABELS, text=text, texttemplate="%{text}",
                                   colorscale=[[0, "rgba(255,255,255,0.0)"], [1, color]], zmin=0, zmax=1,
                                   showscale=False, hovertemplate="true %{y}<br>pred %{x}<extra></extra>"))
        fig.update_layout(xaxis_title="Predicted band", yaxis_title="True band")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        st.plotly_chart(style(fig, 400), width="stretch")
        chart_note("each row sums to 100%: it shows, for CVs of one true band, where the model puts them. "
                   "Mass off the diagonal is misclassification, and neighboring bands are milder errors.")

    st.header("BERT's gain over each baseline")
    gain = read_csv(str(TAB_DIR / "bert_gain.csv"))
    if gain is None:
        missing("The bootstrap gain table")
    else:
        fig = go.Figure(go.Scatter(
            x=gain["baseline"], y=gain["gain (years)"], mode="markers",
            marker=dict(color=MODEL_COLORS["bert"], size=11),
            error_y=dict(type="data", symmetric=False,
                         array=gain["CI95 high"] - gain["gain (years)"],
                         arrayminus=gain["gain (years)"] - gain["CI95 low"], color=MODEL_COLORS["bert"]),
            hovertemplate="%{x}<br>gain %{y:.2f} years<extra></extra>"))
        fig.add_hline(y=0, line_dash="dash", line_color=INK_MUTED)
        fig.update_layout(xaxis_title="Compared with", yaxis_title="MAE reduction (years)")
        st.plotly_chart(style(fig, 360), width="stretch")
        chart_note("a point with its 95% bootstrap interval: if the interval does not cross the dashed "
                   "zero line, the gain is statistically meaningful; negative values mean BERT is worse.")
        st.dataframe(gain.style.format({c: "{:.2f}" for c in gain.columns if gain[c].dtype.kind == "f"}),
                     hide_index=True, width="stretch")

    st.divider()
    st.header("The four analytical questions")

    with st.expander("Q1. What does BERT learn that the baselines cannot?", expanded=True):
        probe = read_csv(str(TAB_DIR / "probe_dates.csv"))
        if probe is not None:
            st.subheader("Probe: shift every date by +5 years on 20 test CVs")
            st.dataframe(probe, hide_index=True, width="stretch")
            st.caption("A model that reads dates should change its prediction when dates move; "
                       "recruiters would expect the years of experience to stay the same if ALL dates move "
                       "together, and to drop if one job is removed (see README for the exact setup).")
        else:
            st.info("The date probe table (output/tables/probe_dates.csv) is not available yet.")
        show_answer("Q1")

    with st.expander("Q2. How much does OCR noise cost?"):
        o = read_csv(str(TAB_DIR / "ocr_vs_reference.csv"))
        if o is None:
            missing("The OCR vs reference table")
        else:
            long = o.melt("Model", value_vars=["MAE OCR text", "MAE reference text"], var_name="Text",
                          value_name="MAE")
            fig = px.bar(long, x="Model", y="MAE", color="Text", barmode="group",
                         color_discrete_map={"MAE OCR text": NEUTRAL, "MAE reference text": "#9aa7ba"})
            fig.update_layout(xaxis_title="", yaxis_title="MAE (years)")
            st.plotly_chart(style(fig, 340), width="stretch")
            chart_note("the same test CVs are scored twice, with OCR text and with the perfect reference "
                       "text; the gap between the bars is the cost of OCR noise for each model.")
            st.dataframe(o.style.format({c: "{:.2f}" for c in o.columns if o[c].dtype.kind == "f"}),
                         hide_index=True, width="stretch")
        show_answer("Q2")

    with st.expander("Q3. Methodological sensitivity"):
        s = read_csv(str(TAB_DIR / "sensitivity.csv"))
        if s is None:
            missing("The sensitivity table")
        else:
            s = s.copy()
            s.columns = ["Setup", "MAE (years)"] + list(s.columns[2:])
            s["family"] = s["Setup"].str.split("_").str[0]
            s = s.sort_values(["family", "MAE (years)"])
            fig = px.bar(s, x="MAE (years)", y="Setup", orientation="h", color="family",
                         color_discrete_map={m: MODEL_COLORS[m] for m in MODEL_ORDER})
            fig.update_layout(yaxis_title="", showlegend=False)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style(fig, 80 + 34 * len(s)), width="stretch")
            chart_note("one bar per training setup, colored by model family, so a change in one choice "
                       "(long-text strategy, digits in TF-IDF, loss, OCR cleaning) is a direct bar comparison.")
        show_answer("Q3")

    with st.expander("Q4. Business value, fairness and ethics"):
        for attr in ("sex", "race"):
            f = read_csv(str(TAB_DIR / "fairness_bert.csv"))
            if f is None:
                missing("The fairness table")
                break
            g = f[f["attribute"] == attr]
            if g.empty:
                continue
            st.subheader(f"By {attr}")
            c1, c2 = st.columns(2)
            fig = px.bar(g, x="group", y="MAE", color_discrete_sequence=[MODEL_COLORS["bert"]], text="n")
            fig.update_traces(texttemplate="n=%{text}", textposition="outside")
            fig.update_layout(xaxis_title="", yaxis_title="MAE (years)")
            c1.plotly_chart(style(fig, 320), width="stretch")
            fig = px.bar(g, x="group", y="mean error (pred - true)", color_discrete_sequence=[MODEL_COLORS["bert"]])
            fig.add_hline(y=0, line_color=INK_MUTED)
            fig.update_layout(xaxis_title="", yaxis_title="Mean error (pred - true), years")
            c2.plotly_chart(style(fig, 320), width="stretch")
        chart_note("MAE tells how large errors are; the mean error tells their direction. A group with a "
                   "consistently positive mean error is systematically over-estimated, which a bar of "
                   "absolute error alone would hide.")
        show_answer("Q4")

    st.header("The 10 largest BERT errors")
    top = read_csv(str(TAB_DIR / "bert_top10_errors.csv"), dtype={"id": str})
    if top is None:
        missing("The top-10 error table")
    else:
        st.dataframe(top.style.format({c: "{:.1f}" for c in top.columns if top[c].dtype.kind == "f"}),
                     hide_index=True, width="stretch")

    sf = read_json(str(MET_DIR / "info_bert.json"))
    if sf:
        st.caption("All numbers come from the test set (15% of the CVs) and a single seed (42).")

# =========================================================================== #
# TAB 4: Interactive estimator
# =========================================================================== #
with tab4:
    st.warning("Demo for a course project. Estimates come from a model trained on synthetic CVs "
               "and must **not** be used to screen or rank real people.")

    # ---- A. Upload and estimate ----
    st.header("Upload a CV")

    @st.cache_resource(show_spinner="Loading the models (first time only)...")
    def get_predictors():
        from inference import load_predictors

        return load_predictors()

    @st.cache_resource(show_spinner="Loading PaddleOCR (first time only)...")
    def get_ocr():
        from inference import load_ocr

        return load_ocr()

    up = st.file_uploader("CV as image (PNG/JPG) or PDF (first page)", type=["png", "jpg", "jpeg", "pdf"])
    pasted = st.text_area("...or paste the CV text instead (skips OCR)", height=100,
                          placeholder="Only needed if PaddleOCR is not installed in this environment.")

    variants, ocr_res, pil = None, None, None
    if up is not None:
        from inference import run_ocr, text_variants, to_pil

        try:
            pil = to_pil(up.getvalue(), up.name)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not open the file: {e}")
        if pil is not None:
            try:
                ocr_res = run_ocr(get_ocr(), pil)
                variants = text_variants(ocr_res)
            except ImportError:
                st.error("PaddleOCR is not installed here. Install `requirements.txt` or paste the text above.")
            except Exception as e:  # noqa: BLE001
                st.error(f"OCR failed: {type(e).__name__}: {e}")
    elif pasted.strip():
        from inference import text_from_pasted

        variants = text_from_pasted(pasted)

    if variants is not None:
        left, right = st.columns(2)
        with left:
            if pil is not None:
                from PIL import ImageDraw

                im = pil.copy()
                dr = ImageDraw.Draw(im)
                for n, ln in enumerate(ocr_res["lines_ordered"], 1):
                    x1, y1, x2, y2 = ln["box"]
                    col = (235, 104, 52) if ln["score"] < 0.6 else (42, 120, 214)
                    dr.rectangle([x1, y1, x2, y2], outline=col, width=2)
                st.image(im, caption="Detected text lines (orange: low confidence)", width="stretch")
        with right:
            if ocr_res:
                st.metric("Lines / mean OCR confidence", f"{ocr_res['n_lines']} / {ocr_res['mean_conf']:.3f}")
            st.text_area("Extracted text (reading order, before cleaning)",
                         ocr_res["text_ordered"] if ocr_res else pasted, height=260)
            st.caption(f"Cleaning masked {variants['n_masked_exp']} experience phrase(s) and "
                       f"{variants['n_masked_age']} age/birth-date phrase(s); personal data was removed "
                       "before prediction. The upload is processed in memory and not stored.")

        predictors, unavailable = get_predictors()
        from inference import predict_all

        res = predict_all(predictors, variants) if predictors else {}
        st.subheader("Predicted years of experience")
        cols = st.columns(len(MODEL_ORDER))
        for col, m in zip(cols, MODEL_ORDER):
            with col:
                st.markdown(f"<span style='color:{MODEL_COLORS[m]}'>■</span> **{MODEL_LABELS[m]}**",
                            unsafe_allow_html=True)
                if m in res:
                    st.metric("years", f"{res[m]:.1f}", label_visibility="collapsed")
                else:
                    st.caption("Not available here (model files are not in the repo).")
        if unavailable:
            with st.expander("Why some models are unavailable"):
                for m, why in unavailable.items():
                    st.write(f"**{MODEL_LABELS[m]}**: {why}")
                st.caption("Model weights are not committed. Re-create them with the scripts in `src/` "
                           "(see README) and place them under `models/`.")
        if res:
            fig = go.Figure(go.Bar(x=[res[m] for m in MODEL_ORDER if m in res],
                                   y=[MODEL_LABELS[m] for m in MODEL_ORDER if m in res], orientation="h",
                                   marker_color=[MODEL_COLORS[m] for m in MODEL_ORDER if m in res],
                                   text=[f"{res[m]:.1f}" for m in MODEL_ORDER if m in res],
                                   textposition="outside"))
            fig.update_xaxes(range=[0, CAP], title="Years")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style(fig, 230), width="stretch")

    # ---- B. Browse test CVs ----
    st.header("Browse test CVs")
    if preds is None:
        missing("The prediction files")
    else:
        have = [m for m in MODEL_ORDER if m in preds.columns]
        ref_model = "bert" if "bert" in have else have[0]
        d = preds.copy()
        d["abs error"] = (d[ref_model] - d["y_true"]).abs()
        c1, c2, c3 = st.columns(3)
        occs = sorted(d["occupation"].dropna().unique())
        sel_occ = c1.multiselect("Occupation", occs, default=occs, format_func=occ_name)
        yr = c2.slider("True years", 0, CAP, (0, CAP))
        emax = float(np.ceil(d["abs error"].max())) if d["abs error"].notna().any() else 10.0
        er = c3.slider(f"{MODEL_LABELS[ref_model]} absolute error (years)", 0.0, max(emax, 1.0),
                       (0.0, max(emax, 1.0)), step=0.5)
        f = d[d["occupation"].isin(sel_occ) & d["y_true"].between(*yr) & d["abs error"].between(*er)]
        out = f.assign(Occupation=f["occupation"].map(occ_name))[
            ["id", "Occupation", "y_true"] + have + ["abs error"]].rename(
            columns={"y_true": "True", **{m: MODEL_LABELS[m] for m in have}})
        st.caption(f"{len(out)} of {len(d)} test CVs match.")
        st.dataframe(out.style.format({c: "{:.1f}" for c in out.columns if out[c].dtype.kind == "f"}),
                     hide_index=True, width="stretch", height=260)

        if len(out):
            pick = st.selectbox("Open a CV", out["id"].tolist())
            r = out[out["id"] == pick].iloc[0]
            texts = load_local_texts()
            a, b = st.columns(2)
            with a:
                img = IMG_DIR / f"{pick}.png"
                if img.exists():
                    st.image(str(img), width="stretch")
                else:
                    st.info("CV image not available locally (images are not stored in the repo).")
            with b:
                st.markdown(f"**True:** {r['True']:.1f} years")
                for m in have:
                    st.markdown(f"<span style='color:{MODEL_COLORS[m]}'>■</span> {MODEL_LABELS[m]}: "
                                f"**{r[MODEL_LABELS[m]]:.1f}**", unsafe_allow_html=True)
                if texts is not None and pick in set(texts["id"]):
                    st.text_area("Cleaned text used by the models",
                                 texts.loc[texts["id"] == pick, "text_clean"].iloc[0], height=260)

    # ---- C. Good and bad examples ----
    st.header("Examples of good and bad predictions")
    if preds is None or "bert" not in preds.columns:
        missing("BERT predictions")
    else:
        d = preds.dropna(subset=["bert"]).copy()
        d["err"] = d["bert"] - d["y_true"]
        d["abs"] = d["err"].abs()
        texts = load_local_texts()
        cause = read_csv(str(TAB_DIR / "bert_top10_errors.csv"), dtype={"id": str})
        cause_col = next((c for c in (cause.columns if cause is not None else []) if c.startswith("cause")), None)

        def explain(row, bad: bool) -> str:
            direction = "over" if row["err"] > 0 else "under"
            base = (f"BERT {direction}estimated by {row['abs']:.1f} years "
                    f"(true {row['y_true']:.1f}, predicted {row['bert']:.1f}).")
            if bad and cause is not None and cause_col and row["id"] in set(cause["id"]):
                v = cause.loc[cause["id"] == row["id"], cause_col].iloc[0]
                if isinstance(v, str) and v.strip():
                    return base + f" Main cause found in the error analysis: {v.strip()}."
            if texts is not None and row["id"] in set(texts["id"]):
                t = texts.loc[texts["id"] == row["id"]].iloc[0]
                return base + f" The CV has {int(t['n_lines'])} OCR lines (mean confidence {t['mean_conf']:.2f})."
            return base

        # good examples spread across true-experience levels: closest prediction in each tercile
        d["tercile"] = pd.qcut(d["y_true"].rank(method="first"), 3, labels=["low", "mid", "high"])
        good = d.sort_values("abs").groupby("tercile", observed=True).head(1).sort_values("y_true")
        bad = d.sort_values("abs", ascending=False).head(3)
        g, bcol = st.columns(2)
        with g:
            st.subheader("Good (one per experience level)")
            for _, r in good.iterrows():
                with st.container(border=True):
                    st.markdown(f"**CV {r['id']}** · {occ_name(r['occupation'])}")
                    st.write(explain(r, False))
        with bcol:
            st.subheader("Bad (three largest errors)")
            for _, r in bad.iterrows():
                with st.container(border=True):
                    st.markdown(f"**CV {r['id']}** · {occ_name(r['occupation'])}")
                    st.write(explain(r, True))
