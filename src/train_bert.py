"""Etapa 5: ajuste fino de un modelo tipo BERT como REGRESOR.

Diseño
- Modelo ligero (por defecto distilbert-base-uncased) con una sola salida:
  AutoModelForSequenceClassification(num_labels=1, problem_type="regression").
- Objetivo escalado: y / CAP (CAP = 20). Las predicciones se devuelven a años y se
  recortan a [0, CAP].
- Pérdida: MSE (por defecto) o Huber (--loss huber, delta en años con --huber-delta).
- CVs largos (> max_len tokens), estrategia --strategy:
    head      : se trunca al inicio (conserva solo los primeros max_len tokens)
    headtail  : primeros --head-tokens + últimos tokens (conserva inicio y final del CV)
    chunks    : se parte en ventanas de max_len; cada ventana se entrena con la etiqueta
                del CV y la predicción final es el PROMEDIO de las ventanas
  El script informa qué % de los CVs excede el límite y qué % de los años (19xx/20xx)
  sobrevive a la estrategia elegida: si se pierden fechas, el modelo no puede leerlas.
- Parada temprana sobre el MAE de validación (paciencia --patience); se restaura el
  mejor modelo. El test no se usa para decidir nada.
- Predice val, test y también test con el texto de REFERENCIA (Pregunta 2).

Uso:
    python src/train_bert.py                                   # modelo principal -> "bert"
    python src/train_bert.py --loss huber --name bert_huber    # Pregunta 3
    python src/train_bert.py --strategy headtail --name bert_headtail
    python src/train_bert.py --limit 64 --epochs 1             # prueba rápida (humo)
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (CAP, MODELS_DIR, SEED, clip_years, ensure_dirs, load_reference_clean,  # noqa: E402
                   load_split, mae, save_predictions, set_seed)

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


# --------------------------------------------------------------------------- #
# Tokenización y estrategias para textos largos
# --------------------------------------------------------------------------- #
def add_special(tok, ids):
    """[CLS] ids [SEP] sin depender de métodos que cambian entre versiones de transformers."""
    start = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    end = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    return [start] + list(ids) + [end]


def encode_doc(tok, text: str, max_len: int, strategy: str, head_tokens: int):
    """Devuelve (lista de secuencias de ids con tokens especiales, ids conservados)."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    room = max_len - 2   # [CLS] + [SEP] (también <s> + </s> en RoBERTa)
    if len(ids) <= room:
        return [add_special(tok, ids)], ids
    if strategy == "head":
        kept = ids[:room]
        return [add_special(tok, kept)], kept
    if strategy == "headtail":
        h = min(head_tokens, room // 2)
        kept = ids[:h] + ids[-(room - h):]
        return [add_special(tok, kept)], kept
    # chunks
    seqs = [add_special(tok, ids[i:i + room]) for i in range(0, len(ids), room)]
    return seqs, ids


def build_items(tok, texts, args):
    items, owner = [], []
    for k, t in enumerate(texts):
        seqs, _ = encode_doc(tok, t, args.max_len, args.strategy, args.head_tokens)
        items.extend(seqs)
        owner.extend([k] * len(seqs))
    return items, np.asarray(owner)


def truncation_report(tok, texts, args) -> dict:
    """% de CVs que exceden el límite y % de años que sobreviven a la estrategia."""
    room = args.max_len - tok.num_special_tokens_to_add(pair=False)
    over, y_total, y_kept = 0, 0, 0
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        over += len(ids) > room
        _, kept = encode_doc(tok, t, args.max_len, args.strategy, args.head_tokens)
        y_total += len(YEAR.findall(t))
        y_kept += len(YEAR.findall(tok.decode(kept)))
    return {"pct_cvs_over_limit": 100 * over / max(len(texts), 1),
            "pct_years_retained": 100 * y_kept / max(y_total, 1)}


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def make_loader(torch, items, labels, pad_id, batch_size, shuffle):
    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            return items[i], labels[i]

    def collate(batch):
        L = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for r, (x, _) in enumerate(batch):
            ids[r, :len(x)] = torch.tensor(x)
            mask[r, :len(x)] = 1
        return ids, mask, torch.tensor([y for _, y in batch], dtype=torch.float32)

    return torch.utils.data.DataLoader(DS(), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def predict(torch, model, items, owner, n_docs, pad_id, batch_size, device, use_amp) -> np.ndarray:
    """Predicción por CV en años (promedia las ventanas de un mismo CV)."""
    model.eval()
    loader = make_loader(torch, items, np.zeros(len(items), dtype=np.float32), pad_id, batch_size, False)
    outs = []
    with torch.no_grad():
        for ids, mask, _ in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                o = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            outs.append(o.float().squeeze(-1).cpu().numpy().reshape(-1))
    flat = np.concatenate(outs) if outs else np.zeros(0)
    sums = np.zeros(n_docs)
    np.add.at(sums, owner, flat)
    counts = np.bincount(owner, minlength=n_docs)
    return clip_years(sums / np.maximum(counts, 1) * CAP, CAP)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--text-col", default="text_clean",
                    choices=["text_raw", "text_clean", "text_clean_lowconf"])
    ap.add_argument("--strategy", default="headtail", choices=["head", "headtail", "chunks"])
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--head-tokens", type=int, default=128)
    ap.add_argument("--loss", default="mse", choices=["mse", "huber"])
    ap.add_argument("--huber-delta", type=float, default=2.0, help="delta de Huber, en años")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--name", default="bert")
    ap.add_argument("--limit", type=int, default=None, help="usa solo N CVs por split (prueba rápida)")
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    ensure_dirs()
    set_seed(args.seed)
    name = args.name + ("_smoke" if args.limit else "")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    hardware = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"Dispositivo: {hardware} | modelo: {args.model} | estrategia: {args.strategy} | pérdida: {args.loss}")

    train, val, test = (load_split(n, args.text_col) for n in ("train", "val", "test"))
    if args.limit:
        train, val, test = train.head(args.limit), val.head(args.limit), test.head(args.limit)
    ref_test = load_reference_clean(test["id"])

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.model_max_length = int(1e9)   # evita el aviso de longitud; truncamos nosotros
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, problem_type="regression").to(device)

    rep = truncation_report(tok, list(train["text"]), args)
    print(f"CVs que exceden {args.max_len} tokens: {rep['pct_cvs_over_limit']:.1f}% | "
          f"años que sobreviven a '{args.strategy}': {rep['pct_years_retained']:.1f}%")

    tr_items, tr_owner = build_items(tok, list(train["text"]), args)
    va_items, va_owner = build_items(tok, list(val["text"]), args)
    y_tr = (train["exp_year"].values / CAP).astype(np.float32)[tr_owner]
    loader = make_loader(torch, tr_items, y_tr, tok.pad_token_id, args.batch_size, True)

    loss_fn = (torch.nn.MSELoss() if args.loss == "mse"
               else torch.nn.HuberLoss(delta=args.huber_delta / CAP))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = max(1, args.epochs * len(loader))
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_mae, best_epoch, best_state, bad = np.inf, 0, None, 0
    t_train = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for ids, mask, y in loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(input_ids=ids, attention_mask=mask).logits.squeeze(-1)
            loss = loss_fn(out.float().reshape(-1), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        p_val = predict(torch, model, va_items, va_owner, len(val), tok.pad_token_id,
                        args.batch_size * 2, device, use_amp)
        v = mae(val["exp_year"], p_val)
        print(f"época {epoch}: pérdida train {run / len(loader):.5f} | MAE val {v:.3f}")
        if v < best_mae - 1e-4:
            best_mae, best_epoch, bad = v, epoch, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Parada temprana (mejor época: {best_epoch}, MAE val {best_mae:.3f})")
                break
    train_time = time.perf_counter() - t_train
    model.load_state_dict(best_state)

    # --- Predicciones finales ---
    def predict_texts(texts):
        items, owner = build_items(tok, texts, args)
        return predict(torch, model, items, owner, len(texts), tok.pad_token_id,
                       args.batch_size * 2, device, use_amp)

    p_val = predict_texts(list(val["text"]))
    t0 = time.perf_counter()
    p_test = predict_texts(list(test["text"]))
    infer_time = (time.perf_counter() - t0) / len(test)
    p_ref = predict_texts(ref_test)

    out_dir = MODELS_DIR / "bert" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    info = {
        "model": name, "base_model": args.model, "text_col": args.text_col,
        "strategy": args.strategy, "max_len": args.max_len, "head_tokens": args.head_tokens,
        "loss": args.loss, "huber_delta_years": args.huber_delta if args.loss == "huber" else None,
        "learning_rate": args.lr, "batch_size": args.batch_size, "epochs_max": args.epochs,
        "best_epoch": best_epoch, "patience": args.patience, "seed": args.seed,
        "hardware": hardware, "val_mae": best_mae,
        "test_mae": mae(test["exp_year"], p_test), "test_mae_reference": mae(test["exp_year"], p_ref),
        "train_time_s": train_time, "inference_time_per_cv_s": infer_time,
        "n_parameters": int(sum(p.numel() for p in model.parameters())), **rep,
    }
    save_predictions(name, {"val": (val, p_val, None), "test": (test, p_test, p_ref)}, info)
    print(f"[{name}] MAE val={best_mae:.3f}  MAE test={info['test_mae']:.3f}  "
          f"(con texto de referencia: {info['test_mae_reference']:.3f})  | {train_time / 60:.1f} min")


if __name__ == "__main__":
    main()
