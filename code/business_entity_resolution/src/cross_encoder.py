"""Transformer cross-encoder: a fine-tuned multilingual encoder that reads both records' raw text.

The pair ("<S1 name> | <S1 address>", "<candidate name> | <candidate address>") is encoded jointly,
so attention compares the two records token by token: native scripts (Telugu, Devanagari, ...)
against their romanised spelling, house numbers, garbled names with an identical address. The
tree models only see hand-made similarity scores; this model's logit becomes one more feature
for them (see stack.py).

Trained on the candidates of train sample C (disjoint from the stage-1 sample A and the matcher
sample B), so its scores on sample B are out-of-sample, just like the test scores.
Base model: intfloat/multilingual-e5-small (MIT license, 118M parameters). Uses a CUDA GPU when
there is one; on CPU, bfloat16 autocast still helps on AVX-512/AMX machines.
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd

from config import CEConfig, SEED


def load_texts(paths, ids) -> pd.Series:
    """entity_id -> "name | address" (raw strings) for the given ids, from the normalised caches."""
    need = pd.Index(pd.unique(np.asarray(ids, dtype=object)))
    parts = []
    for path in paths:
        df = pd.read_parquet(path, columns=["entity_id", "business_name", "business_address"])
        df = df[df["entity_id"].isin(need)]
        parts.append(pd.Series((df["business_name"] + " | " + df["business_address"]).to_numpy(object),
                               index=df["entity_id"].to_numpy(object)))
    return pd.concat(parts)


def pair_texts(pairs: pd.DataFrame, texts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    pos = texts.index.get_indexer
    t = texts.to_numpy(object)
    a, b = pos(pairs["s1_id"].to_numpy(object)), pos(pairs["cand_id"].to_numpy(object))
    if (a < 0).any() or (b < 0).any():
        raise ValueError(f"{(a < 0).sum()} S1 / {(b < 0).sum()} candidate ids without text")
    return t[a], t[b]


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _half(device):
    """Autocast dtype: bfloat16, except float16 on CUDA GPUs older than Ampere (T4, P100, V100),
    which have no native bfloat16."""
    import torch
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        return torch.float16
    return torch.bfloat16


def _autocast(device):
    """Half-precision autocast on CUDA (see _half), and bfloat16 on CPUs with native bf16
    (AMX / AVX512-BF16, e.g. EC2 m7i); elsewhere bf16 is emulated and slower than float32."""
    import torch
    if device.type == "cpu":
        cpu = torch.cpu
        native = getattr(cpu, "_is_amx_tile_supported", lambda: False)() or             getattr(cpu, "_is_avx512_bf16_supported", lambda: False)()
        if not native:
            return nullcontext()
    return torch.autocast(device_type=device.type, dtype=_half(device))


def _lengths(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.fromiter((len(x) + len(y) for x, y in zip(a, b)), dtype=np.int32, count=len(a))


def _batches(n: int, lengths: np.ndarray, bs: int, rng) -> list[np.ndarray]:
    """Shuffled batches of similar length (less padding): sort within random mega-chunks."""
    idx = rng.permutation(n)
    out = []
    mega = bs * 64
    for s in range(0, n, mega):
        m = idx[s:s + mega]
        m = m[np.argsort(lengths[m], kind="stable")]
        out += [m[i:i + bs] for i in range(0, len(m), bs)]
    rng.shuffle(out)
    return out


def train(a: np.ndarray, b: np.ndarray, y: np.ndarray, out_dir: Path, cfg: CEConfig) -> dict:
    """Fine-tune cfg.model as a one-logit pair classifier; saves model + tokenizer to out_dir."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = _device()
    tok = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model, num_labels=1).to(device)
    # the 250k-token word-embedding matrix is ~80% of the parameters: frozen, it needs no optimizer
    # state (~1.5 GB) and no dense update per step, and the pretrained multilingual vocabulary
    # stays intact for the scripts that sample C rarely shows
    model.get_input_embeddings().weight.requires_grad_(False)

    n = len(y)
    perm = rng.permutation(n)
    n_val = max(1, int(n * cfg.val_frac))
    va, tr = perm[:n_val], perm[n_val:]
    if cfg.max_train_pairs and len(tr) > cfg.max_train_pairs:
        tr = tr[:cfg.max_train_pairs]
    lengths = _lengths(a, b)
    y_t = torch.tensor(y, dtype=torch.float32)

    batches = _batches(len(tr), lengths[tr], cfg.batch_size, rng)
    steps = len(batches) * cfg.epochs
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=cfg.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup * steps), steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # float16 needs loss scaling; with bfloat16 / float32 the scaler is a no-op
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and _half(device) == torch.float16)
    print(f"  cross-encoder {cfg.model} on {device}: {len(tr):,} train / {len(va):,} val pairs, "
          f"{steps:,} steps", flush=True)

    model.train()
    t0, step, run = time.perf_counter(), 0, 0.0
    for ep in range(cfg.epochs):
        if ep:
            batches = _batches(len(tr), lengths[tr], cfg.batch_size, rng)
        for bi in batches:
            rows = tr[bi]
            enc = tok(list(a[rows]), list(b[rows]), truncation="longest_first", max_length=cfg.max_len,
                      padding=True, return_tensors="pt").to(device)
            with _autocast(device):
                logit = model(**enc).logits.squeeze(-1)
            loss = loss_fn(logit.float(), y_t[rows].to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 500 == 0 or step == steps:
                el = time.perf_counter() - t0
                print(f"    step {step:,}/{steps:,}  loss {run:.4f}  {step * cfg.batch_size / el:,.0f} pairs/s  "
                      f"eta {el / step * (steps - step) / 60:.0f} min", flush=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    logit = score(out_dir, a[va], b[va], cfg, model=(model, tok))
    from sklearn.metrics import log_loss, roc_auc_score
    p = 1 / (1 + np.exp(-logit.astype(np.float64)))
    rep = {"val_pairs": int(len(va)), "val_logloss": float(log_loss(y[va], p, labels=[0, 1])),
           "val_auc": float(roc_auc_score(y[va], p)) if 0 < y[va].sum() < len(va) else float("nan"),
           "val_acc": float(((p > 0.5) == y[va]).mean())}
    print(f"  cross-encoder held-out: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                   for k, v in rep.items()), flush=True)
    return rep


def score(model_dir: Path, a: np.ndarray, b: np.ndarray, cfg: CEConfig, model=None,
          log_every: int = 100_000) -> np.ndarray:
    """Logit for every (a[i], b[i]) text pair; pairs are batched by length, order is restored."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = _device()
    if model is None:
        tok = AutoTokenizer.from_pretrained(model_dir)
        net = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    else:
        net, tok = model
    net.eval()
    n = len(a)
    out = np.empty(n, np.float32)
    order = np.argsort(_lengths(a, b), kind="stable")
    bs = cfg.score_batch_size
    t0, done, next_log = time.perf_counter(), 0, log_every
    with torch.inference_mode(), _autocast(device):
        for s in range(0, n, bs):
            rows = order[s:s + bs]
            enc = tok(list(a[rows]), list(b[rows]), truncation="longest_first", max_length=cfg.max_len,
                      padding=True, return_tensors="pt").to(device)
            out[rows] = net(**enc).logits.squeeze(-1).float().cpu().numpy()
            done += len(rows)
            if done >= next_log:
                el = time.perf_counter() - t0
                print(f"    scored {done:,}/{n:,}  {done / el:,.0f} pairs/s  "
                      f"eta {el / done * (n - done) / 60:.0f} min", flush=True)
                next_log += log_every
    if n:
        print(f"    scored {n:,} pairs at {n / (time.perf_counter() - t0):,.0f} pairs/s", flush=True)
    if model is not None:
        net.train()
    return out


def band_mask(prob: np.ndarray, band) -> np.ndarray:
    """Pairs to score when only the uncertain ones are sent to the cross-encoder (CPU fallback)."""
    if not band:
        return np.ones(len(prob), bool)
    lo, hi = band
    return (prob >= lo) & (prob <= hi)
