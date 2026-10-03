"""Katalog featurow: jeden przebieg po tokenach zbiera top-konteksty DLA WSZYSTKICH
featurow naraz (wektorowo, numpy) i nadaje automatyczna etykiete (dominujacy token
wsrod najmocniejszych aktywacji + czystosc).

Wyjscie:
  results/feature_catalog.jsonl   pelny katalog (layer, feature, label, purity, examples)
  results/feature_catalog.md      top N featurow na warstwe (czytelna tabela)
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
sys.path.insert(0, HERE)
from train_sae import build_model, capture  # noqa: E402
from topk_sae import TopKSAE  # noqa: E402

D_IN, D_SAE, K = 768, 6144, 50


def load_sae(out_dir, layer):
    sae = TopKSAE(D_IN, D_SAE, K)
    sae.load_state_dict(load_file(os.path.join(out_dir, f"res_v5_layer{layer}_final.safetensors")))
    sae.eval()
    return sae


def merge_top(tv, tr, tt, cand_v, cand_r, cand_t, top):
    """Bieznace top-k per featur [F,k] + kandydaci [F,c] -> nowe top-k."""
    cv = np.concatenate([tv, cand_v], axis=1)
    cr = np.concatenate([tr, cand_r], axis=1)
    ct = np.concatenate([tt, cand_t], axis=1)
    order = np.argsort(-cv, axis=1)[:, :top]
    return (np.take_along_axis(cv, order, 1),
            np.take_along_axis(cr, order, 1),
            np.take_along_axis(ct, order, 1))


def batch_candidates(vals, idx, tokids, n_tok, top_c=3):
    """Z [N,K] aktywacji zwraca top-c per featur: [F,c] (val, row, token)."""
    N = vals.shape[0]
    feat = idx.ravel()
    val = vals.ravel()
    row = np.repeat(np.arange(N), K)
    t = np.repeat(tokids, K)
    order = np.lexsort((val, feat))
    feat, val, row, t = feat[order], val[order], row[order], t[order]
    uniq, counts = np.unique(feat, return_counts=True)
    ends = np.cumsum(counts) - 1
    cv = np.full((D_SAE, top_c), -1e9)
    cr = np.full((D_SAE, top_c), -1, dtype=np.int64)
    ct = np.full((D_SAE, top_c), -1, dtype=np.int64)
    for j in range(top_c):
        pos = ends - j
        ok = pos >= ends - counts + 1
        cv[uniq[ok], j] = val[pos[ok]]
        cr[uniq[ok], j] = row[pos[ok]]
        ct[uniq[ok], j] = t[pos[ok]]
    return cv, cr, ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default=os.path.join(HERE, "..", "data", "tokens_wt103.bin"))
    ap.add_argument("--model-dir", default=os.path.join(HERE, "..", "model"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "out"))
    ap.add_argument("--results-dir", default=os.path.join(HERE, "..", "results"))
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--scan-tokens", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--top", type=int, default=5, help="kontekstow na featur")
    ap.add_argument("--md-per-layer", type=int, default=200)
    a = ap.parse_args()
    torch.manual_seed(1337)
    np.random.seed(1337)

    tok = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    model, cfgd = build_model(a.model_dir)
    n_layer = cfgd["n_layer"]
    layers = a.layers if a.layers is not None else list(range(n_layer))
    saes = {L: load_sae(a.out_dir, L) for L in layers}

    tv = {L: np.full((D_SAE, a.top), -1e9) for L in layers}
    tr = {L: np.full((D_SAE, a.top), -1, dtype=np.int64) for L in layers}
    tt = {L: np.full((D_SAE, a.top), -1, dtype=np.int64) for L in layers}
    hits = {L: np.zeros(D_SAE, dtype=np.int64) for L in layers}

    data = np.memmap(a.tokens, dtype=np.uint16, mode="r")
    win = np.lib.stride_tricks.sliding_window_view(data, a.seq)
    n_tok = min(a.scan_tokens, (len(data) - a.seq - 1))
    steps = max(1, n_tok // (a.batch * a.seq))
    for s in range(steps):
        ix = np.random.randint(0, len(win), size=a.batch)
        ids = torch.from_numpy(win[ix].astype(np.int64))
        with torch.no_grad():
            acts = capture(model, ids)
        tokids = ids.numpy().ravel()
        gpos = (ix[:, None] + np.arange(a.seq)[None, :]).ravel()  # data[ix[b] + p], ulozone row-major jak tokids
        for L in layers:
            with torch.no_grad():
                x = acts[L].reshape(-1, D_IN).float()
                vals, idx = saes[L].encode(x)
            cv, cr, ct = batch_candidates(vals.numpy(), idx.numpy(), tokids, x.shape[0])
            cr = np.where(cr >= 0, gpos[np.clip(cr, 0, len(gpos) - 1)], -1)
            tv[L], tr[L], tt[L] = merge_top(tv[L], tr[L], tt[L], cv, cr, ct, a.top)
            hits[L] += np.bincount(idx.numpy().ravel(), minlength=D_SAE)
        if (s + 1) % 5 == 0:
            print(f"skan {s + 1}/{steps}: {(s + 1) * a.batch * a.seq:,} tokenow", flush=True)

    os.makedirs(a.results_dir, exist_ok=True)
    md = ["# Katalog featurow GoLLeM-v5 (res_v5, automatyczne etykiety)",
          "",
          f"Skan {steps * a.batch * a.seq:,} tokenow WikiText; etykieta = dominujacy token",
          "wsrod najmocniejszych aktywacji (czystosc = jego udzial). Pelny katalog:",
          "`feature_catalog.jsonl` (lokalnie, poza git).", ""]
    with open(os.path.join(a.results_dir, "feature_catalog.jsonl"), "w", encoding="utf-8") as out:
        for L in layers:
            rows = []
            for f in range(D_SAE):
                tk = [int(t) for t in tt[L][f] if t >= 0]
                if not tk:
                    continue
                c = Counter(tk)
                label_id, n = c.most_common(1)[0]
                def ctx_of(g):
                    lo, hi = max(0, g - 8), min(len(data), g + 8)
                    pre = tok.decode(data[lo:g].tolist()).replace("\n", " ")
                    mid = tok.decode([int(data[g])])
                    post = tok.decode(data[g + 1 : hi].tolist()).replace("\n", " ")
                    return f"...{pre}[{mid}]{post}..."

                rec = {
                    "layer": L,
                    "feature": f,
                    "label": tok.decode([label_id]),
                    "purity": round(n / len(tk), 2),
                    "hits": int(hits[L][f]),
                    "examples": [
                        {"act": round(float(tv[L][f, j]), 1), "ctx": ctx_of(int(tr[L][f, j]))}
                        for j in range(a.top) if tr[L][f, j] >= 0
                    ],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows.append((float(tv[L][f].max()), f, rec))
            rows.sort(reverse=True)
            md.append(f"## warstwa {L}")
            md.append("")
            md.append("| feature | etykieta | czystosc | aktywacji | max_act |")
            md.append("|---|---|---:|---:|---:|")
            for _, f, rec in rows[: a.md_per_layer]:
                md.append(f"| {f} | {rec['label']!r} | {rec['purity']:.2f} | {rec['hits']} | {max(tv[L][f]):.0f} |")
            md.append("")
    with open(os.path.join(a.results_dir, "feature_catalog.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"KATALOG: results/feature_catalog.jsonl + .md ({len(layers)} warstw)")


if __name__ == "__main__":
    main()
