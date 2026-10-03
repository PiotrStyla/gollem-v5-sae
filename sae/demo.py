"""Lokalne demo SAE na GoLLeM-v5: probe / find / steer.

- probe  --text "..."             swieczace featury dla zdania (warstwa L)
- find   --feature N              top przykladow z korpusu, gdzie featur N swieci
- steer  --text "..." --feature N pociskanie featura: kierunek W_dec[N] dodany
                                 do rezyduali w warstwie L -> zmiana odpowiedzi

Wagi: out/res_v5_layer{L}_final.safetensors. Tekst przez argv TYLKO ASCII
(Windows argv niszczy nie-ASCII).
"""
import argparse
import heapq
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
sys.path.insert(0, HERE)
from train_sae import build_model, capture  # noqa: E402
from topk_sae import TopKSAE  # noqa: E402

D_IN, D_SAE, K, REAL_VOCAB = 768, 6144, 50, 12288


def load_sae(out_dir, layer):
    sae = TopKSAE(D_IN, D_SAE, K)
    sae.load_state_dict(load_file(os.path.join(out_dir, f"res_v5_layer{layer}_final.safetensors")))
    sae.eval()
    return sae


def encode_text(tok, text):
    return torch.tensor([tok.encode(text).ids], dtype=torch.long)


def fmt(t):
    return repr(t.replace("\n", " "))


def cmd_probe(args, model, tok):
    ids = encode_text(tok, args.text)
    sae = load_sae(args.out_dir, args.layer)
    with torch.no_grad():
        x = capture(model, ids)[args.layer][0].float()
        vals, idx = sae.encode(x)
    print(f"--- probe, warstwa {args.layer}, tekst: {fmt(args.text)}")
    best = {}
    for pos, tid in enumerate(ids[0].tolist()):
        t = tok.decode([tid])
        tops = sorted(zip(vals[pos].tolist(), idx[pos].tolist()), reverse=True)[: args.top]
        print(f"[{pos:3d}] {fmt(t):28s} " + "  ".join(f"f{f}({v:.2f})" for v, f in tops))
        for v, f in tops:
            best[f] = max(best.get(f, 0.0), v)
    print("--- najsilniejsze featury calego tekstu:")
    for f, v in sorted(best.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"  f{f}  max_act={v:.2f}")


def cmd_find(args, model, tok):
    sae = load_sae(args.out_dir, args.layer)
    data = np.memmap(args.tokens, dtype=np.uint16, mode="r")
    win = np.lib.stride_tricks.sliding_window_view(data, args.seq)
    heap = []
    n_tok = min(args.scan_tokens, (len(data) - args.seq - 1))
    steps = max(1, n_tok // (args.batch * args.seq))
    fired = 0
    for s in range(steps):
        ix = np.random.randint(0, len(win), size=args.batch)
        ids = torch.from_numpy(win[ix].astype(np.int64))
        with torch.no_grad():
            x = capture(model, ids)[args.layer].reshape(-1, D_IN).float()
            vals, idx = sae.encode(x)
        hit = idx == args.feature
        rows = hit.any(dim=1)
        fired += int(rows.sum())
        if rows.any():
            v = vals[hit]
            ridx = rows.nonzero().flatten()
            for val, r in zip(v.tolist(), ridx.tolist()):
                pos = r % args.seq
                ctx_ids = ids[r // args.seq].tolist()
                ctx = tok.decode(ctx_ids[max(0, pos - 10) : pos + 10])
                mark = fmt(tok.decode([ctx_ids[pos]]))
                item = (val, f"...{fmt(ctx)}...  <token {mark}>")
                if len(heap) < args.examples:
                    heapq.heappush(heap, item)
                elif val > heap[0][0]:
                    heapq.heapreplace(heap, item)
    print(f"--- featur {args.feature} @ warstwa {args.layer}: {steps * args.batch * args.seq:,} tokenow przeskanowanych, aktywacji: {fired}")
    if not heap:
        print("  featur nie aktywowal sie w tym zakresie")
        return
    for val, ctx in sorted(heap, reverse=True):
        print(f"  act={val:8.2f}  {ctx}")


@torch.no_grad()
def generate(model, ids, steps, inject_layer=None, direction=None, temp=0.0):
    out = []
    for _ in range(steps):
        x = model.tok(ids)
        v0 = None
        for i, blk in enumerate(model.blocks):
            x, v0 = blk(x, v0)
            if inject_layer is not None and i == inject_layer:
                x = x + direction
        logits = model.head(model.lnf(x))[:, -1, :REAL_VOCAB]
        if temp > 0:
            probs = torch.softmax(logits / temp, dim=-1)
            nid = int(torch.multinomial(probs, 1)[0])
        else:
            nid = int(logits.argmax(-1))
        out.append(nid)
        ids = torch.cat([ids, torch.tensor([[nid]])], dim=1)
    return out


def cmd_steer(args, model, tok):
    sae = load_sae(args.out_dir, args.layer)
    ids = encode_text(tok, args.text)
    direction = sae.W_dec[args.feature].clone() * args.weight
    base = generate(model, ids.clone(), args.tokens, temp=args.temp)
    st = generate(model, ids.clone(), args.tokens, inject_layer=args.layer, direction=direction, temp=args.temp)
    print(f"--- steer: featur {args.feature} @ warstwa {args.layer}, waga {args.weight}")
    print(f"BASE   : {args.text}" + tok.decode(base))
    print(f"STEERED: {args.text}" + tok.decode(st))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(HERE, "..", "model"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "out"))
    ap.add_argument("--tokens", default=os.path.join(HERE, "..", "data", "tokens_wt103.bin"))
    ap.add_argument("--layer", type=int, default=8)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe")
    p.add_argument("--text", required=True)
    p.add_argument("--top", type=int, default=3)
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("find")
    p.add_argument("--feature", type=int, required=True)
    p.add_argument("--scan-tokens", type=int, default=400_000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--examples", type=int, default=8)
    p.set_defaults(fn=cmd_find)

    p = sub.add_parser("steer")
    p.add_argument("--text", required=True)
    p.add_argument("--feature", type=int, required=True)
    p.add_argument("--weight", type=float, default=15.0)
    p.add_argument("--tokens", type=int, default=20)
    p.add_argument("--temp", type=float, default=0.0)
    p.set_defaults(fn=cmd_steer)

    args = ap.parse_args()
    torch.manual_seed(1337)
    np.random.seed(1337)
    tok = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    model, _ = build_model(args.model_dir)
    args.fn(args, model, tok)


if __name__ == "__main__":
    main()
