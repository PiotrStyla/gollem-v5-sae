"""Trening TopK SAE na rezydualach GoLLeM-v5 (16 warstw rownolegle z jednego forwardu).

Koszt: forward modela (2*N FLOP/token) + backward 16 SAE. Kazda warstwa = osobny
SAE (d_in 768 -> d_sae = mult*768, k aktywnych). Checkpointy okresowe + metryki
(FVU, martwe featury, tok/s) do metrics.jsonl. Final: sae/res_v5_layer{L}.safetensors.

UWAGA: tekst/id TYLKO przez pliki (Windows argv niszczy nie-ASCII).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
sys.path.insert(0, HERE)
from train_gpt_ref import GPT  # noqa: E402
from topk_sae import TopKSAE, fvu  # noqa: E402


class Cfg:
    """Adapter: GPT/Block czytaja atrybuty cfg z argparse Namespace."""


def build_model(model_dir):
    cfgd = json.load(open(os.path.join(model_dir, "config.json")))
    cfg = Cfg()
    cfg.norm = cfgd["norm"]
    cfg.norm_eps = cfgd["norm_eps"]
    cfg.pos = cfgd["pos"]
    cfg.rope_theta = cfgd["rope_theta"]
    cfg.ffn = cfgd["ffn"]
    cfg.ffn_mult = cfgd["ffn_mult"]
    cfg.value_residual = cfgd["value_residual"]
    cfg.qk_norm = cfgd["qk_norm"]
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    if "head.weight" in sd and sd["head.weight"].shape != sd["tok.weight"].shape:
        del sd["head.weight"]  # tied z tok.weight; padded vocab przinamy
    vocab, n_embd = sd["tok.weight"].shape
    model = GPT(vocab, cfgd["n_layer"], n_embd, cfgd["n_head"], cfgd["block"], cfg)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfgd


@torch.no_grad()
def capture(model, idx):
    """Rezydualy x po kazdym bloku: lista 16 x [B,T,D]."""
    x = model.tok(idx)
    v0 = None
    out = []
    for blk in model.blocks:
        x, v0 = blk(x, v0)
        out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--model-dir", default=os.path.join(HERE, "..", "model"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "out"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--mult", type=int, default=8, help="ekspansja SAE (d_sae = mult*768)")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = bez limitu tokenow")
    ap.add_argument("--max-hours", type=float, default=0.0, help="0 = bez limitu czasu")
    ap.add_argument("--steps", type=int, default=0, help="limit krokow (smoke/bench)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    log_path = os.path.join(a.out_dir, "metrics.jsonl")

    def log(rec):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    data = np.memmap(a.tokens, dtype=np.uint16, mode="r")
    model, cfgd = build_model(a.model_dir)
    n_layer, n_embd = cfgd["n_layer"], cfgd["n_embd"]
    d_sae = a.mult * n_embd
    print(f"model: L{n_layer} d{n_embd}; tokenow={len(data):,}; SAE {n_embd}->{d_sae} k={a.k}", flush=True)

    saes = [TopKSAE(n_embd, d_sae, a.k) for _ in range(n_layer)]
    opts = [torch.optim.Adam(s.parameters(), lr=a.lr, betas=(0.9, 0.999)) for s in saes]
    fired = [torch.zeros(d_sae, dtype=torch.bool) for _ in range(n_layer)]

    def get_batch():
        ix = np.random.randint(0, len(data) - a.seq - 1, size=a.batch)
        w = np.lib.stride_tricks.sliding_window_view(data, a.seq)
        return torch.from_numpy(w[ix].astype(np.int64))

    # warmup: b_dec = srednia rezydualow (standard TopK init) na ~10 batchy
    means = [torch.zeros(n_embd) for _ in range(n_layer)]
    with torch.no_grad():
        for _ in range(10):
            acts = capture(model, get_batch())
            for L in range(n_layer):
                means[L] += acts[L].reshape(-1, n_embd).float().mean(0)
    for L in range(n_layer):
        with torch.no_grad():
            saes[L].b_dec.copy_(means[L] / 10)
    print("b_dec inicjalizowane srednia rezydualow (10 batchy)", flush=True)

    tok_per_step = a.batch * a.seq
    deadline = time.time() + a.max_hours * 3600 if a.max_hours else None
    t0, done, fvu_run = time.time(), 0, np.zeros(n_layer)
    step = 0
    while True:
        if a.steps and step >= a.steps:
            break
        if a.max_tokens and done >= a.max_tokens:
            break
        if deadline and time.time() >= deadline:
            break
        acts = capture(model, get_batch())
        for L in range(n_layer):
            x = acts[L].reshape(-1, n_embd).float()
            x_hat, vals, idx = saes[L](x)
            loss = (x - x_hat).pow(2).mean()
            opts[L].zero_grad(set_to_none=True)
            loss.backward()
            opts[L].step()
            saes[L].renorm_decoder()
            fvu_run[L] += fvu(x, x_hat)
            fired[L][idx.reshape(-1)] = True
        done += tok_per_step
        step += 1
        if step % a.log_every == 0:
            dt = time.time() - t0
            rec = {
                "step": step,
                "tokens": done,
                "tok_s": round(done / dt, 1),
                "fvu": [round(float(v / a.log_every), 4) for v in fvu_run],
                "dead_frac": [round(float(1 - f.float().mean()), 4) for f in fired],
            }
            print(json.dumps(rec), flush=True)
            log(rec)
            fvu_run[:] = 0
        if step % a.ckpt_every == 0:
            save_ckpt(saes, a.out_dir, cfgd, a, step, done, fired)

    save_ckpt(saes, a.out_dir, cfgd, a, step, done, fired, final=True)
    dt = time.time() - t0
    print(f"KONIEC: {step} krokow, {done:,} tokenow, {done/dt:.1f} tok/s", flush=True)
    log({"final": True, "step": step, "tokens": done, "tok_s": round(done / dt, 1)})


def save_ckpt(saes, out_dir, cfgd, a, step, done, fired, final=False):
    for L, sae in enumerate(saes):
        sd = {k: v.contiguous() for k, v in sae.state_dict().items()}
        save_file(
            sd,
            os.path.join(out_dir, f"res_v5_layer{L}{'_final' if final else f'_step{step}'}.safetensors"),
            metadata={
                "layer": str(L),
                "hook": "res_post_block",
                "d_in": str(sae.d_in),
                "d_sae": str(sae.d_sae),
                "k": str(sae.k),
                "model": "gollem-v5-128m-muon-v1",
                "tokens": str(done),
                "step": str(step),
                "dead_frac": str(round(float(1 - fired[L].float().mean()), 4)),
            },
        )
    cfg_out = {
        "model_id": "gollem-v5-128m-muon-v1",
        "hook": "res_post_block",
        "layers": cfgd["n_layer"],
        "d_in": cfgd["n_embd"],
        "d_sae": a.mult * cfgd["n_embd"],
        "k": a.k,
        "lr": a.lr,
        "batch": a.batch,
        "seq": a.seq,
        "step": step,
        "tokens": done,
    }
    with open(os.path.join(out_dir, "sae_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=1)
    print(f"ckpt: {step} ({done:,} tok)", flush=True)


if __name__ == "__main__":
    main()
