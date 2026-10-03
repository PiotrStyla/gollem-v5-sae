"""Polowanie na featury petli powtorzen (repetition loop) + test ablacji.

Pass 1: generacja greedy z promptow, flagowanie pozycji "token juz byl w oknie"
        (petla) i ranking featurow: (srednia aktywacja na powtorce - poza) / std,
        osobno dla kazdej warstwy.
Pass 2: ablacja top-kandydatow (odjecie kierunku W_dec[f] w warstwie L),
        pomiar wspolczynnika powtorzen przed/po.
"""
import argparse
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
from demo import encode_text, generate, fmt  # noqa: E402
from topk_sae import TopKSAE  # noqa: E402

D_IN, D_SAE, K = 768, 6144, 50

PROMPTS = [
    "The capital of France is",
    "In a shocking finding, scientists",
    "The meaning of life is",
    "Once upon a time, there was a",
    "The quick brown fox",
    "My name is John and I am",
]


def rep_flags(gen_ids, window=32):
    """True = token pojawil sie wczesniej w oknie = powtorzenie."""
    out = np.zeros(len(gen_ids), dtype=bool)
    for i, t in enumerate(gen_ids):
        out[i] = t in gen_ids[max(0, i - window) : i]
    return out


def load_sae(out_dir, layer):
    sae = TopKSAE(D_IN, D_SAE, K)
    sae.load_state_dict(load_file(os.path.join(out_dir, f"res_v5_layer{layer}_final.safetensors")))
    sae.eval()
    return sae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(HERE, "..", "model"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "out"))
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--top", type=int, default=15, help="ile kandydatow w rankingu")
    ap.add_argument("--ablate", type=int, default=6, help="ilu kandydatow testujemy")
    ap.add_argument("--scale", type=float, default=0.8, help="od ile x srednia aktywacje odjac kierunek")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="domyslnie wszystkie")
    a = ap.parse_args()
    torch.manual_seed(1337)
    np.random.seed(1337)

    tok = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    model, cfgd = build_model(a.model_dir)
    n_layer = cfgd["n_layer"]
    layers = a.layers or list(range(n_layer))

    # statystyki: per (layer, feature): suma/n dla powtorzen i reszty + suma do wagi
    sR = np.zeros((n_layer, D_SAE))
    nR = np.zeros((n_layer, D_SAE))
    sN = np.zeros((n_layer, D_SAE))
    nN = np.zeros((n_layer, D_SAE))

    saes = {L: load_sae(a.out_dir, L) for L in layers}
    gens = []
    for prompt in PROMPTS:
        ids = encode_text(tok, prompt)
        gen = generate(model, ids.clone(), a.gen)
        full = torch.tensor([ids[0].tolist() + gen], dtype=torch.long)
        flags = rep_flags(gen)
        with torch.no_grad():
            acts = capture(model, full)
        plen = ids.shape[1]
        for L in layers:
            sae = saes[L]
            with torch.no_grad():
                # pozycje plen-1 .. plen+gen-2 przewiduja tokeny gen[0..gen-1]
                x = acts[L][0, plen - 1 : plen + a.gen - 1].float()
                vals, idx = sae.encode(x)
            for r in range(x.shape[0]):
                fs, vs = idx[r].numpy(), vals[r].numpy()
                if flags[r]:
                    sR[L, fs] += vs; nR[L, fs] += 1
                else:
                    sN[L, fs] += vs; nN[L, fs] += 1
        gens.append((prompt, gen, flags))
        print(f"pass1: {fmt(prompt)} rep_rate={flags.mean():.2f}", flush=True)

    meanR = sR / np.maximum(nR, 1)
    meanN = sN / np.maximum(nN, 1)
    score = (meanR - meanN) * np.sqrt(np.minimum(nR, nN))  # roznica * pewnosc
    score[nR < 20] = -1e9
    flat = score.ravel()
    order = np.argsort(-flat)[: a.top]
    print("--- ranking kandydatow na featur petli (layer, feature, score, meanAct@rep, meanAct@nonrep):")
    cands = []
    for o in order:
        L, f = divmod(int(o), D_SAE)
        if L not in layers:
            continue
        cands.append((L, int(f), meanR[L, f]))
        print(f"  L{L:2d} f{f:5d}  score={flat[o]:10.1f}  rep={meanR[L, f]:8.1f}  non={meanN[L, f]:8.1f}")

    base_rates = [float(fl.mean()) for _, _, fl in gens]
    print(f"--- baseline rep_rate srednio: {np.mean(base_rates):.3f}")
    print("--- test ablacji (odjecie kierunku):")
    for L, f, wmean in cands[: a.ablate]:
        sae = load_sae(a.out_dir, L)
        direction = -a.scale * wmean * sae.W_dec[f].clone()
        rates, sample = [], None
        for prompt, _, _ in gens:
            ids = encode_text(tok, prompt)
            gen = generate(model, ids.clone(), a.gen, injects=[(L, direction)])
            flags = rep_flags(gen)
            rates.append(float(flags.mean()))
            if sample is None:
                sample = tok.decode(gen)
        delta = np.mean(rates) - np.mean(base_rates)
        print(f"  L{L:2d} f{f:5d}: rep_rate {np.mean(base_rates):.3f} -> {np.mean(rates):.3f} ({delta:+.3f})")
        print(f"     sample: {fmt(sample[:160])}")


if __name__ == "__main__":
    main()
