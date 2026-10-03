"""Zbiera finalne wagi 16 SAE + metryki do release/ gotowego pod hf upload.

Wejscie: out/res_v5_layer{L}_final.safetensors, out/sae_config.json, out/metrics.jsonl
Wyjscie: release/<nazwa>/{res_v5_layer*.safetensors, topk_sae.py, sae_config.json, README.md}
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(HERE, "..", "out"))
    ap.add_argument("--release-dir", default=os.path.join(HERE, "..", "release", "gollem-v5-128m-sae-res-v5"))
    ap.add_argument("--data-note", default="WikiText-2 + WikiText-103 raw (Wikipedia)")
    a = ap.parse_args()

    os.makedirs(a.release_dir, exist_ok=True)
    cfg = json.load(open(os.path.join(a.in_dir, "sae_config.json")))
    lines = [json.loads(l) for l in open(os.path.join(a.in_dir, "metrics.jsonl"), encoding="utf-8")]
    final = lines[-1]
    last_logged = next(r for r in reversed(lines) if "fvu" in r)

    files = sorted(
        (f for f in os.listdir(a.in_dir) if f.startswith("res_v5_layer") and f.endswith("_final.safetensors")),
        key=lambda s: int(s.split("layer")[1].split("_")[0]),
    )
    assert len(files) == cfg["layers"], f"brak warstw: {len(files)} != {cfg['layers']}"

    sums = {}
    for f in files + ["sae_config.json"]:
        shutil.copy2(os.path.join(a.in_dir, f), os.path.join(a.release_dir, f))
        sums[f] = sha256(os.path.join(a.release_dir, f))
    shutil.copy2(os.path.join(HERE, "topk_sae.py"), os.path.join(a.release_dir, "topk_sae.py"))

    fvu = last_logged["fvu"]
    dead = last_logged["dead_frac"]
    rows = "\n".join(
        f"| {i} | {fvu[i]:.4f} | {dead[i]:.4f} |" for i in range(cfg["layers"])
    )
    readme = f"""# GoLLeM-v5-128M SAE set - residual stream (res_v5)

TopK sparse autoencoders trained on the post-block residual stream of
[gollem-v5-128m-muon-v1](https://huggingface.co/Maggio33/GoLLeM-v5-128M-Muon-v1)
(Neuronpedia model id `gollem-v5-128m-muon-v1`), one SAE per layer, all {cfg['layers']} layers.

## Architecture

- TopK SAE (untied encoder/decoder, unit-norm decoder rows): `f = TopK(ReLU((x - b_dec) @ W_enc + b_enc))`, `x_hat = f_sparse @ W_dec + b_dec`
- d_in {cfg['d_in']} -> d_sae {cfg['d_sae']} ({cfg['d_sae'] // cfg['d_in']}x expansion), k = {cfg['k']} active features
- Hook site: residual stream `x` after each transformer block (`res_post_block`), layer L = block index 0..{cfg['layers'] - 1}
- File `res_v5_layer{{L}}_final.safetensors`: `W_enc` [{cfg['d_in']}, {cfg['d_sae']}], `b_enc` [{cfg['d_sae']}], `W_dec` [{cfg['d_sae']}, {cfg['d_in']}], `b_dec` [{cfg['d_in']}] (all fp32)

## Training

- Data: {a.data_note}; tokenized with the model's BPE tokenizer (12,288 vocab)
- Tokens seen: {final['tokens']:,} ({final['step']} steps x batch {cfg['batch']} x seq {cfg['seq']})
- Adam lr {cfg['lr']}, decoder row renormalization after each step; `b_dec` init = mean residual
- Date: {datetime.date.today().isoformat()}

## Metrics (last logged step {last_logged['step']}, FVU = fraction of variance unexplained)

| layer | FVU | dead frac |
|---|---:|---:|
{rows}

## Usage

```python
from safetensors.torch import load_file
from topk_sae import TopKSAE

sae = TopKSAE(d_in={cfg['d_in']}, d_sae={cfg['d_sae']}, k={cfg['k']})
sae.load_state_dict(load_file("res_v5_layer8_final.safetensors"))
vals, idx = sae.encode(x)   # x: [tokens, {cfg['d_in']}] post-block residuals
x_hat = sae.decode(vals, idx)
```

## Files

| file | sha256 |
|---|---|
{chr(10).join(f"| {f} | `{s}` |" for f, s in sums.items())}

## License and attribution

Base model: no single license (`gollem-v5-see-training-data`), see its model card. Training data:
WikiText (Wikipedia) - CC BY-SA 4.0 / GFDL. These SAE weights are derived artifacts of the base
model activations; any use must respect the base model and WikiText terms.
"""
    with open(os.path.join(a.release_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"RELEASE: {a.release_dir} ({len(files)} warstw + README + config)")
    print(f"tokens: {final['tokens']:,}, sredni FVU: {sum(fvu) / len(fvu):.4f}, martwe (srednio): {sum(dead) / len(dead):.4f}")


if __name__ == "__main__":
    main()
