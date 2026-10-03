"""TopK sparse autoencoder (OpenAI TopK recipe, untied, unit-norm decoder rows).

encode: f = TopK(ReLU((x - b_dec) @ W_enc + b_enc)), k aktywnych featurów
decode: x_hat = f_sparse @ W_dec + b_dec
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, k):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.reset_parameters()

    def reset_parameters(self):
        # W_dec: losowe wiersze o normie 1; W_enc = transponowane * 0.2 (standard TopK init)
        w = torch.randn(self.d_sae, self.d_in)
        with torch.no_grad():
            w /= w.norm(dim=1, keepdim=True)
            self.W_dec.copy_(w)
            self.W_enc.copy_(w.t() * 0.2)

    @torch.no_grad()
    def renorm_decoder(self):
        n = self.W_dec.norm(dim=1, keepdim=True)
        self.W_dec.div_(n.clamp_min(1e-8))

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        pre = F.relu(pre)
        vals, idx = pre.topk(self.k, dim=-1)
        return vals, idx

    def decode(self, vals, idx):
        f = torch.zeros(vals.size(0), self.d_sae, device=vals.device, dtype=vals.dtype)
        f.scatter_(-1, idx, vals)
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        vals, idx = self.encode(x)
        x_hat = self.decode(vals, idx)
        return x_hat, vals, idx


def fvu(x, x_hat):
    """Fraction of variance unexplained (1.0 = brak rekonstrukcji, 0.0 = perfekt)."""
    err = (x - x_hat).pow(2).sum()
    tot = (x - x.mean(dim=0, keepdim=True)).pow(2).sum()
    return (err / tot.clamp_min(1e-12)).item()
