"""
REFERENCE IMPLEMENTATION — do not import or run directly.
This is the complete PTNCN implementation for reference while
writing network.py interactively.

Based on: Ororbia and Mali, 2019 IEEE TNNLS
"""

from __future__ import annotations
import torch
import torch.nn as nn
from layers_torch import normal


class PTNCN(nn.Module):

    def __init__(
        self,
        x_dim:        int,
        hid_dim:      int,
        in_dim:       int   = -1,
        wght_sd:      float = 0.025,
        err_wght_sd:  float = 0.025,
        act_fun:      str   = "tanh",
        out_fun:      str   = "softmax",
        zeta:         float = 1.0,
        fwt_S:        int   = 2,
        fwt_eta:      float = 0.01,
        fwt_lambda:   float = 0.9,
        device:       torch.device | None = None,
    ):
        super().__init__()

        self.x_dim      = x_dim
        self.hid_dim    = hid_dim
        self.in_dim     = x_dim if in_dim <= 0 else in_dim
        self.zeta       = zeta
        self.act_fun    = act_fun
        self.fwt_S      = fwt_S
        self.fwt_eta    = fwt_eta
        self.fwt_lambda = fwt_lambda
        self.device     = device or torch.device("cpu")

        # Bottom-up temporal
        self.M1 = nn.Parameter(normal(torch.empty(self.in_dim, hid_dim), std=wght_sd))
        self.M2 = nn.Parameter(normal(torch.empty(hid_dim,    hid_dim), std=wght_sd))

        # Lateral temporal
        self.V1 = nn.Parameter(normal(torch.empty(hid_dim, hid_dim), std=wght_sd))
        self.V2 = nn.Parameter(normal(torch.empty(hid_dim, hid_dim), std=wght_sd))

        # Top-down temporal (layer 1 only)
        if zeta > 0.0:
            self.U1 = nn.Parameter(normal(torch.empty(hid_dim, hid_dim), std=wght_sd))

        # Top-down prediction
        self.W2 = nn.Parameter(normal(torch.empty(hid_dim, hid_dim), std=wght_sd))
        self.W1 = nn.Parameter(normal(torch.empty(hid_dim, x_dim),   std=wght_sd))

        # Error feedback
        self.E1 = nn.Parameter(normal(torch.empty(x_dim,   hid_dim), std=err_wght_sd))
        self.E2 = nn.Parameter(normal(torch.empty(hid_dim, hid_dim), std=err_wght_sd))

        _act = {"tanh": torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}
        self.act_fx = _act.get(act_fun, torch.tanh)

        _out = {
            "softmax":  lambda x: torch.softmax(x, dim=-1),
            "sigmoid":  torch.sigmoid,
            "tanh":     torch.tanh,
            "identity": lambda x: x,
        }
        self.out_fx = _out.get(out_fun, lambda x: torch.softmax(x, dim=-1))

        self._clear_state()

    def _clear_state(self):
        self.zf0 = None;     self.zf0_tm1 = None
        self.z1  = None;     self.zf1     = None
        self.zf1_tm1 = None; self.y1      = None
        self.e1v = None;     self.e1v_tm1 = None
        self.z2  = None;     self.zf2     = None
        self.zf2_tm1 = None; self.y2      = None
        self.e2v = None;     self.e2v_tm1 = None
        self.e1  = None;     self.ex      = None
        self.A1  = None;     self.A2      = None

    def forward(self, x, mask, beta=0.1, alpha=0.001):
        B = x.shape[0]

        if self.zf1 is None:
            pad_h = torch.zeros(B, self.hid_dim, device=self.device)
            pad_x = torch.zeros(B, self.in_dim,  device=self.device)
            self.zf0_tm1 = pad_x; self.zf1_tm1 = pad_h
            self.zf2_tm1 = pad_h; self.e1v_tm1 = pad_h
            self.e2v_tm1 = pad_h; self.z1 = pad_h
            self.z2 = pad_h;      self.e1 = pad_h
            self.ex = pad_x
        else:
            self.zf0_tm1 = self.zf0
            self.zf1_tm1 = self.y1      # use target, not actual
            self.zf2_tm1 = self.y2
            self.e1v_tm1 = self.e1v
            self.e2v_tm1 = self.e2v

        self.zf0 = x

        # Layer 2
        self.z2  = self.zf1_tm1 @ self.M2 + self.zf2_tm1 @ self.V2
        self.zf2 = self.act_fx(self.z2)
        if self.A2 is None:
            self.A2 = torch.zeros(self.hid_dim, self.hid_dim, device=self.device)
        self.A2 = self.fwt_lambda * self.A2 + self.fwt_eta * (self.zf2.T @ self.zf2)
        for _ in range(self.fwt_S):
            self.zf2 = self.act_fx(self.z2) + self.zf2 @ self.A2
        z1_mu = self.zf2 @ self.W2

        # Layer 1
        self.z1 = self.zf0_tm1 @ self.M1 + self.zf1_tm1 @ self.V1
        if self.zeta > 0.0:
            self.z1 = self.z1 + self.zf2_tm1 @ self.U1
        self.zf1 = self.act_fx(self.z1)
        if self.A1 is None:
            self.A1 = torch.zeros(self.hid_dim, self.hid_dim, device=self.device)
        self.A1 = self.fwt_lambda * self.A1 + self.fwt_eta * (self.zf1.T @ self.zf1)
        for _ in range(self.fwt_S):
            self.zf1 = self.act_fx(self.z1) + self.zf1 @ self.A1

        x_logits = self.zf1 @ self.W1
        x_mu     = self.out_fx(x_logits)

        self.e1 = (z1_mu - self.zf1) * mask
        self.ex = (x_mu  - self.zf0) * mask
        d2 = self.e1 @ self.E2
        d1 = self.ex @ self.E1 - self.e1 * alpha
        self.y2 = self.act_fx(self.z2 - beta * d2)
        self.y1 = self.act_fx(self.z1 - beta * d1)
        self.e2v = (self.zf2 - self.y2) * mask
        self.e1v = (self.zf1 - self.y1) * mask

        return x_logits, x_mu

    def compute_updates(self, gamma=1.0, update_radius=-1.0):
        def clip(dW):
            return torch.clamp(dW, -update_radius, update_radius) if update_radius > 0 else dW

        dW1 = clip(self.zf1.T @ self.ex)
        dE1 = clip(dW1.T * gamma)
        dW2 = clip(self.zf2.T @ self.e1v)
        dE2 = clip(dW2.T * gamma)
        if self.zeta > 0.0:
            dU1 = clip(self.zf2_tm1.T @ self.e1v)
        dM1 = clip(self.zf0_tm1.T @ self.e1v)
        dM2 = clip(self.zf1_tm1.T @ self.e2v)
        dV1 = clip(self.zf1_tm1.T @ self.e1v)
        dV2 = clip(self.zf2_tm1.T @ self.e2v)

        deltas = [dW1, dE1, dW2, dE2, dM1, dM2, dV1, dV2]
        if self.zeta > 0.0:
            deltas.insert(4, dU1)
        return deltas

    def parameter_list(self):
        params = [self.W1, self.E1, self.W2, self.E2, self.M1, self.M2, self.V1, self.V2]
        if self.zeta > 0.0:
            params.insert(4, self.U1)
        return params

    def clear_state(self):
        self._clear_state()

    def num_parameters(self):
        return sum(p.numel() for p in self.parameter_list())


class EmbeddingPTNCN(nn.Module):

    def __init__(self, vocab_size, emb_dim, hid_dim, *, wght_sd=0.025,
                 err_wght_sd=0.025, act_fun="tanh", out_fun="softmax",
                 zeta=1.0, fwt_S=2, fwt_eta=0.01, fwt_lambda=0.9, device=None):
        super().__init__()
        self.device  = device or torch.device("cpu")
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, max_norm=1.0).to(self.device)
        self.ptncn = PTNCN(x_dim=vocab_size, hid_dim=hid_dim, in_dim=emb_dim,
                           wght_sd=wght_sd, err_wght_sd=err_wght_sd,
                           act_fun=act_fun, out_fun=out_fun, zeta=zeta,
                           fwt_S=fwt_S, fwt_eta=fwt_eta, fwt_lambda=fwt_lambda,
                           device=self.device).to(self.device)

    def forward(self, token_ids, mask, beta=0.1, alpha=0.001):
        safe_ids = token_ids.clamp_min(0)
        x_emb    = self.embedding(safe_ids) * mask
        return self.ptncn.forward(x_emb, mask, beta=beta, alpha=alpha)

    def compute_updates(self, **kwargs):
        return self.ptncn.compute_updates(**kwargs)

    def embedding_update(self, token_ids, lr=0.01):
        if self.ptncn.e1v is None:
            return
        with torch.no_grad():
            d_emb    = self.ptncn.e1v @ self.ptncn.M1.T
            safe_ids = token_ids.clamp_min(0)
            self.embedding.weight.index_add_(0, safe_ids, -lr * d_emb)

    def parameter_list(self):
        return self.ptncn.parameter_list()

    def clear_state(self):
        self.ptncn.clear_state()

    def num_parameters(self):
        return self.ptncn.num_parameters() + self.embedding.weight.numel()
