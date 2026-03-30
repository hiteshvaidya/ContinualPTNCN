"""
This file contains components of PTNCN network. One can replace this with other network if needed for benchmarks.

version 1.0
author: Hitesh Vaidya
"""


import torch
import torch.nn as nn
from torch.nn.init import normal_ as normal
import torch.nn.functional as F

class PTNCNCell(nn.Module):
    """
    One layer of PTNCN for one timestep
    """
    def __init__(self,
        input_dim,
        hidden_dim,
        U,
        V,
        M,
        top=False
        ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.U = U
        self.V = V
        self.M = M
        self.top_layer=top
        
        


class PTNCN(nn.Module):
    def __init__(self,
        x_dim: int,                 # vocab/output size
        hid_dim: int,               # hidden layer width
        in_dim: int = -1,           # input width, -1 defaults to x_dim
        wght_sd: float = 0.025,     # weight std for initialization
        err_wght_sd: float = 0.025, # error weight std for initialization
        act_fun: str = "tanh",      # activation function
        out_fun: str = "softmax",   # output function
        zeta: float = 1.0,          # 0.0 disables U1
        fwt_S: int = 2,             # fast weights S value
        fwt_eta: float = 0.01,
        fwt_lambda: float = 0.9,
        device=None
    ):
        super().__init__()
        self.x_dim = x_dim
        self.hid_dim = hid_dim
        self.in_dim = x_dim if in_dim <= 0 else in_dim
        self.wght_sd = wght_sd
        self.err_wght_sd = err_wght_sd
        self.zeta = zeta
        self.fwt_S = fwt_S
        self.fwt_eta = fwt_eta
        self.fwt_lambda = fwt_lambda
        self.device = device or torch.device('cpu')

        # Define parameters
        self.M1 = nn.Parameter(normal(torch.empty(self.in_dim, hid_dim), std=wght_sd))
        self.M2 = nn.Parameter(normal(torch.empty(hid_dim, hid_dim), std=wght_sd))
        self.W1 = nn.Parameter(normal(torch.empty(hid_dim, self.in_dim), std=wght_sd))
        self.W2 = nn.Parameter(normal(torch.empty(hid_dim, hid_dim), std=wght_sd))
        self.V1 = nn.Parameter(normal(torch.empty(hid_dim, hid_dim), std=wght_sd))
        self.V2 = nn.Parameter(normal(torch.empty(hid_dim, hid_dim), std=wght_sd))
        self.E1 = nn.Parameter(normal(torch.empty(self.in_dim, hid_dim), std=err_wght_sd))
        self.E2 = nn.Parameter(normal(torch.empty(hid_dim, hid_dim), std=err_wght_sd))
        if zeta > 0.0:
            self.U1 = nn.Parameter(normal(torch.empty(hid_dim, hid_dim), std=wght_sd))

        self.act_fun = act_fun
        self.out_fun = out_fun
        _act = {'tanh': torch.tanh, 'relu': torch.relu, 'sigmoid': torch.sigmoid}
        _out = {'sigmoid': torch.sigmoid, 
                'softmax': lambda x: torch.softmax(x, dim=-1),
                'tanh': torch.tanh,
                'identity': lambda x: x}
        self.act_fx = _act.get(act_fun, torch.tanh)  # fallback to tanh
        self.out_fx = _out.get(out_fun, lambda x: torch.softmax(x, dim=-1))

        self._clear_state()
        
    
    def _clear_state(self):
        # inputs
        self.zf0, self.zf0_tm1 = None, None

        # layer 1
        self.z1, self.zf1, self.zf1_tm1, self.y1, self.e1y, self.e1y_tm1 = None, None, None, None, None, None

        # layer 2
        self.z2, self.zf2, self.zf2_tm1, self.y2, self.e2y, self.e2y_tm1 = None, None, None, None, None, None

        # errors
        self.e1, self.e0 = None, None

        # fast weight matrices
        self.A1, self.A2 = None, None


    def _init_fast_weights(self):
        self.A1 = torch.zeros(self.hid_dim, self.hid_dim, device=self.device)
        self.A2 = torch.zeros(self.hid_dim, self.hid_dim, device=self.device)

    def forward(self, x, mask, beta=0.1, alpha=0.001, gamma=0.01, lambda_val=0.01):
        B = x.shape[0]
        device = self.M1.device   # always matches wherever parameters live

        if self.zf1 is None:
            # first timestep - no previous state exists yet
            # create zero tensors of the right shape as stand-ins
            pad_h = torch.zeros(B, self.hid_dim, device=device)
            pad_x = torch.zeros(B, self.in_dim, device=device)
            self.zf0_tm1 = pad_x
            self.zf1_tm1 = pad_h
            self.zf2_tm1 = pad_h
            self.e1y_tm1 = pad_h
            self.e2y_tm1 = pad_h
            # also initialise error signals to zero so compute_updates
            # doesn't crash if called before the first LRA pass
            self.z1 = pad_h
            self.z2 = pad_h
            self.e1 = pad_h
            self.e0 = pad_x
        else:
            # shift: use LRA targets as "previous state" for next step
            self.zf0_tm1 = self.zf0
            self.zf1_tm1 = self.y1      # y1_{t-1} — NOT zf1
            self.zf2_tm1 = self.y2      # y2_{t-1} — NOT zf2
            self.e1y_tm1 = self.e1y
            self.e2y_tm1 = self.e2y

        self.zf0 = x   # store current input

        # Layer 2
        self.z2 = self.zf2_tm1 @ self.V2 + self.zf1_tm1 @ self.M2
        h_s = self.act_fx(self.z2)
        if self.A2 is None:
            self.A2 = torch.zeros(self.hid_dim, self.hid_dim, device=device)
        self.A2 = self.fwt_lambda * self.A2 + self.fwt_eta * (h_s.T @ h_s) / B
        for s in range(self.fwt_S):
            h_s = self.act_fx(F.layer_norm(self.z2 + h_s @ self.A2, (self.hid_dim,)))
        self.zf2 = h_s
        
        z1_o_logits = self.zf2 @ self.W2
        z1_o = self.act_fx(z1_o_logits)


        # Layer 1
        u1_term = self.zf2_tm1 @ self.U1 if self.zeta > 0.0 else 0.0
        self.z1 = u1_term + self.zf1_tm1 @ self.V1 + self.zf0_tm1 @ self.M1
        h_s = self.act_fx(self.z1)
        if self.A1 is None:
            self.A1 = torch.zeros(self.hid_dim, self.hid_dim, device=device)
        self.A1 = self.fwt_lambda * self.A1 + self.fwt_eta * (h_s.T @ h_s) / B
        for s in range(self.fwt_S):
            h_s = self.act_fx(F.layer_norm(self.z1 + h_s @ self.A1, (self.hid_dim,)))
        self.zf1 = h_s
        
        z0_o_logits = self.zf1 @ self.W1
        z0_o = self.act_fx(z0_o_logits)

        self.e1 = (z1_o - self.zf1) * mask
        self.e0 = (z0_o - self.zf0) * mask
        d2 = self.e1 @ self.E2
        d1 = self.e0 @ self.E1
        self.y2 = self.act_fx(self.z2 - beta * d2 + lambda_val * torch.sign(self.zf2))
        self.y1 = self.act_fx(self.z1 - beta * d1 + gamma * self.e1 + lambda_val * torch.sign(self.zf1))
        self.e2y = self.zf2 - self.y2
        self.e1y = self.zf1 - self.y1

        return z0_o_logits, z0_o
        
    def compute_updates(self, alpha=0.001, xi=0.4):
        with torch.no_grad():
            self.M2.data -= alpha * self.zf1_tm1.T @ self.e2y
            self.V2.data -= alpha * self.zf2_tm1.T @ self.e2y
            self.M1.data -= alpha * self.zf0_tm1.T @ self.e1y
            self.V1.data -= alpha * self.zf1_tm1.T @ self.e1y
            if self.zeta > 0.0:
                self.U1.data -= alpha * self.zf2_tm1.T @ self.e1y
            mat = self.zf2.T @ self.zf1_tm1
            norm = torch.linalg.norm(mat)
            hebb_factor2 = - mat / norm if norm > 0 else torch.zeros_like(mat)
            self.W2.data -= alpha * (self.zf2.T @ self.e1 + xi * hebb_factor2)
            mat = self.zf1.T @ self.zf0_tm1
            norm = torch.linalg.norm(mat)
            hebb_factor1 = - mat / norm if norm > 0 else torch.zeros_like(mat)
            self.W1.data -= alpha * (self.zf1.T @ self.e0 + xi * hebb_factor1)
            self.E2.data -= alpha * self.e1.T @ (self.e2y - self.e2y_tm1)
            self.E1.data -= alpha * self.e0.T @ (self.e1y - self.e1y_tm1)

class EmbeddingPTNCN(nn.Module):
    def __init__(self,
        vocab_size, 
        emb_dim,
        hid_dim):
        '''
        Embedding layer + PTNCN(x_dim=vocab_size, in_dim=emb_dim, hid_dim=hid_dim)
        '''
        super().__init__()
        self.x_dim = vocab_size
        self.in_dim = emb_dim
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.ptncn = PTNCN(x_dim=vocab_size, hid_dim=hid_dim, in_dim=emb_dim)

    def forward(self, token_ids, mask, **kwargs):
        self.emb_output = self.embedding(token_ids)     # (B, emb_dim)
        _, emb_pred = self.ptncn(self.emb_output, mask, **kwargs)   # emb_pred: (B, emb_dim)
        logits = emb_pred @ self.embedding.weight.T     # (B, vocab_size)
        return logits

    def _clear_state(self):
        self.emb_output = None
        self.ptncn._clear_state()
        

    def compute_updates(self, alpha=0.001, xi=0.4):
        self.ptncn.compute_updates(alpha=alpha, xi=xi)
        