from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weight(shape: tuple[int, int], stddev: float, init_type: str, device: torch.device) -> torch.Tensor:
    if init_type == "normal":
        return torch.randn(shape, device=device) * stddev
    if init_type == "uniform":
        bound = math.sqrt(3.0) * stddev
        return torch.empty(shape, device=device).uniform_(-bound, bound)
    raise ValueError(f"Unsupported init_type: {init_type}")


class PTNCN(nn.Module):
    """
    PyTorch port of the 2-layer P-TNCN used in the TensorFlow baseline.

    The model keeps explicit latent states, prediction errors, and local targets.
    Weight updates are computed manually through `compute_updates()` rather than
    autograd so the predictive-coding learning rule stays visible and editable.
    """

    def __init__(
        self,
        x_dim: int,
        hid_dim: int,
        *,
        act_fun: str = "tanh",
        out_fun: str = "softmax",
        init_type: str = "normal",
        wght_sd: float = 0.025,
        err_wght_sd: float = 0.025,
        in_dim: int | None = None,
        zeta: float = 1.0,
        fast_steps: int = 2,
        fast_eta: float = 0.01,
        fast_lambda: float = 0.9,
        standardize: bool = False,
        use_temporal_error_rule: bool = False,
        l1_penalty: float = 0.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        device = device or torch.device("cpu")
        self.x_dim = x_dim
        self.hid_dim = hid_dim
        self.in_dim = in_dim if in_dim is not None else x_dim
        self.zeta = zeta
        self.standardize = standardize
        self.use_temporal_error_rule = use_temporal_error_rule
        self.l1_penalty = l1_penalty
        self.fast_steps = fast_steps
        self.fast_eta = fast_eta
        self.fast_lambda = fast_lambda
        self.act_fun = act_fun
        self.out_fun = out_fun
        self.device = device

        if zeta > 0.0:
            self.U1 = nn.Parameter(_init_weight((hid_dim, hid_dim), wght_sd, init_type, device))
        else:
            self.U1 = None
        self.M2 = nn.Parameter(_init_weight((hid_dim, hid_dim), wght_sd, init_type, device))
        self.M1 = nn.Parameter(_init_weight((self.in_dim, hid_dim), wght_sd, init_type, device))
        self.W2 = nn.Parameter(_init_weight((hid_dim, hid_dim), wght_sd, init_type, device))
        self.W1 = nn.Parameter(_init_weight((hid_dim, x_dim), wght_sd, init_type, device))
        self.V2 = nn.Parameter(_init_weight((hid_dim, hid_dim), wght_sd, init_type, device))
        self.V1 = nn.Parameter(_init_weight((hid_dim, hid_dim), wght_sd, init_type, device))
        self.E2 = nn.Parameter(_init_weight((hid_dim, hid_dim), err_wght_sd, init_type, device))
        self.E1 = nn.Parameter(_init_weight((x_dim, hid_dim), err_wght_sd, init_type, device))

        self.reset_state(reset_fast_weights=True)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_fun == "tanh":
            return torch.tanh(x)
        if self.act_fun == "relu":
            return F.relu(x)
        if self.act_fun == "relu6":
            return F.relu6(x)
        if self.act_fun == "sigmoid":
            return torch.sigmoid(x)
        if self.act_fun == "identity":
            return x
        raise ValueError(f"Unsupported activation: {self.act_fun}")

    def _out(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_fun == "softmax":
            return F.softmax(x, dim=-1)
        if self.out_fun == "sigmoid":
            return torch.sigmoid(x)
        if self.out_fun == "tanh":
            return torch.tanh(x)
        if self.out_fun == "identity":
            return x
        raise ValueError(f"Unsupported output activation: {self.out_fun}")

    def parameter_list(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = [self.W1, self.E1, self.W2, self.E2]
        if self.U1 is not None:
            params.append(self.U1)
        params.extend([self.M1, self.M2, self.V1, self.V2])
        return params

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameter_list())

    def collect_params(self) -> dict[str, torch.Tensor]:
        params = {
            "W1": self.W1,
            "E1": self.E1,
            "W2": self.W2,
            "E2": self.E2,
            "M1": self.M1,
            "M2": self.M2,
            "V1": self.V1,
            "V2": self.V2,
        }
        if self.U1 is not None:
            params["U1"] = self.U1
        return params

    def reset_state(self, *, reset_fast_weights: bool = True) -> None:
        self.zf0 = None
        self.zf1 = None
        self.zf2 = None
        self.zf0_tm1 = None
        self.zf1_tm1 = None
        self.zf2_tm1 = None
        self.z1 = None
        self.z2 = None
        self.y1 = None
        self.y2 = None
        self.ex = None
        self.e1 = None
        self.e1v = None
        self.e2v = None
        self.e1v_tm1 = None
        self.e2v_tm1 = None
        self.x_tm1 = None
        self.z_pad = None
        self.x_pad = None
        if reset_fast_weights:
            self.A1 = None
            self.A2 = None

    def _clip_update(self, update: torch.Tensor, radius: float) -> torch.Tensor:
        if radius <= 0.0:
            return update
        norm = torch.linalg.norm(update)
        if norm <= radius:
            return update
        return update * (radius / (norm + 1e-8))

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (x - mean) / std

    def _ensure_state(self, x: torch.Tensor) -> None:
        batch_size = x.size(0)
        device = x.device

        if self.zf1 is not None:
            self.zf0_tm1 = self.x_tm1 if self.x_tm1 is not None else self.zf0
            self.zf1_tm1 = self.y1
            self.zf2_tm1 = self.y2
            self.e1v_tm1 = self.e1v
            self.e2v_tm1 = self.e2v
            return

        self.z_pad = torch.zeros(batch_size, self.hid_dim, device=device)
        self.x_pad = torch.zeros(batch_size, self.in_dim, device=device)
        self.zf0_tm1 = self.x_pad
        self.zf1_tm1 = self.z_pad
        self.zf2_tm1 = self.z_pad
        self.e1v_tm1 = self.z_pad
        self.e2v_tm1 = self.z_pad
        self.z1 = self.z_pad
        self.z2 = self.z_pad
        self.e1 = self.z_pad
        self.ex = torch.zeros(batch_size, self.x_dim, device=device)

    def forward_step(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        beta: float = 0.2,
        alpha: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones(x.size(0), 1, device=x.device)
        mask = mask.to(dtype=x.dtype)
        x = x.to(dtype=torch.float32)

        self._ensure_state(x)
        self.zf0 = x

        if self.zeta > 0.0:
            self.z2 = (self.zf1_tm1 @ self.M2) + (self.zf2_tm1 @ self.V2)
        else:
            self.z2 = self.zf2_tm1 @ self.V2
        if self.standardize:
            self.z2 = self._standardize(self.z2)

        self.zf2 = self._act(self.z2)
        if self.A2 is None:
            self.A2 = torch.zeros(self.hid_dim, self.hid_dim, device=x.device)
        self.A2 = (self.fast_lambda * self.A2) + (self.fast_eta * (self.zf2.transpose(0, 1) @ self.zf2))
        for _ in range(self.fast_steps):
            self.zf2 = self._act(self.z2) + (self.zf2 @ self.A2)

        z1_mu = self.zf2 @ self.W2

        if self.zeta > 0.0 and self.U1 is not None:
            self.z1 = (self.zf0_tm1 @ self.M1) + (self.zf2_tm1 @ self.U1) + (self.zf1_tm1 @ self.V1)
        else:
            self.z1 = (self.zf0_tm1 @ self.M1) + (self.zf1_tm1 @ self.V1)
        if self.standardize:
            self.z1 = self._standardize(self.z1)

        self.zf1 = self._act(self.z1)
        if self.A1 is None:
            self.A1 = torch.zeros(self.hid_dim, self.hid_dim, device=x.device)
        self.A1 = (self.fast_lambda * self.A1) + (self.fast_eta * (self.zf1.transpose(0, 1) @ self.zf1))
        for _ in range(self.fast_steps):
            self.zf1 = self._act(self.z1) + (self.zf1 @ self.A1)

        x_logits = self.zf1 @ self.W1
        x_mu = self._out(x_logits)

        self.e1 = (z1_mu - self.zf1) * mask
        self.ex = (x_mu - self.zf0) * mask

        d2 = self.e1 @ self.E2
        d1 = (self.ex @ self.E1) - (self.e1 * alpha)
        if self.l1_penalty > 0.0:
            d2 = d2 + (torch.sign(self.z2) * self.l1_penalty)
            d1 = d1 + (torch.sign(self.z1) * self.l1_penalty)

        self.y2 = self._act(self.z2 - (d2 * beta))
        self.y1 = self._act(self.z1 - (d1 * beta))
        self.e2v = (self.zf2 - self.y2) * mask
        self.e1v = (self.zf1 - self.y1) * mask
        self.x_tm1 = self.zf0

        return x_logits, x_mu

    def compute_updates(self, *, gamma: float = 1.0, update_radius: float = -1.0) -> list[torch.Tensor]:
        if self.zf1 is None or self.zf2 is None:
            raise RuntimeError("forward_step() must run before compute_updates().")

        updates: list[torch.Tensor] = []

        d_w1 = self._clip_update(self.zf1.transpose(0, 1) @ self.ex, update_radius)
        updates.append(d_w1)

        if self.use_temporal_error_rule:
            d_e1 = (self.ex.transpose(0, 1) @ (self.e1v - self.e1v_tm1)) * -gamma
        else:
            d_e1 = d_w1.transpose(0, 1) * gamma
        updates.append(self._clip_update(d_e1, update_radius))

        d_w2 = self._clip_update(self.zf2.transpose(0, 1) @ self.e1v, update_radius)
        updates.append(d_w2)

        if self.use_temporal_error_rule:
            d_e2 = (self.e1.transpose(0, 1) @ (self.e2v - self.e2v_tm1)) * -gamma
        else:
            d_e2 = d_w2.transpose(0, 1) * gamma
        updates.append(self._clip_update(d_e2, update_radius))

        if self.U1 is not None:
            d_u1 = self._clip_update(self.zf2_tm1.transpose(0, 1) @ self.e1v, update_radius)
            updates.append(d_u1)

        d_m1 = self._clip_update(self.zf0_tm1.transpose(0, 1) @ self.e1v, update_radius)
        d_m2 = self._clip_update(self.zf1_tm1.transpose(0, 1) @ self.e2v, update_radius)
        d_v1 = self._clip_update(self.zf1_tm1.transpose(0, 1) @ self.e1v, update_radius)
        d_v2 = self._clip_update(self.zf2_tm1.transpose(0, 1) @ self.e2v, update_radius)
        updates.extend([d_m1, d_m2, d_v1, d_v2])

        return updates
