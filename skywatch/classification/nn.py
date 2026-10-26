"""Small dependency-free neural network (NumPy) used by the learned models.

A fully-connected network with ReLU hidden layers, trained with Adam and
weight decay. Supports a softmax/cross-entropy head (classification, with
temperature calibration) and a masked mean-squared-error head (regression of
future trajectories, where some horizons may be unavailable).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class MLP:
    def __init__(self, sizes: list[int], seed: int = 0, task: str = "classification") -> None:
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.task = task
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.x_mean = np.zeros(sizes[0])
        self.x_std = np.ones(sizes[0])
        self.temperature = 1.0

    # --------------------------------------------------------------- forward
    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        acts = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        return acts, h

    def _norm(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self._forward(self._norm(np.atleast_2d(X)))[1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.logits(X) / self.temperature
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.logits(X)

    # ----------------------------------------------------------------- train
    def fit(self, X: np.ndarray, Y: np.ndarray, mask: Optional[np.ndarray] = None, epochs: int = 60, lr: float = 2e-3,
            batch: int = 256, weight_decay: float = 1e-4, sample_weight: Optional[np.ndarray] = None,
            seed: int = 0, label_smoothing: float = 0.05, verbose: bool = False) -> list[float]:
        rng = np.random.default_rng(seed)
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-6
        Xn = self._norm(X)
        n = len(X)
        if sample_weight is None:
            sample_weight = np.ones(n)
        m = [np.zeros_like(W) for W in self.W] + [np.zeros_like(b) for b in self.b]
        v = [np.zeros_like(p) for p in m]
        step = 0
        b1, b2 = 0.9, 0.999
        losses = []
        for ep in range(epochs):
            perm = rng.permutation(n)
            ep_loss = 0.0
            lr_ep = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)))  # cosine schedule
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                xb, yb, wb = Xn[idx], Y[idx], sample_weight[idx]
                acts, out = self._forward(xb)
                if self.task == "classification":
                    k = out.shape[1]
                    z = out - out.max(axis=1, keepdims=True)
                    p = np.exp(z)
                    p /= p.sum(axis=1, keepdims=True)
                    target = np.eye(k)[yb] * (1 - label_smoothing) + label_smoothing / k
                    loss = -np.sum(wb[:, None] * target * np.log(p + 1e-12)) / wb.sum()
                    grad = wb[:, None] * (p - target) / wb.sum()
                else:
                    mb = np.ones_like(out) if mask is None else mask[idx]
                    diff = (out - yb) * mb
                    denom = max(mb.sum(), 1.0)
                    # Huber loss: robust to occasional bad targets
                    a = np.abs(diff)
                    hub = np.where(a < 1.0, 0.5 * diff**2, a - 0.5)
                    loss = float((wb[:, None] * hub).sum() / denom)
                    grad = wb[:, None] * np.clip(diff, -1.0, 1.0) / denom
                ep_loss += float(loss) * len(idx)
                # backward
                grads_W, grads_b = [], []
                g = grad
                for i in range(len(self.W) - 1, -1, -1):
                    grads_W.insert(0, acts[i].T @ g + weight_decay * self.W[i])
                    grads_b.insert(0, g.sum(axis=0))
                    if i > 0:
                        g = (g @ self.W[i].T) * (acts[i] > 0)
                step += 1
                params = self.W + self.b
                grads = grads_W + grads_b
                for j, (pp, gg) in enumerate(zip(params, grads)):
                    m[j] = b1 * m[j] + (1 - b1) * gg
                    v[j] = b2 * v[j] + (1 - b2) * gg * gg
                    mh = m[j] / (1 - b1**step)
                    vh = v[j] / (1 - b2**step)
                    pp -= lr_ep * mh / (np.sqrt(vh) + 1e-8)
            losses.append(ep_loss / n)
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:3d} loss {losses[-1]:.4f}")
        return losses

    def calibrate_temperature(self, X: np.ndarray, y: np.ndarray) -> float:
        """Temperature scaling on held-out data (minimise NLL)."""
        z = self.logits(X)
        best_t, best_nll = 1.0, np.inf
        for t in np.linspace(0.5, 4.0, 36):
            zz = z / t
            zz = zz - zz.max(axis=1, keepdims=True)
            lp = zz - np.log(np.exp(zz).sum(axis=1, keepdims=True))
            nll = -lp[np.arange(len(y)), y].mean()
            if nll < best_nll:
                best_t, best_nll = float(t), float(nll)
        self.temperature = best_t
        return best_t

    # ------------------------------------------------------------- persist
    def state_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        d = {f"{prefix}sizes": np.array(self.sizes), f"{prefix}x_mean": self.x_mean, f"{prefix}x_std": self.x_std,
             f"{prefix}temperature": np.array(self.temperature), f"{prefix}task": np.array(self.task)}
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            d[f"{prefix}W{i}"] = W
            d[f"{prefix}b{i}"] = b
        return d

    @staticmethod
    def from_state(d, prefix: str = "") -> "MLP":
        sizes = [int(s) for s in d[f"{prefix}sizes"]]
        net = MLP(sizes, task=str(d[f"{prefix}task"]))
        net.x_mean = np.asarray(d[f"{prefix}x_mean"])
        net.x_std = np.asarray(d[f"{prefix}x_std"])
        net.temperature = float(d[f"{prefix}temperature"])
        net.W = [np.asarray(d[f"{prefix}W{i}"]) for i in range(len(sizes) - 1)]
        net.b = [np.asarray(d[f"{prefix}b{i}"]) for i in range(len(sizes) - 1)]
        return net
