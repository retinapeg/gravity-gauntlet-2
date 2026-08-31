"""The policy network (plan.md 3.1), written in numpy.

    o (d) -> [input gate: o' = o * sigmoid(g)]        <- M1 necessity probe
          -> Linear(d, 64)  -> tanh
          -> Linear(64, 64) -> tanh
          -> Linear(64, |A|)                          logits for the Categorical
          -> [value head Linear(64, 1)]               baseline, plan.md 3.2 step 5

Why numpy and not torch: torch has no wheel for this dev machine (Intel Mac, py3.13),
and the worker only ever needs a forward pass. Keeping the *whole* net in numpy means
the controller's gradient step and the sandbox's forward pass run the identical
arithmetic, so there is no training/serving skew to debug. See requirements-train.txt.

This module lives in `gg/` - baked into the snapshot AND imported by the controller -
so the architecture has exactly one definition, the same discipline as `spec.py`.
"""
from __future__ import annotations

import base64
import io

import numpy as np

HIDDEN = (64, 64)
N_ACTIONS = 9            # 0 = coast, 1..8 = compass burns (physics.action_to_thrust)
GATE_INIT = 2.0          # sigmoid(2.0) = 0.88: gates start mostly open, L1 closes them
PARAM_KEYS = ("g", "W1", "b1", "W2", "b2", "W3", "b3", "Wv", "bv")


def init_params(in_dim: int, seed: int = 0, n_actions: int = N_ACTIONS) -> dict:
    """Glorot-ish init for tanh, with a deliberately tiny final layer so generation 0
    is near-uniform over actions - the entropy has to be *lost* by learning, not
    fought for from a peaked start."""
    rng = np.random.default_rng(seed)
    h1, h2 = HIDDEN

    def glorot(fan_in, fan_out):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

    return {
        "g":  np.full(in_dim, GATE_INIT, dtype=np.float32),
        "W1": glorot(in_dim, h1), "b1": np.zeros(h1, dtype=np.float32),
        "W2": glorot(h1, h2),     "b2": np.zeros(h2, dtype=np.float32),
        "W3": (0.01 * glorot(h2, n_actions)).astype(np.float32),
        "b3": np.zeros(n_actions, dtype=np.float32),
        "Wv": (0.01 * glorot(h2, 1)).astype(np.float32),
        "bv": np.zeros(1, dtype=np.float32),
    }


def n_params(p: dict) -> int:
    return int(sum(v.size for v in p.values()))


# ------------------------------------------------------------------ forward

def forward(p: dict, o: np.ndarray) -> dict:
    """o: (N, d) -> cache of every intermediate the backward pass needs."""
    o = np.atleast_2d(np.asarray(o, dtype=np.float32))
    gate = _sigmoid(p["g"])
    x = o * gate
    z1 = x @ p["W1"] + p["b1"]; h1 = np.tanh(z1)
    z2 = h1 @ p["W2"] + p["b2"]; h2 = np.tanh(z2)
    logits = h2 @ p["W3"] + p["b3"]
    value = (h2 @ p["Wv"] + p["bv"]).ravel()
    return {"o": o, "gate": gate, "x": x, "h1": h1, "h2": h2,
            "logits": logits, "value": value}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(z))


def act_sample(p: dict, obs: np.ndarray, rng: np.random.Generator):
    """One stochastic action for one observation (S4 - on from Stage A, plan.md 2.6)."""
    logits = forward(p, obs)["logits"][0]
    logp = log_softmax(logits)
    a = int(rng.choice(len(logp), p=np.exp(logp)))
    return a, float(logp[a])


def act_greedy(p: dict, obs: np.ndarray):
    """Argmax action - for deterministic evaluation of a champion."""
    logits = forward(p, obs)["logits"][0]
    return int(np.argmax(logits)), float(log_softmax(logits)[int(np.argmax(logits))])


# ------------------------------------------------------- (de)serialisation

def pack(p: dict) -> str:
    """Weights -> base64 npz, small enough to ride in the GG_INPUT env var
    (~29 KB of float32 at d=38, well under the ~100 KB limit in plan.md 4.5)."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: p[k] for k in PARAM_KEYS})
    return base64.b64encode(buf.getvalue()).decode()


def unpack(b64: str) -> dict:
    with np.load(io.BytesIO(base64.b64decode(b64))) as z:
        return {k: z[k].astype(np.float32) for k in PARAM_KEYS}


# ----------------------------------------------------------------- backward

def loss_and_grads(p: dict, obs, actions, adv, beta: float, gate_l1: float = 0.0,
                   value_targets=None, value_coef: float = 0.0):
    """Gradient of the entropy-regularised REINFORCE loss (plan.md A4):

        L = -(1/N) sum_t [ A_t log pi(a_t|o_t) + beta H(pi(.|o_t)) ]
            + gate_l1 * ||sigma(g)||_1

    `||sigma(g)||_1` is just `sum(sigma(g))` because sigma > 0. Returns
    (loss, grads, stats). Verified against finite differences in
    tests/test_reinforce.py - do not "tidy" the algebra without re-running it.
    """
    obs = np.atleast_2d(np.asarray(obs, dtype=np.float32))
    actions = np.asarray(actions, dtype=int)
    adv = np.asarray(adv, dtype=np.float32)
    n = obs.shape[0]

    c = forward(p, obs)
    logp_all = log_softmax(c["logits"])
    prob = np.exp(logp_all)
    logp_a = logp_all[np.arange(n), actions]
    entropy = -(prob * logp_all).sum(axis=1)                  # (N,)

    onehot = np.zeros_like(prob)
    onehot[np.arange(n), actions] = 1.0

    # d/dlogits of  -(A logp)/N  and of  -(beta H)/N
    dz = (-adv[:, None] * (onehot - prob)
          + beta * prob * (logp_all + entropy[:, None])) / n

    dh2 = dz @ p["W3"].T
    g_W3 = c["h2"].T @ dz
    g_b3 = dz.sum(axis=0)

    g_Wv = np.zeros_like(p["Wv"])
    g_bv = np.zeros_like(p["bv"])
    value_mse = 0.0
    if value_coef and value_targets is not None:
        vt = np.asarray(value_targets, dtype=np.float32)
        resid = c["value"] - vt
        value_mse = float(np.mean(resid ** 2))
        dv = (2.0 * value_coef / n) * resid
        g_Wv = c["h2"].T @ dv[:, None]
        g_bv = np.array([dv.sum()], dtype=np.float32)
        dh2 = dh2 + dv[:, None] @ p["Wv"].T

    dz2 = dh2 * (1.0 - c["h2"] ** 2)
    g_W2 = c["h1"].T @ dz2
    g_b2 = dz2.sum(axis=0)

    dz1 = (dz2 @ p["W2"].T) * (1.0 - c["h1"] ** 2)
    g_W1 = c["x"].T @ dz1
    g_b1 = dz1.sum(axis=0)

    # x = o * sigma(g)  ->  dL/dg = (dL/dx . o) * sigma'(g)
    s = c["gate"]
    dsig = s * (1.0 - s)
    g_g = ((dz1 @ p["W1"].T) * c["o"]).sum(axis=0) * dsig + gate_l1 * dsig

    loss = float(-(adv * logp_a).mean() - beta * entropy.mean()
                 + gate_l1 * float(s.sum()) + value_coef * value_mse)
    grads = {"g": g_g.astype(np.float32),
             "W1": g_W1.astype(np.float32), "b1": g_b1.astype(np.float32),
             "W2": g_W2.astype(np.float32), "b2": g_b2.astype(np.float32),
             "W3": g_W3.astype(np.float32), "b3": g_b3.astype(np.float32),
             "Wv": g_Wv.astype(np.float32), "bv": g_bv.astype(np.float32)}
    stats = {"entropy": float(entropy.mean()), "value_mse": value_mse,
             "gate_mean": float(s.mean())}
    return loss, grads, stats


def global_norm(grads: dict) -> float:
    return float(np.sqrt(sum(float((v ** 2).sum()) for v in grads.values())))


def clip_grads(grads: dict, max_norm: float = 1.0) -> float:
    """Grad-norm clip (plan.md 3.3). Returns the pre-clip norm."""
    norm = global_norm(grads)
    if norm > max_norm and norm > 0.0:
        scale = max_norm / norm
        for k in grads:
            grads[k] *= scale
    return norm


class Adam:
    """Adam (plan.md 3.3: lr 3e-4). Kept here so the optimiser state can be
    checkpointed next to the weights it belongs to."""

    def __init__(self, params: dict, lr: float = 3e-4, betas=(0.9, 0.999), eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for k, gr in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gr
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (gr ** 2)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= (self.lr * mhat / (np.sqrt(vhat) + self.eps)).astype(np.float32)
