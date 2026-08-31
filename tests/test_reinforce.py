"""The learning half of Stage A: the network, its gradient, and the RL bookkeeping.

The gradient is hand-written numpy (gg/net.py), so the finite-difference check here
is the only thing standing between a subtly wrong derivative and a training run that
looks plausible and learns nothing. Run it after ANY edit to `loss_and_grads`.
"""
from __future__ import annotations

import numpy as np
import pytest

from controller.train import discounted
from gg import net, spec


def _fixture(d=11, n=17, seed=7):
    rng = np.random.default_rng(seed)
    p = net.init_params(d, seed=seed + 1)
    p["W3"] = (rng.normal(size=p["W3"].shape) * 0.3).astype(np.float32)
    p["Wv"] = (rng.normal(size=p["Wv"].shape) * 0.3).astype(np.float32)
    p["g"] = rng.normal(size=d).astype(np.float32)
    return p, {
        "obs": rng.normal(size=(n, d)).astype(np.float32),
        "acts": rng.integers(0, net.N_ACTIONS, n),
        "adv": rng.normal(size=n).astype(np.float32),
        "vt": rng.normal(size=n).astype(np.float32),
    }


@pytest.mark.parametrize("key", net.PARAM_KEYS)
def test_analytic_gradient_matches_finite_differences(key):
    """Central differences in float64, on a random subset of each parameter block."""
    beta, l1, vc = 0.01, 0.003, 0.5
    p, b = _fixture()
    _, grads, _ = net.loss_and_grads(p, b["obs"], b["acts"], b["adv"], beta, l1, b["vt"], vc)

    p64 = {k: v.astype(np.float64) for k, v in p.items()}
    flat = p64[key].ravel()
    rng = np.random.default_rng(0)
    idxs = rng.choice(flat.size, size=min(10, flat.size), replace=False)

    h = 1e-6
    for i in idxs:
        orig = flat[i]
        flat[i] = orig + h
        lp = net.loss_and_grads(p64, b["obs"], b["acts"], b["adv"], beta, l1, b["vt"], vc)[0]
        flat[i] = orig - h
        lm = net.loss_and_grads(p64, b["obs"], b["acts"], b["adv"], beta, l1, b["vt"], vc)[0]
        flat[i] = orig
        num = (lp - lm) / (2 * h)
        ana = grads[key].ravel()[i]
        assert abs(num - ana) / max(abs(num) + abs(ana), 1e-8) < 2e-4, (
            f"{key}[{i}]: analytic {ana} vs numeric {num}")


def test_entropy_bonus_pushes_toward_uniform():
    """With no advantage signal the only force left is the entropy bonus, so plain
    gradient descent must make a peaked policy flatter - that is what beta is for.

    Deliberately plain SGD with a small step, not Adam: Adam's bias-corrected first
    steps are ~lr in size regardless of curvature and briefly overshoot to a MORE
    peaked policy before recovering. That is optimiser dynamics, not the sign of the
    gradient, and this test is about the sign.
    """
    d = 6
    p = net.init_params(d, seed=2)
    p["W3"] = (np.random.default_rng(1).normal(size=p["W3"].shape) * 2.0).astype(np.float32)
    obs = np.random.default_rng(3).normal(size=(64, d)).astype(np.float32)
    acts = np.zeros(64, dtype=int)
    zero_adv = np.zeros(64, dtype=np.float32)

    seen = []
    for _ in range(40):
        loss, grads, stats = net.loss_and_grads(p, obs, acts, zero_adv, 0.05)
        seen.append(stats["entropy"])
        for k in p:
            p[k] -= (0.5 * grads[k]).astype(np.float32)

    assert seen[-1] > seen[0], f"entropy fell: {seen[0]:.4f} -> {seen[-1]:.4f}"
    assert all(b >= a - 1e-6 for a, b in zip(seen, seen[1:])), "entropy must rise monotonically"
    assert seen[-1] <= np.log(net.N_ACTIONS) + 1e-5, "cannot exceed the uniform maximum"


def test_advantage_sign_moves_the_chosen_action():
    """A positive advantage must raise log pi(a|o); a negative one must lower it."""
    d = 5
    obs = np.ones((1, d), dtype=np.float32)
    for sign in (+1.0, -1.0):
        p = net.init_params(d, seed=4)
        p["W3"] = (np.random.default_rng(5).normal(size=p["W3"].shape) * 0.5).astype(np.float32)
        a = np.array([3])
        before = net.log_softmax(net.forward(p, obs)["logits"])[0, 3]
        opt = net.Adam(p, lr=0.02)
        for _ in range(30):
            _, grads, _ = net.loss_and_grads(p, obs, a, np.array([sign], dtype=np.float32), 0.0)
            opt.step(p, grads)
        after = net.log_softmax(net.forward(p, obs)["logits"])[0, 3]
        assert (after > before) if sign > 0 else (after < before)


def test_gate_l1_closes_gates_when_nothing_else_pushes():
    """M1's premise: with no task signal, the L1 term drives sigma(g) toward 0."""
    d = 7
    p = net.init_params(d, seed=6)
    obs = np.random.default_rng(7).normal(size=(32, d)).astype(np.float32)
    acts = np.zeros(32, dtype=int)
    zero_adv = np.zeros(32, dtype=np.float32)
    before = float(net._sigmoid(p["g"]).mean())
    opt = net.Adam(p, lr=0.05)
    for _ in range(100):
        _, grads, _ = net.loss_and_grads(p, obs, acts, zero_adv, 0.0, gate_l1=0.5)
        opt.step(p, grads)
    assert float(net._sigmoid(p["g"]).mean()) < before


def test_grad_clip_caps_the_global_norm():
    grads = {"a": np.full((3, 3), 10.0, dtype=np.float32),
             "b": np.full(4, -10.0, dtype=np.float32)}
    pre = net.clip_grads(grads, 1.0)
    assert pre > 1.0
    assert np.isclose(net.global_norm(grads), 1.0, atol=1e-5)


def test_clip_leaves_small_gradients_alone():
    grads = {"a": np.array([0.1, 0.1], dtype=np.float32)}
    before = grads["a"].copy()
    net.clip_grads(grads, 1.0)
    assert np.array_equal(grads["a"], before)


def test_discounted_returns_match_the_closed_form():
    """G_t = sum_k gamma^k r_{t+k}  (plan.md A3)."""
    r = [1.0, 2.0, 3.0]
    g = 0.9
    assert np.allclose(discounted(r, g), [1 + 0.9*2 + 0.81*3, 2 + 0.9*3, 3.0])
    assert np.allclose(discounted([5.0], g), [5.0])
    assert np.allclose(discounted([1.0]*4, 1.0), [4, 3, 2, 1])


def test_weights_survive_the_base64_round_trip():
    """Weights ride into every sandbox as an env var - a lossy round trip would be
    training/serving skew that no gate would catch."""
    p = net.init_params(spec.OBS_DIM_2D, seed=11)
    q = net.unpack(net.pack(p))
    for k in net.PARAM_KEYS:
        assert np.array_equal(p[k], q[k]), k
    assert len(net.pack(p)) < 100_000, "must fit the plan.md 4.5 env-var budget"


def test_policy_is_near_uniform_at_init():
    p = net.init_params(spec.OBS_DIM_2D, seed=0)
    logits = net.forward(p, np.random.default_rng(0).normal(size=(20, spec.OBS_DIM_2D)))["logits"]
    probs = net.softmax(logits)
    assert np.allclose(probs, 1.0 / net.N_ACTIONS, atol=0.02)


def test_greedy_is_deterministic_and_sampling_is_not():
    p = net.init_params(spec.OBS_DIM_2D, seed=1)
    p["W3"] = (np.random.default_rng(2).normal(size=p["W3"].shape) * 1.0).astype(np.float32)
    o = np.random.default_rng(3).normal(size=spec.OBS_DIM_2D).astype(np.float32)
    assert net.act_greedy(p, o)[0] == net.act_greedy(p, o)[0]
    draws = {net.act_sample(p, o, np.random.default_rng(s))[0] for s in range(40)}
    assert len(draws) > 1
