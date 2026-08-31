"""REINFORCE with a baseline (plan.md 3), in numpy.

One generation = freeze theta -> M Daytona rollouts -> discounted returns ->
batch-mean baseline -> normalise -> entropy-regularised policy gradient -> ONE Adam
step. Vanilla REINFORCE is on-policy, so a batch buys exactly one step (plan.md 3.2).

The gradient itself lives in `gg/net.py` (finite-difference checked in
tests/test_reinforce.py); this module owns only the RL bookkeeping: returns,
advantages, entropy decay, champion selection and the metrics log.

    python -m controller.train --stage A --generations 200
    python -m controller.train --stage A --generations 30 --fresh   # new sandboxes each gen
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import numpy as np
from daytona import AsyncDaytona

from controller.adapter import adapt_rollout, generation_payload
from controller.build_snapshot import ensure_snapshot
from controller.config import (REPO_ROOT, daytona_config, load_stage_config,
                               load_stage_keys, results_dir)
from controller.fanout import SandboxPool, gather_rollouts, sample_keys, worker_input
from controller.record import write_generation
from gg import net, spec


def discounted(rewards, gamma: float) -> np.ndarray:
    """G_t = sum_{k>=0} gamma^k r_{t+k}  (plan.md A3)."""
    out = np.empty(len(rewards), dtype=np.float64)
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        g = rewards[i] + gamma * g
        out[i] = g
    return out


class Trainer:
    """Owns theta, the optimiser state, and the champion."""

    def __init__(self, stage: str, cfg: dict, seed: int = 0, params: dict | None = None,
                 lr: float | None = None):
        self.stage, self.cfg = stage, cfg
        tcfg, pcfg = cfg["training"], cfg["policy"]
        self.gamma = cfg["episode"]["gamma"]
        self.gate_l1 = float(pcfg.get("gate_l1", 0.0)) if pcfg.get("input_gate") else 0.0
        self.grad_clip = float(tcfg["grad_clip"])
        self.value_coef = float(tcfg.get("value_coef", 0.5))
        self.beta0 = float(tcfg["entropy_beta_start"])
        self.beta1 = float(tcfg["entropy_beta_end"])
        self.params = params if params is not None else net.init_params(spec.OBS_DIM_2D, seed=seed)
        self.lr0 = float(lr if lr is not None else tcfg["lr"])
        self.lr_final_frac = float(tcfg.get("lr_final_frac", 0.1))
        self.opt = net.Adam(self.params, lr=self.lr0)
        self.version = 0
        self.champion = {"version": 0, "params": {k: v.copy() for k, v in self.params.items()},
                         "score": -np.inf, "generation": 0, "kind": "mean_reward"}

    def set_lr(self, gen: int, total: int) -> float:
        """Linearly decay the step size to `lr_final_frac` of its start.

        Stage A learns fast and then falls over: the fleet run peaked near generation
        30 (best reward +98.7) and degraded afterwards with the gradient norm climbing
        into the clip. That is the "lower lr if generation-to-generation KL spikes"
        note in plan.md 3.3, applied on a schedule instead of by hand.
        """
        frac = 0.0 if total <= 1 else min(1.0, max(0.0, (gen - 1) / (total - 1)))
        self.opt.lr = self.lr0 * (1.0 + (self.lr_final_frac - 1.0) * frac)
        return self.opt.lr

    def beta(self, gen: int, total: int) -> float:
        """Linear entropy decay across the run (plan.md 3.3)."""
        frac = 0.0 if total <= 1 else min(1.0, max(0.0, (gen - 1) / (total - 1)))
        return self.beta0 + (self.beta1 - self.beta0) * frac

    def update(self, rollouts: list[dict], beta: float) -> dict:
        """The gradient step. `rollouts` are RAW worker payloads (plan.md 3.6).

        Baseline: plan.md 3.2 step 5 offers batch-mean first and a fitted V(o) "if
        variance is high". At Stage A it IS high, and for a structural reason - every
        sandbox flies the SAME universe, so G_t varies far more along t than across
        the batch. A batch-mean baseline therefore leaves the advantage encoding
        mostly "how late is this step", and the policy gradient learns nothing about
        the actions. V(o) removes that time trend, which is what makes Stage A move.
        Returns are standardised first so the value head sees an O(1) target.
        """
        obs = np.concatenate([np.asarray(r["observations"], dtype=np.float32) for r in rollouts])
        acts = np.concatenate([np.asarray(r["actions"], dtype=int) for r in rollouts])
        rets = np.concatenate([discounted(r["rewards"], self.gamma) for r in rollouts])

        # Standardise returns WITHIN each key. From Stage B on a batch spans several
        # layouts whose returns differ by the full +/-100 goal bonus, so a batch-wide
        # baseline makes the advantage mostly encode "which layout did I draw" - the
        # same pathology the value head fixes for elapsed time at Stage A, one level
        # up. Per-key centring removes the layout's difficulty and leaves the part the
        # policy is actually responsible for.
        # Group by the WORLD, not by layout_id alone: at Stage D every rollout carries a
        # `seed_universe` and they all share layout_id 0, so keying on layout_id would
        # collapse the whole batch into one group and lose the per-world baseline.
        def world_of(r):
            su = r.get("seed_universe")
            return int(su) if su is not None else int(r.get("layout_id") or 0)

        key_of = np.concatenate([
            np.full(len(r["rewards"]), world_of(r)) for r in rollouts])
        z = np.empty_like(rets)
        for k in np.unique(key_of):
            sel = key_of == k
            z[sel] = (rets[sel] - rets[sel].mean()) / (rets[sel].std() + 1e-8)
        z = z.astype(np.float32)
        if self.value_coef > 0.0:
            baseline = net.forward(self.params, obs)["value"]
            adv = z - baseline
        else:
            adv = z                                          # batch-mean baseline only
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        loss, grads, stats = net.loss_and_grads(
            self.params, obs, acts, adv.astype(np.float32), beta, self.gate_l1,
            value_targets=z, value_coef=self.value_coef)
        gnorm = net.clip_grads(grads, self.grad_clip)
        self.opt.step(self.params, grads)
        self.version += 1
        return {"loss": round(float(loss), 5), "grad_norm": round(float(gnorm), 5),
                "entropy": round(stats["entropy"], 5), "gate_mean": round(stats["gate_mean"], 5),
                "value_mse": round(stats["value_mse"], 5),
                "n_samples": int(obs.shape[0]), "beta": round(beta, 5),
                "lr": round(float(self.opt.lr), 6),
                "return_mean": round(float(rets.mean()), 4)}

    def consider_champion(self, score: float, generation: int, kind: str = "mean_reward") -> bool:
        """Keep the best policy seen so far.

        A `kind="heldout"` score beats any `kind="mean_reward"` score: training-batch
        reward is what the gradient already chased and is measured on the seeds the
        policy just trained on, whereas the held-out probe asks the gate's question -
        does this policy reach the portal on action seeds it has not seen.
        """
        prev = self.champion.get("kind", "mean_reward")
        better = (kind == "heldout" and prev != "heldout") or \
                 (kind == prev and score > self.champion["score"])
        if better:
            self.champion = {"version": self.version, "generation": generation, "score": score,
                             "kind": kind,
                             "params": {k: v.copy() for k, v in self.params.items()}}
            return True
        return False

    def save_champion(self, path) -> None:
        np.savez_compressed(path, **self.champion["params"],
                            _meta=np.array(json.dumps({
                                "version": self.champion["version"],
                                "generation": self.champion["generation"],
                                "score": float(self.champion["score"]),
                                "kind": self.champion.get("kind", "mean_reward"),
                                "obs_spec_hash": spec.obs_spec_hash(2),
                                "stage": self.stage}), dtype=object))


def quiet_new_inputs(params: dict, stage: str) -> list[str]:
    """Zero the first-layer weights reading the groups that go live AT this stage.

    The observation vector is a frozen superset, so a group that is zero-filled until
    rung X has fed nothing but zeros into W1 for every earlier rung - which means its
    W1 rows never received a gradient and are still random init. Warm-starting into
    rung X switches those inputs on, and 37% of the input dimensions start injecting
    random weights of the same magnitude as the trained ones straight into the trunk.
    The "warm" start is then not warm at all: measured, Stage B inherited 50% success
    at generation 1 and had lost it by generation 30.

    Zeroing them makes the inherited policy behave EXACTLY as it did on the previous
    rung, and lets it learn to use the new inputs from zero instead of first having to
    undo noise. The gate logits are left alone - M1 still gets to close them.
    """
    newly = [g for g, (_, _, first) in spec.OBS_LAYOUT_2D.items() if first == stage]
    for g in newly:
        lo, hi, _ = spec.OBS_LAYOUT_2D[g]
        params["W1"][lo:hi, :] = 0.0
    return newly


def curriculum_at(cfg: dict, gen: int) -> float | None:
    """Difficulty for this generation, or None when the stage declares no curriculum.

    Ramps 0 -> 1 linearly over `training.curriculum_generations` and stays at 1
    afterwards, so the last stretch of training is on the real distribution.
    """
    ramp = cfg.get("training", {}).get("curriculum_generations")
    if not ramp:
        return None
    return float(min(1.0, max(0.0, (gen - 1) / float(ramp))))


def load_champion(path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k].astype(np.float32) for k in net.PARAM_KEYS}


def champion_stage(path) -> str | None:
    """The rung a saved champion was trained on, from its `_meta` blob."""
    with np.load(path, allow_pickle=True) as z:
        if "_meta" not in z:
            return None
        try:
            return json.loads(str(z["_meta"])).get("stage")
        except (ValueError, TypeError):
            return None


def champion_meta(path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return json.loads(str(z["_meta"]))


# ---------------------------------------------------------------- the loop

async def train(stage: str, generations: int, m: int, *, seed: int = 0, fresh: bool = False,
                record_every: int = 1, quiet: bool = False, eval_every: int = 10,
                lr: float | None = None, episodes: int | None = None,
                cfg_override: dict | None = None):
    """`m` is CONCURRENCY (how many sandboxes exist at once, plan.md 3.3 M=8);
    `episodes` is BATCH DEPTH (how many rollouts the gradient averages over). They
    were the same number until Stage A showed its hand: the whole batch flies one
    universe, so the only signal is action-sampling spread (std ~0.1 of reward), and
    REINFORCE variance falls as 1/episodes. Running 32 episodes through 8 warm
    sandboxes keeps the fleet story exactly as specified while giving the gradient
    four times the sample. Fresh-sandbox mode always uses episodes = m."""
    cfg = cfg_override or load_stage_config(stage)
    keys_file = load_stage_keys(stage)
    train_keys, eval_keys = keys_file["train"], keys_file["eval_held_out"]
    n_ep = m if fresh else int(episodes or m)
    warm = cfg["training"].get("warm_start_from")
    warm_params = None
    if warm:
        wp = REPO_ROOT / warm
        if wp.exists():
            warm_params = load_champion(wp)
            # Quieting is only right for a checkpoint from an EARLIER rung, where the
            # newly-live groups really are still random init. Continuing a rung from its
            # own champion (a re-run after a reward change, say) would otherwise zero the
            # very weights that rung already trained - silently throwing away the transfer
            # the warm start exists to keep.
            from_stage = champion_stage(wp)
            if from_stage == stage:
                print(f"[train] warm start from a stage-{from_stage} champion: continuing "
                      f"the same rung, so newly-live groups are NOT quieted", flush=True)
            else:
                quieted = quiet_new_inputs(warm_params, stage)
                if quieted:
                    print(f"[train] warm start: zeroed W1 rows for newly-live groups {quieted}",
                          flush=True)
        else:
            print(f"[train] warm start {wp} not found - starting from scratch", flush=True)
    trainer = Trainer(stage, cfg, seed=seed, lr=lr, params=warm_params)
    history, t0 = [], time.time()

    def log(*a):
        if not quiet:
            print(*a, flush=True)

    log(f"[train] stage {stage}  generations={generations}  M={m}  "
        f"dispatch={'fresh sandboxes' if fresh else 'warm pool'}  "
        f"episodes/gen={n_ep}  params={net.n_params(trainer.params)}"
        f"{'  warm-started' if warm_params is not None else ''}")

    async with AsyncDaytona(daytona_config()) as d:
        await ensure_snapshot(d, quiet=True)
        pool = None if fresh else await SandboxPool(d, m).__aenter__()
        try:
            for gen in range(1, generations + 1):
                weights = net.pack(trainer.params)
                base_seed = seed * 1_000_003 + gen * 10_000
                distinct = m if cfg.get("dispatch", {}).get("repeat_across_batch") else None
                keys = sample_keys(train_keys, n_ep, np.random.default_rng(base_seed),
                                   distinct=distinct)
                # Curriculum (stage D on). It goes INTO the key, not alongside it: a
                # stage-D world is a function of (seed_universe, curriculum), so a
                # recorded generation that stored only the seed would replay at the
                # wrong difficulty. Held-out evaluation below always pins curriculum=1.0
                # - ramping the EVAL distribution would be measuring the gate on an
                # easier world than the one being claimed.
                cur = curriculum_at(cfg, gen)
                if cur is not None:
                    keys = [dict(k, curriculum=cur) for k in keys]
                payloads = [worker_input(stage, cfg, weights_b64=weights, key=keys[i],
                                         seed=base_seed + i, policy_version=trainer.version)
                            for i in range(n_ep)]

                if fresh:
                    indexed, failures = await gather_rollouts(d, payloads)
                    raw = [o for _, o in indexed]
                    ids = [(i, o) for i, o in indexed]
                else:
                    raw, bad = await pool.map(payloads, f"g{gen}w")
                    failures = [{"worker_id": b["i"], "error": b["error"]} for b in bad]
                    ids = list(enumerate(raw))

                # plan.md 4.5: never update from a half-empty batch
                need = int(np.ceil(cfg["training"]["min_successful_rollouts_frac"] * n_ep))
                if len(raw) < need:
                    log(f"  gen {gen}: only {len(raw)}/{m} rollouts returned (< {need}) - "
                        f"skipping the update")
                    continue

                mean_r = float(np.mean([r["reward"] for r in raw]))
                succ = float(np.mean([bool(r["success"]) for r in raw]))
                trainer.set_lr(gen, generations)
                stats = trainer.update(raw, trainer.beta(gen, generations))
                improved = trainer.consider_champion(mean_r, gen)

                rec = {"generation": gen, "policy_version": trainer.version,
                       "mean_reward": round(mean_r, 4),
                       "best_reward": round(max(r["reward"] for r in raw), 4),
                       "success_rate": round(succ, 4),
                       "mean_steps": round(float(np.mean([r["steps"] for r in raw])), 1),
                       "n_rollouts": len(raw), "n_failed": len(failures), **stats}
                history.append(rec)
                log(f"  gen {gen:3d}  P{trainer.version:03d}  mean={mean_r:+8.2f}  "
                    f"best={rec['best_reward']:+8.2f}  succ={succ:.2f}  "
                    f"H={stats['entropy']:.3f}  |g|={stats['grad_norm']:.3f}  "
                    f"gate={stats['gate_mean']:.3f}{'  *champion' if improved else ''}")

                # Probe the champion on HELD-OUT action seeds, sampling exactly as the
                # fleet does. REINFORCE optimises the stochastic policy - S4 is on from
                # Stage A (plan.md 2.6) and every rollout the worker runs draws
                # a_t ~ Categorical(pi_theta) - so the argmax policy is one that was
                # never trained, and scoring the champion by it measures the wrong
                # object. Greedy is still reported at the end, as a diagnostic of how
                # peaked the policy has become.
                if eval_every and gen % eval_every == 0:
                    ev_keys_g = ([dict(k, curriculum=1.0) for k in eval_keys]
                                 if curriculum_at(cfg, gen) is not None else eval_keys)
                    gp = [worker_input(stage, cfg, weights_b64=net.pack(trainer.params),
                                       key=ev_keys_g[i % len(ev_keys_g)],
                                       seed=880_000 + 97 * gen + i,
                                       policy_version=trainer.version, greedy=False)
                          for i in range(m)]
                    if fresh:
                        gidx, _ = await gather_rollouts(d, gp)
                        gouts = [o for _, o in gidx]
                    else:
                        gouts, _ = await pool.map(gp, f"greedy{gen}")
                    if gouts:
                        gsucc = float(np.mean([bool(o["success"]) for o in gouts]))
                        gdist = float(np.mean([o["min_portal_dist"] for o in gouts]))
                        rec["heldout_success"] = round(gsucc, 3)
                        rec["heldout_min_portal_dist"] = round(gdist, 4)
                        # The generalisation gap only means something from Stage D on:
                        # before it, train and held-out are different keys into one fixed
                        # world set, so the "gap" measures nothing. Here they are
                        # different UNIVERSES drawn from disjoint seed blocks, which is
                        # the first point at which train-minus-held-out is a real
                        # overfitting signal (plan.md 5, stage D gate).
                        rec["generalisation_gap"] = round(succ - gsucc, 3)
                        if trainer.consider_champion(gsucc, gen, kind="heldout"):
                            log(f"        held-out probe: success={gsucc:.2f} "
                                f"closest={gdist:.3f}  *champion")

                if record_every and gen % record_every == 0:
                    adapted = [adapt_rollout(o, i, gen, keys[i]) for i, o in ids[:m]]
                    gp = generation_payload(gen, trainer.version, base_seed, keys, adapted, failures)
                    write_generation(gp)

            # ---- final held-out evaluation, greedy and sampled
            log("\n[train] evaluating the champion on held-out keys ...")
            champ_w = net.pack(trainer.champion["params"])
            _fin = ([dict(k, curriculum=1.0) for k in eval_keys]
                    if curriculum_at(cfg, 10**9) is not None else eval_keys)
            ev_keys = [_fin[i % len(_fin)] for i in range(m)]
            ev = {}
            for label, greedy in (("greedy", True), ("sampled", False)):
                pl = [worker_input(stage, cfg, weights_b64=champ_w, key=ev_keys[i],
                                   seed=777_000 + i, policy_version=trainer.version,
                                   greedy=greedy) for i in range(m)]
                if fresh:
                    idx, _ = await gather_rollouts(d, pl)
                    outs = [o for _, o in idx]
                else:
                    outs, _ = await pool.map(pl, f"eval-{label}")
                ev[label] = {
                    "success_rate": round(float(np.mean([bool(o["success"]) for o in outs])), 4),
                    "mean_reward": round(float(np.mean([o["reward"] for o in outs])), 4),
                    "terminations": {t: sum(o["termination"] == t for o in outs)
                                     for t in {o["termination"] for o in outs}},
                    "n": len(outs)}
                log(f"  {label:8s} success={ev[label]['success_rate']:.2f}  "
                    f"mean_reward={ev[label]['mean_reward']:+.2f}  {ev[label]['terminations']}")
        finally:
            if pool is not None:
                await pool.__aexit__(None, None, None)

    rd = results_dir(stage)
    trainer.save_champion(rd / "champion.npz")
    metrics = {"stage": stage, "generations": generations, "m": m, "seed": seed,
               "dispatch": "fresh" if fresh else "pool",
               "obs_spec_hash": spec.obs_spec_hash(2),
               "wallclock_s": round(time.time() - t0, 1),
               "champion": {"version": trainer.champion["version"],
                            "generation": trainer.champion["generation"],
                            "score": round(float(trainer.champion["score"]), 4)},
               "held_out_eval": ev, "history": history}
    (rd / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"\n[train] wrote {rd/'metrics.json'} and {rd/'champion.npz'} "
        f"({metrics['wallclock_s']}s)")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="A8 - REINFORCE training loop")
    ap.add_argument("--stage", default="A")
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("-n", "--m", type=int, default=None, help="sandboxes per generation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true",
                    help="new sandboxes every generation instead of a warm pool")
    ap.add_argument("--record-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=None, help="override config lr")
    ap.add_argument("--beta-end", type=float, default=None, help="override entropy floor")
    ap.add_argument("--episodes", type=int, default=None,
                    help="rollouts per gradient step (batch depth); default = M")
    args = ap.parse_args()

    cfg = load_stage_config(args.stage)
    if args.beta_end is not None:
        cfg["training"]["entropy_beta_end"] = args.beta_end
    gens = args.generations or cfg["training"]["generations"]
    m = args.m or cfg["training"]["sandboxes_per_generation"]
    metrics = asyncio.run(train(args.stage, gens, m, seed=args.seed, fresh=args.fresh,
                                record_every=args.record_every, lr=args.lr,
                                episodes=args.episodes, cfg_override=cfg))

    hist = metrics["history"]
    if not hist:
        print("GATE A8: FAIL - no generations completed")
        return 1
    k = max(1, len(hist) // 5)
    first = float(np.mean([h["mean_reward"] for h in hist[:k]]))
    last = float(np.mean([h["mean_reward"] for h in hist[-k:]]))
    succ_first = float(np.mean([h["success_rate"] for h in hist[:k]]))
    succ_last = float(np.mean([h["success_rate"] for h in hist[-k:]]))
    ev = metrics["held_out_eval"]

    print("\n" + "=" * 72)
    print(f"  mean reward   first {k} gens: {first:+8.2f}   last {k} gens: {last:+8.2f}")
    print(f"  success rate  first {k} gens: {succ_first:8.2f}   last {k} gens: {succ_last:8.2f}")
    print(f"  held-out      greedy success {ev['greedy']['success_rate']:.2f} | "
          f"sampled success {ev['sampled']['success_rate']:.2f}")
    print("=" * 72)
    # The gate is about the policy as trained AND as deployed: stochastic. Greedy is
    # reported beside it as a diagnostic, not as the bar.
    checks = {
        "reward improved over the run": last > first,
        "success rate rose over the run": succ_last > succ_first,
        "champion reaches the portal reliably (held-out, sampled)":
            ev["sampled"]["success_rate"] >= 0.8,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    good = all(checks.values())
    print("\nGATE A8:", "PASS - the policy learned to fly to the portal" if good else "FAIL - see above")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
