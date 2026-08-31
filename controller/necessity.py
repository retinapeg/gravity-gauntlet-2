"""Which Newtonian variables does the craft actually need? (plan.md 6)

Four probes on the stage champion, reported per OBSERVATION GROUP (the units the
necessity matrix in plan.md 6.1 is written in):

  M1  Learned input gate     sigma(g) per group. Trained with an L1 penalty, so a
                             group the policy does not use is free to close.
  M2  Ablation & permutation THE HEADLINE. Re-fly held-out keys with a group (a)
                             zeroed and (b) resampled from the champion's own
                             empirical distribution - the "shuffled across the
                             batch" arm. Importance = Delta return.
  M3  Gradient saliency      mean |d logits / d o_i| over champion trajectories.
                             Pure analysis of stored observations: no rollouts.
  M4  Nested feature sets    train a fresh policy on each nested set (plan.md 6
                             table A-E) and compare. The smallest set that matches
                             the full set is the empirically necessary set.

Only M2 and M4 need sandboxes. They use a warm pool (plan.md 4.6) because the
rollout count runs to several hundred and create latency would otherwise dominate.

    python -m controller.necessity --stage A                  # M1-M3, 3 repeats
    python -m controller.necessity --stage A --with-m4        # + nested-set training
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import numpy as np
from daytona import AsyncDaytona

from controller.build_snapshot import ensure_snapshot
from controller.config import (daytona_config, load_stage_config, load_stage_keys,
                               results_dir)
from controller.fanout import SandboxPool, worker_input
from controller.train import load_champion
from gg import net, spec

# Groups that carry real signal at this rung; the rest are zero-filled by design
# and asking whether a zero vector is "necessary" is not a question.
def active_groups(stage: str) -> list[str]:
    return [g for g, (_, _, first) in spec.OBS_LAYOUT_2D.items()
            if spec.stage_active(first, stage)]


# ------------------------------------------------------------------- M1 gate

def m1_gate(params: dict, stage: str) -> dict:
    """sigma(g) averaged over each group's slots. -> 0 unnecessary, -> 1 load-bearing."""
    s = 1.0 / (1.0 + np.exp(-params["g"]))
    return {g: round(float(s[lo:hi].mean()), 5)
            for g in active_groups(stage)
            for lo, hi, _ in [spec.OBS_LAYOUT_2D[g]]}


# -------------------------------------------------------------- M3 saliency

def m3_saliency(params: dict, observations: np.ndarray, stage: str) -> dict:
    """mean_t sum_a |d logit_a / d o_i|, aggregated per group and normalised.

    Backprop of a vector of ones through the logits gives sum_a d logit_a / d o_i;
    to get the sum of ABSOLUTE per-action sensitivities we push one action at a
    time. |A| = 9, so this is nine cheap backward passes, not a loop over samples.
    """
    obs = np.asarray(observations, dtype=np.float32)
    c = net.forward(params, obs)
    total = np.zeros(obs.shape[1])
    for a in range(c["logits"].shape[1]):
        seed = np.zeros_like(c["logits"])
        seed[:, a] = 1.0
        dh2 = seed @ params["W3"].T
        dz2 = dh2 * (1.0 - c["h2"] ** 2)
        dz1 = (dz2 @ params["W2"].T) * (1.0 - c["h1"] ** 2)
        dx = dz1 @ params["W1"].T
        total += np.abs(dx * c["gate"]).mean(axis=0)     # chain through the input gate
    out = {g: float(total[lo:hi].mean())
           for g in active_groups(stage)
           for lo, hi, _ in [spec.OBS_LAYOUT_2D[g]]}
    peak = max(out.values()) or 1.0
    return {k: round(v / peak, 5) for k, v in out.items()}


# ------------------------------------------- M2 ablation / permutation (fleet)

async def m2_ablation(pool: SandboxPool, stage: str, cfg: dict, weights: str,
                      eval_keys: list, groups: list[str], repeats: int, n_eps: int,
                      seed0: int, obs_pool: dict) -> dict:
    """Delta return when a group is zeroed, and when it is resampled.

    Baseline and every arm use the SAME action-sampling seeds, so a difference is
    the intervention and not luck.
    """
    def payloads(extra: dict, rep: int):
        return [worker_input(stage, cfg, weights_b64=weights,
                             key=eval_keys[i % len(eval_keys)],
                             seed=seed0 + 1000 * rep + i, policy_version=0, **extra)
                for i in range(n_eps)]

    def summarise(outs):
        return (float(np.mean([o["reward"] for o in outs])),
                float(np.mean([bool(o["success"]) for o in outs])),
                float(np.mean([o["min_portal_dist"] for o in outs])))

    result = {"baseline": [], "zero": {g: [] for g in groups},
              "resample": {g: [] for g in groups}}

    for rep in range(repeats):
        base, _ = await pool.map(payloads({}, rep), f"m2-base{rep}")
        result["baseline"].append(summarise(base))
        for g in groups:
            z, _ = await pool.map(payloads({"obs_ablate": [g]}, rep), f"m2-zero-{g}{rep}")
            result["zero"][g].append(summarise(z))
            r, _ = await pool.map(
                payloads({"obs_resample": {g: obs_pool[g]}}, rep), f"m2-res-{g}{rep}")
            result["resample"][g].append(summarise(r))
    return result


def m2_scores(raw: dict, groups: list[str]) -> dict:
    """Per-group Delta return / Delta success, averaged over repeats."""
    base_r = float(np.mean([b[0] for b in raw["baseline"]]))
    base_s = float(np.mean([b[1] for b in raw["baseline"]]))
    base_d = float(np.mean([b[2] for b in raw["baseline"]]))
    out = {}
    for g in groups:
        z = np.array(raw["zero"][g]); r = np.array(raw["resample"][g])
        # Delta return carries the +/-100 goal bonus, so a single lucky capture moves
        # it by ~100/n_eps. Delta closest-approach is continuous and therefore far
        # better conditioned; it is reported alongside as the low-variance witness.
        out[g] = {
            "delta_return_zeroed": round(base_r - float(z[:, 0].mean()), 4),
            "delta_return_resampled": round(base_r - float(r[:, 0].mean()), 4),
            "delta_success_zeroed": round(base_s - float(z[:, 1].mean()), 4),
            "delta_portal_dist_zeroed": round(float(z[:, 2].mean()) - base_d, 4),
            "delta_portal_dist_resampled": round(float(r[:, 2].mean()) - base_d, 4),
            "return_zeroed_by_repeat": [round(float(x), 3) for x in z[:, 0]],
            "return_zeroed_spread": round(float(z[:, 0].std()), 3),
        }
    return {"baseline": {"return": round(base_r, 4), "success": round(base_s, 4),
                         "min_portal_dist": round(base_d, 4)},
            "per_group": out}


# ------------------------------------------------------- classification + rank

def corroborated(delta_zeroed: float, delta_resampled: float, span: float) -> bool:
    """Does the in-distribution arm agree that this group matters?

    Zeroing a group hands the policy a value it may never see in play - zeroing
    `fuel_fraction` says "you have no fuel" on step one - so a large delta from the
    zeroed arm can be pure out-of-distribution shock rather than evidence the policy
    USES the group. The resampled arm replaces the group with a draw from the
    champion's own trajectory pool, which stays in distribution, so it is the honest
    control. MEASURED at Stage C: fuel_fraction zeroed +143.6 but resampled -4.0, and
    craft_vel zeroed +51.3 but resampled -4.4 - both "necessary" verdicts were shock,
    not necessity. At Stage A the two arms agreed (portal_rel_pos 39.0 / 16.7), which
    is what a real dependence looks like.
    """
    if delta_zeroed < 0.10 * span:          # nothing claimed, nothing to corroborate
        return True
    return delta_resampled >= 0.25 * delta_zeroed


def classify(delta_return: float, delta_success: float, span: float) -> str:
    """plan.md 6.1 legend: necessary / marginal / predicted unnecessary."""
    if delta_success >= 0.25 or delta_return >= 0.35 * span:
        return "necessary"
    if delta_success >= 0.05 or delta_return >= 0.10 * span:
        return "marginal"
    return "unnecessary"


# plan.md 6.1 column A, as predicted before any of this was measured. Reporting
# measured against predicted IS the result (stages/stage_a_fixed_star/README.md).
PREDICTED_A = {
    "portal_rel_pos": "necessary", "craft_vel": "necessary", "a_grav": "necessary",
    "fuel_fraction": "necessary", "time_fraction": "necessary",
    "speed": "marginal", "potential": "marginal",
    "closing_speed_portal": "unnecessary", "a_grav_magdir": "unnecessary",
    "craft_mass": "unnecessary", "control_authority": "unnecessary",
    "orbital_energy": "unnecessary", "angular_momentum": "unnecessary",
    "eccentricity_vec": "unnecessary",
}


def verdict_stability(raw: dict, groups: list[str], repeats: int) -> dict:
    """Does each group land in the SAME verdict class on every re-run?

    A full-rank Spearman is the wrong instrument once most variables are genuinely
    unnecessary: their delta-returns are all within noise of zero, so their relative
    order is arbitrary and re-shuffling it tanks the correlation even though nothing
    meaningful moved. This asks the question the study actually cares about - does a
    variable keep its verdict - and reports the unstable groups by name.
    """
    per_repeat = []
    for r in range(repeats):
        base_r, base_s = raw["baseline"][r][0], raw["baseline"][r][1]
        span = max(1e-9, max(base_r - raw["zero"][g][r][0] for g in groups))
        per_repeat.append({g: classify(base_r - raw["zero"][g][r][0],
                                       base_s - raw["zero"][g][r][1], span)
                           for g in groups})
    agree = {g: len({pr[g] for pr in per_repeat}) == 1 for g in groups}
    unstable = sorted(g for g, ok in agree.items() if not ok)
    return {
        "per_repeat": per_repeat,
        "n_groups": len(groups),
        "n_stable": sum(agree.values()),
        "fraction_stable": round(sum(agree.values()) / len(groups), 4),
        "unstable_groups": unstable,
        "stable": not unstable,
    }


def topk_stability(rankings: list[list[str]], k: int = 3) -> dict:
    """Is the leading set the same every time, ignoring order below it?"""
    sets = [frozenset(r[:k]) for r in rankings]
    return {"k": k, "identical": len(set(sets)) == 1,
            "intersection": sorted(set.intersection(*[set(x) for x in sets])),
            "per_repeat": [sorted(x) for x in sets]}


def rank_stability(rankings: list[list[str]]) -> dict:
    """Spearman-style agreement between per-repeat rankings (the A9 gate)."""
    if len(rankings) < 2:
        return {"pairs": 0, "min_spearman": None, "stable": False}
    order = {g: i for i, g in enumerate(rankings[0])}
    rhos = []
    for a in range(len(rankings)):
        for b in range(a + 1, len(rankings)):
            x = np.array([order[g] for g in rankings[a]], dtype=float)
            y = np.array([order[g] for g in rankings[b]], dtype=float)
            xr = np.argsort(np.argsort(x)); yr = np.argsort(np.argsort(y))
            rho = float(np.corrcoef(xr, yr)[0, 1])
            rhos.append(rho)
    return {"pairs": len(rhos), "min_spearman": round(min(rhos), 4),
            "mean_spearman": round(float(np.mean(rhos)), 4),
            "stable": bool(min(rhos) >= 0.7)}


# ------------------------------------------------------ M4 nested feature sets

# plan.md 6, the M4 table. At Stage A, sets C and D add only groups that are
# zero-filled until Stage B/C - so C and D are expected to be INDISTINGUISHABLE
# from B here. That is a property of the rung, not a bug, and it is reported.
NESTED_SETS = {
    "A": ["portal_rel_pos", "craft_vel", "speed", "closing_speed_portal", "bearing_error",
          "fuel_fraction", "craft_mass", "control_authority", "time_fraction"],
    "B": None,   # A + gravity field
    "C": None,   # B + nearest-body data
    "D": None,   # C + per-body top-k
    "E": None,   # D + engineered invariants
}
NESTED_SETS["B"] = NESTED_SETS["A"] + ["a_grav", "a_grav_magdir", "potential"]
NESTED_SETS["C"] = NESTED_SETS["B"] + ["nearest_body", "tca_nearest", "nearest_rel_vel", "dphi_dt"]
NESTED_SETS["D"] = NESTED_SETS["C"] + ["body_count", "perbody_topk"]
NESTED_SETS["E"] = NESTED_SETS["D"] + ["orbital_energy", "angular_momentum", "eccentricity_vec"]


def groups_outside(keep: list[str], stage: str) -> list[str]:
    """Everything active at this stage that the nested set excludes -> ablate it."""
    return [g for g in active_groups(stage) if g not in keep]


async def m4_nested(pool: SandboxPool, stage: str, cfg: dict, keys: list, *,
                    generations: int, n_eps: int, lr: float, seed: int = 0,
                    log=print) -> dict:
    """Train a fresh policy per nested set and compare final performance.

    The smallest set whose curve matches the full set is the empirically necessary
    variable set (plan.md 6, M4).
    """
    from controller.train import Trainer

    out = {}
    for name, keep in NESTED_SETS.items():
        drop = groups_outside(keep, stage)
        tr = Trainer(stage, cfg, seed=seed, lr=lr)
        curve = []
        for gen in range(1, generations + 1):
            w = net.pack(tr.params)
            pl = [worker_input(stage, cfg, weights_b64=w, key=keys[i % len(keys)],
                               seed=500_000 + 1000 * gen + i, policy_version=tr.version,
                               obs_ablate=drop) for i in range(n_eps)]
            outs, _ = await pool.map(pl, f"m4-{name}-g{gen}")
            if not outs:
                continue
            tr.update(outs, tr.beta(gen, generations))
            curve.append({"generation": gen,
                          "mean_reward": round(float(np.mean([o["reward"] for o in outs])), 3),
                          "success_rate": round(float(np.mean([bool(o["success"]) for o in outs])), 3),
                          "min_portal_dist": round(float(np.mean([o["min_portal_dist"] for o in outs])), 4)})
        tail = curve[-max(1, len(curve) // 5):]
        out[name] = {
            "groups": keep, "n_ablated": len(drop), "generations": len(curve),
            "final_mean_reward": round(float(np.mean([c["mean_reward"] for c in tail])), 3),
            "final_success_rate": round(float(np.mean([c["success_rate"] for c in tail])), 3),
            "final_min_portal_dist": round(float(np.mean([c["min_portal_dist"] for c in tail])), 4),
            "curve": curve,
        }
        log(f"  M4 set {name}: {len(keep)} groups kept, {len(drop)} ablated -> "
            f"reward {out[name]['final_mean_reward']:+.2f}  "
            f"success {out[name]['final_success_rate']:.2f}  "
            f"closest {out[name]['final_min_portal_dist']:.3f}")
    return out


# ------------------------------------------------------------------- driver

async def study(stage: str, *, repeats: int, n_eps: int, pool_size: int,
                with_m4: bool, m4_generations: int, m4_lr: float, quiet: bool = False):
    cfg = load_stage_config(stage)
    keys_file = load_stage_keys(stage)
    eval_keys = keys_file["eval_held_out"]
    rd = results_dir(stage)
    champ_path = rd / "champion.npz"
    if not champ_path.exists():
        raise SystemExit(f"no champion at {champ_path} - run controller.train first (A8)")

    params = load_champion(champ_path)
    weights = net.pack(params)
    groups = active_groups(stage)

    def log(*a):
        if not quiet:
            print(*a, flush=True)

    log(f"[necessity] stage {stage}  {len(groups)} active groups  "
        f"repeats={repeats}  episodes/arm={n_eps}  pool={pool_size}")
    t0 = time.time()

    async with AsyncDaytona(daytona_config()) as d:
        await ensure_snapshot(d, quiet=True)
        async with SandboxPool(d, pool_size) as pool:
            # champion trajectories: the observation pool for the resample arm, and
            # the sample M3 differentiates over.
            probe = [worker_input(stage, cfg, weights_b64=weights,
                                  key=eval_keys[i % len(eval_keys)], seed=4242 + i)
                     for i in range(n_eps)]
            champ_runs, _ = await pool.map(probe, "probe")
            obs_all = np.concatenate([np.asarray(o["observations"], dtype=np.float32)
                                      for o in champ_runs])
            log(f"[necessity] champion probe: {len(champ_runs)} rollouts, "
                f"{obs_all.shape[0]} observations, "
                f"success={np.mean([o['success'] for o in champ_runs]):.2f}")

            rng = np.random.default_rng(0)
            take = rng.choice(obs_all.shape[0], size=min(200, obs_all.shape[0]), replace=False)
            obs_pool = {g: obs_all[np.ix_(take, np.arange(*spec.OBS_LAYOUT_2D[g][:2]))].tolist()
                        for g in groups}

            log("[necessity] M2 ablation + permutation on the fleet ...")
            raw = await m2_ablation(pool, stage, cfg, weights, eval_keys, groups,
                                    repeats, n_eps, 90_000, obs_pool)

            m4 = None
            if with_m4:
                log(f"[necessity] M4 nested sets ({m4_generations} generations each) ...")
                m4 = await m4_nested(pool, stage, cfg, eval_keys,
                                     generations=m4_generations, n_eps=n_eps,
                                     lr=m4_lr, log=log)

    m2 = m2_scores(raw, groups)
    m1 = m1_gate(params, stage)
    m3 = m3_saliency(params, obs_all, stage)

    span = max(1e-9, max(v["delta_return_zeroed"] for v in m2["per_group"].values()))
    combined = {g: {
        "m1_gate": m1[g],
        "m2_delta_return_zeroed": m2["per_group"][g]["delta_return_zeroed"],
        "m2_delta_return_resampled": m2["per_group"][g]["delta_return_resampled"],
        "m2_delta_success_zeroed": m2["per_group"][g]["delta_success_zeroed"],
        "m3_saliency": m3[g],
        "corroborated_by_resampling": corroborated(
            m2["per_group"][g]["delta_return_zeroed"],
            m2["per_group"][g]["delta_return_resampled"], span),
        "verdict": classify(m2["per_group"][g]["delta_return_zeroed"],
                            m2["per_group"][g]["delta_success_zeroed"], span),
    } for g in groups}

    ranking = sorted(groups, key=lambda g: -combined[g]["m2_delta_return_zeroed"])
    # Rank each repeat against ITS OWN baseline: a repeat's absolute returns drift
    # with the action seeds, but the ordering is what the gate is about.
    per_repeat = []
    for r in range(repeats):
        b = raw["baseline"][r][0]
        per_repeat.append(sorted(groups, key=lambda g: -(b - raw["zero"][g][r][0])))
    stability = rank_stability(per_repeat)

    # The same question asked of the low-variance statistic. If the return ranking is
    # unstable but this one is not, the instability is the sparse goal bonus, not the
    # absence of a real ordering - and that distinction is the difference between
    # "needs more episodes" and "there is nothing here".
    per_repeat_dist = []
    for r in range(repeats):
        bd = raw["baseline"][r][2]
        per_repeat_dist.append(sorted(groups, key=lambda g: -(raw["zero"][g][r][2] - bd)))
    stability_dist = rank_stability(per_repeat_dist)
    verdicts = verdict_stability(raw, groups, repeats)
    topk = topk_stability(per_repeat, k=3)

    result = {
        "stage": stage,
        "obs_spec_hash": spec.obs_spec_hash(2),
        "champion": json.loads(str(np.load(champ_path, allow_pickle=True)["_meta"])),
        "repeats": repeats, "episodes_per_arm": n_eps,
        "wallclock_s": round(time.time() - t0, 1),
        "baseline": m2["baseline"],
        "groups": combined,
        "per_group_raw": m2["per_group"],
        "ranking_by_m2": ranking,
        "ranking_by_portal_dist": sorted(
            groups, key=lambda g: -m2["per_group"][g]["delta_portal_dist_zeroed"]),
        "ranking_per_repeat": per_repeat,
        "stability": stability,
        "stability_portal_dist": stability_dist,
        "stability_verdict": verdicts,
        "stability_top3": topk,
        "measured_vs_predicted": {
            g: {"predicted": PREDICTED_A.get(g), "measured": combined[g]["verdict"],
                "agrees": PREDICTED_A.get(g) == combined[g]["verdict"]}
            for g in groups},
        "warning": (None if m2["baseline"]["success"] >= 0.5 else
                    f"baseline success is only {m2['baseline']['success']:.2f} - delta-return "
                    f"importances are dominated by whether a rollout happens to capture, "
                    f"so read the closest-approach column as the reliable one"),
        "m4_nested": m4,
    }
    (rd / "necessity.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="A9 - necessity study M1-M4")
    ap.add_argument("--stage", default="A")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=16, help="episodes per arm per repeat")
    ap.add_argument("--pool", type=int, default=8, help="concurrent sandboxes")
    ap.add_argument("--with-m4", action="store_true", help="also train the nested sets")
    ap.add_argument("--m4-generations", type=int, default=150)
    ap.add_argument("--m4-lr", type=float, default=0.015)
    a = ap.parse_args()

    res = asyncio.run(study(a.stage, repeats=a.repeats, n_eps=a.episodes,
                            pool_size=a.pool, with_m4=a.with_m4,
                            m4_generations=a.m4_generations, m4_lr=a.m4_lr))

    print("\n" + "=" * 88)
    print(f"  Stage {res['stage']} necessity - baseline return {res['baseline']['return']:+.2f}, "
          f"success {res['baseline']['success']:.2f}")
    print("=" * 88)
    print(f"  {'group':<22}{'M1 gate':>9}{'M2 dR(0)':>10}{'+/-':>7}{'M2 dSucc':>10}"
          f"{'M2 dDist':>10}{'M3 sal':>8}  verdict")
    for g in res["ranking_by_m2"]:
        v, raw_g = res["groups"][g], res["per_group_raw"][g]
        print(f"  {g:<22}{v['m1_gate']:>9.3f}{v['m2_delta_return_zeroed']:>10.2f}"
              f"{raw_g['return_zeroed_spread']:>7.1f}{v['m2_delta_success_zeroed']:>10.2f}"
              f"{raw_g['delta_portal_dist_zeroed']:>10.3f}{v['m3_saliency']:>8.3f}  {v['verdict']}")
    if res.get("warning"):
        print("\n  ! " + res["warning"])
    st, sd = res["stability"], res["stability_portal_dist"]
    print("=" * 88)
    print(f"  rank stability over {res['repeats']} re-runs:")
    print(f"    by delta return         min Spearman {st['min_spearman']}  mean {st['mean_spearman']}")
    print(f"    by closest approach     min Spearman {sd['min_spearman']}  mean {sd['mean_spearman']}")
    v, tk = res["stability_verdict"], res["stability_top3"]
    print(f"    by VERDICT class        {v['n_stable']}/{v['n_groups']} groups keep their "
          f"verdict on every re-run" + (f"  (unstable: {', '.join(v['unstable_groups'])})"
                                        if v["unstable_groups"] else ""))
    print(f"    top-3 set identical     {tk['identical']}  {tk['intersection']}")
    print()
    print("  measured vs PREDICTED (plan.md 6.1 column A):")
    for g in res["ranking_by_m2"]:
        mp = res["measured_vs_predicted"][g]
        mark = "  " if mp["agrees"] else "<-"
        print(f"    {g:<22} predicted {str(mp['predicted']):<12} measured "
              f"{mp['measured']:<12} {mark}")
    ok = bool(st["stable"])
    print(f"\n  [{'PASS' if ok else 'FAIL'}] ranking stable across {res['repeats']} re-runs")
    print("GATE A9:", "PASS - necessity column written" if ok else "FAIL - unstable ranking")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
