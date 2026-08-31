"""Universe generation - one new branch per stage (plan.md 5.2).

Step 1 implements Stage A only: one fixed central star, a fixed portal, and a
fixed craft start pose on a partial orbit. Later steps add stages B, C, ...
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import G as G_CONST


def _lerp(t: float, easy, hard):
    """Curriculum interpolation: t=0 gives the easy anchor, t=1 the sampled value."""
    return easy + (hard - easy) * t


@dataclass
class Universe:
    dims: int
    body_pos: np.ndarray      # (n, d)  - static in Stage A/B
    body_mass: np.ndarray     # (n,)
    body_radius: np.ndarray   # (n,)    - contact => collision
    body_vel: np.ndarray      # (n, d)  - zeros in Stage A/B
    portal_pos: np.ndarray    # (d,)
    portal_vel: np.ndarray    # (d,)
    r_capture: float          # A7 capture radius
    v_capture: float          # A7 speed gate (relative to portal)
    craft_r0: np.ndarray      # (d,)
    craft_v0: np.ndarray      # (d,)
    fuel0: float
    craft_mass0: float
    a_thrust_mag: float       # fixed burn magnitude
    fuel_burn_c: float        # c in the discrete A5 fuel update
    arena_radius: float       # |r| beyond this => escape
    stage: str
    layout_id: int = 0        # which fixed layout (stage B onward); 0 at stage A
    seed_universe: int | None = None   # the seed a stage-D world was drawn from
    # Stage C: prescribed circular orbits about the star. Kinematic on purpose -
    # the planets orbit but do NOT pull on each other, so a universe stays a pure
    # function of its key and replays exactly (plan.md 3.4).
    orbit_a: np.ndarray | None = None       # (n,) orbital radius; 0 for the fixed star
    orbit_omega: np.ndarray | None = None   # (n,) angular rate
    orbit_theta0: np.ndarray | None = None  # (n,) angle at t = 0
    # Stage D: the orbits become genuine ellipses. Left None at Stage C, where
    # `bodies_at` keeps its exact circular arithmetic (that rung's recorded physics
    # checksums must not move).
    orbit_ecc: np.ndarray | None = None     # (n,) eccentricity, 0 = circular
    orbit_argp: np.ndarray | None = None    # (n,) argument of periapsis
    orbital_phase: float = 0.0

    def bodies_at(self, t: float):
        """(positions, velocities) at time t. Static universes ignore t."""
        if self.orbit_omega is None:
            return self.body_pos, self.body_vel
        if self.orbit_ecc is None:
            th = self.orbit_theta0 + self.orbit_omega * t
            c, s_ = np.cos(th), np.sin(th)
            pos = np.stack([self.orbit_a * c, self.orbit_a * s_], axis=1)
            vel = np.stack([-self.orbit_a * self.orbit_omega * s_,
                            self.orbit_a * self.orbit_omega * c], axis=1)
            return pos, vel
        return self._kepler_at(t)

    def _kepler_at(self, t: float):
        """Closed-form ellipse: solve Kepler's equation for the eccentric anomaly.

        Still kinematic - the planets follow prescribed ellipses about the star and
        do not pull on each other - so a Stage D universe stays a pure function of
        its seed and replays exactly, exactly as Stage C's circles do (plan.md 3.4).
        `orbit_a` is the SEMI-MAJOR axis here and `orbit_omega` the mean motion.
        """
        a, e = self.orbit_a, self.orbit_ecc
        m_anom = self.orbit_theta0 + self.orbit_omega * t
        # Newton on E - e sin E = M. Five iterations is plenty for e <= 0.3, and a
        # FIXED count keeps this a pure function of t (a tolerance-based loop would
        # make the trajectory depend on floating-point luck).
        ecc_anom = m_anom.copy()
        for _ in range(5):
            f = ecc_anom - e * np.sin(ecc_anom) - m_anom
            ecc_anom = ecc_anom - f / np.maximum(1.0 - e * np.cos(ecc_anom), 1e-9)
        cos_e, sin_e = np.cos(ecc_anom), np.sin(ecc_anom)
        sqrt1me2 = np.sqrt(np.maximum(1.0 - e * e, 0.0))
        # perifocal frame
        xp, yp = a * (cos_e - e), a * sqrt1me2 * sin_e
        edot = self.orbit_omega / np.maximum(1.0 - e * cos_e, 1e-9)
        vxp, vyp = -a * sin_e * edot, a * sqrt1me2 * cos_e * edot
        cw, sw = np.cos(self.orbit_argp), np.sin(self.orbit_argp)
        pos = np.stack([xp * cw - yp * sw, xp * sw + yp * cw], axis=1)
        vel = np.stack([vxp * cw - vyp * sw, vxp * sw + vyp * cw], axis=1)
        # the central star carries a = 0 and stays pinned at the origin
        fixed = a <= 1e-9
        if np.any(fixed):
            pos[fixed] = 0.0
            vel[fixed] = 0.0
        return pos, vel


def _stage_a() -> Universe:
    # One solar mass at the origin (G = 1). The craft starts at r = 2.5 with a
    # mild prograde velocity (circular speed here is sqrt(GM/r) = 0.632); the
    # portal sits further out on the same side, so the whole episode is a fight
    # against the inward pull - reach r = 3.75 and null the speed for capture,
    # without falling into the star or coasting off past the arena. The direct
    # craft->portal corridor never crosses the well, so a proportional pursuit
    # controller is enough to close the loop (the transfer-orbit problem is
    # Stage C's slingshot, not Stage A's).
    return Universe(
        dims=2,
        body_pos=np.array([[0.0, 0.0]]),
        body_mass=np.array([1.0]),
        body_radius=np.array([0.15]),
        body_vel=np.zeros((1, 2)),
        portal_pos=np.array([2.6, 2.7]),
        portal_vel=np.zeros(2),
        r_capture=0.35,
        v_capture=0.6,
        craft_r0=np.array([2.5, 0.0]),
        craft_v0=np.array([0.0, 0.55]),
        fuel0=1.0,
        craft_mass0=1.0,
        a_thrust_mag=0.30,        # Pi_1 = a_thrust_max / g0 ~ 1.9
        fuel_burn_c=0.30,
        arena_radius=12.0,
        stage="A",
    )


# --------------------------------------------------------------------- stage B

# Twelve layouts that differ in CHARACTER, not just coordinates (stages/
# stage_b_fixed_planets/keys.json). The craft flies from (2.5, 0) to the portal at
# (2.6, 2.7), so "the direct path" is the corridor at x ~ 2.5, y in 0..2.7 - each
# layout is placed relative to that corridor. Masses stay well under the star's 1.0
# so the far field is still Keplerian and the planets matter locally, which is
# exactly the question this rung asks: is the aggregated field plus one nearest-body
# term enough, or must the policy enumerate the bodies?
#
#   (x, y, mass, radius) per planet
FIXED_LAYOUTS = {
    0:  [(2.55, 1.35, 0.06, 0.13)],                                   # straddles the path
    1:  [(1.15, 1.40, 0.14, 0.16)],                                   # mass off to one side
    2:  [(2.10, 1.30, 0.04, 0.11), (3.05, 1.45, 0.04, 0.11)],         # flanking gate
    3:  [(0.80, 0.55, 0.22, 0.18)],                                   # heavy, near the star
    4:  [(1.00, 2.25, 0.03, 0.10), (3.85, 0.80, 0.03, 0.10),
         (3.40, 2.60, 0.03, 0.10)],                                   # three light, wide
    5:  [(2.78, 3.35, 0.09, 0.14)],                                   # behind the portal
    6:  [(1.95, 1.60, 0.05, 0.12), (3.25, 1.60, 0.05, 0.12)],         # corridor between
    7:  [(-1.50, 1.50, 0.26, 0.20)],                                  # far side, slingshot
    8:  [(4.50, -1.10, 0.02, 0.09)],                                  # near-empty field
    9:  [(1.45, 0.80, 0.05, 0.11), (1.35, 1.30, 0.05, 0.11),
         (1.80, 1.15, 0.05, 0.11)],                                   # dense flank cluster
    10: [(2.20, 0.90, 0.05, 0.12), (3.15, 2.05, 0.07, 0.13)],         # HELD OUT
    11: [(3.10, 2.90, 0.15, 0.15)],                                   # HELD OUT, by portal
}


def _stage_b(key=None) -> Universe:
    """Stage A's star plus stationary planets. The planets have mass and a contact
    radius; they do not move and they do not pull on each other (plan.md 5)."""
    u = _stage_a()
    layout_id = int((key or {}).get("layout_id", 0))
    if layout_id not in FIXED_LAYOUTS:
        raise SystemExit(f'{{"error": "unknown layout_id {layout_id} for stage B"}}')
    planets = FIXED_LAYOUTS[layout_id]

    u.body_pos = np.vstack([u.body_pos, [[p[0], p[1]] for p in planets]])
    u.body_mass = np.concatenate([u.body_mass, [p[2] for p in planets]])
    u.body_radius = np.concatenate([u.body_radius, [p[3] for p in planets]])
    u.body_vel = np.zeros((len(u.body_mass), 2))
    u.stage = "B"
    u.layout_id = layout_id
    return u


def _stage_c(key=None) -> Universe:
    """Stage B's layouts, now ORBITING. Each planet is put on the circular orbit
    that passes through its Stage B position, so a layout keeps its character while
    the field becomes time-dependent - the one new complication this rung adds.

    `orbital_phase` in [0, 1) advances every planet along its own orbit, which is
    what turns 12 layouts into an effectively continuous key space (plan.md 7.1).
    """
    u = _stage_b(key)
    phase = float((key or {}).get("orbital_phase", 0.0))
    m_star = float(u.body_mass[0])

    a = np.linalg.norm(u.body_pos, axis=1)                 # star sits at the origin: a = 0
    theta = np.arctan2(u.body_pos[:, 1], u.body_pos[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.where(a > 1e-9, np.sqrt(G_CONST * m_star / np.maximum(a, 1e-9) ** 3), 0.0)
    theta0 = theta + 2.0 * np.pi * phase

    u.orbit_a, u.orbit_omega, u.orbit_theta0 = a, omega, theta0
    u.orbital_phase = phase
    u.body_pos, u.body_vel = u.bodies_at(0.0)
    u.stage = "C"
    return u


# --------------------------------------------------------------------- stage D

# Kept in ONE place so the sampler is auditable: every number a Stage D universe
# depends on is here, and nothing else is random (plan.md 2.6, S1).
D_SAMPLING = dict(
    n_bodies=(2, 5),            # INCLUSIVE total incl. the star -> 1..4 planets
    mass=(0.02, 0.30),          # log-uniform, ~1.2 decades
    semi_major=(0.9, 4.4),      # annulus the planets orbit in
    ecc=(0.0, 0.28),            # "near-circular with eccentricity jitter"
    portal_radius=(3.3, 4.1),
    portal_arc_deg=(25.0, 100.0),   # angular separation from the craft start
    craft_radius=(2.2, 2.8),
    craft_tangential=(0.82, 1.02),  # fraction of the local circular speed
    clearance=0.45,             # keep planets off the craft start and the portal
    seed_offset=911_000,
    # Whether the corridor's ABSOLUTE orientation is randomised, as opposed to its
    # angular separation (the "arc"), which always is. See the note in sample_universe.
    # MEASURED both ways at FULL randomisation: fixing the orientation does NOT help
    # (C champion 0.031 fixed vs 0.062 random). Orientation dominates only when every
    # other knob is held at the easy anchor; once they are all open the rest of the
    # randomisation is just as damaging. So the full spec stands.
    randomise_orientation=True,
)


def sample_universe(seed: int, dims: int = 2, difficulty: float = 1.0) -> Universe:
    """A whole universe from one integer (plan.md 5.2, stage D onward).

    Everything a rollout needs is a pure function of `(seed, difficulty)`, so a Stage D
    world replays exactly like a Stage A/B/C one - the randomisation is across episodes,
    never within one. The SCALES are deliberately inherited from Stage A (star mass,
    thrust, fuel, capture radii, arena) so a warm start from the Stage C champion is
    a warm start into the same units, and only the layout and the noise are new.

    `difficulty` in [0, 1] is the curriculum knob the rung's plan calls for. At 0 the
    draw is deliberately Stage-C-like - one planet, a circular orbit, and the SAME
    craft/portal orientation Stages A-C always used. At 1 it is the full spec. The
    orientation is ramped along with N and eccentricity because that is what actually
    breaks the transfer: several observation components (`craft_vel`, `a_grav`) are in
    the absolute frame, so a rotated world is unrecognisable to a policy trained on one
    fixed corridor. MEASURED: warm-starting the Stage C champion straight into full
    randomisation sits at 0.04-0.08 held-out for 120 generations with no trend.

    Held-out evaluation must always use difficulty=1.0, or the gate is being measured
    on an easier distribution than the one claimed.
    """
    if dims != 2:
        raise NotImplementedError("stage D is 2D; the 3D lift is stage E")
    S = D_SAMPLING
    d = float(np.clip(difficulty, 0.0, 1.0))
    # The knobs are NOT ramped together. Measured, holding everything else at the easy
    # anchor and opening ONE knob for the Stage C champion:
    #     nothing randomised        0.62
    #     orientation free          0.17   <- dominant
    #     two planets               0.42
    #     three planets             0.21
    # Orientation is the cliff, because several observation components (`craft_vel`,
    # `a_grav`) are in the ABSOLUTE frame, so a rotated world is unrecognisable to a
    # policy trained on one fixed corridor. So orientation opens FIRST and alone, on
    # one-planet worlds: learn rotation invariance where the hazards are mildest, then
    # add bodies. Ramping both at once destroyed the transfer before either was learned.
    d_orient = float(np.clip(d * 2.0, 0.0, 1.0))          # full by the halfway point
    d_bodies = float(np.clip((d - 0.5) * 2.0, 0.0, 1.0))  # only after that
    d_ecc = d
    rng = np.random.default_rng(int(seed) + S["seed_offset"])
    u = _stage_a()                       # canonical scales, then overwrite the layout

    # --- craft start and portal: a random arc, not a random pair of points. Keeping
    # the SEPARATION in family with Stage A-C keeps difficulty comparable; letting the
    # absolute orientation run free is what stops the policy memorising one corridor.
    craft_r = _lerp(d, 2.5, rng.uniform(*S["craft_radius"]))
    # Stage A-C always started the craft at angle 0 with the portal ~47 degrees round.
    th_free = rng.uniform(0.0, 2.0 * np.pi)      # drawn either way, to keep the seed
    base_th = _lerp(d_orient, 0.0, th_free) if S["randomise_orientation"] else 0.0
    arc_deg = _lerp(d, 47.0, rng.uniform(*S["portal_arc_deg"]))
    sign = 1.0 if (d_orient < 1e-9 or rng.random() < 0.5) else -1.0
    arc = np.deg2rad(arc_deg) * sign
    portal_r = _lerp(d, 3.75, rng.uniform(*S["portal_radius"]))
    u.craft_r0 = craft_r * np.array([np.cos(base_th), np.sin(base_th)])
    u.portal_pos = portal_r * np.array([np.cos(base_th + arc), np.sin(base_th + arc)])
    u.portal_vel = np.zeros(2)
    v_circ = float(np.sqrt(G_CONST * u.body_mass[0] / craft_r))
    tangential = np.array([-np.sin(base_th), np.cos(base_th)]) * np.sign(arc)
    u.craft_v0 = tangential * v_circ * _lerp(d, 0.87, rng.uniform(*S["craft_tangential"]))

    # --- planets on kinematic ellipses about the star
    n_hi = int(round(_lerp(d_bodies, S["n_bodies"][0], S["n_bodies"][1])))
    n_planets = int(rng.integers(S["n_bodies"][0], n_hi + 1)) - 1
    pos, vel, mass, radius, sma, ecc, argp, theta0 = [], [], [], [], [], [], [], []
    for _ in range(n_planets):
        for _attempt in range(40):
            a_i = rng.uniform(*S["semi_major"])
            e_i = _lerp(d_ecc, 0.0, rng.uniform(*S["ecc"]))
            w_i = rng.uniform(0.0, 2.0 * np.pi)
            m0_i = rng.uniform(0.0, 2.0 * np.pi)
            m_i = 10.0 ** rng.uniform(np.log10(S["mass"][0]), np.log10(S["mass"][1]))
            rad_i = 0.08 + 0.30 * m_i ** (1.0 / 3.0)
            probe = Universe(
                dims=2, body_pos=np.zeros((1, 2)), body_mass=np.array([1.0]),
                body_radius=np.array([0.1]), body_vel=np.zeros((1, 2)),
                portal_pos=u.portal_pos, portal_vel=u.portal_vel,
                r_capture=u.r_capture, v_capture=u.v_capture, craft_r0=u.craft_r0,
                craft_v0=u.craft_v0, fuel0=1.0, craft_mass0=1.0,
                a_thrust_mag=u.a_thrust_mag, fuel_burn_c=u.fuel_burn_c,
                arena_radius=u.arena_radius, stage="D",
                orbit_a=np.array([a_i]), orbit_omega=np.array([
                    float(np.sqrt(G_CONST * u.body_mass[0] / a_i ** 3))]),
                orbit_theta0=np.array([m0_i]), orbit_ecc=np.array([e_i]),
                orbit_argp=np.array([w_i]))
            p_i = probe.bodies_at(0.0)[0][0]
            # A planet sitting on the start pose or inside the capture bubble makes
            # the episode unwinnable before the policy acts; reject those draws.
            clear = (np.linalg.norm(p_i - u.craft_r0) > S["clearance"] + rad_i
                     and np.linalg.norm(p_i - u.portal_pos) > u.r_capture + rad_i + 0.12
                     and a_i * (1.0 - e_i) > 0.55)
            if clear:
                break
        mass.append(m_i); radius.append(rad_i); sma.append(a_i)
        ecc.append(e_i); argp.append(w_i); theta0.append(m0_i)

    u.body_mass = np.concatenate([u.body_mass, mass]) if mass else u.body_mass
    u.body_radius = np.concatenate([u.body_radius, radius]) if radius else u.body_radius
    n = len(u.body_mass)
    u.orbit_a = np.concatenate([[0.0], sma]) if sma else np.zeros(n)
    u.orbit_ecc = np.concatenate([[0.0], ecc]) if ecc else np.zeros(n)
    u.orbit_argp = np.concatenate([[0.0], argp]) if argp else np.zeros(n)
    u.orbit_theta0 = np.concatenate([[0.0], theta0]) if theta0 else np.zeros(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        u.orbit_omega = np.where(u.orbit_a > 1e-9,
                                 np.sqrt(G_CONST * u.body_mass[0]
                                         / np.maximum(u.orbit_a, 1e-9) ** 3), 0.0)
    u.body_pos, u.body_vel = u.bodies_at(0.0)
    u.stage = "D"
    u.seed_universe = int(seed)
    return u


def _stage_d(key=None) -> Universe:
    """Key is `seed_universe`, NOT `seed`. `seed` is the action-sampling seed (S4) and
    the two must stay separable: the same world flown with different action draws is
    exactly the Monte-Carlo comparison the fleet exists to make (plan.md 5.3)."""
    k = key or {}
    return sample_universe(int(k.get("seed_universe", k.get("seed", 0))),
                           difficulty=float(k.get("curriculum", 1.0)))


_BUILDERS = {"A": lambda key=None: _stage_a(), "B": _stage_b, "C": _stage_c,
             "D": _stage_d}


def build_universe(stage: str, key=None) -> Universe:
    if stage not in _BUILDERS:
        raise NotImplementedError(f"stage {stage!r} lands in a later build-order step")
    return _BUILDERS[stage](key)
