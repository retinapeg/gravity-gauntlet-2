# Gravity Gauntlet — Architecture

> **One universe gives an agent one experience. Daytona gives it many universes at once.**

## System Overview

Gravity Gauntlet is a reinforcement-learning research platform where a spacecraft policy learns to navigate gravitational fields by training across **parallel Daytona sandboxes**. Each rollout runs inside a real, isolated sandbox — not a simulation, not a mock.

```mermaid
graph LR
    subgraph HOST["🖥️ Host Machine (Controller)"]
        API["FastAPI + SSE<br/>controller/api.py"]
        TR["REINFORCE Trainer<br/>controller/train.py"]
        FAN["Fan-out<br/>controller/fanout.py"]
        NEC["Necessity Study<br/>controller/necessity.py"]
        REC["Generation Recorder<br/>controller/record.py"]
        ADP["Adapter<br/>controller/adapter.py"]
    end

    subgraph DAYTONA["☁️ Daytona Fleet (8 Sandboxes)"]
        S1["🚀 Sandbox 1<br/>sbx_7ca2…"]
        S2["🚀 Sandbox 2<br/>sbx_fd05…"]
        S3["🚀 Sandbox 3<br/>sbx_048b…"]
        S4["🚀 ..."]
        S8["🚀 Sandbox 8<br/>sbx_3a91…"]
    end

    subgraph BROWSER["🌐 Browser UI"]
        LIVE["Live Mode<br/>SSE Events"]
        REPLAY["Replay Mode<br/>Stored JSON"]
    end

    TR -->|"freeze θ + 8 distinct keys"| FAN
    FAN -->|"asyncio.gather"| S1 & S2 & S3 & S4 & S8
    S1 & S2 & S3 & S4 & S8 -->|"trajectories + rewards"| FAN
    FAN -->|"raw rollouts"| TR
    FAN -->|"adapted payloads"| ADP
    ADP -->|"UI payloads"| REC
    REC -->|"gen_N.json"| REPLAY
    API -->|"SSE lifecycle events"| LIVE
    TR -->|"∇θ J(θ) → Adam step"| TR
    NEC -->|"M1-M4 ablation"| FAN

    style DAYTONA fill:#1a1a2e,stroke:#3FD6C4,color:#E9ECF7
    style HOST fill:#151C31,stroke:#F2B33D,color:#E9ECF7
    style BROWSER fill:#111728,stroke:#8C6BE0,color:#E9ECF7
```

---

## Training Loop — One Generation

Each press of **TRAIN NEXT GENERATION** executes the full RL loop end-to-end against real Daytona infrastructure:

```mermaid
sequenceDiagram
    participant UI as 🌐 Browser
    participant API as FastAPI
    participant Trainer as REINFORCE
    participant Fan as Fan-out
    participant D1 as Sandbox 1
    participant D2 as Sandbox 2
    participant DN as Sandbox 8

    UI->>API: POST /api/train/next
    API->>Trainer: freeze θ_k, serialize weights
    Trainer->>Fan: weights_b64 + 8 distinct keys

    par 8 concurrent sandboxes
        Fan->>D1: create(snapshot, key_1, θ_k)
        Fan->>D2: create(snapshot, key_2, θ_k)
        Fan->>DN: create(snapshot, key_8, θ_k)
    end

    Note over D1,DN: Each sandbox runs python -m gg.rollout_worker<br/>500 steps of Verlet integration

    par SSE events stream to UI
        D1-->>API: phase: creating → ready → running
        D2-->>API: phase: creating → ready → running
        DN-->>API: phase: creating → ready → running
    end

    D1-->>Fan: {trajectory, rewards, actions, logprobs}
    D2-->>Fan: {trajectory, rewards, actions, logprobs}
    DN-->>Fan: {trajectory, rewards, actions, logprobs}

    Fan->>Trainer: batch of 8 rollouts
    Note over Trainer: G_t = Σ γ^k r_{t+k}<br/>Â = normalize(G - baseline)<br/>loss = -(Â · log π) - β·H

    Trainer->>Trainer: ONE Adam step → θ_{k+1}
    Trainer-->>API: generation stats
    API-->>UI: SSE: generation_complete
```

---

## Repository Structure

```mermaid
graph TD
    subgraph GG["📦 gg/ — Worker Package<br/>(runs INSIDE Daytona sandbox)"]
        SPEC["spec.py<br/>Frozen obs layout<br/>38-dim vector, hash-locked"]
        PHYS["physics.py<br/>Softened gravity<br/>Velocity Verlet"]
        UNI["universe.py<br/>World builder<br/>Stages A/B/C branches"]
        NET["net.py<br/>MLP in pure numpy<br/>Forward + backward + Adam"]
        POL["policy.py<br/>NoThrust / Pursuit / MLP<br/>One act() interface"]
        ROLL["rollout_worker.py<br/>Episode loop<br/>GG_INPUT → JSON stdout"]
    end

    subgraph CTRL["📦 controller/ — Host Side<br/>(never simulates, only orchestrates)"]
        CFG["config.py<br/>Credentials + snapshot name"]
        SNAP["build_snapshot.py<br/>Debian + numpy + torch"]
        FOUT["fanout.py<br/>asyncio.gather × M<br/>SandboxPool"]
        ADAPT["adapter.py<br/>Worker → UI payload"]
        RECORD["record.py<br/>→ runs/generations/"]
        TRAIN["train.py<br/>REINFORCE + baseline"]
        NECESS["necessity.py<br/>M1-M4 ablation"]
        APIMOD["api.py<br/>FastAPI + SSE"]
    end

    subgraph WEB["🌐 web/ — Demo UI"]
        LIVE["live.html<br/>SSE-driven live view"]
        REPL["replay.html<br/>Stored generation playback"]
        INDEX["index.html<br/>Entry point"]
    end

    subgraph STAGES["📋 stages/ — Research Output"]
        SA["stage_a_fixed_star/"]
        SB["stage_b_fixed_planets/"]
        SC["stage_c_orbiting/"]
        SD["stage_d_stochastic/"]
        SE["stage_e_3d/"]
        SF["stage_f_pomdp/"]
    end

    SPEC --> ROLL
    PHYS --> ROLL
    UNI --> ROLL
    NET --> POL
    POL --> ROLL

    CFG --> SNAP & FOUT
    SNAP --> FOUT
    FOUT --> ADAPT
    ADAPT --> RECORD
    FOUT --> TRAIN
    TRAIN --> NECESS
    APIMOD --> FOUT

    style GG fill:#0d1b2a,stroke:#3FD6C4,color:#E9ECF7
    style CTRL fill:#1b2838,stroke:#F2B33D,color:#E9ECF7
    style WEB fill:#1a1a2e,stroke:#8C6BE0,color:#E9ECF7
    style STAGES fill:#16213e,stroke:#FF7A66,color:#E9ECF7
```

---

## Worker I/O Contract

The worker and controller share exactly one interface — `spec.py`. The worker rejects a mismatched `obs_spec_hash`, so the two can never silently drift.

```mermaid
flowchart LR
    subgraph INPUT["📥 GG_INPUT (env var)"]
        direction TB
        I1["stage: A|B|C"]
        I2["layout_id + orbital_phase"]
        I3["policy_weights_b64"]
        I4["obs_spec_hash: 16a7d04e"]
        I5["max_steps: 500, gamma: 0.99"]
    end

    subgraph WORKER["⚙️ rollout_worker.py"]
        direction TB
        W1["1. Verify obs_spec_hash"]
        W2["2. build_universe(stage, key)"]
        W3["3. Nondimensionalise (L₀, g₀, T₀)"]
        W4["4. 500 Verlet steps"]
        W5["5. Observe → Act → Step → Reward"]
    end

    subgraph OUTPUT["📤 stdout (one JSON line)"]
        direction TB
        O1["sandbox_id, reward, success"]
        O2["termination: portal|collision|timeout|escape"]
        O3["observations[500×38]"]
        O4["actions[500], logprobs[500]"]
        O5["trajectory[500×6], bodies_trajectory"]
    end

    INPUT --> WORKER --> OUTPUT

    style INPUT fill:#1a1a2e,stroke:#3FD6C4,color:#E9ECF7
    style WORKER fill:#151C31,stroke:#F2B33D,color:#E9ECF7
    style OUTPUT fill:#111728,stroke:#8C6BE0,color:#E9ECF7
```

---

## The Observation Superset (38 dimensions, frozen from Stage A)

```mermaid
graph TD
    subgraph OBS["Frozen Observation Vector — 38 dims (2D)"]
        subgraph KINE["Kinematic (7)"]
            K1["portal_rel_pos (2) — A"]
            K2["craft_vel (2) — A"]
            K3["speed (1) — A"]
            K4["closing_speed_portal (1) — A"]
            K5["bearing_error (1) — A"]
        end
        subgraph FIELD["Field (23)"]
            F1["a_grav (2) — A"]
            F2["a_grav_magdir (2) — A"]
            F3["potential (1) — A"]
            F4["nearest_body (4) — B"]
            F5["tca_nearest (1) — C"]
            F6["nearest_rel_vel (2) — C"]
            F7["dphi_dt (1) — C"]
            F8["body_count (1) — B"]
            F9["perbody_topk (9) — B"]
        end
        subgraph RES["Resource (3)"]
            R1["fuel_fraction (1) — A"]
            R2["craft_mass (1) — A"]
            R3["control_authority (1) — A"]
        end
        subgraph INV["Invariants (5)"]
            I1["time_fraction (1) — A"]
            I2["orbital_energy (1) — A"]
            I3["angular_momentum (1) — A"]
            I4["eccentricity_vec (2) — A"]
        end
    end

    OBS -->|"Groups labelled B/C are<br/>zero-filled until that stage"| NOTE["Necessity scores stay<br/>comparable across the ladder"]

    style OBS fill:#0d1b2a,stroke:#3FD6C4,color:#E9ECF7
    style KINE fill:#16213e,stroke:#3FD6C4,color:#E9ECF7
    style FIELD fill:#16213e,stroke:#F2B33D,color:#E9ECF7
    style RES fill:#16213e,stroke:#FF7A66,color:#E9ECF7
    style INV fill:#16213e,stroke:#8C6BE0,color:#E9ECF7
```

---

## Stage Ladder — Curriculum Design

Each rung adds **one physical complication** and asks: which Newtonian variables did that complication force into the observation?

```mermaid
graph TB
    subgraph LADDER["The Newtonian Variable Ladder"]
        A["🟩 Stage A — Fixed Star<br/>One body, deterministic<br/>Calibration rung<br/><b>A1–A9 implemented</b>"]
        B["🟨 Stage B — Fixed Planets<br/>Star + 2–3 stationary planets<br/>12 layouts, 8 per generation<br/><b>Implemented, training partial</b>"]
        C["🟩 Stage C — Orbiting<br/>Planets on ellipses<br/>Time-varying field<br/><b>Demo endpoint — 0.83/0.96 success</b>"]
        D["⬜ Stage D — Stochastic<br/>Random n-body + noise<br/><i>Plan only</i>"]
        E["⬜ Stage E — 3D Lift<br/>Inclined orbits, ℝ³<br/><i>Plan only</i>"]
        F["⬜ Stage F — POMDP<br/>Sensor-range masking<br/><i>Stretch goal</i>"]
    end

    A -->|"warm-start θ"| B
    B -->|"warm-start θ"| C
    C -.->|"planned"| D
    D -.->|"planned"| E
    E -.->|"planned"| F

    A -->|"Δ necessity"| AB["Column A→B:<br/>+ nearest_body?"]
    B -->|"Δ necessity"| BC["Column B→C:<br/>+ anticipatory vars?"]
    C -.->|"Δ necessity"| CD["Column C→D:<br/>+ stochastic vars?"]

    style A fill:#1a472a,stroke:#3FD6C4,color:#E9ECF7
    style B fill:#3d3d00,stroke:#F2B33D,color:#E9ECF7
    style C fill:#1a472a,stroke:#3FD6C4,color:#E9ECF7
    style D fill:#2a2a2a,stroke:#63709A,color:#9AA6C4
    style E fill:#2a2a2a,stroke:#63709A,color:#9AA6C4
    style F fill:#2a2a2a,stroke:#63709A,color:#9AA6C4
```

---

## Necessity Study — Four Probes (M1–M4)

The ablation study is the research output. It determines which variables the policy **must** observe to fly.

```mermaid
flowchart TB
    CHAMP["🏆 Stage Champion<br/>(trained policy)"] --> M1 & M2 & M3 & M4

    subgraph M1["M1 — Learned Input Gate"]
        M1D["σ(g) per group<br/>→1 load-bearing<br/>→0 unused"]
    end

    subgraph M2["M2 — Ablation & Permutation<br/>(THE HEADLINE)"]
        M2A["Zero a group → re-fly<br/>Resample from empirical dist"]
        M2B["ΔReturn = importance"]
        M2A --> M2B
    end

    subgraph M3["M3 — Gradient Saliency"]
        M3D["mean |∂logits/∂o_i|<br/>over champion trajectories<br/>No rollouts needed"]
    end

    subgraph M4["M4 — Nested Feature Sets"]
        M4D["Train fresh policy on<br/>each nested subset<br/>Smallest set that matches<br/>= empirically necessary set"]
    end

    M1 & M2 & M3 & M4 --> VERDICT["📊 Necessity Column<br/>necessary / marginal / unnecessary"]

    style CHAMP fill:#3d3d00,stroke:#F2B33D,color:#E9ECF7
    style M1 fill:#16213e,stroke:#3FD6C4,color:#E9ECF7
    style M2 fill:#16213e,stroke:#FF7A66,color:#E9ECF7
    style M3 fill:#16213e,stroke:#8C6BE0,color:#E9ECF7
    style M4 fill:#16213e,stroke:#F2B33D,color:#E9ECF7
    style VERDICT fill:#1a472a,stroke:#3FD6C4,color:#E9ECF7
```

---

## The Five-Beat Loop (per stage)

```mermaid
graph LR
    EXT["1️⃣ Extend<br/>Add universe branch<br/>in gg/universe.py"] --> FLIP["2️⃣ Flip<br/>Switch obs groups<br/>from zero → live"]
    FLIP --> TRAIN["3️⃣ Train<br/>REINFORCE over M<br/>Daytona sandboxes"]
    TRAIN --> ATTR["4️⃣ Attribute<br/>M1–M4 ablation<br/>on champion"]
    ATTR --> REC["5️⃣ Record<br/>necessity.json<br/>+ Δ vs previous rung"]

    style EXT fill:#16213e,stroke:#3FD6C4,color:#E9ECF7
    style FLIP fill:#16213e,stroke:#F2B33D,color:#E9ECF7
    style TRAIN fill:#16213e,stroke:#FF7A66,color:#E9ECF7
    style ATTR fill:#16213e,stroke:#8C6BE0,color:#E9ECF7
    style REC fill:#16213e,stroke:#3FD6C4,color:#E9ECF7
```

---

## Sandbox Lifecycle

```mermaid
stateDiagram-v2
    [*] --> CREATING: create(snapshot, key, θ)
    CREATING --> READY: sandbox provisioned
    READY --> RUNNING: exec rollout_worker
    RUNNING --> SUCCESS: portal capture (dist < r && speed < v)
    RUNNING --> CRASHED: body collision
    RUNNING --> TIMEOUT: steps exhausted
    RUNNING --> ESCAPED: left arena
    RUNNING --> WORKER_FAILED: error / timeout
    SUCCESS --> [*]: delete sandbox
    CRASHED --> [*]: delete sandbox
    TIMEOUT --> [*]: delete sandbox
    ESCAPED --> [*]: delete sandbox
    WORKER_FAILED --> [*]: delete sandbox
```

---

## Data Pipeline — Live vs Replay

```mermaid
flowchart LR
    subgraph LIVE_MODE["Live Mode"]
        D["Daytona Fleet"] -->|"SSE events"| API["FastAPI /api/events"]
        API -->|"real-time"| UI1["Browser<br/>live.html"]
        D -->|"rollouts"| REC2["record.py"]
        REC2 -->|"gen_N.json"| DISK["runs/generations/"]
    end

    subgraph REPLAY_MODE["Replay Mode"]
        DISK2["runs/generations/<br/>gen_N.json"] -->|"GET /api/generations/N"| UI2["Browser<br/>replay.html"]
    end

    NOTE["Same renderer.<br/>Same data format.<br/>Replay = genuine stored rollouts."]

    style LIVE_MODE fill:#16213e,stroke:#3FD6C4,color:#E9ECF7
    style REPLAY_MODE fill:#16213e,stroke:#F2B33D,color:#E9ECF7
```
