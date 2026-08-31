"""Gravity Gauntlet worker package (plan.md 4).

One strict interface shared by controller and Daytona worker: `spec` defines the
frozen observation layout + reward config; `universe` builds the world; `physics`
integrates it; `policy` acts; `rollout_worker` runs one episode and prints one
JSON line. Daytona never renders - it only calculates.
"""
