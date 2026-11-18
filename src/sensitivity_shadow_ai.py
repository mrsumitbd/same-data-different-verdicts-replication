"""
sensitivity_shadow_ai.py

Sensitivity analysis for shadow AI contamination in the human baseline.

We know 5.0% of human-baseline PRs may contain undetected AI assistance
(shadow AI). This script simulates three scenarios for that 5%:

  - Scenario A (pessimistic): shadow AI code is maximally stable
    (modification rate = 0%, i.e., all retained). This is the scenario
    most favourable to the null hypothesis.
  - Scenario B (neutral): shadow AI code has the same modification rate
    as the observed human-supervised tier (best-case agent tier).
  - Scenario C (optimistic): shadow AI code has the same modification rate
    as the observed human baseline (contamination has no effect).

For each scenario we re-compute the adjusted human baseline modification
rate and re-derive the implied hazard ratio using the relationship:

    HR_adjusted = log(1 - MR_agent) / log(1 - MR_human_adjusted)

This is an approximation valid for constant-hazard models and is used
only for sensitivity illustration. The exact Cox HR is not re-fitted
since the contaminated units are not individually identifiable.

TWO-TIER TAXONOMY:
  human_supervised = pair_programmer + batch_generator
  fully_autonomous = autonomous (Devin only)
  human            = human baseline

Usage:
    python src/sensitivity_shadow_ai.py \
        --line-data data/survival_events_line_v5_analysis.csv \
        --output results/sensitivity_shadow_ai.csv

Robustness check (verified merges only; outputs to results/robustness_verified_merge/):
    VERIFIED_MERGE=1 python src/sensitivity_shadow_ai.py
"""

import argparse
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.observation_window import apply_observation_window, output_path
from config.paths import P


# AI-attributed share of human-baseline PRs in the double-coded review sample
SHADOW_AI_RATE = 0.055       # 5.5% point estimate
SHADOW_AI_CI_LOW = 0.028     # 95% CI lower
SHADOW_AI_CI_HIGH = 0.089    # 95% CI upper

HUMAN_SUPERVISED_TIERS = {"pair_programmer", "batch_generator"}
FULLY_AUTONOMOUS_TIERS = {"autonomous"}
TIERS = ["human_supervised", "fully_autonomous"]


def adjusted_hr(mr_agent: float, mr_human_adj: float) -> float:
    """
    Approximate HR from modification rates using constant-hazard assumption.
    HR < 1 means agent tier has lower modification risk than human.
    """
    if mr_human_adj >= 1.0 or mr_agent >= 1.0:
        return float("nan")
    return np.log(1 - mr_agent) / np.log(1 - mr_human_adj)


def run(line_data: str, output: str) -> None:
    df = apply_observation_window(pd.read_csv(line_data), label='line')

    # Observed modification rates per tier
    tiers = {
        "human_supervised": df[df["is_agent"] & df["agent_tier"].isin(HUMAN_SUPERVISED_TIERS)],
        "fully_autonomous": df[df["is_agent"] & df["agent_tier"].isin(FULLY_AUTONOMOUS_TIERS)],
        "human":            df[df["is_agent"] == False],
    }

    obs = {}
    for name, sub in tiers.items():
        n = len(sub)
        events = sub["event"].sum()
        mr = events / n
        obs[name] = {"n": n, "events": int(events), "mod_rate": round(mr, 4)}

    mr_human_obs = obs["human"]["mod_rate"]
    mr_hs_obs = obs["human_supervised"]["mod_rate"]
    n_human = obs["human"]["n"]

    print("\n=== Observed modification rates ===")
    for name, v in obs.items():
        print(f"  {name:20s}: n={v['n']:>7,}  mod_rate={v['mod_rate']:.3f}")

    rows = []

    for contamination_rate, label_rate in [
        (SHADOW_AI_CI_LOW,  f"CI low ({SHADOW_AI_CI_LOW:.1%})"),
        (SHADOW_AI_RATE,    f"Point est. ({SHADOW_AI_RATE:.1%})"),
        (SHADOW_AI_CI_HIGH, f"CI high ({SHADOW_AI_CI_HIGH:.1%})"),
    ]:
        for scenario, shadow_mr in [
            ("A: shadow AI fully stable (MR=0.0)",        0.000),
            (f"B: shadow AI = human-supervised (MR={mr_hs_obs:.3f})", mr_hs_obs),
            ("C: shadow AI = human baseline (no effect)", mr_human_obs),
        ]:
            # Number of contaminated human units
            n_contaminated = contamination_rate * n_human

            # Adjusted human event count
            # Remove contaminated units and replace with scenario events
            events_clean = obs["human"]["events"] - (n_contaminated * mr_human_obs)
            events_shadow = n_contaminated * shadow_mr
            events_adj = events_clean + events_shadow
            mr_human_adj = events_adj / n_human

            for tier in TIERS:
                mr_agent = obs[tier]["mod_rate"]
                hr = adjusted_hr(mr_agent, mr_human_adj)
                rows.append({
                    "contamination_rate": label_rate,
                    "scenario": scenario,
                    "tier": tier,
                    "mr_agent": round(mr_agent, 4),
                    "mr_human_obs": round(mr_human_obs, 4),
                    "mr_human_adj": round(mr_human_adj, 4),
                    "hr_approx": round(hr, 4),
                    "hr_direction": "< 1 (agent more stable)" if hr < 1 else "> 1 (agent less stable)",
                })

    results = pd.DataFrame(rows)
    results.to_csv(output, index=False)
    print(f"\n=== Sensitivity results written to {output} ===")

    # Summary: does HR remain < 1 under all scenarios?
    print("\n=== HR range across all scenarios ===")
    for tier in TIERS:
        sub = results[results["tier"] == tier]["hr_approx"]
        print(f"  {tier:20s}: HR range [{sub.min():.4f}, {sub.max():.4f}]  "
              f"{'ALL < 1 (robust)' if sub.max() < 1 else 'SOME >= 1 (check)'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--line-data", default=P("data/survival_events_line_v5_analysis.csv"))
    parser.add_argument("--output",    default=P("results/sensitivity_shadow_ai.csv"))
    parser.add_argument("--rates", nargs=3, type=float, metavar=("LOW", "POINT", "HIGH"),
                        help="contamination rates (default 0.028 0.055 0.089, from the double-coded "
                             "human-baseline review sample)")
    args = parser.parse_args()
    if args.rates:
        SHADOW_AI_CI_LOW, SHADOW_AI_RATE, SHADOW_AI_CI_HIGH = args.rates
    args.output = output_path(args.output)
    run(args.line_data, args.output)