"""
sensitivity_observation_window.py

Re-runs the primary RQ1 survival analysis restricted to code units
born in PRs with a minimum observation window of 5, 8, and 10 months.

The observation cutoff is December 31, 2025. A PR merged on
August 1, 2025 has exactly a 5-month window. By restricting to
PRs merged on or before April 30, 2025 (8-month window) or
February 28, 2025 (10-month window), we test whether right-censoring
of recently merged PRs is masking long-term modification patterns.

Modifications after the observation cutoff are censored there
(config.observation_window, censoring only). The 5-month baseline uses the
same merge cutoff as the main analysis (birth_date < 2025-08-01), so it
reproduces the main RQ1 Step 2 models.

TWO-TIER TAXONOMY:
  human_supervised = pair_programmer + batch_generator
  fully_autonomous = autonomous (Devin only)
  human            = human baseline

Outputs:
    results/sensitivity_obs_window.csv  — modification rates and
                                          Cox HRs per tier per window

Usage:
    python src/sensitivity_observation_window.py \
        --line-data data/survival_events_line_v5_analysis.csv \
        --output results/sensitivity_obs_window.csv

Robustness check (verified merges only; outputs to results/robustness_verified_merge/):
    VERIFIED_MERGE=1 python src/sensitivity_observation_window.py
"""

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.observation_window import MERGE_CUTOFF, apply_observation_window, output_path
from config.paths import P


OBS_CUTOFF = datetime(2025, 12, 31)

HUMAN_SUPERVISED_TIERS = {"pair_programmer", "batch_generator"}
FULLY_AUTONOMOUS_TIERS = {"autonomous"}
TIERS = ["human_supervised", "fully_autonomous"]

WINDOW_MONTHS = {
    "5 months (baseline)": 5,
    "8 months":            8,
    "10 months":           10,
}


def months_before_cutoff(n_months: int) -> datetime:
    """Return the latest birth date that still gives n_months of window."""
    # Approximate: subtract n_months * 30 days
    return OBS_CUTOFF - timedelta(days=n_months * 30)


def add_two_tier_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'two_tier' column mapping agent_tier to the two-tier taxonomy."""
    def map_tier(row):
        if not row["is_agent"]:
            return "human"
        if row["agent_tier"] in HUMAN_SUPERVISED_TIERS:
            return "human_supervised"
        if row["agent_tier"] in FULLY_AUTONOMOUS_TIERS:
            return "fully_autonomous"
        return "unknown"
    df = df.copy()
    df["two_tier"] = df.apply(map_tier, axis=1)
    return df


def run_window(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Run KM + Cox for all tiers on a filtered dataframe."""
    rows = []
    human = df[df["two_tier"] == "human"]

    for tier in TIERS:
        agent = df[df["two_tier"] == tier]
        if len(agent) == 0:
            continue

        n_a = len(agent)
        n_h = len(human)
        mr_a = agent["event"].mean()
        mr_h = human["event"].mean()

        # Log-rank test
        lr = logrank_test(
            agent["duration_days"], human["duration_days"],
            event_observed_A=agent["event"],
            event_observed_B=human["event"],
        )

        # Cox regression (simplified — tier binary vs human)
        cox_df = pd.concat([
            agent[["duration_days", "event"]].assign(is_agent_tier=1),
            human[["duration_days", "event"]].assign(is_agent_tier=0),
        ]).dropna()

        hr = np.nan
        hr_ci_low = np.nan
        hr_ci_high = np.nan
        try:
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col="duration_days", event_col="event")
            summary = cph.summary
            hr = float(np.exp(summary.loc["is_agent_tier", "coef"]))
            hr_ci_low  = float(np.exp(summary.loc["is_agent_tier", "coef lower 95%"]))
            hr_ci_high = float(np.exp(summary.loc["is_agent_tier", "coef upper 95%"]))
        except Exception as e:
            print(f"  Cox failed for {tier} / {label}: {e}")

        rows.append({
            "window":       label,
            "tier":         tier,
            "n_agent":      n_a,
            "n_human":      n_h,
            "mr_agent":     round(mr_a, 4),
            "mr_human":     round(mr_h, 4),
            "delta_pp":     round((mr_a - mr_h) * 100, 2),
            "logrank_p":    round(lr.p_value, 4),
            "hr":           round(hr, 4) if not np.isnan(hr) else "N/A",
            "hr_ci_low":    round(hr_ci_low, 4) if not np.isnan(hr_ci_low) else "N/A",
            "hr_ci_high":   round(hr_ci_high, 4) if not np.isnan(hr_ci_high) else "N/A",
        })

    return pd.DataFrame(rows)


def run(line_data: str, output: str) -> None:
    df = add_two_tier_column(apply_observation_window(
        pd.read_csv(line_data, parse_dates=["birth_date"]),
        drop_late_births=False, label="line"))

    all_results = []

    for label, n_months in WINDOW_MONTHS.items():
        if n_months == 5:
            # Same merge cutoff as the main analysis
            cutoff_date = MERGE_CUTOFF
            filtered = df[df["birth_date"] < cutoff_date].copy()
        else:
            cutoff_date = months_before_cutoff(n_months)
            filtered = df[df["birth_date"] <= cutoff_date].copy()
        n_total = len(filtered)
        n_prs = len(filtered[["slug", "pr_number"]].drop_duplicates())

        print(f"\n--- Window: {label} ---")
        print(f"    Birth date cutoff : {cutoff_date.date()}")
        print(f"    Units retained    : {n_total:,}")
        print(f"    PRs retained      : {n_prs}")

        result = run_window(filtered, label)
        all_results.append(result)
        print(result[["tier", "n_agent", "mr_agent", "mr_human",
                       "delta_pp", "logrank_p", "hr"]].to_string(index=False))

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output, index=False)
    print(f"\n=== Results written to {output} ===")

    # Summary: do HRs remain directionally consistent?
    print("\n=== HR consistency across windows ===")
    for tier in TIERS:
        sub = combined[combined["tier"] == tier][["window", "hr"]]
        print(f"\n  {tier}:")
        print(sub.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--line-data", default=P("data/survival_events_line_v5_analysis.csv"))
    parser.add_argument("--output",    default=P("results/sensitivity_obs_window.csv"))
    args = parser.parse_args()
    args.output = output_path(args.output)
    run(args.line_data, args.output)