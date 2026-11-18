"""
analyze_rq1_bot_excluded.py

Re-runs the RQ1 pooled and two-tier analyses after excluding:

  1. Bot-authored PRs from the human baseline
     Identified via username patterns in human_pull_request.parquet
     (Dependabot, Renovate, github-actions, mergify, etc.)

  2. Lockfile and manifest paths from ALL groups
     These files are churned continuously by dependency bots regardless
     of who authored the PR, and were not covered by the original
     generated-file exclusion list.

Motivation: the ~17.5% automation-bot contamination of the human
baseline is measured but not corrected in the main analysis. Dependency bots produce high-churn lines (version
pins, lockfile entries) that inflate the human modification rate
(71.1%), which may manufacture part of the naive agent retention
advantage (HR = 0.827). This script produces the bot-excluded
specification as a pre-correction step ("Step 0") in the ladder.

Produces the same four-step output as analyze_rq1_master.py but on
the cleaned cohort, so numbers can be compared directly.

Usage:
    python src/analyze_rq1_bot_excluded.py \
        --line-data data/survival_events_line_v5_analysis.csv \
        --hunk-data data/survival_events_hunk_v5.csv \
        --human-prs data/human_pull_request.parquet \
        --output results/rq1_bot_excluded_results.txt \
        --n-bootstrap 500

Robustness check (verified merges only; outputs to results/robustness_verified_merge/):
    VERIFIED_MERGE=1 python src/analyze_rq1_bot_excluded.py
"""

import argparse
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.agent_logins import is_agent_login
from config.observation_window import apply_observation_window, output_path
from config.paths import P

# ── Two-tier taxonomy ──────────────────────────────────────────────
HUMAN_SUPERVISED_TIERS = {'pair_programmer', 'batch_generator'}
FULLY_AUTONOMOUS_TIERS  = {'autonomous'}
TIERS = ['human_supervised', 'fully_autonomous']
SEED = 42

# Bot/agent logins: shared rule in config/agent_logins.py

# ── Lockfile / manifest path patterns (not in original exclusion) ──
LOCKFILE_PATTERNS = [
    r'package-lock\.json$',
    r'yarn\.lock$',
    r'npm-shrinkwrap\.json$',
    r'Cargo\.lock$',
    r'go\.sum$',
    r'go\.mod$',
    r'poetry\.lock$',
    r'Pipfile\.lock$',
    r'requirements.*\.txt$',
    r'setup\.cfg$',
    r'pyproject\.toml$',
    r'Gemfile\.lock$',
    r'composer\.lock$',
    r'pom\.xml$',
    r'build\.gradle(\.kts)?$',
    r'\.csproj$',
    r'packages\.lock\.json$',
]

# ── Original generated-file exclusion patterns ────────────────────
GENERATED_PATH_PATTERNS = [
    r'/dist/', r'/third_party/', r'/vendor/', r'/generated/',
    r'_pb2\.py$', r'\.min\.js$', r'immutable_entry',
    r'_generated\.', r'zz_generated\.',
]

OUTLIER_PR_KEYS = {('microsoft/typespec', 7115)}


def is_bot(username: str) -> bool:
    return is_agent_login(username)


def is_excluded_path(unit_id: str) -> bool:
    filepath = str(unit_id).split('::')[0]
    all_patterns = GENERATED_PATH_PATTERNS + LOCKFILE_PATTERNS
    return any(re.search(p, filepath) for p in all_patterns)


def add_two_tier_column(df: pd.DataFrame) -> pd.DataFrame:
    def map_tier(row):
        if not row['is_agent']:
            return 'human'
        if row['agent_tier'] in HUMAN_SUPERVISED_TIERS:
            return 'human_supervised'
        if row['agent_tier'] in FULLY_AUTONOMOUS_TIERS:
            return 'fully_autonomous'
        return 'unknown'
    df = df.copy()
    df['two_tier'] = df.apply(map_tier, axis=1)
    return df


def pooled_cox(agent: pd.DataFrame, human: pd.DataFrame,
               label: str, out: list) -> None:
    cox_df = pd.concat([
        agent[['duration_days', 'event']].assign(is_tier=1),
        human[['duration_days', 'event']].assign(is_tier=0)
    ], ignore_index=True).dropna()

    lr = logrank_test(
        agent['duration_days'], human['duration_days'],
        event_observed_A=agent['event'], event_observed_B=human['event']
    )
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='duration_days', event_col='event')
    s = cph.summary
    hr      = float(np.exp(s.loc['is_tier', 'coef']))
    ci_low  = float(np.exp(s.loc['is_tier', 'coef lower 95%']))
    ci_high = float(np.exp(s.loc['is_tier', 'coef upper 95%']))
    p_cox   = float(s.loc['is_tier', 'p'])

    kmf = KaplanMeierFitter()
    kmf.fit(agent['duration_days'], agent['event'])
    median_agent = kmf.median_survival_time_
    kmf.fit(human['duration_days'], human['event'])
    median_human = kmf.median_survival_time_

    out.append({
        'label': label,
        'n_agent': len(agent),
        'n_human': len(human),
        'mod_rate_agent': round(agent['event'].mean() * 100, 1),
        'mod_rate_human': round(human['event'].mean() * 100, 1),
        'median_agent': round(float(median_agent), 1),
        'median_human': round(float(median_human), 1),
        'hr':      round(hr, 4),
        'ci_low':  round(ci_low, 4),
        'ci_high': round(ci_high, 4),
        'p_cox':   f'{p_cox:.2e}',
        'logrank_p': f'{lr.p_value:.2e}',
    })


def fit_bootstrap_draw(strat_input: pd.DataFrame, sampled: np.ndarray) -> float:
    """Fit one cluster-bootstrap resample; NaN if the fit fails."""
    pieces = []
    for i, repo in enumerate(sampled):
        piece = strat_input[strat_input['slug'] == repo].copy()
        piece['_boot_stratum'] = f'{repo}__draw{i}'
        pieces.append(piece)
    resampled = pd.concat(pieces, ignore_index=True)
    try:
        cph_b = CoxPHFitter()
        cph_b.fit(resampled[['duration_days', 'event', 'is_tier', '_boot_stratum']],
                  duration_col='duration_days', event_col='event',
                  strata=['_boot_stratum'])
        return float(np.exp(cph_b.summary.loc['is_tier', 'coef']))
    except Exception:
        return np.nan


def parallel_bootstrap_hrs(strat_input: pd.DataFrame, rng: np.random.Generator,
                           n_bootstrap: int) -> list:
    """Bootstrap HRs in draw order (NaN = failed fit).

    All resamples are drawn up front from rng in the same order as a serial
    loop, so results match it; fits run in parallel with one BLAS thread per
    worker.
    """
    repos = strat_input['slug'].unique()
    draws = [rng.choice(repos, size=len(repos), replace=True)
             for _ in range(n_bootstrap)]
    with parallel_config(backend='loky', inner_max_num_threads=1):
        return Parallel(n_jobs=-1)(
            delayed(fit_bootstrap_draw)(strat_input, sampled) for sampled in draws
        )


def stratified_cox_bootstrap(agent: pd.DataFrame, human: pd.DataFrame,
                              label: str, out: list,
                              n_bootstrap: int = 500) -> None:
    combined = pd.concat([
        agent.assign(is_tier=1),
        human.assign(is_tier=0)
    ], ignore_index=True)
    combined['_cluster'] = combined['slug'].copy()
    strat_input = combined[['duration_days', 'event', 'is_tier',
                             'slug', '_cluster']].dropna()

    n_contributing = int(
        (strat_input.groupby('slug')['is_tier'].nunique() >= 2).sum()
    )
    n_total_strata = int(strat_input['slug'].nunique())

    hr_a = ci_l = ci_h = p_a = np.nan
    try:
        cph = CoxPHFitter()
        cph.fit(strat_input, duration_col='duration_days', event_col='event',
                strata=['slug'], cluster_col='_cluster')
        s = cph.summary
        hr_a = float(np.exp(s.loc['is_tier', 'coef']))
        ci_l = float(np.exp(s.loc['is_tier', 'coef lower 95%']))
        ci_h = float(np.exp(s.loc['is_tier', 'coef upper 95%']))
        p_a  = float(s.loc['is_tier', 'p'])
    except Exception as e:
        print(f'  Stratified Cox failed for {label}: {e}')

    rng = np.random.default_rng(SEED)
    hrs_boot = [hr_b for hr_b in parallel_bootstrap_hrs(strat_input, rng, n_bootstrap)
                if np.isfinite(hr_b)]

    out.append({
        'label': label,
        'n_contributing_strata': n_contributing,
        'n_total_strata': n_total_strata,
        'hr_analytic': round(hr_a, 4),
        'p_analytic': f'{p_a:.2e}',
        'boot_median_hr': round(np.median(hrs_boot), 4) if hrs_boot else np.nan,
        'boot_ci_low':  round(np.percentile(hrs_boot, 2.5), 4) if hrs_boot else np.nan,
        'boot_ci_high': round(np.percentile(hrs_boot, 97.5), 4) if hrs_boot else np.nan,
        'boot_n_valid': len(hrs_boot),
    })


def run(line_data: str, hunk_data: str, human_prs: str,
        output: str, n_bootstrap: int) -> None:

    print('Loading data...')
    line = apply_observation_window(pd.read_csv(line_data), label='line')
    hunk = apply_observation_window(pd.read_csv(hunk_data), label='hunk')
    hpr  = pd.read_parquet(human_prs)

    # ── Identify bot PRs in human baseline ────────────────────────
    bot_users = hpr[hpr['user'].apply(is_bot)]['user'].unique()
    print(f'\nBot usernames identified ({len(bot_users)}):')
    for u in sorted(bot_users):
        n = (hpr['user'] == u).sum()
        print(f'  {u}: {n} PRs')

    # Extract (slug, pr_number) for bot PRs
    hpr['slug'] = hpr['repo_url'].str.extract(r'repos/(.+)$')
    bot_pr_keys = set(zip(hpr[hpr['user'].apply(is_bot)]['slug'],
                          hpr[hpr['user'].apply(is_bot)]['number']))
    print(f'\nTotal bot (slug, pr_number) pairs to exclude: {len(bot_pr_keys)}')

    # Step 1 pooled on the cohort before exclusions, for the comparison summary
    orig_out = []
    pooled_cox(line[line['is_agent'] == True], line[line['is_agent'] == False],
               'pooled_all_agents', orig_out)

    # ── Apply exclusions to line data ─────────────────────────────
    line_orig = len(line)

    # 1. Remove bot human-baseline PRs
    line['_key'] = list(zip(line['slug'], line['pr_number']))
    n_bot_lines = ((~line['is_agent']) &
                   (line['_key'].isin(bot_pr_keys))).sum()
    line = line[~((~line['is_agent']) & (line['_key'].isin(bot_pr_keys)))]

    # 2. Remove lockfile/manifest paths (additional exclusion)
    n_lockfile_lines = line['unit_id'].apply(is_excluded_path).sum()
    line = line[~line['unit_id'].apply(is_excluded_path)]
    line = line.drop(columns=['_key'])

    print(f'\nLine exclusions:')
    print(f'  Original lines:         {line_orig:,}')
    print(f'  Bot human-baseline:     {n_bot_lines:,}')
    print(f'  Lockfile/manifest:      {n_lockfile_lines:,}')
    print(f'  Remaining:              {len(line):,}')

    # ── Also apply outlier PR exclusion ───────────────────────────
    hunk['_key'] = list(zip(hunk['slug'], hunk['pr_number']))
    hunk = hunk[~hunk['_key'].isin(OUTLIER_PR_KEYS)]
    hunk = hunk[~hunk['unit_id'].apply(is_excluded_path)]
    hunk = hunk[~((~hunk['is_agent']) &
                  (hunk['_key'].isin(bot_pr_keys)))]
    hunk = hunk.drop(columns=['_key'])

    # ── Apply two-tier taxonomy ────────────────────────────────────
    line = add_two_tier_column(line)
    hunk = add_two_tier_column(hunk)

    print(f'\nTwo-tier distribution after exclusions (line):')
    print(line['two_tier'].value_counts().to_string())

    human_line = line[line['two_tier'] == 'human'].copy()
    results = {}
    out_lines = []

    # ── STEP 1: Pooled vs human, bot-excluded ─────────────────────
    print('\n=== STEP 1 (bot-excluded): Pooled vs human ===')
    step1_out = []
    all_agent = line[line['is_agent'] == True].copy()
    pooled_cox(all_agent, human_line, 'pooled_all_agents_bot_excl', step1_out)
    results['step1'] = pd.DataFrame(step1_out)
    print(results['step1'].to_string(index=False))

    out_lines.append('=== STEP 1: Pooled vs human (bot-excluded) ===')
    out_lines.append(results['step1'].to_string(index=False))

    # ── STEP 2: Two-tier, bot-excluded ────────────────────────────
    print('\n=== STEP 2 (bot-excluded): Two-tier ===')
    step2_out = []
    for tier in TIERS:
        agent = line[line['two_tier'] == tier].copy()
        pooled_cox(agent, human_line, f'{tier}_bot_excl', step2_out)
    results['step2'] = pd.DataFrame(step2_out)
    print(results['step2'][['label', 'n_agent', 'mod_rate_agent',
                             'mod_rate_human', 'median_agent',
                             'median_human', 'hr', 'ci_low', 'ci_high',
                             'p_cox', 'logrank_p']].to_string(index=False))

    out_lines.append('\n=== STEP 2: Two-tier (bot-excluded) ===')
    out_lines.append(results['step2'].to_string(index=False))

    # ── STEP 3: AST structural, bot-excluded ──────────────────────
    print('\n=== STEP 3 (bot-excluded): AST structural ===')
    step3_out = []
    line_def2 = line.copy()
    line_def2['event_def2'] = (line_def2['ast_death'] == 1.0).astype(int)
    human_def2 = line_def2[line_def2['two_tier'] == 'human'][
        ['duration_days', 'event_def2', 'slug']
    ].rename(columns={'event_def2': 'event'})

    for tier in TIERS:
        agent_def2 = line_def2[line_def2['two_tier'] == tier][
            ['duration_days', 'event_def2', 'slug']
        ].rename(columns={'event_def2': 'event'})
        pooled_cox(agent_def2, human_def2, f'{tier}_ast_bot_excl', step3_out)
    results['step3'] = pd.DataFrame(step3_out)
    print(results['step3'][['label', 'n_agent', 'mod_rate_agent',
                             'mod_rate_human', 'hr', 'ci_low', 'ci_high',
                             'p_cox', 'logrank_p']].to_string(index=False))

    out_lines.append('\n=== STEP 3: AST structural (bot-excluded) ===')
    out_lines.append(results['step3'].to_string(index=False))

    # ── STEP 4: Repo-stratified + bootstrap, bot-excluded ─────────
    print(f'\n=== STEP 4 (bot-excluded): Repo-stratified + bootstrap ===')
    step4_out = []
    for tier in TIERS:
        print(f'  Running {tier}...')
        agent = line[line['two_tier'] == tier].copy()
        stratified_cox_bootstrap(agent, human_line, f'{tier}_bot_excl',
                                 step4_out, n_bootstrap)
        print(f'  Done: {step4_out[-1]}')
    results['step4'] = pd.DataFrame(step4_out)
    print(results['step4'].to_string(index=False))

    out_lines.append('\n=== STEP 4: Repo-stratified + bootstrap (bot-excluded) ===')
    out_lines.append(results['step4'].to_string(index=False))

    # ── Comparison summary ─────────────────────────────────────────
    summary = [
        '\n=== COMPARISON: Original vs Bot-excluded (Step 1 pooled) ===',
        f'  Original:     HR={orig_out[0]["hr"]}, '
        f'mod_rate_agent={orig_out[0]["mod_rate_agent"]}%, '
        f'mod_rate_human={orig_out[0]["mod_rate_human"]}%',
        f'  Bot-excluded: HR={step1_out[0]["hr"]}, '
        f'mod_rate_agent={step1_out[0]["mod_rate_agent"]}%, '
        f'mod_rate_human={step1_out[0]["mod_rate_human"]}%',
        f'  Change in human mod_rate: '
        f'{step1_out[0]["mod_rate_human"] - orig_out[0]["mod_rate_human"]:+.1f}pp',
        f'  Change in HR: {step1_out[0]["hr"] - orig_out[0]["hr"]:+.4f}',
    ]
    for s in summary:
        print(s)
    out_lines.extend(summary)

    with open(output, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f'\nAll results written to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--line-data',  default=P('data/survival_events_line_v5_analysis.csv'))
    parser.add_argument('--hunk-data',  default=P('data/survival_events_hunk_v5.csv'))
    parser.add_argument('--human-prs',  default=P('data/human_pull_request.parquet'))
    parser.add_argument('--output',     default=P('results/rq1_bot_excluded_results.txt'))
    parser.add_argument('--n-bootstrap', type=int, default=500)
    args = parser.parse_args()
    args.output = output_path(args.output)
    run(args.line_data, args.hunk_data, args.human_prs,
        args.output, args.n_bootstrap)