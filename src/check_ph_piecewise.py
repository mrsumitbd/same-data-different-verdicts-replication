"""
check_ph_piecewise.py

Proportional-hazards diagnosis and remedy for the tier-vs-human Cox models.

Schoenfeld tests (check_ph_assumption.py) reject proportionality in every
model. This script shows HOW the tier effect changes over time and re-fits
the models piecewise:

  - Diagnosis: the tier HR in successive time intervals since merge
    (0-7, 7-30, 30-90, 90-180, >180 days), unadjusted, as a figure.
  - Remedy: a piecewise Cox model (0-30, 30-90, >90 days) fitted per interval
    with delayed entry (units enter an interval only if unmodified at its
    start), both unadjusted and repository-stratified with cluster-robust SEs
    by repository. Within each interval the HR is an average over a short
    window, so proportionality matters much less.

Usage:
    python src/check_ph_piecewise.py
Output:
    results/ph_piecewise.txt, paper-overfeal-files/visuals/ph_piecewise.pdf
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.observation_window import apply_observation_window
from config.paths import P
from analyze_rq1_master import add_two_tier_column, is_generated, OUTLIER_PR_KEYS

TIERS = {'human_supervised': 'Human-supervised', 'fully_autonomous': 'Fully-autonomous'}
FINE = [0, 7, 30, 90, 180, np.inf]
PIECES = [0, 30, 90, np.inf]


def interval(df, a, b):
    x = df[df['duration_days'] > a].copy()
    x['entry'] = a
    x['stop'] = np.minimum(x['duration_days'], b)
    x['ev'] = ((x['event'] == 1) & (x['duration_days'] <= b)).astype(int)
    return x


def fit(df, tier, a, b, stratified):
    x = interval(df[df['two_tier'].isin([tier, 'human'])], a, b)
    d = pd.DataFrame({'entry': x['entry'], 'stop': x['stop'], 'ev': x['ev'],
                      't': (x['two_tier'] == tier).astype(int)})
    if stratified:
        d['s'] = x['slug'].values
        d = d[d.groupby('s')['t'].transform('nunique') == 2]
        d['c'] = d['s']
        c = CoxPHFitter().fit(d, 'stop', 'ev', entry_col='entry', strata=['s'], cluster_col='c')
    else:
        c = CoxPHFitter().fit(d, 'stop', 'ev', entry_col='entry')
    r = c.summary.loc['t']
    return r['exp(coef)'], r['exp(coef) lower 95%'], r['exp(coef) upper 95%'], r['p'], int(d['ev'].sum())


def label(a, b):
    return f'{int(a)}-{int(b)} d' if np.isfinite(b) else f'>{int(a)} d'


def main():
    df = add_two_tier_column(apply_observation_window(
        pd.read_csv(P('data/survival_events_line_v5_analysis.csv'), low_memory=False), label='line'))
    df = df[~df['unit_id'].apply(is_generated)]
    df = df[~pd.Series(list(zip(df['slug'], df['pr_number'])), index=df.index).isin(OUTLIER_PR_KEYS)]
    out = ['=' * 70, 'Proportional hazards: tier HR by time since merge', '=' * 70,
           '', 'Diagnosis (unadjusted, delayed entry):']
    diag = {}
    for t in TIERS:
        diag[t] = []
        for a, b in zip(FINE[:-1], FINE[1:]):
            hr, lo, hi, p, n = fit(df, t, a, b, False)
            diag[t].append((label(a, b), hr, lo, hi))
            out.append(f'  {t:17s} {label(a, b):>10s}: HR={hr:.3f} [{lo:.3f}, {hi:.3f}] p={p:.2g} events={n:,}')
    out += ['', 'Piecewise Cox (0-30, 30-90, >90 days):']
    for strat in (False, True):
        out.append(f'  {"repository-stratified, cluster-robust SE" if strat else "unadjusted"}:')
        for t in TIERS:
            for a, b in zip(PIECES[:-1], PIECES[1:]):
                hr, lo, hi, p, n = fit(df, t, a, b, strat)
                out.append(f'    {t:17s} {label(a, b):>10s}: HR={hr:.3f} [{lo:.3f}, {hi:.3f}] p={p:.2g} events={n:,}')
    text = '\n'.join(out) + '\n'
    open(P('results/ph_piecewise.txt'), 'w').write(text)
    print(text)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for k, (t, name) in enumerate(TIERS.items()):
        xs = np.arange(len(diag[t])) + (k - 0.5) * 0.18
        hr = [r[1] for r in diag[t]]
        err = [[r[1] - r[2] for r in diag[t]], [r[3] - r[1] for r in diag[t]]]
        ax.errorbar(xs, hr, yerr=err, fmt='o', capsize=3, label=name, color=['#0072B2', '#D55E00'][k])
    ax.axhline(1, color='#555555', lw=0.8, ls='--')
    ax.set_xticks(range(len(diag['human_supervised'])))
    ax.set_xticklabels([r[0] for r in diag['human_supervised']])
    ax.set_ylabel('HR vs human (95% CI)')
    ax.set_xlabel('Time since merge')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(P('paper-overfeal-files/visuals/ph_piecewise.pdf'))
    fig.savefig(P('paper-overfeal-files/visuals/ph_piecewise.png'), dpi=200)


if __name__ == '__main__':
    main()
