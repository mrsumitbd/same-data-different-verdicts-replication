"""
compare_tracking_methods.py

Effect of the line-tracking method alone: the SAME v4 birth units (capped
sampling, squash/rebase merges only) tracked two ways.

  v2 tracking (data/survival_events_line_v4_ast.csv): blame of the birth line
      NUMBER at HEAD, first-touch fallback, renamed files counted as deleted.
  v3 tracking (data/survival_events_line_v4units_v5track.csv, from
      measure_retention_v5.py --phase track, same tracker as the main data): reverse blame by line identity, deaths
      verified in the diff; unverified units excluded.

Reports modification rates and unadjusted Cox HRs (tier vs human) and the
repository-stratified analytic HR (no bootstrap) for each tracking method.

On the conference born lines (DATA_TAG=v6conf: position-based units
data/survival_events_line_pos_v6conf_ast.csv and their identity-tracker twins
data/survival_events_line_pos_v6conf_idtrack.csv, which carry the conference
tracker's result as conf_* columns) a third method is added on the same units:
  conference tracking (earlier_study/scripts/mine_survival.py): blame
      --reverse without -M/-C/--first-parent, death = first later commit that
      touches the file, not verified.
plus the pairwise agreement of the three methods (modified vs not) on the
units all three observe.

Usage:
    python src/compare_tracking_methods.py
Output:
    results/tracking_method_comparison.txt
"""

import os
import sys

import pandas as pd
from lifelines import CoxPHFitter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.constants import OBSERVATION_CUTOFF
from config.observation_window import apply_observation_window
from config.paths import P
from analyze_rq1_master import add_two_tier_column, is_generated, OUTLIER_PR_KEYS


def prep(df):
    df = add_two_tier_column(df)
    df = df[~df['unit_id'].apply(is_generated)]
    df = df[~pd.Series(list(zip(df['slug'], df['pr_number'])), index=df.index).isin(OUTLIER_PR_KEYS)]
    return df


def cox(df, tier, strata=False):
    x = df[df['two_tier'].isin([tier, 'human'])]
    d = pd.DataFrame({'d': x['duration_days'], 'e': x['event'].astype(int), 't': (x['two_tier'] == tier).astype(int)})
    if strata:
        d['s'] = x['slug'].values
        keep = d.groupby('s')['t'].transform('nunique') == 2
        d = d[keep]
        d['c'] = d['s']
        c = CoxPHFitter().fit(d, 'd', 'e', strata=['s'], cluster_col='c')
    else:
        c = CoxPHFitter().fit(d, 'd', 'e')
    r = c.summary.loc['t']
    return f"HR={r['exp(coef)']:.3f} [{r['exp(coef) lower 95%']:.3f}, {r['exp(coef) upper 95%']:.3f}] p={r['p']:.2g}"


def main():
    v2 = prep(apply_observation_window(pd.read_csv(P('data/survival_events_line_v4_ast.csv')), label='v2'))
    v3 = pd.read_csv(P('data/survival_events_line_v4units_v5track.csv'), low_memory=False)
    conf = None
    if 'conf_event' in v3.columns:
        conf = v3.copy()
        conf['event'] = conf['conf_event'].astype(int)
        dd = pd.to_datetime(conf['conf_death_date'])
        bd = pd.to_datetime(conf['birth_date'])
        end = dd.where(conf['event'] == 1, pd.Timestamp(OBSERVATION_CUTOFF))
        conf['duration_days'] = ((end - bd).dt.total_seconds() / 86400).clip(lower=0.001)
        conf['death_date'] = conf['conf_death_date']
        conf = prep(apply_observation_window(conf, label='conference'))
    n_unv = int((v3['event_type'] == 'unverified').sum())
    v3 = v3[v3['event_type'] != 'unverified'].copy()
    v3['event'] = v3['event'].astype(int)
    v3 = prep(apply_observation_window(v3, label='v3'))
    lines = ['=' * 70, 'Tracking method on the same v4 birth units', '=' * 70,
             f'v2 units {len(v2):,}; v3 units {len(v3):,} (unverified excluded: {n_unv:,})', '']
    methods = [('v2 (line number)', v2), ('v3 (identity)', v3)] + ([('conference (blame --reverse, file touch)', conf)]
                                                                 if conf is not None else [])
    if conf is not None:
        lines.insert(4, f'conference units {len(conf):,}')
    for name, df in methods:
        lines.append(f'--- {name}')
        lines.append('  modification rate %: ' + ', '.join(f'{t} {v:.1f}' for t, v in df.groupby('two_tier')['event'].mean().mul(100).items()))
        for t in ['human_supervised', 'fully_autonomous']:
            lines.append(f'  {t:17s} unadjusted {cox(df, t)}')
            lines.append(f'  {t:17s} repo-stratified (analytic, cluster-robust SE by repo) {cox(df, t, strata=True)}')
        lines.append('')
    if conf is not None:
        key = ['slug', 'pr_number', 'unit_id']
        m = (v2[key + ['event']].rename(columns={'event': 'v2'})
             .merge(v3[key + ['event']].rename(columns={'event': 'v3'}), on=key)
             .merge(conf[key + ['event']].rename(columns={'event': 'conf'}), on=key))
        lines.append(f'Pairwise agreement on modified vs not, units observed by all three ({len(m):,}):')
        for a_, b_ in [('v2', 'v3'), ('v2', 'conf'), ('v3', 'conf')]:
            lines.append(f'  {a_} vs {b_}: {(m[a_] == m[b_]).mean() * 100:.1f}%  '
                         f'(modified: {a_} {m[a_].mean() * 100:.1f}%, {b_} {m[b_].mean() * 100:.1f}%)')
        lines.append('')
    if 'event_type' in v3:
        lines.append('v3 event_type mix by tier (%):')
        lines.append(pd.crosstab(v3['two_tier'], v3['event_type'], normalize='index').mul(100).round(1).to_string())
    text = '\n'.join(lines) + '\n'
    open(P('results/tracking_method_comparison.txt'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
