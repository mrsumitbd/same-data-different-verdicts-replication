"""
c3_refit_r1.py 

RQ2 Definition 2 (AST-structural) refitted with the lines whose first
modification is leaf-only followed past that edit
(results_v6/round1/c3_ast_continued.csv, src/measure_ast_continue_r1.py).

For those lines: event at their first structural modification, or censored at
the cutoff / where they could not be followed further. All other lines are
unchanged: a structural first modification is an event, an unknown-AST first
modification censors the line there (as now), a leaf-only line deletion
censors there (as now).

Reports, per tier:
  - share of lines (and of modified lines with known AST status) whose first
    modification is leaf-only;
  - Definition 2 current vs continued: modification rate; across-repository HR
    (lifelines, model-based SE as in the paper); within-repository HR with the
    500-resample repository bootstrap (seed 42; src/boot_r1.py);
and writes results_v6/round1/work/c3_def2_continued.csv for the R clustered-SE
fits (src/c3_cluster_r1.R).

Usage: DATA_TAG=v6confnv RESULTS_DIR=results_v6 python src/c3_refit_r1.py
Output: results_v6/round1/c3_def2_continued.txt
"""

import os
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
from config.constants import OBSERVATION_CUTOFF
from config.observation_window import apply_observation_window
from config.paths import P
from analyze_rq1_master import add_two_tier_column
from boot_r1 import boot_within

TIERS = ['human_supervised', 'fully_autonomous']
KEY = ['slug', 'pr_number', 'unit_id', 'birth_sha']


def unadj(df, tier, col):
    x = df[df['two_tier'].isin([tier, 'human'])]
    d = pd.DataFrame({'d': x['duration_days'], 'e': x[col], 't': (x['two_tier'] == tier).astype(int)})
    s = CoxPHFitter().fit(d, 'd', 'e').summary.loc['t']
    return s['exp(coef)'], s['exp(coef) lower 95%'], s['exp(coef) upper 95%']


def main():
    line = add_two_tier_column(apply_observation_window(
        pd.read_csv(P('data/survival_events_line_v5_analysis.csv'), low_memory=False), label='line'))
    line['e2'] = (line['ast_death'] == 1.0).astype(int)
    c = pd.read_csv('results_v6/round1/c3_ast_continued.csv')
    m = line[KEY].reset_index().merge(c, on=KEY, how='inner').set_index('index')
    print(f'continued lines matched to the analysis data: {len(m):,} of {len(c):,}')
    bd = pd.to_datetime(line.loc[m.index, 'birth_date'])
    fd = pd.to_datetime(m['final_date'])
    end = fd.where(m['final_reason'] != 'reached end snapshot', pd.Timestamp(OBSERVATION_CUTOFF))
    line['d2c'] = line['duration_days']
    line['e2c'] = line['e2']
    line.loc[m.index, 'd2c'] = ((end - bd).dt.total_seconds() / 86400).clip(lower=0.001).values
    line.loc[m.index, 'e2c'] = (m['final_status'] == 'event').astype(int).values
    line['leaf_first'] = 0
    lf = (line['event'] == 1) & (line['ast_death'] == 0)
    line.loc[lf, 'leaf_first'] = 1

    out = ['=' * 78, 'C3. Definition 2 with lines followed past leaf-only edits', '=' * 78, '',
           f'leaf-only first modifications followed: {len(m):,} content changes (+ {int((lf & (line["event_type"] == "line_deleted")).sum()) if "event_type" in line else "n/a"} leaf-only deletions, censored there as before)',
           'final status of the followed lines: ' + ', '.join(f'{k} {v:,}' for k, v in m['final_reason'].value_counts().items()), '',
           'A. Share of lines whose first modification is leaf-only']
    for t in TIERS + ['human']:
        s = line[line['two_tier'] == t]
        mk = s[(s['event'] == 1) & s['ast_death'].notna()]
        out.append(f'  {t:17s} {s["leaf_first"].mean() * 100:5.2f}% of all lines; '
                   f'{(mk["ast_death"] == 0).mean() * 100:5.1f}% of modified lines with known AST status '
                   f'(n lines {len(s):,}, modified known {len(mk):,})')
    out += ['', 'B. Definition 2: current (censored at the first leaf-only edit) vs continued', '']
    rows = []
    for t in TIERS:
        r = {'tier': t}
        for lab, dcol, ecol in [('current', 'duration_days', 'e2'), ('continued', 'd2c', 'e2c')]:
            x = line[['two_tier', 'slug', dcol, ecol]].rename(columns={dcol: 'duration_days', ecol: 'ev'})
            hr, lo, hi = unadj(x, t, 'ev')
            a = x[x['two_tier'] == t]
            h = x[x['two_tier'] == 'human']
            df = pd.concat([a.assign(is_tier=1), h.assign(is_tier=0)], ignore_index=True).rename(columns={'ev': 'event'})
            b = boot_within(df, f'c3_within_def2_{lab}_{t}')
            r.update({f'{lab}_mr_tier': a['ev'].mean() * 100, f'{lab}_mr_human': h['ev'].mean() * 100,
                      f'{lab}_across_hr': hr, f'{lab}_across_lo': lo, f'{lab}_across_hi': hi,
                      f'{lab}_within_hr': b['hr'], f'{lab}_within_lo95': b['lo95'], f'{lab}_within_hi95': b['hi95'],
                      f'{lab}_within_lo975': b['lo975'], f'{lab}_within_hi975': b['hi975']})
            out.append(f'  {t:17s} {lab:9s} mod. rate {a["ev"].mean() * 100:5.1f}% vs human {h["ev"].mean() * 100:5.1f}%   '
                       f'across HR {hr:.3f} [{lo:.3f}, {hi:.3f}] (model-based SE)   '
                       f'within HR {b["hr"]:.3f} boot95 [{b["lo95"]:.3f}, {b["hi95"]:.3f}] boot97.5 [{b["lo975"]:.3f}, {b["hi975"]:.3f}]')
        rows.append(r)
    pd.DataFrame(rows).to_csv('results_v6/round1/c3_def2_continued.csv', index=False)
    exp = line[line['two_tier'].isin(TIERS + ['human'])]
    pd.DataFrame({'d_cur': exp['duration_days'].values, 'e_cur': exp['e2'].values, 'd_con': exp['d2c'].values,
                  'e_con': exp['e2c'].values, 'grp': exp['two_tier'].values, 'slug': exp['slug'].values,
                  'pr': (exp['slug'] + '#' + exp['pr_number'].astype(str)).values}).to_csv(
        'results_v6/round1/work/c3_def2_continued.csv', index=False)
    text = '\n'.join(out) + '\n'
    open('results_v6/round1/c3_def2_continued.txt', 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
