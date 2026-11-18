"""
hunk_size_clustered_r3.py 

Table tab:hunk_stratified: the same unadjusted Cox model per tier x hunk-size
group as src/analyze_rq1_master.py (section hunk_size), with 95% CIs from
standard errors clustered by repository (lifelines cluster_col = slug, robust
sandwich). Point estimates are unaffected by clustering and are checked
against results_v6/rq1_master_results.txt. Model-based CIs are printed too.

Run from the repository root:
  DATA_TAG=v6confnv RESULTS_DIR=results_v6 python src/hunk_size_clustered_r3.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.observation_window import apply_observation_window
from config.paths import P
from analyze_rq1_master import preprocess_hunk, add_two_tier_column

OUT = 'results_v6/round3'
SIZES = {'single_line': lambda s: s == 1, 'small_block': lambda s: (s >= 2) & (s <= 10), 'large_block': lambda s: s >= 11}


def paper_points():
    txt = open('results_v6/rq1_master_results.txt').read().split('SECTION: hunk_size')[1]
    pts = {}
    for ln in txt.splitlines():
        m = re.match(r'\s*(human_supervised|fully_autonomous)\s+(\w+)\s+(\d+)\s+(\d+)\s+([\d.]+)', ln)
        if m:
            pts[(m.group(1), m.group(2))] = (float(m.group(5)), int(m.group(3)), int(m.group(4)))
    return pts


def main():
    path = P('data/survival_events_hunk_v5.csv')
    hunk = apply_observation_window(pd.read_csv(path), label='hunk')
    hunk = add_two_tier_column(preprocess_hunk(hunk))
    human = hunk[hunk['two_tier'] == 'human']
    pts = paper_points()
    rows, out = [], [f'Task 5: hunk-size Cox (unadjusted), repo-clustered 95% CIs; input {path} ({len(hunk):,} hunks)', '']
    for tier in ('human_supervised', 'fully_autonomous'):
        th = hunk[hunk['two_tier'] == tier]
        for sz, f in SIZES.items():
            a, h = th[f(th['hunk_size'])], human[f(human['hunk_size'])]
            df = pd.concat([a[['duration_days', 'event', 'slug']].assign(is_tier=1),
                            h[['duration_days', 'event', 'slug']].assign(is_tier=0)], ignore_index=True).dropna()
            res = {}
            for kind, kw in (('model', {}), ('cluster', {'cluster_col': 'slug'})):
                d = df if kind == 'cluster' else df.drop(columns='slug')
                cph = CoxPHFitter().fit(d, duration_col='duration_days', event_col='event', **kw)
                s = cph.summary.loc['is_tier']
                res[kind] = (np.exp(s['coef']), np.exp(s['coef lower 95%']), np.exp(s['coef upper 95%']), s['p'])
            hr = res['cluster'][0]
            old = pts[(tier, sz)]
            match = round(hr, 4) == old[0] and len(a) == old[1] and len(h) == old[2]
            r = dict(tier=tier, size=sz, n_agent=len(a), n_human=len(h), n_repos=df['slug'].nunique(), hr=hr,
                     paper_hr=old[0], reproduces=match,
                     cl_lo=res['cluster'][1], cl_hi=res['cluster'][2], cl_p=res['cluster'][3],
                     mb_lo=res['model'][1], mb_hi=res['model'][2], mb_p=res['model'][3])
            rows.append(r)
            out.append(f'{tier:17s} {sz:12s} n={len(a):6,d} vs {len(h):6,d}  HR {hr:.4f} (results file {old[0]:.4f}, '
                       f'{"reproduces" if match else "MISMATCH"})  repo-cl [{r["cl_lo"]:.3f}, {r["cl_hi"]:.3f}] p={r["cl_p"]:.3g}'
                       f'  | model-based [{r["mb_lo"]:.3f}, {r["mb_hi"]:.3f}] p={r["mb_p"]:.2g}')
    pd.DataFrame(rows).to_csv(f'{OUT}/hunk_size_clustered.csv', index=False)
    txt = '\n'.join(out) + '\n'
    open(f'{OUT}/hunk_size_clustered.txt', 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
