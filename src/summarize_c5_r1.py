"""
summarize_c5_r1.py 

The observation window as a dimension of the specification curve. Reads the
R refits (src/cox_variance_r1.R) for windows:
  full  all PRs followed to the cutoff (paper; >= 5 months)       c1_cox_variance.csv
  c8    PRs with >= 8 months possible follow-up, to the cutoff      c5_cox_window_c8.csv
  c8h   same cohort, censored 240 days after merge (equal follow-up) c5_cox_window_c8h.csv
  c10h  PRs with >= 10 months, censored 300 days after merge         c5_cox_window_c10h.csv
Reports verdict counts per tier x comparison x window under the paper's
estimators and with repository clustering, and the RQ1 headline rung
(identity, all lines, per line / per PR, syntactic) per window.
Output: results_v6/round1/c5_window_curve.txt (+ c5_window_specs.csv)
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = 'results_v6/round1'
SHORT = {'full (>=5 months, to cutoff)': 'full', '>=8 months, to cutoff': '8m-cut', '>=8 months, censored at 240 d': '8m-240d',
         '>=10 months, censored at 300 d': '10m-300d'}
FILES = {'full (>=5 months, to cutoff)': 'c1_cox_variance.csv', '>=8 months, to cutoff': 'c5_cox_window_c8.csv',
         '>=8 months, censored at 240 d': 'c5_cox_window_c8h.csv', '>=10 months, censored at 300 d': 'c5_cox_window_c10h.csv'}


def main():
    parts = []
    for w, f in FILES.items():
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            x = pd.read_csv(p)
            x['window'] = w
            parts.append(x)
    r = pd.concat(parts, ignore_index=True)
    r = r[r['tier'].isin(['human_supervised', 'fully_autonomous']) & r['definition'].isin(['syntactic', 'structural'])]
    cur = np.where(r['comparison'] == 'within', 'cl_repo', np.where(r['unit'] == 'PR-weighted', 'robust', 'model'))
    r['lo_current'] = [r.loc[i, f'lo_{c}'] for i, c in zip(r.index, cur)]
    r['hi_current'] = [r.loc[i, f'hi_{c}'] for i, c in zip(r.index, cur)]
    # a tier with fewer than 10 PRs in the cohort cannot be tested (HS at 8 months: 3 PRs, 1 shared repository)
    r['testable'] = r['n_pr_tier'] >= 10
    r.to_csv(os.path.join(OUT, 'c5_window_specs.csv'), index=False)
    out = ['=' * 78, 'C5. Observation window as a specification choice (R refits, Efron)', '=' * 78, '',
           'A. Verdict counts longer / none / sooner (specifications fitted; a spec is missing when the tier has no units)']
    for est in ['current', 'cl_repo']:
        out.append(f'  estimator: {"as in the paper" if est == "current" else "clustered by repository"}')
        for (t, cmp), g in r.groupby(['tier', 'comparison']):
            line = f'    {t:17s} {cmp:14s}'
            for w in FILES:
                x = g[(g['window'] == w) & g['testable']]
                if not len(x):
                    line += f' | {SHORT[w]:8s} not testable (<10 tier PRs)'
                    continue
                lo, hi = x[f'lo_{est}'], x[f'hi_{est}']
                L = int(((hi < 1)).sum()); S = int((lo > 1).sum())
                line += f' | {SHORT[w]:8s} {L:2d}/{len(x) - L - S:2d}/{S:2d} (n={len(x)}, med HR {x["hr"].median():.2f})'
            out.append(line)
    out += ['', 'B. Headline rung (identity tracking, vendored/generated removed, syntactic) per window']
    h = r[(r['key'] == 'v3all_line') & (r['definition'] == 'syntactic')]
    for (t, unit, cmp), g in h.groupby(['tier', 'unit', 'comparison']):
        for _, x in g[g['testable']].iterrows():
            out.append(f'  {t:17s} {unit:11s} {cmp:14s} {x["window"]:32s} HR {x["hr"]:.3f}  paper-SE [{x["lo_current"]:.3f}, '
                       f'{x["hi_current"]:.3f}]  repo-cl [{x["lo_cl_repo"]:.3f}, {x["hi_cl_repo"]:.3f}]  '
                       f'n={x["n"]:,} tier repos {x["n_repo_tier"]} tier PRs {x["n_pr_tier"]}')
    text = '\n'.join(out) + '\n'
    open(os.path.join(OUT, 'c5_window_curve.txt'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
