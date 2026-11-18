"""
def2_continued_r2.py 

Definition 2 (AST-structural) with lines followed past leaf-only edits (round-1 C3).
Across-repository HRs with repository-clustered CIs come from round 1
(results_v6/round1/c3_def2_clustered.csv, R coxph Efron, src/c3_cluster_r1.R).
Round 1's within-repository bootstraps used the R engine (R RNG); here they are
re-run with the headline engine (lifelines Efron, numpy default_rng(42), 500
repository resamples, src/boot_r1.py with BOOT_ENGINE unset), for the current
Definition 2 (check: must equal the paper's 0.717 [0.477, 0.921] and
1.155 [0.519, 2.176]) and the continued version, both tiers.
Input: results_v6/round1/work/c3_def2_continued.csv (same row order as the analysis data).

Usage: python src/def2_continued_r2.py
Output: results_v6/round2/e5_def2_continued.txt / .csv
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
os.environ.pop('BOOT_ENGINE', None)
import boot_r1

boot_r1.OUT = 'results_v6/round2/boot_draws'


def main():
    x = pd.read_csv('results_v6/round1/work/c3_def2_continued.csv')
    cl = pd.read_csv('results_v6/round1/c3_def2_clustered.csv')
    rows, out = [], ['E5. Definition 2 continued past leaf-only edits', '']
    out.append('A. Across repositories, R coxph (Efron), repository-clustered 95% CI (round 1, c3_def2_clustered.csv)')
    for _, r in cl[cl['comparison'] == 'across'].iterrows():
        out.append(f'  {r["version"]:32s} {r["tier"]:17s} HR {r["hr"]:.3f} repo-cl [{r["lo_cl_repo"]:.3f}, {r["hi_cl_repo"]:.3f}] '
                   f'p={r["p_cl_repo"]:.3f}  (model [{r["lo_model"]:.3f}, {r["hi_model"]:.3f}])')
    out += ['', 'B. Within repository, lifelines Efron, 500-resample repository bootstrap, numpy default_rng(42)']
    for tier in ['human_supervised', 'fully_autonomous']:
        for v, lab in [('cur', 'current (censored at leaf-only)'), ('con', 'continued past leaf-only')]:
            y = x[['grp', 'slug', f'd_{v}', f'e_{v}']].rename(columns={f'd_{v}': 'duration_days', f'e_{v}': 'event'})
            a, h = y[y['grp'] == tier], y[y['grp'] == 'human']
            df = pd.concat([a.drop(columns='grp').assign(is_tier=1), h.drop(columns='grp').assign(is_tier=0)], ignore_index=True)
            b = boot_r1.boot_within(df, f'e5_within_def2_{v}_{tier}')
            b.pop('covariates')
            b.update(tier=tier, version=lab)
            print(json.dumps(b), flush=True)
            rows.append(b)
            out.append(f'  {lab:32s} {tier:17s} HR {b["hr"]:.3f} boot95 [{b["lo95"]:.3f}, {b["hi95"]:.3f}] '
                       f'boot97.5 [{b["lo975"]:.3f}, {b["hi975"]:.3f}] p_boot {b["p_boot"]:.3f} (valid {b["boot_valid"]})')
            pd.DataFrame(rows).to_csv('results_v6/round2/e5_def2_continued.csv', index=False)
    open('results_v6/round2/e5_def2_continued.txt', 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
