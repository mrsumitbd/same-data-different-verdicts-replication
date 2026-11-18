"""
c8_covariate_within_r1.py 

Covariate-adjusted within-repository Cox model per tier, line level,
Definition 1 (syntactic) and Definition 2 (AST-structural): repository strata,
tier indicator plus file covariates at merge (src/file_covariates_r1.py):
  log1p(file age in days), log1p(first-parent commits to the file in the 90
  days before merge), file created by the PR (0/1), test-file path (0/1).
Intervals: 500-resample repository cluster bootstrap, seed 42 (src/boot_r1.py),
as for the other within-repository models. The unadjusted counterparts are the
C4 bootstraps (results_v6/round1/c4_bootstrap.csv).

Usage: DATA_TAG=v6confnv RESULTS_DIR=results_v6 python src/c8_covariate_within_r1.py
Output: results_v6/round1/c8_covariate_within.txt (+ .csv)
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
from config.observation_window import apply_observation_window
from config.paths import P
from analyze_rq1_master import add_two_tier_column
from boot_r1 import boot_within

TIERS = ['human_supervised', 'fully_autonomous']
COV = ['log_file_age', 'log_commits_90d', 'created_by_pr', 'test_file']


def main():
    line = add_two_tier_column(apply_observation_window(
        pd.read_csv(P('data/survival_events_line_v5_analysis.csv'), low_memory=False), label='line'))
    line['event_def2'] = (line['ast_death'] == 1.0).astype(int)
    line['path'] = line['unit_id'].str.rsplit('::', n=1).str[0]
    cov = pd.read_csv('results_v6/round1/work/file_covariates.csv')
    line = line.merge(cov, on=['slug', 'birth_sha', 'path'], how='left')
    miss = line['file_age_days'].isna().sum()
    line['log_file_age'] = np.log1p(line['file_age_days'])
    line['log_commits_90d'] = np.log1p(line['commits_90d'])
    out = ['=' * 78, 'C8. Covariate-adjusted within-repository Cox (repository strata, 500-resample cluster bootstrap)',
           f'covariates: {", ".join(COV)}; lines without covariates (dropped): {miss:,}', '=' * 78, '',
           'Covariate means by group:']
    out.append(line.groupby('two_tier')[['file_age_days', 'commits_90d', 'created_by_pr', 'test_file']].mean()
               .round(3).to_string())
    out.append('')
    c4 = pd.read_csv('results_v6/round1/c4_bootstrap.csv') if os.path.exists('results_v6/round1/c4_bootstrap.csv') else None
    rows = []
    for tier in TIERS:
        for dfn, col in [('def1_syntactic', 'event'), ('def2_structural', 'event_def2')]:
            cols = ['duration_days', col, 'slug'] + COV
            a = line[line['two_tier'] == tier][cols].rename(columns={col: 'event'})
            h = line[line['two_tier'] == 'human'][cols].rename(columns={col: 'event'})
            df = pd.concat([a.assign(is_tier=1), h.assign(is_tier=0)], ignore_index=True)
            r = boot_within(df, f'c8_within_{dfn}_{tier}', covariates=COV)
            r.update(tier=tier, definition=dfn)
            print(json.dumps(r), flush=True)
            unadj = ''
            if c4 is not None:
                u = c4[(c4['tier'] == tier) & (c4['definition'] == dfn)].iloc[0]
                unadj = f'   (unadjusted {u["hr"]:.3f} [{u["lo95"]:.3f}, {u["hi95"]:.3f}])'
            out.append(f'{tier:17s} {dfn:16s} adjusted HR {r["hr"]:.3f} boot95 [{r["lo95"]:.3f}, {r["hi95"]:.3f}] '
                       f'boot97.5 [{r["lo975"]:.3f}, {r["hi975"]:.3f}]{unadj}')
            out.append('    covariate HRs (point, analytic p): ' + '; '.join(
                f'{k} {v[0]:.3f} (p={v[1]:.2g})' for k, v in r['covariates'].items()))
            flat = {k: v for k, v in r.items() if k != 'covariates'}
            flat.update({f'cov_{k}_hr': v[0] for k, v in r['covariates'].items()})
            rows.append(flat)
    pd.DataFrame(rows).to_csv('results_v6/round1/c8_covariate_within.csv', index=False)
    text = '\n'.join(out) + '\n'
    open('results_v6/round1/c8_covariate_within.txt', 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
