"""
c4_bootstrap_r1.py 

Re-runs the headline within-repository cluster bootstraps with the same data
preparation, seed (42) and B (500) as src/analyze_rq1_master.py (step 4,
Definition 1; also the RQ4 within-repository numbers) and
src/analyze_rq1_def2_stratified.py (Definition 2), saving every draw, and
reports 95% and 97.5% (Bonferroni for two tiers) percentile intervals and the
minimum detectable HR (80% power, two-sided alpha .05) from the bootstrap SE
of the log HR: MDHR = exp((1.96 + 0.84) * SE).

Usage: DATA_TAG=v6confnv RESULTS_DIR=results_v6 python src/c4_bootstrap_r1.py
Output: results_v6/round1/c4_bootstrap.csv, results_v6/round1/boot_draws/*.csv
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


def main():
    line = add_two_tier_column(apply_observation_window(
        pd.read_csv(P('data/survival_events_line_v5_analysis.csv')), label='line'))
    line['event_def2'] = (line['ast_death'] == 1.0).astype(int)
    rows = []
    for tier in TIERS:
        for dfn, col in [('def1_syntactic', 'event'), ('def2_structural', 'event_def2')]:
            a = line[line['two_tier'] == tier][['duration_days', col, 'slug']].rename(columns={col: 'event'})
            h = line[line['two_tier'] == 'human'][['duration_days', col, 'slug']].rename(columns={col: 'event'})
            df = pd.concat([a.assign(is_tier=1), h.assign(is_tier=0)], ignore_index=True)
            r = boot_within(df, f'within_line_{dfn}_{tier}')
            r.update(tier=tier, definition=dfn)
            print(json.dumps(r), flush=True)
            rows.append(r)
    out = pd.DataFrame(rows)
    out['sig95'] = (out['hi95'] < 1) | (out['lo95'] > 1)
    out['sig975'] = (out['hi975'] < 1) | (out['lo975'] > 1)
    out['mdhr_below1'] = np.exp(-(1.959964 + 0.841621) * out['se_log_boot'])
    out['mdhr_above1'] = np.exp((1.959964 + 0.841621) * out['se_log_boot'])
    out.drop(columns=['covariates']).to_csv('results_v6/round1/c4_bootstrap.csv', index=False)
    print(out.drop(columns=['covariates']).to_string())


if __name__ == '__main__':
    main()
