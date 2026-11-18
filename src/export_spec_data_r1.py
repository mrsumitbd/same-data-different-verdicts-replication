"""
export_spec_data_r1.py 

Exports every dataset of the specification curve (src/specification_curve.py),
prepared exactly as that script prepares it (observation window, generated-path
exclusion, outlier PR, two-tier mapping), as compact CSVs for the R refits in
src/cox_variance_r1.R and src/spec_null_r1.R.

Columns: d (duration days), e1 (syntactic event), e2 (AST-structural event:
event==1 & ast_death==1), ast_known (0 if modified with unknown AST status),
grp (human_supervised / fully_autonomous / human), slug, pr (slug#number),
w (1 / units in the PR), plus for lines: unit_id.

Usage:
    DATA_TAG=v6confnv RESULTS_DIR=results_v6 python src/export_spec_data_r1.py [--followup-only]
    (--followup-only: write only followup_<key>.csv, the days from birth to the cutoff, row-aligned; C5)
Output:
    results_v6/round1/work/spec_<key>.csv, results_v6/round1/work/spec_datasets.csv
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.paths import P
from specification_curve import prep, conference_units
from config.constants import OBSERVATION_CUTOFF

BIRTH_ONLY = '--followup-only' in sys.argv

OUT = os.path.join(P('results/round1'), 'work')


def export(df, key, name, unit_kind):
    x = df[df['two_tier'].isin(['human_supervised', 'fully_autonomous', 'human'])]
    e1 = x['event'].astype(int)
    ast = x['ast_death'] if 'ast_death' in x else pd.Series(np.nan, index=x.index)
    out = pd.DataFrame({
        'd': x['duration_days'].values,
        'e1': e1.values,
        'e2': ((e1 == 1) & (ast == 1.0)).astype(int).values,
        'ast_known': (~((e1 == 1) & ast.isna())).astype(int).values,
        'grp': x['two_tier'].values,
        'slug': x['slug'].values,
        'pr': (x['slug'] + '#' + x['pr_number'].astype(str)).values,
        'unit_id': x['unit_id'].values,
    })
    out['w'] = 1.0 / out.groupby('pr')['d'].transform('size')
    path = os.path.join(OUT, f'spec_{key}.csv')
    if BIRTH_ONLY:
        # days of possible follow-up (birth to the observation cutoff), row-aligned with spec_<key>.csv
        fu = (pd.Timestamp(OBSERVATION_CUTOFF) - pd.to_datetime(x['birth_date'])).dt.total_seconds() / 86400
        pd.DataFrame({'pr': out['pr'].values, 'followup': fu.values}).to_csv(
            os.path.join(OUT, f'followup_{key}.csv'), index=False)
        print(f'{key}: follow-up written', flush=True)
        return dict(key=key)
    out.to_csv(path, index=False)
    has_ast = bool(ast.notna().any())
    print(f'{key}: {len(out):,} rows, AST {has_ast}', flush=True)
    return dict(key=key, data=name, unit_kind=unit_kind, has_ast=has_ast, path=path, n=len(out))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    tag = os.environ.get('DATA_TAG', '')
    base = tag[:-2] if tag.endswith('nv') else tag
    if base.startswith('v6conf'):
        cl, ch = conference_units(base)
        rows.append(export(prep(cl, 'conference', drop_generated=False), 'conf_line',
                           'conference tracking, all lines, with vendored', 'line'))
        rows.append(export(prep(ch, 'conference hunks', drop_generated=False), 'conf_hunk',
                           'conference tracking, all lines, with vendored', 'hunk'))
        del cl, ch
        rows.append(export(prep(pd.read_csv(f'data/survival_events_line_{base}_analysis.csv', low_memory=False),
                                'v3 with vendored', drop_generated=False), 'v3wv_line',
                           'v3 tracking, all lines, with vendored', 'line'))
        rows.append(export(prep(pd.read_csv(f'data/survival_events_hunk_{base}.csv', low_memory=False),
                                'hunk with vendored', drop_generated=False), 'v3wv_hunk',
                           'v3 tracking, all lines, with vendored', 'hunk'))
    rows.append(export(prep(pd.read_csv(P('data/survival_events_line_v4_ast.csv'), low_memory=False), 'v2'),
                       'v2cap_line', 'v2 tracking, capped', 'line'))
    v4u = pd.read_csv(P('data/survival_events_line_v4units_v5track_ast.csv'), low_memory=False)
    v4u = v4u[v4u['event_type'] != 'unverified'].copy()
    v4u['event'] = v4u['event'].astype(int)
    rows.append(export(prep(v4u, 'v3 capped'), 'v3cap_line', 'v3 tracking, capped', 'line'))
    rows.append(export(prep(pd.read_csv(P('data/survival_events_line_v5_analysis.csv'), low_memory=False), 'v5'),
                       'v3all_line', 'v3 tracking, all lines', 'line'))
    rows.append(export(prep(pd.read_csv(P('data/survival_events_hunk_v3.csv'), low_memory=False), 'hunk v2'),
                       'v2cap_hunk', 'v2 tracking, capped', 'hunk'))
    rows.append(export(prep(pd.read_csv(P('data/survival_events_hunk_v5.csv'), low_memory=False), 'hunk v5'),
                       'v3all_hunk', 'v3 tracking, all lines', 'hunk'))
    if not BIRTH_ONLY:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'spec_datasets.csv'), index=False)


if __name__ == '__main__':
    main()
