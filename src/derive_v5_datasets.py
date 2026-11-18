"""
derive_v5_datasets.py

Builds the analysis datasets from the v5 line tracking (with AST status).

Line level (data/survival_events_line_v5_analysis.csv):
  every birth line except those whose modification could not be verified
  (event_type == 'unverified'); their count is reported, they are not
  treated as modifications or as survivals.

Hunk level (data/survival_events_hunk_v5.csv):
  a hunk is a maximal run of consecutive lines added in one file of a PR's
  merge diff (hunk_id from the births). A hunk is modified at the first
  verified modification of any of its lines; its event_type and death commit
  are those of that line. A hunk with no modified line is censored at the
  observation cutoff. Lines that could not be verified are ignored; a hunk
  whose lines are all unverified is dropped.

Conference cohort (--cohort-prs, a CSV with slug, number, author_type): after
the steps above, keep only the units of those PRs, matched on (slug, PR number,
agent or human). Used with the conf215 mining (measure_retention_v5.py).

Conference born lines (v6conf, --keep-shared-merges): PRs that share a merge
commit are all kept, as the conference did, instead of keeping only the PR
whose AIDev timeline names the commit. Extra columns present in the input
(EXTRA: vendored, blank, pr_id, conf_*) are carried into the line data; the
hunk data carries vendored.

Usage:
    python src/derive_v5_datasets.py
    python src/derive_v5_datasets.py --input data/survival_events_line_v6_ast.csv \
        --line-out data/survival_events_line_v6_analysis.csv --hunk-out data/survival_events_hunk_v6.csv \
        --cohort-prs earlier_study/prs_cohort256.csv
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config.constants import OBSERVATION_CUTOFF

KEEP = ['slug', 'pr_number', 'is_agent', 'agent_name', 'agent_tier', 'unit_id', 'event', 'event_type',
        'duration_days', 'birth_sha', 'birth_date', 'death_sha', 'death_date', 'birth_content',
        'death_content', 'whitespace_only', 'merge_type', 'match_reason', 'hunk_id', 'hunk_size', 'path_at_death',
        'ast_death', 'ast_enclosing_node', 'ast_exclusion_reason', 'granularity']
EXTRA = ['vendored', 'blank', 'pr_id', 'conf_event', 'conf_kind', 'conf_death_sha', 'conf_death_date']


def main(inp='data/survival_events_line_v5_ast.csv', line_out='data/survival_events_line_v5_analysis.csv',
         hunk_out='data/survival_events_hunk_v5.csv', cohort_prs=None, keep_shared_merges=False):
    df = pd.read_csv(inp, low_memory=False)
    # A merge commit matched to several PRs (merged_at within 60 s of each other):
    # keep the PR whose AIDev timeline names that commit, drop the others.
    prs = df[['slug', 'birth_sha', 'pr_number', 'match_reason']].drop_duplicates()
    shared = prs[prs.duplicated(['slug', 'birth_sha'], keep=False)]
    drop = shared[shared['match_reason'] != 'timeline merge sha'][['slug', 'birth_sha', 'pr_number']]
    if keep_shared_merges:
        print(f'merge commits matched to several PRs: {shared[["slug", "birth_sha"]].drop_duplicates().shape[0]} '
              f'({len(shared)} PRs), all kept (--keep-shared-merges)')
        drop = drop.iloc[:0]
    elif len(shared):
        print(f'merge commits matched to several PRs: {shared[["slug", "birth_sha"]].drop_duplicates().shape[0]}; '
              f'PRs dropped: {len(drop)}')
        df = df.merge(drop.assign(_drop=1), on=['slug', 'birth_sha', 'pr_number'], how='left')
        df = df[df['_drop'].isna()].drop(columns='_drop')
    if cohort_prs:
        c = pd.read_csv(cohort_prs)
        c = pd.DataFrame({'slug': c['slug'].str.lower(), 'pr_number': c['number'],
                          'is_agent': c['author_type'].eq('Agent'), '_keep': 1}).drop_duplicates()
        df = df.merge(c, on=['slug', 'pr_number', 'is_agent'], how='left')
        n_prs = df[['slug', 'pr_number', 'is_agent']].drop_duplicates()
        kept = df.loc[df['_keep'] == 1, ['slug', 'pr_number', 'is_agent']].drop_duplicates()
        print(f'cohort {cohort_prs}: {len(c):,} PRs listed; mined PRs {len(n_prs):,}, kept {len(kept):,}; '
              f'units kept {(df["_keep"] == 1).sum():,} of {len(df):,}')
        df = df[df['_keep'] == 1].drop(columns='_keep')
    dups = df.duplicated(['slug', 'pr_number', 'unit_id'])
    if dups.any():
        print(f'duplicate units removed: {dups.sum():,}')
        df = df[~dups]
    unver = df['event_type'] == 'unverified'
    print(f'line units {len(df):,}; unverified {unver.sum():,} ({unver.mean() * 100:.2f}%) dropped')
    line = df.loc[~unver, KEEP + [c for c in EXTRA if c in df.columns]].copy()
    line['event'] = line['event'].astype(int)
    line.to_csv(line_out, index=False)

    line['death_dt'] = pd.to_datetime(line['death_date'])
    ev = line[line['event'] == 1].sort_values('death_dt')
    first = ev.drop_duplicates(['slug', 'pr_number', 'hunk_id'])
    keys = ['slug', 'pr_number', 'hunk_id']
    h = line.groupby(keys, as_index=False).agg(
        is_agent=('is_agent', 'first'), agent_name=('agent_name', 'first'), agent_tier=('agent_tier', 'first'),
        birth_sha=('birth_sha', 'first'), birth_date=('birth_date', 'first'), hunk_size=('hunk_size', 'first'),
        lines_tracked=('unit_id', 'size'), lines_modified=('event', 'sum'), merge_type=('merge_type', 'first'),
        **({'vendored': ('vendored', 'first')} if 'vendored' in line.columns else {}))
    h = h.merge(first[keys + ['death_sha', 'death_date', 'event_type']], on=keys, how='left')
    h['event'] = h['death_sha'].notna().astype(int)
    h['event_type'] = h['event_type'].fillna('censored')
    bd, dd = pd.to_datetime(h['birth_date']), pd.to_datetime(h['death_date'])
    h['duration_days'] = ((dd.where(h['event'] == 1, OBSERVATION_CUTOFF) - bd).dt.total_seconds() / 86400).clip(lower=0.001)
    h['unit_id'] = h['hunk_id']
    h['granularity'] = 'hunk'
    h.to_csv(hunk_out, index=False)
    print(f'hunks {len(h):,}; modified {h["event"].mean() * 100:.1f}%; size bins:',
          pd.cut(h['hunk_size'], [0, 1, 10, 10 ** 9], labels=['1', '2-10', '11+']).value_counts().to_dict())
    print('wrote', line_out, hunk_out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/survival_events_line_v5_ast.csv')
    ap.add_argument('--line-out', default='data/survival_events_line_v5_analysis.csv')
    ap.add_argument('--hunk-out', default='data/survival_events_hunk_v5.csv')
    ap.add_argument('--cohort-prs', default=None)
    ap.add_argument('--keep-shared-merges', action='store_true')
    a = ap.parse_args()
    main(a.input, a.line_out, a.hunk_out, a.cohort_prs, a.keep_shared_merges)
