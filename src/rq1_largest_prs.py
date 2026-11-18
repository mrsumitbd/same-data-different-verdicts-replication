"""
rq1_largest_prs.py

How much of each tier's line count comes from its largest PRs, and how
often those lines are modified (RQ1, weighting of units). Supports the
statement that one Devin PR (pdfme/pdfme#1059) holds about half of all
Devin lines.

Per agent tier (pair_programmer, batch_generator, autonomous, human): the
five largest PRs by line units, each PR's share of the tier's lines and
the share of its lines modified, then the pooled modification share of
the top-5 PRs against the other PRs of the tier.

Lines are those of the v5 analysis table as written by
derive_v5_datasets.py (its own window; no further filtering), so the
totals equal the cohort size reported in the paper.

Usage:
    python src/rq1_largest_prs.py [--line-data data/survival_events_line_v5_analysis.csv]
Output:
    results/rq1_largest_prs.txt
"""

import argparse
from pathlib import Path

import pandas as pd
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import P, DATA_LABEL

TIERS = ['autonomous', 'pair_programmer', 'batch_generator', 'human']
TOP_N = 5


def main(line_data: str, output: str) -> None:
    df = pd.read_csv(line_data, usecols=['slug', 'pr_number', 'is_agent', 'agent_tier', 'event'],
                     low_memory=False)
    df['tier'] = df['agent_tier'].where(df['is_agent'].astype(bool), 'human')
    df['event'] = df['event'].astype(int)
    out = [f'Largest PRs by line units, {DATA_LABEL} analysis data (window as in the file)']
    for tier in TIERS:
        g = df[df['tier'] == tier]
        prs = (g.groupby(['slug', 'pr_number'])['event'].agg(['size', 'mean'])
               .sort_values('size', ascending=False))
        total = len(g)
        out += ['', f'{tier}: {total:,} lines in {len(prs)} PRs']
        for (slug, pr), r in prs.head(TOP_N).iterrows():
            out.append(f'  {slug}#{pr}: {int(r["size"]):,} lines ({100 * r["size"] / total:.1f}% of tier), '
                       f'{100 * r["mean"]:.1f}% modified')
        top_keys = set(prs.head(TOP_N).index)
        in_top = pd.Series(list(zip(g['slug'], g['pr_number'])), index=g.index).isin(top_keys)
        out.append(f'  top {TOP_N} PRs: {100 * in_top.mean():.1f}% of lines, '
                   f'{100 * g.loc[in_top, "event"].mean():.1f}% modified; '
                   f'other PRs: {100 * g.loc[~in_top, "event"].mean():.1f}% modified')
    text = '\n'.join(out) + '\n'
    print(text)
    Path(output).write_text(text)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--line-data', default=P('data/survival_events_line_v5_analysis.csv'))
    ap.add_argument('--output', default=P('results/rq1_largest_prs.txt'))
    a = ap.parse_args()
    main(a.line_data, a.output)
