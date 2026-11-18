"""
make_intent_topup_sheet_v6.py

Second intent-label top-up, for the v6conf corpus (conference born lines,
identity tracker): data/death_commit_messages_v6conf.csv, 5,373 modification
commits. 369 of the 533 human-labelled commits (385 sample + 148 top-up) are in
it, all in the 156 journal repositories; the commits of the new repositories
have no labels.

Pool: modification commits of the v6conf corpus in repositories outside
data/aidev_filtered_v1.csv (the new repositories). Each commit gets the
majority tier among the hunks whose lines it modifies (as in
sample_intent_validation.py), on the two-tier scale used by the analyses:
human / human_supervised (Copilot, Cursor, Claude Code, Codex) / Devin.

Sample: n commits (default 256), proportional allocation to the pool's tiers
(largest remainder), seeded (SEED). Row order: the first 100 sample ids
(V001-V100) are themselves a proportional stratified sample of the pool, so
the sheet can be cut at 100; ids V101-V256 complete the 256. Within each block
the rows are shuffled, so tiers are not visible from the order. The sheet is
blind (no tier, keyword label or LLM label), same columns and format as
results/intent_annotation_topup.xlsx (src/make_annotation_sheet.py).

Output:
  results_v6/intent_topup_v6_sheet.xlsx (+ .csv)   blind sheet for the annotator
  results_v6/intent_topup_v6_sample.csv            key: sample_id, slug, sha, tier, block, keyword_label
  results_v6/intent_topup_v6_instructions.md

Usage:
    python src/make_intent_topup_sheet_v6.py [--n 256] [--first 100]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import make_annotation_sheet as mas

SEED = 20261103
TIER = {'pair_programmer': 'human_supervised', 'batch_generator': 'human_supervised', 'autonomous': 'devin',
        'human': 'human'}


def allocate(sizes: pd.Series, n: int) -> pd.Series:
    """Proportional allocation with largest remainders."""
    q = sizes / sizes.sum() * n
    a = np.floor(q).astype(int)
    a[(q - a).sort_values(ascending=False).index[:n - a.sum()]] += 1
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--first', type=int, default=100)
    ap.add_argument('--diff-cache', default='results_v6/diff_cache_full_corpus.csv')
    a = ap.parse_args()
    msgs = pd.read_csv('data/death_commit_messages_v6conf.csv').drop_duplicates(['slug', 'sha'])
    journal = set(pd.read_csv('data/aidev_filtered_v1.csv')['slug'])
    line = pd.read_csv('data/survival_events_line_v6conf_analysis.csv',
                       usecols=['slug', 'pr_number', 'hunk_id', 'agent_tier', 'death_sha', 'event'], low_memory=False)
    line = line[line['event'] == 1]
    hunks = line.drop_duplicates(['slug', 'death_sha', 'pr_number', 'hunk_id'])
    tier = (hunks.assign(t=hunks['agent_tier'].map(TIER))
            .groupby(['slug', 'death_sha'])['t'].agg(lambda s: s.value_counts().sort_index().idxmax()))
    msgs['tier'] = [tier.get((s, h)) for s, h in zip(msgs['slug'], msgs['sha'])]
    pool = msgs[~msgs['slug'].isin(journal)].copy()
    labelled = set()
    for f in ['results/intent_validation_sample_full.csv', 'results/intent_topup_sample.csv']:
        x = pd.read_csv(f)
        labelled |= set(zip(x['slug'], x['sha']))
    pool = pool[[(s, h) not in labelled for s, h in zip(pool['slug'], pool['sha'])]]
    sizes = pool['tier'].value_counts().sort_index()
    n_all, n_first = allocate(sizes, a.n), allocate(sizes, a.first)
    rng = np.random.default_rng(SEED)
    first, rest = [], []
    for t in sizes.index:
        g = pool[pool['tier'] == t]
        g = g.iloc[rng.permutation(len(g))[:n_all[t]]]
        first.append(g.iloc[:n_first[t]].assign(block='first'))
        rest.append(g.iloc[n_first[t]:].assign(block='rest'))
    first = pd.concat(first).sample(frac=1, random_state=SEED)
    rest = pd.concat(rest).sample(frac=1, random_state=SEED + 1)
    top = pd.concat([first, rest]).reset_index(drop=True)
    top['sample_id'] = [f'V{i + 1:03d}' for i in range(len(top))]
    cache = pd.read_csv(a.diff_cache).drop_duplicates(['slug', 'sha']).set_index(['slug', 'sha'])['diff_text']
    top['diff_text'] = [str(cache.get((s, h), '')).replace('--- DIFF ---', mas.DIFF_MARKER, 1)
                        for s, h in zip(top['slug'], top['sha'])]
    top['commit_url'] = 'https://github.com/' + top['slug'] + '/commit/' + top['sha']
    top = top.rename(columns={'message': 'commit_message'})
    rows = mas.build_rows(top)
    out = Path('results_v6/intent_topup_v6_sheet.xlsx')
    out.parent.mkdir(exist_ok=True)
    mas.write_xlsx(rows, out)
    mas.write_csv(rows, out.with_suffix('.csv'))
    top[['sample_id', 'slug', 'sha', 'tier', 'block', 'intent']].rename(columns={'intent': 'keyword_label'}).to_csv(
        'results_v6/intent_topup_v6_sample.csv', index=False)
    missing = int((top['diff_text'].str.len() < 10).sum())
    summary = pd.DataFrame({'pool': sizes, f'sample_{a.n}': n_all, f'first_{a.first}': n_first})
    print(f'corpus {len(msgs):,} commits; new-repo pool {len(pool):,}; journal-repo commits {len(msgs) - len(pool):,}')
    print(summary.to_string())
    print(f'seed {SEED}; missing diffs: {missing}')
    print('written', out, out.with_suffix('.csv'), 'results_v6/intent_topup_v6_sample.csv')
    return summary, len(pool), missing


if __name__ == '__main__':
    main()
