"""
make_intent_topup_sheet.py

Top-up sample for the human intent labels after the v5 re-mine.

The 385 labelled commits were drawn from the earlier modification-commit
corpus; 208 of them are modification commits in the v5 corpus
(data/death_commit_messages_v5.csv). The v5 corpus splits into two strata:
  overlap  commits also in the earlier corpora (data/death_commit_messages.csv
           or data/death_commit_messages_v4.csv); the 208 labelled commits
           are a random sample of this stratum
  new      commits only in v5; no labels yet
This script draws a simple random sample of n_new commits (seed 42) from the
new stratum and writes a blind annotation sheet in the same format as
results/intent_annotation_sheet.xlsx (src/make_annotation_sheet.py).

Stratified margin of error for a proportion (worst case p = 0.5, 95%):
  MoE = 1.96 * sqrt( sum_h W_h^2 * p(1-p)/n_h * (N_h - n_h)/(N_h - 1) ),  W_h = N_h / N

Usage:
    python src/make_intent_topup_sheet.py --n-new 148
Output:
    results/intent_annotation_topup.xlsx (+ .csv), results/intent_topup_sample.csv (key, not blind)
"""

import argparse
import math
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import make_annotation_sheet as mas


def margin(strata):
    N = sum(Nh for Nh, _ in strata)
    v = sum((Nh / N) ** 2 * 0.25 / nh * (Nh - nh) / (Nh - 1) for Nh, nh in strata)
    return 1.96 * math.sqrt(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-new', type=int, default=148)
    a = ap.parse_args()
    v5 = pd.read_csv('data/death_commit_messages_v5.csv').drop_duplicates(['slug', 'sha'])
    old = set()
    for f in ['data/death_commit_messages.csv', 'data/death_commit_messages_v4.csv']:
        o = pd.read_csv(f)
        old |= set(zip(o['slug'], o['sha']))
    v5['stratum'] = ['overlap' if (s, h) in old else 'new' for s, h in zip(v5['slug'], v5['sha'])]
    samp = pd.read_csv('results/intent_validation_sample_full.csv')
    labelled = set(zip(samp['slug'], samp['sha']))
    n_overlap = sum((s, h) in labelled for s, h in zip(v5['slug'], v5['sha']))
    N_over, N_new = (v5['stratum'] == 'overlap').sum(), (v5['stratum'] == 'new').sum()
    top = v5[v5['stratum'] == 'new'].sample(a.n_new, random_state=42).reset_index(drop=True)
    top['sample_id'] = [f'T{i + 1:03d}' for i in range(len(top))]
    cache = pd.read_csv('results/diff_cache_full_corpus.csv').drop_duplicates(['slug', 'sha']).set_index(['slug', 'sha'])['diff_text']
    top['diff_text'] = [str(cache.get((s, h), '')).replace('--- DIFF ---', mas.DIFF_MARKER, 1) for s, h in zip(top['slug'], top['sha'])]
    top['commit_url'] = 'https://github.com/' + top['slug'] + '/commit/' + top['sha']
    top = top.rename(columns={'message': 'commit_message'})
    rows = mas.build_rows(top)
    out = Path('results/intent_annotation_topup.xlsx')
    mas.write_xlsx(rows, out)
    mas.write_csv(rows, out.with_suffix('.csv'))
    top[['sample_id', 'slug', 'sha', 'stratum', 'intent']].rename(columns={'intent': 'keyword_label'}).to_csv('results/intent_topup_sample.csv', index=False)
    moe = margin([(N_over, n_overlap), (N_new, a.n_new)])
    print(f'v5 corpus {len(v5):,}: overlap {N_over:,} (labelled {n_overlap}), new {N_new:,} (top-up {a.n_new})')
    print(f'stratified 95% margin of error (p = 0.5): +/-{moe * 100:.2f}%')
    print(f'missing diffs in top-up: {(top["diff_text"].str.len() < 10).sum()}')
    print('written', out, out.with_suffix('.csv'), 'results/intent_topup_sample.csv')


if __name__ == '__main__':
    main()
