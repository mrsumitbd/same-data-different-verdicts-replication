"""
rq3_keyword_llm_disagreement.py

Direction of disagreement between the keyword rules and the three LLM
classifiers on the full corpus of modification commits (RQ3). Counts the
commits the keyword rules label corrective while all three LLMs agree on
adaptive or agree on perfective, and the reverse (all three corrective),
and prints the worked example used in the paper (lingodotdev/lingo.dev ccd6f315f0).

Inputs are the same as analyze_intent_aggregation.py --full-corpus:
keyword labels from data/death_commit_messages_v5.csv and the cached LLM
labels in results/intent_{deepseek,gpt_oss,minimax}_full_corpus_t0.csv
(deduplicated on (slug, sha), last run kept).

Usage:
    python src/rq3_keyword_llm_disagreement.py [--corpus-kw data/death_commit_messages_v5.csv]
Output:
    results/rq3_keyword_llm_disagreement.txt
"""

import argparse
from pathlib import Path

import pandas as pd
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import P

LLMS = ['deepseek', 'gpt_oss', 'minimax']
LABELS = ['corrective', 'adaptive', 'perfective', 'preventive', 'unknown']
NON_CORRECTIVE = {'adaptive', 'perfective'}
EXAMPLES = [('lingodotdev/lingo.dev', 'ccd6f315f0')]


def main(corpus_kw: str, output: str) -> None:
    kw = pd.read_csv(corpus_kw).drop_duplicates(['slug', 'sha'])
    df = kw[['slug', 'sha', 'message', 'intent']].rename(columns={'intent': 'keyword'}) \
        if 'message' in kw.columns else kw[['slug', 'sha', 'intent']].rename(columns={'intent': 'keyword'})
    for m in LLMS:
        run = pd.read_csv(P(f'results/intent_{m}_full_corpus_t0.csv')).drop_duplicates(['slug', 'sha'], keep='last')
        df = df.merge(run[['slug', 'sha', 'label']].rename(columns={'label': m}), on=['slug', 'sha'], how='left')
    for c in ['keyword'] + LLMS:
        df[c] = df[c].fillna('missing').astype(str).str.strip().str.lower()
        df.loc[~df[c].isin(LABELS), c] = 'missing'

    unanimous = (df[LLMS].nunique(axis=1) == 1) & (df['deepseek'] != 'missing')
    # all three LLMs agree on one label: adaptive, perfective, or corrective
    llm_noncorr = unanimous & df['deepseek'].isin(NON_CORRECTIVE)
    llm_corr = unanimous & (df['deepseek'] == 'corrective')
    kw_to_llm = (df['keyword'] == 'corrective') & llm_noncorr
    llm_to_kw = df['keyword'].isin(NON_CORRECTIVE) & llm_corr

    out = [f'commits: {len(df):,}',
           f'unanimous LLM label: {unanimous.sum():,}',
           f'keyword corrective, all three LLMs adaptive or perfective: {kw_to_llm.sum():,}',
           f'keyword adaptive or perfective, all three LLMs corrective: {llm_to_kw.sum():,}',
           '', 'examples (keyword corrective, LLMs not):']
    for slug, prefix in EXAMPLES:
        r = df[(df['slug'] == slug) & df['sha'].str.startswith(prefix)].iloc[0]
        msg = str(r['message']).splitlines()[0] if 'message' in df.columns else ''
        out.append(f'  {slug} {prefix}: {msg!r} keyword={r["keyword"]} '
                   + ' '.join(f'{m}={r[m]}' for m in LLMS))
    text = '\n'.join(out) + '\n'
    print(text)
    Path(output).write_text(text)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus-kw', default=P('data/death_commit_messages_v5.csv'))
    ap.add_argument('--output', default=P('results/rq3_keyword_llm_disagreement.txt'))
    a = ap.parse_args()
    main(a.corpus_kw, a.output)
