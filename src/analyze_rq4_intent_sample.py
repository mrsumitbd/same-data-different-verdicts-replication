"""
analyze_rq4_intent_sample.py

RQ4 intent by modifier type on the 385 human-labelled commits.

For the human labels, the keyword classifier and each LLM, reports intent
shares for human-modified and agent-modified commits and a chi-square test
(agent vs human modifier). The full-corpus counterpart is Section B of
results/intent_aggregation_full_corpus.txt; this script adds the human-label
reference, which only exists for the sample.

Usage:
    python src/analyze_rq4_intent_sample.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import P
from compute_intent_kappa import read_xlsx

ORDER = ['corrective', 'perfective', 'adaptive', 'preventive']
LLMS = ['deepseek', 'gpt_oss', 'minimax']


def shares(ct: pd.DataFrame, row: str) -> str:
    sh = ct.div(ct.sum(axis=1), axis=0) * 100
    out = '/'.join(f'{sh.loc[row].get(o, 0):.1f}' for o in ORDER)
    if 'unknown' in sh.columns:
        out += f'/{sh.loc[row, "unknown"]:.1f}u'
    return out


def run(sample: str, gold: str, authors: str, output: str, labels_dir: str = None) -> None:
    if labels_dir:
        # all human labels in the corpus (src/build_intent_labels_v6.py): gold.csv, sample.csv,
        # keyword.csv and the per-commit LLM labels intent_llm_<m>_t0.csv
        s = pd.read_csv(os.path.join(labels_dir, 'sample.csv'))[['sample_id', 'slug', 'sha']]
        kw = pd.read_csv(os.path.join(labels_dir, 'keyword.csv')).rename(columns={'intent': 'keyword_label'})
        s = s.merge(kw[['slug', 'sha', 'keyword_label']], on=['slug', 'sha'], how='left')
        g = pd.read_csv(os.path.join(labels_dir, 'gold.csv'))[['sample_id', 'human_label']]
        llm_path = lambda llm: os.path.join(labels_dir, f'intent_llm_{llm}_t0.csv')
    else:
        s = pd.read_csv(sample)[['sample_id', 'slug', 'sha', 'keyword_label']]
        g = read_xlsx(gold)[['sample_id', 'human_label']]
        llm_path = lambda llm: P(f'results/intent_llm_{llm}_t0.csv')
    g['human_label'] = g['human_label'].astype(str).str.strip().str.lower()
    a = pd.read_csv(authors)[['slug', 'sha', 'modifier_type']]
    m = s.merge(g, on='sample_id').merge(a, on=['slug', 'sha'], how='left')
    for llm in LLMS:
        d = (pd.read_csv(llm_path(llm))[['slug', 'sha', 'label']]
             .drop_duplicates(['slug', 'sha']).rename(columns={'label': llm}))
        m = m.merge(d, on=['slug', 'sha'], how='left')

    n_agent = int((m['modifier_type'] == 'agent').sum())
    n_human = int((m['modifier_type'] == 'human').sum())
    lines = [
        '=' * 70,
        f'RQ4 intent by modifier type, {len(m)} human-labelled commits' + (f' ({labels_dir})' if labels_dir else ''),
        f'Modifier: human={n_human}, agent={n_agent}, missing={int(m["modifier_type"].isna().sum())}',
        'Shares (%) corrective/perfective/adaptive/preventive[/unknown]; chi-square agent vs human modifier',
        '=' * 70,
    ]
    for col, name in [('human_label', 'human labels'), ('keyword_label', 'keyword')] + [(x, x) for x in LLMS]:
        ct = pd.crosstab(m['modifier_type'], m[col])
        chi2, p, _, _ = chi2_contingency(ct)
        v = np.sqrt(chi2 / ct.values.sum() / (min(ct.shape) - 1))
        lines.append(f'{name:13s}  human-modified {shares(ct, "human"):24s}  '
                     f'agent-modified {shares(ct, "agent"):24s}  | p={p:.3f} V={v:.2f}')

    text = '\n'.join(lines) + '\n'
    print(text)
    with open(output, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample',  default=P('results/intent_validation_sample_full.csv'))
    parser.add_argument('--gold',    default=P('results/intent_annotation_sheet.xlsx'))
    parser.add_argument('--authors', default=P('results/modifier_authors.csv'))
    parser.add_argument('--output',  default=P('results/rq4_intent_by_modifier_sample.txt'))
    parser.add_argument('--labels-dir', default=None,
                        help='use every human label in the corpus, e.g. results_v6/labels618 (src/build_intent_labels_v6.py)')
    args = parser.parse_args()
    run(args.sample, args.gold, args.authors, args.output, args.labels_dir)
