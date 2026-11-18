"""
build_intent_labels_v6.py

Combines all human intent labels (385 sample + 148 top-up + v6 top-up) into the
inputs analyze_intent_aggregation.py reads for its gold standard, so that its
validation-sample analysis, kappa-weighted vote and confusion-corrected shares
use every label in the corpus instead of the 385 only.

Restricted to the labelled commits that are modification commits of --corpus.
Per commit:
  gold     human_label (notes kept for the 385 and 148; the v6 top-up has a note
           on every row, written as a rationale, so its notes go to `rationale`
           and are not treated as 'unclear' flags)
  tier     majority tier among the hunks the commit modifies in --line-data, on the
           two-tier scale (human-supervised written as pair_programmer),
           for all labelled commits (the 385's sampling-time tier is kept as
           tier_at_sampling)
  keyword  data/death_commit_messages.csv for the 385 (as before), else the key's
           keyword_label
  LLMs     results/intent_llm_<m>_t0.csv for the 385 (as before), else the
           full-corpus runs in --llm-dir

Output directory (--out, default results_v6/labels618): gold.csv, sample.csv,
keyword.csv, intent_llm_<m>_t0.csv. Use with
    python src/analyze_intent_aggregation.py --gold <out>/gold.csv --sample <out>/sample.csv \
        --intent-kw <out>/keyword.csv --sample-llm-dir <out> [--full-corpus ...]
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from compute_intent_kappa import read_sheet

MODELS = ['deepseek', 'gpt_oss', 'minimax']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='data/death_commit_messages_v6confnv.csv')
    ap.add_argument('--line-data', default='data/survival_events_line_v6confnv_analysis.csv')
    ap.add_argument('--topup-v6-sheet', default='results_v6/intent-top-up-v6-labeled-sheet-v001-v256.xlsx')
    ap.add_argument('--llm-dir', default='results_v6')
    ap.add_argument('--out', default='results_v6/labels618')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    corpus = set(map(tuple, pd.read_csv(a.corpus)[['slug', 'sha']].drop_duplicates().values))
    ld = pd.read_csv(a.line_data, usecols=['slug', 'pr_number', 'hunk_id', 'agent_tier', 'death_sha', 'event'],
                     low_memory=False)
    ld = ld[ld['event'] == 1].drop_duplicates(['slug', 'death_sha', 'pr_number', 'hunk_id'])
    # majority on the two-tier scale (as intent_kappa_v6.py and the analyses), written back as a raw
    # tier name that analyze_intent_aggregation.load() maps to the same two-tier group
    two = {'pair_programmer': 'pair_programmer', 'batch_generator': 'pair_programmer', 'autonomous': 'autonomous',
           'human': 'human'}
    tier = (ld.assign(t=ld['agent_tier'].map(two)).groupby(['slug', 'death_sha'])['t']
            .agg(lambda s: s.value_counts().sort_index().idxmax()))

    s385 = pd.read_csv('results/intent_validation_sample_full.csv')[['sample_id', 'slug', 'sha', 'tier']]
    g385 = read_sheet('results/intent_annotation_sheet.xlsx')[['sample_id', 'human_label', 'notes']]
    k148 = pd.read_csv('results/intent_topup_sample.csv')[['sample_id', 'slug', 'sha', 'keyword_label']]
    g148 = read_sheet('results/intent_annotation_topup.xlsx')[['sample_id', 'human_label', 'notes']]
    k256 = pd.read_csv('results_v6/intent_topup_v6_sample.csv')[['sample_id', 'slug', 'sha', 'keyword_label']]
    g256 = read_sheet(a.topup_v6_sheet)[['sample_id', 'human_label', 'notes']].rename(columns={'notes': 'rationale'})
    g256['notes'] = ''

    a385 = s385.merge(g385, on='sample_id').assign(set='sample385')
    a148 = k148.merge(g148, on='sample_id').assign(set='topup148')
    a256 = k256.merge(g256, on='sample_id').assign(set='topup_v6')
    a385 = a385.rename(columns={'tier': 'tier_at_sampling'})
    for d in (a385, a148, a256):
        d['tier'] = [tier.get((x, h)) for x, h in zip(d['slug'], d['sha'])]
    allr = pd.concat([a385, a148, a256], ignore_index=True)
    allr = allr[[(x, h) in corpus for x, h in zip(allr['slug'], allr['sha'])]]
    if 'tier_at_sampling' in allr:
        x = allr[allr['set'] == 'sample385']
        print(f'385 sample in corpus: tier now == tier at sampling for {(x["tier"] == x["tier_at_sampling"]).sum()} of {len(x)}')
    if allr['tier'].isna().any():
        sys.exit(f'{allr["tier"].isna().sum()} labelled commits without a tier')
    allr[['sample_id', 'human_label', 'notes', 'rationale', 'set']].to_csv(os.path.join(a.out, 'gold.csv'), index=False)
    allr[['sample_id', 'slug', 'sha', 'tier', 'tier_at_sampling', 'set']].to_csv(os.path.join(a.out, 'sample.csv'), index=False)

    kw_old = pd.read_csv('data/death_commit_messages.csv')[['slug', 'sha', 'intent']].drop_duplicates(['slug', 'sha'])
    kw385 = a385[['slug', 'sha']].merge(kw_old, on=['slug', 'sha'], how='left')
    kwtop = pd.concat([a148, a256])[['slug', 'sha', 'keyword_label']].rename(columns={'keyword_label': 'intent'})
    pd.concat([kw385, kwtop]).drop_duplicates(['slug', 'sha']).to_csv(os.path.join(a.out, 'keyword.csv'), index=False)

    for m in MODELS:
        old = pd.read_csv(f'results/intent_llm_{m}_t0.csv')[['slug', 'sha', 'label']]
        full = pd.read_csv(os.path.join(a.llm_dir, f'intent_{m}_full_corpus_t0.csv'))[['slug', 'sha', 'label']]
        full = full.drop_duplicates(['slug', 'sha'], keep='last')
        top = pd.concat([a148, a256])[['slug', 'sha']].merge(full, on=['slug', 'sha'], how='left')
        pd.concat([old, top]).drop_duplicates(['slug', 'sha']).to_csv(
            os.path.join(a.out, f'intent_llm_{m}_t0.csv'), index=False)
    print(f'labelled commits in the corpus: {len(allr)} ({allr["set"].value_counts().to_dict()})')
    print('tiers:', allr['tier'].value_counts().to_dict())
    print('written to', a.out)


if __name__ == '__main__':
    main()
