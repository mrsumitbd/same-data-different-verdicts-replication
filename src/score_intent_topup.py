"""
score_intent_topup.py

Scores the intent classifiers once the top-up sheet
(results/intent_annotation_topup.xlsx, 148 commits from the new stratum) is
labelled, together with the original 385 labels.

Reports:
  - Cohen's kappa for every approach (keyword, DeepSeek, gpt-oss, MiniMax,
    majority vote, cross-fitted weighted vote, Dawid-Skene) on all labelled
    commits (385 + 148 = 533), and on the stratified corpus sample
    (the 208 of the 385 that are v5 modification commits + the 148), weighted
    by stratum (N_h / n_h) so it estimates agreement on the whole v5 corpus;
  - stratified human-label intent shares by tier for the v5 corpus;
  - confusion-corrected corpus shares (confusion estimated on the stratified
    sample) for each LLM and the majority vote.
No LLM calls are made: every v5 commit already has labels from all three models.

Usage:
    python src/score_intent_topup.py            # requires the top-up sheet fully labelled
    python src/score_intent_topup.py --partial  # score the rows labelled so far
Output:
    results/intent_topup_scores.txt
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.paths import P
import analyze_intent_aggregation as agg
from compute_intent_kappa import read_xlsx

MODELS = ['deepseek', 'gpt_oss', 'minimax']
LABELS = agg.INTENT_LABELS


def llm_labels(df):
    for m in MODELS:
        run = pd.read_csv(P(f'results/intent_{m}_full_corpus_t0.csv'))[['slug', 'sha', 'label']].drop_duplicates(['slug', 'sha'])
        df = df.drop(columns=[m], errors='ignore').merge(run.rename(columns={'label': m}), on=['slug', 'sha'], how='left')
        df[m] = df[m].fillna('missing').astype(str).str.strip().str.lower()
    return df


def tier_map():
    l = pd.read_csv(P('data/survival_events_line_v5_analysis.csv'), usecols=['slug', 'death_sha', 'is_agent', 'agent_tier'])
    l = l.dropna(subset=['death_sha'])
    l['two_tier'] = np.where(~l['is_agent'], 'human',
                             np.where(l['agent_tier'] == 'autonomous', 'fully_autonomous', 'human_supervised'))
    return (l.groupby(['slug', 'death_sha'])['two_tier'].agg(lambda s: s.value_counts().index[0])
            .reset_index().rename(columns={'death_sha': 'sha'}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partial', action='store_true')
    a = ap.parse_args()

    # original 385 (sample LLM labels, keyword from the sample file)
    base = agg.load('results/intent_annotation_sheet.xlsx', P('results/intent_validation_sample_full.csv'),
                    P('data/death_commit_messages.csv'))
    base['set'] = 'original'
    # top-up 148
    key = pd.read_csv(P('results/intent_topup_sample.csv'))
    sheet = read_xlsx(P('results/intent_annotation_topup.xlsx'))
    sheet['human_label'] = sheet['human_label'].fillna('').astype(str).str.strip().str.lower()
    top = key.merge(sheet[['sample_id', 'human_label', 'notes']], on='sample_id')
    done = top['human_label'].isin(LABELS)
    if not done.all() and not a.partial:
        sys.exit(f'{(~done).sum()} top-up rows are unlabelled or invalid; label them or use --partial.')
    top = top[done].rename(columns={'human_label': 'gold', 'keyword_label': 'keyword'})
    top = llm_labels(top)
    top = top.merge(tier_map(), on=['slug', 'sha'], how='left')
    top['set'] = 'topup'
    cols = ['slug', 'sha', 'gold', 'keyword', 'two_tier', 'set'] + MODELS
    lab = pd.concat([base[cols], top[cols]], ignore_index=True)
    lab['keyword'] = lab['keyword'].fillna('missing').astype(str).str.lower()

    v5 = pd.read_csv(P('data/death_commit_messages_v5.csv')).drop_duplicates(['slug', 'sha'])
    old = set()
    for f in [P('data/death_commit_messages.csv'), P('data/death_commit_messages_v4.csv')]:
        o = pd.read_csv(f)
        old |= set(zip(o['slug'], o['sha']))
    N_over = sum((s, h) in old for s, h in zip(v5['slug'], v5['sha']))
    N_new = len(v5) - N_over
    in_v5 = set(zip(v5['slug'], v5['sha']))
    lab['in_v5'] = [(s, h) in in_v5 for s, h in zip(lab['slug'], lab['sha'])]
    lab['stratum'] = np.where(lab['set'] == 'topup', 'new', np.where(lab['in_v5'], 'overlap', 'outside v5'))
    # tier of a v5 modification commit = majority birth tier of the lines it modifies (v5 data)
    tm = tier_map().rename(columns={'two_tier': 'tier_v5'})
    lab = lab.merge(tm, on=['slug', 'sha'], how='left')
    lab.loc[lab['in_v5'], 'two_tier'] = lab.loc[lab['in_v5'], 'tier_v5']
    n_over, n_new = (lab['stratum'] == 'overlap').sum(), (lab['stratum'] == 'new').sum()
    w = {'overlap': N_over / max(n_over, 1), 'new': N_new / max(n_new, 1)}

    # combined labels
    lab['majority'] = agg.vote(lab[MODELS], {m: 1.0 for m in MODELS}, [])
    lab['weighted_cv'] = agg.cv_vote(lab, MODELS, weighted=True, tie_best=True)
    lab['dawid_skene'] = agg.dawid_skene(lab, MODELS)
    approaches = ['keyword'] + MODELS + ['majority', 'weighted_cv', 'dawid_skene']

    out = ['=' * 70, 'Intent classifiers vs human labels: original 385 + top-up', '=' * 70,
           f'labelled: original {(lab["set"] == "original").sum()}, top-up {(lab["set"] == "topup").sum()} '
           f'(of {len(key)}){" PARTIAL" if a.partial else ""}',
           f'v5 strata: overlap N={N_over} (labelled {n_over}), new N={N_new} (labelled {n_new}); '
           f'weights {w["overlap"]:.2f} / {w["new"]:.2f}', '']
    strat = lab[lab['stratum'].isin(['overlap', 'new'])]
    sw = strat['stratum'].map(w)
    rng = np.random.default_rng(42)

    def boot_ci(d, c, weights=None, B=1000):
        vals = []
        for _ in range(B):
            i = rng.integers(0, len(d), len(d))
            dd = d.iloc[i]
            vals.append(cohen_kappa_score(dd['gold'], dd[c], sample_weight=None if weights is None else weights.iloc[i]))
        return np.percentile(vals, [2.5, 97.5])

    top_only = lab[lab['set'] == 'topup']
    out.append(f'{"approach":12s} {"kappa all 533 [95% CI]":>26s} {"kappa v5 weighted 356 [95% CI]":>32s} {"kappa top-up 148":>17s}')
    for c in approaches:
        k_all = cohen_kappa_score(lab['gold'], lab[c]); ci_a = boot_ci(lab, c)
        k_v5 = cohen_kappa_score(strat['gold'], strat[c], sample_weight=sw); ci_v = boot_ci(strat, c, sw)
        k_t = cohen_kappa_score(top_only['gold'], top_only[c]) if len(top_only) else float('nan')
        out.append(f'{c:12s} {k_all:8.3f} [{ci_a[0]:.3f}, {ci_a[1]:.3f}] {k_v5:14.3f} [{ci_v[0]:.3f}, {ci_v[1]:.3f}] {k_t:17.3f}')
    out.append(f'\ntop-up label distribution: ' + ', '.join(f'{k} {v}' for k, v in top_only['gold'].value_counts().items()))

    out += ['', 'Stratified human-label intent shares in the v5 corpus (%), with 95% margin of error (Kish effective n):']
    for t, g in strat.groupby('two_tier'):
        ww = g['stratum'].map(w)
        neff = ww.sum() ** 2 / (ww ** 2).sum()
        parts = []
        for l in LABELS:
            p_ = ww[g['gold'] == l].sum() / ww.sum()
            parts.append(f'{l} {100 * p_:.1f} +/-{196 * np.sqrt(p_ * (1 - p_) / neff):.1f}')
        out.append(f'  {t:17s} ' + ', '.join(parts) + f'  (n={len(g)}, effective n={neff:.0f})')
    from scipy.stats import chi2_contingency
    out.append('Human labels, tier vs human baseline (chi-square on the 356 stratified-sample commits, unweighted):')
    for t in ['human_supervised', 'fully_autonomous']:
        sub = strat[strat['two_tier'].isin([t, 'human'])]
        ct = pd.crosstab(sub['two_tier'], sub['gold'])
        chi2, p_, _, _ = chi2_contingency(ct)
        v = np.sqrt(chi2 / ct.values.sum() / (min(ct.shape) - 1))
        out.append(f'  {t:17s} p={p_:.3g} V={v:.2f} (n={len(sub)})')

    corpus = llm_labels(v5[['slug', 'sha']].copy()).merge(tier_map(), on=['slug', 'sha'], how='left')
    corpus['majority'] = agg.vote(corpus[MODELS], {m: 1.0 for m in MODELS}, [])
    out += ['', 'Confusion-corrected corpus shares (confusion from the stratified sample):']
    for c in MODELS + ['majority']:
        point, ci, clipped, n_clip = agg.corrected_full(strat, corpus, c)
        out.append(f'  {c}  [{n_clip} bootstrap draws clipped]')
        for t in agg.TIERS:
            out.append('    ' + f'{t:17s} ' + '  '.join(
                f'{l[:4]} {point[t][i]:5.1f} [{ci[t][0][i]:5.1f}, {ci[t][1][i]:5.1f}]' for i, l in enumerate(LABELS)))
    text = '\n'.join(out) + '\n'
    open(P('results/intent_topup_scores.txt'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
