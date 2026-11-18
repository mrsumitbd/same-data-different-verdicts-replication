"""
intent_kappa_v6.py

Agreement of every intent approach with the human labels, on the labelled
commits that are modification commits of the v6conf corpus
(data/death_commit_messages_v6conf.csv), next to the values on all 533 labels
(385 sample + 148 top-up) as in score_intent_topup.py.

Labels per commit exactly as in score_intent_topup.py: the 385 use the sample
LLM runs (results/intent_llm_<m>_t0.csv) and the sample keyword label, the 148
top-up use the full-corpus LLM runs and their keyword label. Combined approaches
(majority vote, cross-fitted weighted vote, Dawid-Skene) are computed on all 533
and then restricted, so they are the same labels in both columns.

With --topup-v6 the second top-up (results_v6/intent_topup_v6_sheet.xlsx, key
results_v6/intent_topup_v6_sample.csv) is added once labelled: columns for all
labels (533 + top-up) and for the labelled commits in the v6conf corpus
(369 + top-up), LLM labels for the top-up from results_v6/.

--topup-v6-sheet points at the labelled sheet (default
results_v6/intent_topup_v6_sheet.xlsx). For the labelled commits in the corpus
the script also reports kappa weighted by stratum (old = the 156 journal
repositories, new = the others; weight N_h / n_h, N_h = corpus commits in the
stratum) and the human-label intent shares per tier with chi-square tests of
each agent tier vs human (unweighted, and shares weighted by stratum). Tier of
a commit = majority tier among the hunks it modifies in --line-data.

Output: results_v6/intent_kappa_v6.txt
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import analyze_intent_aggregation as agg
from compute_intent_kappa import read_xlsx

MODELS = ['deepseek', 'gpt_oss', 'minimax']
TIER = {'pair_programmer': 'human_supervised', 'batch_generator': 'human_supervised', 'autonomous': 'fully_autonomous',
        'human': 'human'}
LABELS = agg.INTENT_LABELS


def llm_labels(df, results_dir):
    for m in MODELS:
        run = pd.read_csv(os.path.join(results_dir, f'intent_{m}_full_corpus_t0.csv'))[['slug', 'sha', 'label']]
        run = run.drop_duplicates(['slug', 'sha'], keep='last')
        df = df.drop(columns=[m], errors='ignore').merge(run.rename(columns={'label': m}), on=['slug', 'sha'], how='left')
        df[m] = df[m].fillna('missing').astype(str).str.strip().str.lower()
    return df


def topup(key_path, sheet_path, results_dir, partial):
    key = pd.read_csv(key_path)
    sheet = read_xlsx(sheet_path)
    sheet['human_label'] = sheet['human_label'].fillna('').astype(str).str.strip().str.lower()
    t = key.merge(sheet[['sample_id', 'human_label', 'notes']], on='sample_id')
    done = t['human_label'].isin(LABELS)
    if not done.all() and not partial:
        sys.exit(f'{sheet_path}: {(~done).sum()} rows unlabelled or invalid; label them or use --partial.')
    t = t[done].rename(columns={'human_label': 'gold', 'keyword_label': 'keyword'})
    return llm_labels(t, results_dir), len(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='data/death_commit_messages_v6conf.csv')
    ap.add_argument('--topup-v6', action='store_true')
    ap.add_argument('--topup-v6-sheet', default='results_v6/intent_topup_v6_sheet.xlsx')
    ap.add_argument('--line-data', default=None,
                    help='line data for commit tiers (default: matches --corpus, v6conf or v6confnv)')
    ap.add_argument('--partial', action='store_true')
    ap.add_argument('--output', default='results_v6/intent_kappa_v6.txt')
    a = ap.parse_args()
    base = agg.load('results/intent_annotation_sheet.xlsx', 'results/intent_validation_sample_full.csv',
                    'data/death_commit_messages.csv')
    base['set'] = 'sample385'
    t1, _ = topup('results/intent_topup_sample.csv', 'results/intent_annotation_topup.xlsx', 'results', False)
    t1['set'] = 'topup148'
    sets = [base, t1]
    n_key2 = 0
    if a.topup_v6:
        t2, n_key2 = topup('results_v6/intent_topup_v6_sample.csv', a.topup_v6_sheet, 'results_v6', a.partial)
        t2['set'] = 'topup_v6'
        sets.append(t2)
    cols = ['slug', 'sha', 'gold', 'keyword', 'set'] + MODELS
    lab = pd.concat([s[cols] for s in sets], ignore_index=True)
    lab['keyword'] = lab['keyword'].fillna('missing').astype(str).str.lower()
    lab['majority'] = agg.vote(lab[MODELS], {m: 1.0 for m in MODELS}, [])
    lab['weighted_cv'] = agg.cv_vote(lab, MODELS, weighted=True, tie_best=True)
    lab['dawid_skene'] = agg.dawid_skene(lab, MODELS)
    corpus = pd.read_csv(a.corpus)[['slug', 'sha']].drop_duplicates()
    inc = set(zip(corpus['slug'], corpus['sha']))
    lab['in_v6'] = [(s, h) in inc for s, h in zip(lab['slug'], lab['sha'])]
    approaches = ['keyword'] + MODELS + ['majority', 'weighted_cv', 'dawid_skene']
    rng = np.random.default_rng(42)

    def k_ci(d, c, B=1000):
        if not len(d):
            return float('nan'), (float('nan'), float('nan'))
        vals = [cohen_kappa_score(d['gold'].iloc[i], d[c].iloc[i])
                for i in (rng.integers(0, len(d), len(d)) for _ in range(B))]
        return cohen_kappa_score(d['gold'], d[c]), np.percentile(vals, [2.5, 97.5])

    old = lab[lab['set'] != 'topup_v6']
    groups = [(f'all {len(old)} (385+148)', old), (f'in v6conf {old["in_v6"].sum()}', old[old['in_v6']])]
    if a.topup_v6:
        groups += [(f'all {len(lab)}', lab), (f'in v6conf {lab["in_v6"].sum()}', lab[lab['in_v6']]),
                   (f'top-up v6 {(lab["set"] == "topup_v6").sum()}', lab[lab['set'] == 'topup_v6'])]
    out = ['Intent approaches vs human labels (Cohen kappa, bootstrap 95% CI, 1,000 draws, seed 42)',
           f'corpus: {a.corpus} ({len(corpus):,} commits); labelled commits in it: {lab["in_v6"].sum()} of {len(lab)}'
           + (f'; v6 top-up labelled {(lab["set"] == "topup_v6").sum()} of {n_key2}' if a.topup_v6 else ''),
           'labelled commits in the corpus by set: ' + ', '.join(
               f'{k} {v}' for k, v in lab[lab['in_v6']]['set'].value_counts().items()), '']
    out.append(f'{"approach":12s} ' + ' '.join(f'{g:>26s}' for g, _ in groups))
    for c in approaches:
        cells = []
        for _, d in groups:
            k, ci = k_ci(d, c)
            cells.append(f'{k:7.3f} [{ci[0]:.3f}, {ci[1]:.3f}]')
        out.append(f'{c:12s} ' + ' '.join(f'{x:>26s}' for x in cells))
    kw = lab[lab['keyword'] != 'unknown']
    out.append(f'\nkeyword on commits it labels: all {cohen_kappa_score(kw["gold"], kw["keyword"]):.3f} (n={len(kw)}), '
               f'in v6conf {cohen_kappa_score(kw[kw["in_v6"]]["gold"], kw[kw["in_v6"]]["keyword"]):.3f} '
               f'(n={kw["in_v6"].sum()})')
    out.append('missing LLM labels: ' + ', '.join(f'{m} {(lab[m] == "missing").sum()}' for m in MODELS))

    # stratum-weighted kappa and human-label tier tests on the labelled commits in the corpus
    journal = set(pd.read_csv('data/aidev_filtered_v1.csv')['slug'])
    corpus['stratum'] = np.where(corpus['slug'].isin(journal), 'old', 'new')
    Nh = corpus['stratum'].value_counts()
    inc_lab = lab[lab['in_v6']].copy()
    inc_lab['stratum'] = np.where(inc_lab['slug'].isin(journal), 'old', 'new')
    nh = inc_lab['stratum'].value_counts()
    w = inc_lab['stratum'].map(lambda h: Nh[h] / nh[h])
    out.append(f'\nLabelled commits in the corpus by stratum: ' + ', '.join(f'{h} n={nh.get(h, 0)} of N={Nh[h]}' for h in Nh.index)
               + '; weights ' + ', '.join(f'{h} {Nh[h] / nh[h]:.2f}' for h in nh.index))
    out.append(f'{"approach":12s} {"unweighted [95% CI]":>26s} {"stratum-weighted [95% CI]":>28s}')
    for c in approaches:
        k_u, ci_u = k_ci(inc_lab, c)
        vals = []
        for _ in range(1000):
            i = rng.integers(0, len(inc_lab), len(inc_lab))
            vals.append(cohen_kappa_score(inc_lab['gold'].iloc[i], inc_lab[c].iloc[i], sample_weight=w.iloc[i]))
        k_w = cohen_kappa_score(inc_lab['gold'], inc_lab[c], sample_weight=w)
        ci_w = np.percentile(vals, [2.5, 97.5])
        out.append(f'{c:12s} {k_u:7.3f} [{ci_u[0]:.3f}, {ci_u[1]:.3f}]       {k_w:7.3f} [{ci_w[0]:.3f}, {ci_w[1]:.3f}]')

    line_data = a.line_data or a.corpus.replace('death_commit_messages_', 'survival_events_line_').replace('.csv', '_analysis.csv')
    ld = pd.read_csv(line_data, usecols=['slug', 'pr_number', 'hunk_id', 'agent_tier', 'death_sha', 'event'], low_memory=False)
    ld = ld[ld['event'] == 1].drop_duplicates(['slug', 'death_sha', 'pr_number', 'hunk_id'])
    tier = (ld.assign(t=ld['agent_tier'].map(TIER)).groupby(['slug', 'death_sha'])['t']
            .agg(lambda s_: s_.value_counts().sort_index().idxmax()))
    inc_lab['two_tier'] = [tier.get((x, h)) for x, h in zip(inc_lab['slug'], inc_lab['sha'])]
    out.append(f'\nHuman-label intent shares by tier (labelled commits in the corpus; tier from {line_data}):')
    out.append(f'  {"tier":17s} {"n":>4s}  ' + '  '.join(f'{l[:5]:>14s}' for l in LABELS) + '   (unweighted %, stratum-weighted %)')
    for t, g in inc_lab.groupby('two_tier'):
        ww = w.loc[g.index]
        cells = [f'{(g["gold"] == l).mean() * 100:5.1f} / {ww[g["gold"] == l].sum() / ww.sum() * 100:5.1f}' for l in LABELS]
        out.append(f'  {t:17s} {len(g):4d}  ' + '  '.join(f'{c:>14s}' for c in cells))
    out.append('Chi-square, tier vs human baseline on the human labels (unweighted counts):')
    for t in ['human_supervised', 'fully_autonomous']:
        sub = inc_lab[inc_lab['two_tier'].isin([t, 'human'])]
        ct = pd.crosstab(sub['two_tier'], sub['gold'])
        chi2, p_, _, _ = chi2_contingency(ct)
        v = np.sqrt(chi2 / ct.values.sum() / (min(ct.shape) - 1))
        out.append(f'  {t:17s} chi2={chi2:.2f} p={p_:.3g} V={v:.3f} (n={len(sub)})')
    text = '\n'.join(out) + '\n'
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    open(a.output, 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
