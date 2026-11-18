"""
intent_weighted_tier_test_v6.py

Tier-weighted test of the RQ2 intent question on the labelled commits (618 in the
no-vendored corpus): do agent-tier modifications have a different intent mix
than human ones, once the labelled commits are weighted to the corpus?

Weights: each labelled commit gets w = N_{h,t} / n_{h,t}, h = stratum (old = the 156
journal repositories, new = the others), t = two-tier group (human /
human_supervised / fully_autonomous; majority tier among the hunks the commit
modifies). N = corpus commits, n = labelled commits in the cell. The weighted
sample then has the corpus's stratum-by-tier composition.

Test, per agent tier vs human, for the human labels and for each LLM approach on
the same commits:
  - weighted intent shares;
  - Rao-Scott first-order corrected chi-square: Pearson X^2 on the weighted table
    (weights normalised to the sample size), divided by the Kish design effect
    deff = n * sum(w^2) / sum(w)^2 of the commits in the comparison; df = (r-1)(c-1);
  - Cramer's V from the corrected statistic;
  - a weighted permutation test (tier labels permuted within stratum, 5,000
    permutations, seed 42) as a check that does not rely on the deff approximation;
  - Bonferroni over the two tier tests (p x 2).
The unweighted chi-square is shown alongside.

Inputs: results_v6/labels618/ (src/build_intent_labels_v6.py), the corpus and line data.
Output: results_v6/intent_human_label_weighted_test.txt
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import analyze_intent_aggregation as agg

LABELS = ['corrective', 'perfective', 'adaptive', 'preventive']
MODELS = ['deepseek', 'gpt_oss', 'minimax']
TWO = {'pair_programmer': 'human_supervised', 'batch_generator': 'human_supervised',
       'autonomous': 'fully_autonomous', 'human': 'human'}
SEED = 42
N_PERM = 5000


def weighted_table(g, t, w, tiers):
    return np.array([[w[(t == a) & (g == l)].sum() for l in LABELS] for a in tiers])


def pearson(tab):
    tab = tab[:, tab.sum(0) > 0]
    e = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    return ((tab - e) ** 2 / e).sum(), (tab.shape[0] - 1) * (tab.shape[1] - 1)


def test(df, col, tier):
    d = df[df['two_tier'].isin([tier, 'human']) & df[col].isin(LABELS)]
    g, t, s = d[col].values, d['two_tier'].values, d['stratum'].values
    w = d['w'].values * len(d) / d['w'].sum()
    deff = len(w) * (w ** 2).sum() / w.sum() ** 2
    x2, df_ = pearson(weighted_table(g, t, w, [tier, 'human']))
    x2c = x2 / deff
    p = chi2.sf(x2c, df_)
    v = np.sqrt(x2c / len(d))
    rng = np.random.default_rng(SEED)
    ge = 0
    for _ in range(N_PERM):
        tp = t.copy()
        for h in np.unique(s):
            m = s == h
            tp[m] = rng.permutation(tp[m])
        ge += pearson(weighted_table(g, tp, w, [tier, 'human']))[0] >= x2
    p_perm = (ge + 1) / (N_PERM + 1)
    ct = pd.crosstab(t, g)
    _, p_u, _, _ = chi2_contingency(ct)
    shares = {a: [w[(t == a) & (g == l)].sum() / w[t == a].sum() * 100 for l in LABELS] for a in [tier, 'human']}
    return dict(n=len(d), deff=deff, x2c=x2c, df=df_, p=p, p_bonf=min(1, 2 * p), v=v, p_perm=p_perm,
                p_perm_bonf=min(1, 2 * p_perm), p_unw=p_u, shares=shares)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default='results_v6/labels618')
    ap.add_argument('--corpus', default='data/death_commit_messages_v6confnv.csv')
    ap.add_argument('--line-data', default='data/survival_events_line_v6confnv_analysis.csv')
    ap.add_argument('--output', default='results_v6/intent_human_label_weighted_test.txt')
    a = ap.parse_args()
    L = a.labels
    agg.LLM_RUNS.update({m: os.path.join(L, f'intent_llm_{m}_t0.csv') for m in MODELS})
    lab = agg.load(os.path.join(L, 'gold.csv'), os.path.join(L, 'sample.csv'), os.path.join(L, 'keyword.csv'))
    lab['majority'] = agg.vote(lab[MODELS], {m: 1.0 for m in MODELS}, MODELS)
    lab['dawid_skene'] = agg.dawid_skene(lab, MODELS)

    journal = set(pd.read_csv('data/aidev_filtered_v1.csv')['slug'])
    corpus = pd.read_csv(a.corpus)[['slug', 'sha']].drop_duplicates()
    ld = pd.read_csv(a.line_data, usecols=['slug', 'pr_number', 'hunk_id', 'agent_tier', 'death_sha', 'event'],
                     low_memory=False)
    ld = ld[ld['event'] == 1].drop_duplicates(['slug', 'death_sha', 'pr_number', 'hunk_id'])
    tier = (ld.assign(t=ld['agent_tier'].map(TWO)).groupby(['slug', 'death_sha'])['t']
            .agg(lambda s_: s_.value_counts().sort_index().idxmax()))
    corpus['two_tier'] = [tier.get((x, h)) for x, h in zip(corpus['slug'], corpus['sha'])]
    corpus['stratum'] = np.where(corpus['slug'].isin(journal), 'old', 'new')
    lab['stratum'] = np.where(lab['slug'].isin(journal), 'old', 'new')
    N = corpus.groupby(['stratum', 'two_tier']).size()
    n = lab.groupby(['stratum', 'two_tier']).size()
    lab['w'] = [N[(h, t)] / n[(h, t)] for h, t in zip(lab['stratum'], lab['two_tier'])]

    out = ['Tier-weighted test of intent by tier on the labelled commits (RQ2), no-vendored v6 corpus',
           f'labelled commits: {len(lab)}; corpus: {len(corpus):,} ({a.corpus})',
           'weights w = N_(stratum,tier) / n_(stratum,tier):']
    cells = pd.DataFrame({'N corpus': N, 'n labelled': n}).fillna(0).astype(int)
    cells['weight'] = (cells['N corpus'] / cells['n labelled']).round(2)
    out.append(cells.to_string())
    out.append('\nRao-Scott first-order corrected chi-square (Kish deff), weighted permutation p (tier permuted within '
               f'stratum, {N_PERM:,} permutations), Bonferroni x2 over the two tier tests; unweighted p for comparison.')
    for col in ['gold', 'deepseek', 'gpt_oss', 'minimax', 'majority', 'dawid_skene', 'keyword']:
        out.append(f'\n--- {col}' + (' (human labels)' if col == 'gold' else ''))
        for t in ['human_supervised', 'fully_autonomous']:
            r = test(lab, col, t)
            sh = '  '.join(f'{x}: ' + '/'.join(f'{v:.1f}' for v in r['shares'][x]) for x in [t, 'human'])
            out.append(f'  {t:17s} n={r["n"]:3d} deff={r["deff"]:.2f}  X2_RS={r["x2c"]:.2f} df={r["df"]} '
                       f'p={r["p"]:.3g} (Bonf {r["p_bonf"]:.3g})  V={r["v"]:.3f}  perm p={r["p_perm"]:.3g} '
                       f'(Bonf {r["p_perm_bonf"]:.3g})  | unweighted p={r["p_unw"]:.3g}')
            out.append(f'      weighted shares corr/perf/adap/prev %  {sh}')
    text = '\n'.join(out) + '\n'
    open(a.output, 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
