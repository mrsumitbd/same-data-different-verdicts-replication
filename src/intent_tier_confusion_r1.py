"""
intent_tier_confusion_r1.py 

Does classifier accuracy differ by tier, and does a per-tier correction change
the misclassification-corrected corrective shares?

On the 618 human-labelled commits (results_v6/labels618/), for each approach of
analyze_intent_aggregation.py (keyword, the three LLMs, majority vote with the
best-kappa tie-break, kappa-weighted vote, Dawid-Skene):
  1. accuracy and kappa per tier (human-supervised / Devin / human) and per
     modifier type (agent- vs human-modified commits);
  2. tests of equal accuracy: Fisher exact (2x2 correct/incorrect) HS vs human,
     Devin vs human, agent- vs human-modified; chi-square across the three tiers;
  3. per-tier confusion matrices P(pred | true) (+0.5 smoothing, as the pooled
     correction) and the corpus shares of each tier corrected with ITS OWN
     matrix, next to the pooled-matrix correction of the paper; bootstrap 95%
     CIs (1,000 draws, seed 42) resampling the labelled commits within tier and
     the corpus commits within tier.
The same per-group correction is also reported by modifier type (RQ4).

Usage: DATA_TAG=v6confnv RESULTS_DIR=results_v6 python src/intent_tier_confusion_r1.py
Output: results_v6/round1/c6_intent_tier_confusion.txt (+ .csv)
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import analyze_intent_aggregation as A
from config.paths import P

L = 'results_v6/labels618'
A.LLM_RUNS.update({m: os.path.join(L, f'intent_llm_{m}_t0.csv') for m in A.NEW_LLMS})
RESP = A.INTENT_LABELS + ['unknown', 'missing', 'undecided']
TIERS = A.TIERS
N_BOOT, SEED = 1000, 42


def build():
    smp = A.load(f'{L}/gold.csv', f'{L}/sample.csv', f'{L}/keyword.csv')
    k = {m: max(A.kappa(smp['gold'], smp[m]), 0.0) for m in A.NEW_LLMS}
    order = sorted(A.NEW_LLMS, key=lambda m: -k[m])
    kw = pd.read_csv(P('data/death_commit_messages_v5.csv')).drop_duplicates(['slug', 'sha'])
    corpus = kw[['slug', 'sha', 'intent']].rename(columns={'intent': 'keyword'})
    for m, path in A.FULL_RUNS.items():
        run = pd.read_csv(path).drop_duplicates(['slug', 'sha'], keep='last')
        corpus = corpus.merge(run[['slug', 'sha', 'label']].rename(columns={'label': m}), on=['slug', 'sha'], how='left')
    for col in ['keyword'] + A.NEW_LLMS:
        corpus[col] = corpus[col].fillna('missing').astype(str).str.strip().str.lower()
        corpus.loc[~corpus[col].isin(A.INTENT_LABELS + ['unknown']), col] = 'missing'
    for d in (corpus, smp):
        d['mv3_tiebest'] = A.vote(d[A.NEW_LLMS], {m: 1.0 for m in A.NEW_LLMS}, order)
        d['wv3'] = A.vote(d[A.NEW_LLMS], k, order)
        d['ds3'] = A.dawid_skene(d, A.NEW_LLMS)
    corpus = corpus.merge(A.tier_map_from_line(P('data/survival_events_line_v5_analysis.csv')),
                          on=['slug', 'sha'], how='inner')
    authors = pd.read_csv(P('results/modifier_authors.csv'))[['slug', 'sha', 'modifier_type']].drop_duplicates(['slug', 'sha'])
    corpus = corpus.merge(authors, on=['slug', 'sha'], how='left')
    smp = smp.merge(authors, on=['slug', 'sha'], how='left')
    return smp, corpus


def conf_matrix(smp, col):
    c = pd.crosstab(smp['gold'], smp[col]).reindex(index=A.INTENT_LABELS, columns=RESP, fill_value=0).to_numpy(float) + 0.5
    return c / c.sum(1, keepdims=True)


def correct(conf, q_labels):
    q = q_labels.value_counts(normalize=True).reindex(RESP, fill_value=0).to_numpy()
    p, *_ = np.linalg.lstsq(conf.T, q, rcond=None)
    clipped = (p < 0).any()
    p = np.clip(p, 0, None)
    return p / p.sum() * 100, clipped


def shares_by(smp, corpus, col, gcol, groups, per_group):
    """Corrected shares per group; per_group=True uses the group's own matrix."""
    out = {}
    for g in groups:
        conf = conf_matrix(smp[smp[gcol] == g] if per_group else smp, col)
        out[g] = correct(conf, corpus.loc[corpus[gcol] == g, col])[0]
    return out


def boot(smp, corpus, col, gcol, groups, per_group):
    rng = np.random.default_rng(SEED)
    draws = {g: [] for g in groups}
    sidx = {g: np.flatnonzero(smp[gcol].values == g) for g in groups}
    cidx = {g: np.flatnonzero(corpus[gcol].values == g) for g in groups}
    for _ in range(N_BOOT):
        if per_group:
            s = smp.iloc[np.concatenate([rng.choice(sidx[g], len(sidx[g])) for g in groups])]
        else:
            s = smp.iloc[rng.integers(0, len(smp), len(smp))]
        c = corpus.iloc[np.concatenate([rng.choice(cidx[g], len(cidx[g])) for g in groups])]
        r = shares_by(s, c, col, gcol, groups, per_group)
        for g in groups:
            draws[g].append(r[g])
    return {g: (np.percentile(draws[g], 2.5, axis=0), np.percentile(draws[g], 97.5, axis=0)) for g in groups}


def main():
    smp, corpus = build()
    approaches = ['keyword'] + A.NEW_LLMS + ['mv3_tiebest', 'wv3', 'ds3']
    out = ['=' * 78, 'C6. Classifier accuracy and misclassification correction per tier',
           f'labelled commits: {len(smp)} (tiers: ' + ', '.join(f'{t} {n}' for t, n in smp['two_tier'].value_counts().items()) +
           '; modifier: ' + ', '.join(f'{t} {n}' for t, n in smp['modifier_type'].value_counts().items()) + ')',
           f'corpus commits mapped to a tier: {len(corpus):,}', '=' * 78, '',
           'A. Accuracy (% exact agreement with the human label) and kappa per group; '
           'Fisher exact (correct vs not) vs human / human-modified; chi-square across the three tiers', '']
    rows = []
    for col in approaches:
        smp['ok'] = (smp[col] == smp['gold']).astype(int)
        r = {'approach': col}
        line = f'{col:12s}'
        for g, gcol in [(t, 'two_tier') for t in TIERS] + [(m, 'modifier_type') for m in ['agent', 'human']]:
            s = smp[smp[gcol] == g]
            key = g if gcol == 'two_tier' else f'{g}_modified'
            r[f'acc_{key}'] = s['ok'].mean() * 100
            r[f'kappa_{key}'] = cohen_kappa_score(s['gold'], s[col])
            r[f'n_{key}'] = len(s)
            line += f'  {key[:12]:>12s} {r[f"acc_{key}"]:5.1f}% k={r[f"kappa_{key}"]:.2f}'
        for t in ['human_supervised', 'fully_autonomous']:
            tab = pd.crosstab(smp.loc[smp['two_tier'].isin([t, 'human']), 'two_tier'],
                              smp.loc[smp['two_tier'].isin([t, 'human']), 'ok'])
            r[f'fisher_p_{t}_vs_human'] = fisher_exact(tab.values)[1]
        r['chi2_p_three_tiers'] = chi2_contingency(pd.crosstab(smp['two_tier'], smp['ok']).values)[1]
        tab = pd.crosstab(smp['modifier_type'], smp['ok'])
        r['fisher_p_agent_vs_human_modified'] = fisher_exact(tab.values)[1]
        line += (f'  | p HS={r["fisher_p_human_supervised_vs_human"]:.3f} FA={r["fisher_p_fully_autonomous_vs_human"]:.3f}'
                 f' 3-tier={r["chi2_p_three_tiers"]:.3f} agent-mod={r["fisher_p_agent_vs_human_modified"]:.3f}')
        out.append(line)
        rows.append(r)

    out += ['', 'B. Corrective share (%) of the corpus, corrected with the pooled matrix (paper) vs each tier\'s own matrix',
            '   bootstrap 95% CI; all four intents in the CSV', '']
    for col in A.NEW_LLMS + ['mv3_tiebest', 'wv3', 'ds3']:
        r = next(x for x in rows if x['approach'] == col)
        pooled = shares_by(smp, corpus, col, 'two_tier', TIERS, False)
        own = shares_by(smp, corpus, col, 'two_tier', TIERS, True)
        ci_p = boot(smp, corpus, col, 'two_tier', TIERS, False)
        ci_o = boot(smp, corpus, col, 'two_tier', TIERS, True)
        out.append(f'  {col}')
        for t in TIERS:
            out.append(f'    {t:17s} pooled corr {pooled[t][0]:5.1f} [{ci_p[t][0][0]:4.1f}, {ci_p[t][1][0]:4.1f}]'
                       f'   per-tier corr {own[t][0]:5.1f} [{ci_o[t][0][0]:4.1f}, {ci_o[t][1][0]:4.1f}]'
                       f'   per-tier perf/adap/prev {own[t][1]:.1f}/{own[t][2]:.1f}/{own[t][3]:.1f}')
            for i, lab in enumerate(A.INTENT_LABELS):
                r[f'pooled_{t}_{lab}'] = pooled[t][i]
                r[f'pertier_{t}_{lab}'] = own[t][i]
                r[f'pertier_{t}_{lab}_lo'] = ci_o[t][0][i]
                r[f'pertier_{t}_{lab}_hi'] = ci_o[t][1][i]
        # modifier type (RQ4)
        mods = ['agent', 'human']
        pm = shares_by(smp, corpus, col, 'modifier_type', mods, False)
        om = shares_by(smp, corpus, col, 'modifier_type', mods, True)
        cm = boot(smp, corpus, col, 'modifier_type', mods, True)
        for m in mods:
            out.append(f'    {m + "-modified":17s} pooled corr {pm[m][0]:5.1f}   per-group corr {om[m][0]:5.1f} '
                       f'[{cm[m][0][0]:4.1f}, {cm[m][1][0]:4.1f}]   per-group perf/adap/prev '
                       f'{om[m][1]:.1f}/{om[m][2]:.1f}/{om[m][3]:.1f}')
            for i, lab in enumerate(A.INTENT_LABELS):
                r[f'pergroup_{m}_modified_{lab}'] = om[m][i]
    os.makedirs('results_v6/round1', exist_ok=True)
    pd.DataFrame(rows).to_csv('results_v6/round1/c6_intent_tier_confusion.csv', index=False)
    text = '\n'.join(out) + '\n'
    open('results_v6/round1/c6_intent_tier_confusion.txt', 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
