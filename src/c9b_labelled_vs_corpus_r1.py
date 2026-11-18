"""
c9b_labelled_vs_corpus_r1.py 

Are the 618 human-labelled commits representative of the corpus, and what
explains the human-supervised corrective gap seen in the labelled sample
(human labels: HS 28% vs human 21% corrective, p = .038) but not in the corpus?

1. Labelled (618) vs unlabelled corpus commits (the 5,300 tier-mapped
   commits minus the labelled ones) on observable features: tier, repository,
   modification month, message length, birth agent, label round (sample set).
2. Corrective share (majority vote of the three LLMs, mv3 with best-kappa
   tie-break; available for every commit) for HS vs human in the labelled and
   the unlabelled commits: is the gap a property of the labelled commits?
3. For each feature, the labelled commits are reweighted (within tier) to the
   unlabelled distribution of that feature; the feature whose reweighting
   removes the labelled gap (human and LLM labels) is the one that explains it.

Usage: DATA_TAG=v6confnv RESULTS_DIR=results_v6 python src/c9b_labelled_vs_corpus_r1.py
Output: results_v6/round1/c9b_labelled_vs_corpus.txt
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import analyze_intent_aggregation as A
from config.paths import P

L = 'results_v6/labels618'


def wshare(lab, w, target='corrective'):
    lab = np.asarray(lab)
    w = np.broadcast_to(np.asarray(w, float), lab.shape)
    return float(np.sum(w * (lab == target)) / np.sum(w)) * 100


def main():
    corpus = pd.read_csv(P('data/death_commit_messages_v5.csv')).drop_duplicates(['slug', 'sha'])
    for m in A.NEW_LLMS:
        r = pd.read_csv(P(f'results/intent_{m}_full_corpus_t0.csv')).drop_duplicates(['slug', 'sha'], keep='last')
        corpus = corpus.merge(r[['slug', 'sha', 'label']].rename(columns={'label': m}), on=['slug', 'sha'], how='left')
        corpus[m] = corpus[m].fillna('missing').str.lower()
    smp = A.load(f'{L}/gold.csv', f'{L}/sample.csv', f'{L}/keyword.csv')
    k = {m: max(A.kappa(smp['gold'], smp[m]), 0.0) for m in A.NEW_LLMS}
    corpus['mv3'] = A.vote(corpus[A.NEW_LLMS], {m: 1.0 for m in A.NEW_LLMS}, sorted(A.NEW_LLMS, key=lambda m: -k[m]))
    corpus = corpus.merge(A.tier_map_from_line(P('data/survival_events_line_v5_analysis.csv')), on=['slug', 'sha'])
    line = pd.read_csv(P('data/survival_events_line_v5_analysis.csv'), usecols=['slug', 'death_sha', 'death_date', 'agent_name'],
                       low_memory=False).dropna(subset=['death_sha'])
    meta = line.groupby(['slug', 'death_sha']).agg(date=('death_date', 'min'),
                                                   agent=('agent_name', lambda s: s.value_counts().index[0])).reset_index()
    corpus = corpus.merge(meta.rename(columns={'death_sha': 'sha'}), on=['slug', 'sha'], how='left')
    corpus['month'] = corpus['date'].str[:7]
    corpus['msg_len'] = corpus['message'].fillna('').str.len()
    corpus['msg_len_q'] = pd.qcut(corpus['msg_len'], 4, labels=['Q1 short', 'Q2', 'Q3', 'Q4 long']).astype(str)
    sm = pd.read_csv(f'{L}/sample.csv')[['slug', 'sha', 'set']]
    g = pd.read_csv(f'{L}/gold.csv')[['sample_id', 'human_label']]
    sm = pd.read_csv(f'{L}/sample.csv').merge(g, on='sample_id')[['slug', 'sha', 'set', 'human_label']]
    corpus = corpus.merge(sm, on=['slug', 'sha'], how='left')
    corpus['labelled'] = corpus['set'].notna()
    # repository groups: top 20 repositories by corpus commits, rest pooled
    top = corpus['slug'].value_counts().index[:20]
    corpus['repo_grp'] = np.where(corpus['slug'].isin(top), corpus['slug'], 'other repositories')
    corpus['tier'] = corpus['two_tier']
    lab, unl = corpus[corpus['labelled']], corpus[~corpus['labelled']]
    out = ['=' * 78, 'C9b. Labelled (618) vs unlabelled corpus commits', '=' * 78,
           f'tier-mapped corpus commits {len(corpus):,}: labelled {len(lab)}, unlabelled {len(unl):,}', '',
           '1. Feature distributions (labelled vs unlabelled)']
    for f in ['tier', 'repo_grp', 'month', 'msg_len_q', 'agent']:
        tab = pd.crosstab(corpus[f].fillna('n/a'), corpus['labelled'])
        chi2, p, _, _ = chi2_contingency(tab)
        share = (tab / tab.sum() * 100).round(1)
        out.append(f'  {f}: chi-square p = {p:.2g}')
        if f in ('tier', 'msg_len_q'):
            out.append('    ' + share.rename(columns={False: 'unlabelled %', True: 'labelled %'}).to_string().replace('\n', '\n    '))
    out.append(f'  message length median: labelled {lab["msg_len"].median():.0f}, unlabelled {unl["msg_len"].median():.0f} '
               f'(Mann-Whitney p = {mannwhitneyu(lab["msg_len"], unl["msg_len"])[1]:.2g})')
    hs_rep = lambda d: d[d['tier'] == 'human_supervised']['slug'].value_counts(normalize=True).head(3).round(3).to_dict()
    out.append(f'  top HS repositories, labelled: {hs_rep(lab)}; unlabelled: {hs_rep(unl)}')
    out.append(f'  labelled by round: {lab["set"].value_counts().to_dict()}')

    out += ['', '2. Corrective share (%) HS vs human, mv3 LLM label (and human label for the labelled commits)']
    for nm, d in [('labelled', lab), ('unlabelled', unl), ('all corpus', corpus)]:
        hs, hu = d[d['tier'] == 'human_supervised'], d[d['tier'] == 'human']
        line_ = f'  {nm:11s} mv3: HS {wshare(hs["mv3"], np.ones(len(hs))):.1f} vs human {wshare(hu["mv3"], np.ones(len(hu))):.1f}'
        if nm == 'labelled':
            line_ += f'   human label: HS {wshare(hs["human_label"], np.ones(len(hs))):.1f} vs human {wshare(hu["human_label"], np.ones(len(hu))):.1f}'
        out.append(line_ + f'   (n HS {len(hs)}, human {len(hu)})')
    for st in sorted(lab['set'].unique()):
        d = lab[lab['set'] == st]
        hs, hu = d[d['tier'] == 'human_supervised'], d[d['tier'] == 'human']
        out.append(f'    round {st:9s} human label: HS {wshare(hs["human_label"], np.ones(len(hs))):.1f} (n {len(hs)}) vs human '
                   f'{wshare(hu["human_label"], np.ones(len(hu))):.1f} (n {len(hu)})')

    out += ['', '3. Labelled gap (HS minus human corrective, pp) after reweighting the labelled commits, within tier,',
            '   to the unlabelled distribution of one feature (weight = unlabelled share / labelled share of the cell)']
    base_h = (wshare(lab[lab.tier == 'human_supervised']['human_label'], 1) - wshare(lab[lab.tier == 'human']['human_label'], 1))
    base_m = (wshare(lab[lab.tier == 'human_supervised']['mv3'], 1) - wshare(lab[lab.tier == 'human']['mv3'], 1))
    out.append(f'  {"none":12s} human-label gap {base_h:+.1f}   mv3 gap {base_m:+.1f}')
    for f in ['repo_grp', 'slug', 'month', 'msg_len_q', 'agent', 'set']:
        gaps = {}
        for col in ['human_label', 'mv3']:
            sh = {}
            for t in ['human_supervised', 'human']:
                lt, ut = lab[lab['tier'] == t], unl[unl['tier'] == t]
                if f == 'set':
                    w = np.ones(len(lt))   # rounds have no unlabelled counterpart: drop one round at a time below
                else:
                    pu = ut[f].fillna('n/a').value_counts(normalize=True)
                    pl = lt[f].fillna('n/a').value_counts(normalize=True)
                    w = lt[f].fillna('n/a').map(lambda v: pu.get(v, 0) / pl[v]).values
                sh[t] = wshare(lt[col].values, w)
            gaps[col] = sh['human_supervised'] - sh['human']
        if f != 'set':
            out.append(f'  {f:12s} human-label gap {gaps["human_label"]:+.1f}   mv3 gap {gaps["mv3"]:+.1f}')
    # leave-one-repository-out on the labelled HS commits: which repository carries the gap?
    out += ['', '4. Labelled human-label gap when one repository is left out (largest changes)']
    res = []
    for s in lab['slug'].unique():
        d = lab[lab['slug'] != s]
        gap = wshare(d[d.tier == 'human_supervised']['human_label'], 1) - wshare(d[d.tier == 'human']['human_label'], 1)
        res.append((s, gap, int(((lab.slug == s) & (lab.tier == 'human_supervised')).sum()),
                    int(((lab.slug == s) & (lab.tier == 'human_supervised') & (lab.human_label == 'corrective')).sum())))
    for s, gap, n, nc in sorted(res, key=lambda r: r[1])[:6]:
        out.append(f'  without {s:35s} gap {gap:+.1f} pp (that repo: {n} labelled HS commits, {nc} corrective)')
    text = '\n'.join(out) + '\n'
    open('results_v6/round1/c9b_labelled_vs_corpus.txt', 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
