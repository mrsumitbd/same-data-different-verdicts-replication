"""
intent_topup_sample_size_v6.py

Sample-size arithmetic for the human intent labels on the v6 primary corpus
(no-vendored: data/death_commit_messages_v6confnv.csv). Reads the top-up key
(results_v6/intent_topup_v6_sample.csv), never the sheet being labelled.

Strata: old = commits in the 156 journal repositories (data/aidev_filtered_v1.csv),
new = commits in the other repositories. Tier of a commit = majority tier among
the hunks it modifies (as in sample_intent_validation.py), from the no-vendored
line data.

Reports:
  existing human labels (385 + 148) in the corpus by stratum and tier;
  stratum sizes; Cochran n with finite-population correction (p = 0.5) and its
  proportional allocation to the strata; worst-case margin of error of the
  post-stratified (stratum-weighted) proportion,
      MoE = z * sqrt( sum_h W_h^2 * 0.25 / n_h * (N_h - n_h) / (N_h - 1) ),  W_h = N_h / N,
  for several new-stratum sizes; whether the first k sheet rows are a
  tier-proportional sample of the new stratum, and if not the row ids that are.

Output: results_v6/intent_topup_sample_size.txt
"""

import math

import numpy as np
import pandas as pd

Z = {95: 1.959964, 99: 2.575829}
TIER = {'pair_programmer': 'human_supervised', 'batch_generator': 'human_supervised', 'autonomous': 'devin',
        'human': 'human'}


def cochran(N, e, conf):
    n0 = Z[conf] ** 2 * 0.25 / e ** 2
    return n0, n0 / (1 + (n0 - 1) / N)


def moe(strata, conf):
    N = sum(Nh for Nh, _ in strata)
    v = sum((Nh / N) ** 2 * 0.25 / nh * (Nh - nh) / (Nh - 1) for Nh, nh in strata if nh > 0)
    return Z[conf] * math.sqrt(v)


def allocate(sizes, n):
    q = sizes / sizes.sum() * n
    a = np.floor(q).astype(int)
    a[(q - a).sort_values(ascending=False).index[:int(n - a.sum())]] += 1
    return a


def main():
    corpus = pd.read_csv('data/death_commit_messages_v6confnv.csv')[['slug', 'sha']].drop_duplicates()
    journal = set(pd.read_csv('data/aidev_filtered_v1.csv')['slug'])
    line = pd.read_csv('data/survival_events_line_v6confnv_analysis.csv',
                       usecols=['slug', 'pr_number', 'hunk_id', 'agent_tier', 'death_sha', 'event'], low_memory=False)
    line = line[line['event'] == 1].drop_duplicates(['slug', 'death_sha', 'pr_number', 'hunk_id'])
    tier = (line.assign(t=line['agent_tier'].map(TIER)).groupby(['slug', 'death_sha'])['t']
            .agg(lambda s: s.value_counts().sort_index().idxmax()))
    corpus['tier'] = [tier.get((s, h)) for s, h in zip(corpus['slug'], corpus['sha'])]
    corpus['stratum'] = np.where(corpus['slug'].isin(journal), 'old', 'new')
    key = set(zip(corpus['slug'], corpus['sha']))
    lab = pd.concat([pd.read_csv('results/intent_validation_sample_full.csv')[['slug', 'sha']].assign(set='sample385'),
                     pd.read_csv('results/intent_topup_sample.csv')[['slug', 'sha']].assign(set='topup148')])
    lab = lab[[(s, h) in key for s, h in zip(lab['slug'], lab['sha'])]].merge(corpus, on=['slug', 'sha'])
    N_old, N_new = (corpus['stratum'] == 'old').sum(), (corpus['stratum'] == 'new').sum()
    N = N_old + N_new
    n_old = (lab['stratum'] == 'old').sum()
    out = ['Intent label sample size, v6 primary corpus (no vendored lines)',
           f'corpus data/death_commit_messages_v6confnv.csv: N = {N:,} modification commits',
           f'  old stratum (156 journal repos) N_old = {N_old:,}; new stratum (other repos) N_new = {N_new:,} '
           f'({N_old / N * 100:.1f}% / {N_new / N * 100:.1f}%)', '',
           '(1) existing human labels (533 = 385 sample + 148 top-up) in this corpus: '
           f'{len(lab)} (old {n_old}, new {(lab["stratum"] == "new").sum()})']
    out.append(pd.crosstab([lab['stratum'], lab['set']], lab['tier'], margins=True).to_string())
    out.append('\ncorpus tiers by stratum:')
    out.append(pd.crosstab(corpus['stratum'], corpus['tier'], margins=True).to_string())

    out.append('\n(2)-(3) Cochran n with finite-population correction (p = 0.5), whole corpus, and its proportional '
               'allocation:')
    alloc = {}
    for conf, e in [(95, 0.05), (95, 0.04), (99, 0.05)]:
        n0, n = cochran(N, e, conf)
        n = math.ceil(n)
        a = allocate(pd.Series({'old': N_old, 'new': N_new}), n)
        alloc[(conf, e)] = a
        out.append(f'  {conf}% / +/-{e * 100:.0f}%: n0 = {n0:.1f}, n = {n} -> old {a["old"]}, new {a["new"]} '
                   f'(old labels available {n_old}; '
                   f'{"enough" if n_old >= a["old"] else "short by " + str(a["old"] - n_old)})')

    out.append('\n(4) worst-case (p = 0.5) margin of the post-stratified estimate, fpc, with the '
               f'{n_old} existing old-stratum labels:')
    k95 = alloc[(95, 0.05)]['new']
    sizes = sorted({100, int(k95), 147, 200, 256})
    out.append(f'  {"n_new":>6s} {"n total":>8s} {"MoE 95%":>9s} {"MoE 99%":>9s}')
    for nn in sizes:
        out.append(f'  {nn:6d} {n_old + nn:8d} {moe([(N_old, n_old), (N_new, nn)], 95) * 100:8.2f}% '
                   f'{moe([(N_old, n_old), (N_new, nn)], 99) * 100:8.2f}%')

    k = pd.read_csv('results_v6/intent_topup_v6_sample.csv')
    pool = corpus[corpus['stratum'] == 'new']
    inpool = set(zip(pool['slug'], pool['sha']))
    k['in_nv_pool'] = [(s, h) in inpool for s, h in zip(k['slug'], k['sha'])]
    k = k.merge(corpus[['slug', 'sha', 'tier']].rename(columns={'tier': 'tier_nv'}), on=['slug', 'sha'], how='left')
    n_in = k['in_nv_pool'].sum()
    out.append(f'\ntop-up sheet (drawn from the with-vendored new-repo pool of 2,202): {len(k)} rows, {n_in} are in the '
               f'no-vendored new stratum ({len(k) - n_in} touch only vendored lines and drop out); '
               f'tier changed by removing vendored lines: '
               f'{(k["in_nv_pool"] & (k["tier"] != k["tier_nv"])).sum()}')
    k_eff = k[k['in_nv_pool']]
    out.append(f'  with all {len(k)} labelled: new-stratum labels {n_in}, total {n_old + n_in}; MoE 95% '
               f'{moe([(N_old, n_old), (N_new, n_in)], 95) * 100:.2f}%, 99% {moe([(N_old, n_old), (N_new, n_in)], 99) * 100:.2f}%')

    psize = pool['tier'].value_counts().sort_index()
    out.append('\n(5) tier-proportional prefixes of the sheet (row order V001...), new-stratum pool tiers: '
               + ', '.join(f'{t} {v}' for t, v in psize.items()))
    for kk in sorted({100, int(k95)}):
        target = allocate(psize, kk)
        pref = k.iloc[:kk]
        pref_in = pref[pref['in_nv_pool']]
        got = pref_in['tier_nv'].value_counts().reindex(target.index, fill_value=0)
        ok = (got == target).all() and len(pref_in) == kk
        out.append(f'  k = {kk}: target ' + ', '.join(f'{t} {v}' for t, v in target.items())
                   + f'; first {kk} rows give ' + ', '.join(f'{t} {v}' for t, v in got.items())
                   + f' ({len(pref) - len(pref_in)} not in pool) -> {"proportional" if ok else "NOT proportional"}')
        if not ok:
            ids = []
            for t, n in target.items():
                ids += k_eff[k_eff['tier_nv'] == t]['sample_id'].iloc[:n].tolist()
            ids = sorted(ids)
            out.append(f'    proportional set of {len(ids)} rows (first rows of each tier in sheet order): '
                       + ' '.join(ids))
            out.append(f'    i.e. label up to {ids[-1]} and skip: '
                       + ' '.join(sorted(set(k.iloc[:k.index[k["sample_id"] == ids[-1]][0] + 1]['sample_id']) - set(ids))))
    text = '\n'.join(out) + '\n'
    open('results_v6/intent_topup_sample_size.txt', 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
