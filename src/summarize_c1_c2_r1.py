"""
summarize_c1_c2_r1.py 

C1: from results_v6/round1/c1_cox_variance.csv (src/cox_variance_r1.R)
  - check that the R refits reproduce the hazard ratios of results_v6/spec_curve.csv
  - verdict counts (95% CI excludes 1: longer / none / sooner) per tier and
    comparison under the variance estimator the paper uses ('current': model-based
    for unweighted across-repository fits, per-observation robust for PR-weighted
    across fits, repository-clustered within repository), clustered by repository,
    clustered by PR; and the new shared-repository pooled specification
  - the RQ1 ladder / RQ2 / RQ4 across-repository numbers under each estimator
C2: from results_v6/round1/work/c2_null_draws.csv.gz (src/spec_null_r1.py)
  - observed vs permutation-null share of significant specifications per
    direction and median HR, per tier and scope, per variance estimator
  - direction of every point estimate (HR above / below 1)
Outputs: results_v6/round1/c1_variance_verdicts.txt (+ c1_spec_curve_clustered.csv),
         results_v6/round1/c2_spec_inference.txt (+ .csv)
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = 'results_v6/round1'
CMP = {'across': 'across repositories', 'within': 'within repository', 'across_shared': 'across, shared repositories'}


def verdict(lo, hi, hr):
    return np.where((hi < 1) | (lo > 1), np.where(hr < 1, 'longer', 'sooner'), 'none')


def counts(df, est):
    v = pd.Series(verdict(df[f'lo_{est}'], df[f'hi_{est}'], df['hr']), index=df.index)
    c = v.groupby([df['tier'], df['comparison']]).value_counts().unstack(fill_value=0)
    return c.reindex(columns=['longer', 'none', 'sooner'], fill_value=0)


def c1():
    r = pd.read_csv(f'{OUT}/c1_cox_variance.csv')
    sc = pd.read_csv('results_v6/spec_curve.csv')
    spec = r[r['tier'].isin(['human_supervised', 'fully_autonomous']) & r['definition'].isin(['syntactic', 'structural'])].copy()
    # 'current' estimator: what the paper used for each specification
    cur = np.where(spec['comparison'] == 'within', 'cl_repo', np.where(spec['unit'] == 'PR-weighted', 'robust', 'model'))
    for k in ['lo', 'hi', 'p', 'se']:
        spec[f'{k}_current'] = [spec.loc[i, f'{k}_{c}'] for i, c in zip(spec.index, cur)]
    m = spec[spec['comparison'] != 'across_shared'].assign(comparison=lambda d: d['comparison'].map(CMP)).merge(
        sc, on=['tier', 'data', 'unit', 'definition', 'comparison'], suffixes=('', '_sc'))
    out = ['=' * 78, 'C1. Variance estimators and the specification curve', '=' * 78, '',
           f'R refits matched to spec_curve.csv: {len(m)} of {len(sc)} specifications; '
           f'max |log HR difference| {np.abs(np.log(m["hr"] / m["hr_sc"])).max():.2e}; '
           f'max |log CI-bound difference| (current estimator vs lifelines) '
           f'{max(np.abs(np.log(m["lo_current"] / m["ci_low"])).max(), np.abs(np.log(m["hi_current"] / m["ci_high"])).max()):.2e}',
           '']
    out.append('A. Verdict counts (95% CI excludes 1): longer / none / sooner')
    tab = []
    for est, name in [('current', 'as in the paper'), ('model', 'model-based'), ('robust', 'robust, per observation'),
                      ('cl_repo', 'clustered by repository'), ('cl_pr', 'clustered by PR')]:
        c = counts(spec, est)
        for (t, cmp), row in c.iterrows():
            tab.append(dict(estimator=name, tier=t, comparison=CMP[cmp], longer=row['longer'], none=row['none'],
                            sooner=row['sooner'], n=row.sum()))
    tab = pd.DataFrame(tab)
    tab.to_csv(f'{OUT}/c1_verdict_counts.csv', index=False)
    for (t, cmp), g in tab.groupby(['tier', 'comparison'], sort=False):
        out.append(f'  {t} / {cmp}:')
        for _, x in g.iterrows():
            out.append(f'    {x["estimator"]:26s} {x["longer"]:2d} / {x["none"]:2d} / {x["sooner"]:2d}  (of {x["n"]})')
    spec.to_csv(f'{OUT}/c1_spec_curve_clustered.csv', index=False)

    out += ['', 'B. Headline across-repository models (identity tracking, all lines, line unit) under each estimator']
    hl = r[(r['key'] == 'v3all_line') & (r['unit'] == 'line')]
    lab = {('pooled', 'syntactic'): 'RQ1 step 1 (pooled agents), Def 1', ('human_supervised', 'syntactic'): 'RQ1 step 2 / RQ4 across, Def 1',
           ('fully_autonomous', 'syntactic'): 'RQ1 step 2 / RQ4 across, Def 1',
           ('human_supervised', 'structural'): 'RQ1 step 3 / RQ2, Def 2', ('fully_autonomous', 'structural'): 'RQ1 step 3 / RQ2, Def 2',
           ('human_supervised', 'structural_known'): 'RQ2 Def 2, unknown AST dropped',
           ('fully_autonomous', 'structural_known'): 'RQ2 Def 2, unknown AST dropped'}
    for cmp in ['across', 'across_shared', 'within']:
        for (t, d), name in lab.items():
            x = hl[(hl['tier'] == t) & (hl['definition'] == d) & (hl['comparison'] == cmp)]
            if not len(x):
                continue
            x = x.iloc[0]
            out.append(f'  [{CMP[cmp]}] {t:17s} {name:36s} HR {x["hr"]:.3f}  model [{x["lo_model"]:.3f}, {x["hi_model"]:.3f}]  '
                       f'repo-cl [{x["lo_cl_repo"]:.3f}, {x["hi_cl_repo"]:.3f}] p={x["p_cl_repo"]:.3f}  '
                       f'PR-cl [{x["lo_cl_pr"]:.3f}, {x["hi_cl_pr"]:.3f}] p={x["p_cl_pr"]:.3f}  '
                       f'(n repos {x["n_repo"]}, tier repos {x["n_repo_tier"]})')
    out += ['', 'C. RQ1 ladder rungs (syntactic, across repositories) under each estimator']
    for (key, unit) in [('conf_line', 'line'), ('conf_line', 'PR-weighted'), ('conf_hunk', 'hunk'), ('v3wv_line', 'line'),
                        ('v3wv_line', 'PR-weighted'), ('v3all_line', 'line'), ('v3all_line', 'PR-weighted'), ('v3all_hunk', 'hunk'),
                        ('v3cap_line', 'line'), ('v3cap_line', 'PR-weighted'), ('v2cap_line', 'line'), ('v2cap_line', 'PR-weighted'),
                        ('v2cap_hunk', 'hunk')]:
        for t in ['human_supervised', 'fully_autonomous']:
            x = r[(r['key'] == key) & (r['unit'] == unit) & (r['tier'] == t) & (r['definition'] == 'syntactic') &
                  (r['comparison'] == 'across')]
            if len(x):
                x = x.iloc[0]
                cur = 'robust' if unit == 'PR-weighted' else 'model'
                out.append(f'  {key:10s} {unit:11s} {t:17s} HR {x["hr"]:.3f}  paper [{x[f"lo_{cur}"]:.3f}, {x[f"hi_{cur}"]:.3f}]  '
                           f'repo-cl [{x["lo_cl_repo"]:.3f}, {x["hi_cl_repo"]:.3f}]  PR-cl [{x["lo_cl_pr"]:.3f}, {x["hi_cl_pr"]:.3f}]')
    text = '\n'.join(out) + '\n'
    open(f'{OUT}/c1_variance_verdicts.txt', 'w').write(text)
    print(text)


def c2():
    p = 'results_v6/round1/work/c2_null_draws.csv.gz'
    if not os.path.exists(p):
        print('no C2 draws yet')
        return
    d = pd.read_csv(p)
    d['hr'] = np.exp(d['coef'])
    z = 1.959964
    rows, out = [], ['=' * 78, 'C2. Specification-curve inference under the null (PR-level label permutation)',
                     'Breslow Cox (src/fastcox_r1.py); across: labels permuted among all PRs; within and across-shared: within repository',
                     '=' * 78, '']
    for est in ['current', 'cl_repo', 'cl_pr']:
        if est == 'current':
            se = np.where(d['comparison'] == 'within', d['se_cl_repo'],
                          np.where(d['unit'] == 'PR-weighted', d['se_robust'], d['se_model']))
        else:
            se = d[f'se_{est}']
        lo, hi = d['coef'] - z * se, d['coef'] + z * se
        d['sig_longer'] = hi < 0
        d['sig_sooner'] = lo > 0
        g = d.groupby(['tier', 'comparison', 'draw']).agg(n=('hr', 'size'), longer=('sig_longer', 'sum'),
                                                          sooner=('sig_sooner', 'sum'), median_hr=('hr', 'median'),
                                                          below1=('hr', lambda h: (h < 1).sum())).reset_index()
        for (t, cmp), x in g.groupby(['tier', 'comparison']):
            obs = x[x['draw'] == 0].iloc[0]
            null = x[x['draw'] > 0]
            r = dict(estimator=est, tier=t, comparison=CMP[cmp], n_specs=int(obs['n']), draws=len(null),
                     obs_longer=int(obs['longer']), obs_sooner=int(obs['sooner']), obs_median_hr=obs['median_hr'],
                     obs_hr_below1=int(obs['below1']),
                     null_longer_mean=null['longer'].mean(), null_sooner_mean=null['sooner'].mean(),
                     null_longer_95=np.percentile(null['longer'], 95), null_sooner_95=np.percentile(null['sooner'], 95),
                     p_longer=(1 + (null['longer'] >= obs['longer']).sum()) / (1 + len(null)),
                     p_sooner=(1 + (null['sooner'] >= obs['sooner']).sum()) / (1 + len(null)),
                     null_median_hr_2_5=np.percentile(null['median_hr'], 2.5),
                     null_median_hr_97_5=np.percentile(null['median_hr'], 97.5),
                     p_median_two_sided=(1 + (np.abs(np.log(null['median_hr'])) >= abs(np.log(obs['median_hr']))).sum()) / (1 + len(null)))
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(f'{OUT}/c2_spec_inference.csv', index=False)
    for est, g in res.groupby('estimator', sort=False):
        out.append(f'Variance estimator: {est}')
        for _, r in g.iterrows():
            out.append(f'  {r["tier"]:17s} {r["comparison"]:28s} specs {r["n_specs"]:2d}: significant longer {r["obs_longer"]:2d} '
                       f'(null mean {r["null_longer_mean"]:.1f}, 95th pct {r["null_longer_95"]:.0f}, p={r["p_longer"]:.3f}); '
                       f'sooner {r["obs_sooner"]:2d} (null mean {r["null_sooner_mean"]:.1f}, 95th {r["null_sooner_95"]:.0f}, p={r["p_sooner"]:.3f}); '
                       f'median HR {r["obs_median_hr"]:.3f} (null 95% [{r["null_median_hr_2_5"]:.3f}, {r["null_median_hr_97_5"]:.3f}], '
                       f'p={r["p_median_two_sided"]:.3f}); HR<1 in {r["obs_hr_below1"]}/{r["n_specs"]}')
        out.append('')
    text = '\n'.join(out) + '\n'
    open(f'{OUT}/c2_spec_inference.txt', 'w').write(text)
    print(text)


if __name__ == '__main__':
    if 'c1' in sys.argv[1:] or len(sys.argv) == 1:
        c1()
    if 'c2' in sys.argv[1:] or len(sys.argv) == 1:
        c2()
