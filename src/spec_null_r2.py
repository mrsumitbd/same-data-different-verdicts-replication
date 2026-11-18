"""
spec_null_r2.py 

Specification-curve refits and PR-level permutation test (Simonsohn et al. 2020)
per observation window / cohort, with Breslow (src/fastcox_r1.py, the round-1
engine) or Efron ties (src/fastcox_efron_r2.py, matches R coxph ties = "efron",
which produced the verdict tables). Same scheme and seed as src/spec_null_r1.py:
labels permuted among PRs (across) or among PRs within each repository (within,
across_shared), one permutation per draw shared by all datasets and units;
draw 0 is the observed data. Permutations are generated for both tiers in the
round-1 order, so the full-window draws equal round 1's.

Window / cohort variants (rows of results_v6/round1/work/spec_<key>.csv, follow-up
from followup_<key>.csv; follow-up = days from merge to the cutoff, minimum 154):
  full        all PRs, followed to the cutoff (paper)
  c8          PRs with >= 240 days of follow-up (all groups), followed to the cutoff
  c8h         same cohort, censored 240 days after merge
  c10h        PRs with >= 300 days, censored 300 days after merge
  nopdfme     full window without pdfme/pdfme PR #1059                         (E1a)
  coh270_150  full window, Devin restricted to its 270 PRs with >= 240 days
              (the 142 PRs of c5_devin_prs_dropped_8m.csv removed; human and
              human-supervised PRs unchanged), every unit censored at 150 days    (E1b)
  coh8m_150   the c8 cohort (all groups >= 240 days) censored at 150 days       (E1b, extra)
  c6 / c6h    PRs with >= 180 days (all groups), to the cutoff / censored at 180 days (E10c)

Usage:
  python src/spec_null_r2.py run --variant c8 --engine efron [--tiers fully_autonomous] [--draws 500] [--jobs 8]
  python src/spec_null_r2.py summarize
Output: results_v6/round2/work/null_<variant>_<engine>.csv.gz (one row per spec x draw);
        results_v6/round2/e1_window_curve.txt / .csv, e4_efron_breslow.txt (summaries)
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import fastcox_r1
import fastcox_efron_r2

WD1 = 'results_v6/round1/work'
OUT = 'results_v6/round2'
WD = os.path.join(OUT, 'work')
TIERS = ['human_supervised', 'fully_autonomous']
SEED = 42
VARIANTS = {
    'full': dict(min_fu=0, horizon=None),
    'c8': dict(min_fu=240, horizon=None),
    'c8h': dict(min_fu=240, horizon=240),
    'c10h': dict(min_fu=300, horizon=300),
    'nopdfme': dict(min_fu=0, horizon=None, drop_pr='pdfme/pdfme#1059'),
    'coh270_150': dict(min_fu=0, horizon=150, fa_min_fu=240),
    'coh8m_150': dict(min_fu=240, horizon=150),
    'c6': dict(min_fu=180, horizon=None),
    'c6h': dict(min_fu=180, horizon=180),
}
LABEL = {'full': 'Full (>= 5 months)', 'c8': '>= 8 months', 'c8h': '>= 8 months, censored at 240 days',
         'c10h': '>= 10 months, censored at 300 days', 'nopdfme': 'Full, without pdfme #1059',
         'coh270_150': 'Full, without the 142 PRs, censored at 150 days',
         'coh8m_150': '>= 8-month cohort (all groups), censored at 150 days',
         'c6': '>= 6 months', 'c6h': '>= 6 months, censored at 180 days'}


def load(key, path, variant, cols=('d', 'e1', 'e2', 'grp', 'slug', 'pr', 'w')):
    x = pd.read_csv(path, usecols=list(cols))
    v = VARIANTS[variant]
    fu = pd.read_csv(os.path.join(WD1, f'followup_{key}.csv'))
    assert len(fu) == len(x) and (fu['pr'].values == x['pr'].values).all()
    keep = fu['followup'].values >= v['min_fu']
    if v.get('fa_min_fu'):
        keep &= ~((x['grp'].values == 'fully_autonomous') & (fu['followup'].values < v['fa_min_fu']))
    if v.get('drop_pr'):
        keep &= x['pr'].values != v['drop_pr']
    x = x[keep].copy()
    if v['horizon'] is not None:
        late = x['d'] > v['horizon']
        x.loc[late, ['e1', 'e2']] = 0
        x.loc[late, 'd'] = float(v['horizon'])
    return x


def pr_universe(ds, variant):
    prs = [load(r.key, r.path, variant, cols=('d', 'e1', 'e2', 'pr', 'slug', 'grp'))[['pr', 'slug', 'grp']].drop_duplicates('pr')
           for r in ds.itertuples()]
    return pd.concat(prs).drop_duplicates('pr').reset_index(drop=True)


def permutations(u, draws):
    rng = np.random.default_rng(SEED)
    out = {}
    for tier in TIERS:
        elig = u['grp'].isin([tier, 'human']).values
        obs = (u['grp'] == tier).values
        glob_ = np.zeros((draws + 1, len(u)), bool)
        within = np.zeros((draws + 1, len(u)), bool)
        glob_[0] = within[0] = obs
        idx = np.flatnonzero(elig)
        groups = [np.flatnonzero(elig & (u['slug'] == s).values) for s in u.loc[elig, 'slug'].unique()]
        for b in range(1, draws + 1):
            lab = obs[idx].copy()
            rng.shuffle(lab)
            glob_[b, idx] = lab
            for g in groups:
                lab = obs[g].copy()
                rng.shuffle(lab)
                within[b, g] = lab
        out[tier] = {'global': glob_, 'within': within}
    return out


def run_spec(meta, path, labels, pr_index, variant, engine, draws):
    t0 = time.time()
    eng = fastcox_efron_r2 if engine == 'efron' else fastcox_r1
    x = load(meta['key'], path, variant)
    x = x[x['grp'].isin([meta['tier'], 'human'])]
    obs_t = (x['grp'] == meta['tier']).values
    if meta['comparison'] != 'across':
        both = pd.Series(obs_t).groupby(x['slug'].values).transform('nunique').values == 2
        x, obs_t = x[both], obs_t[both]
    if obs_t.sum() == 0:
        return []
    pr_i = x['pr'].map(pr_index).values
    e = (x['e1'] if meta['definition'] == 'syntactic' else x['e2']).values
    w = x['w'].values if meta['unit'] == 'PR-weighted' else None
    repo = pd.factorize(x['slug'])[0]
    prc = pd.factorize(x['pr'])[0]
    p = eng.prepare(x['d'].values, e, w, repo if meta['comparison'] == 'within' else None, {'cl_repo': repo, 'cl_pr': prc})
    L = labels[meta['tier']]['global' if meta['comparison'] == 'across' else 'within']
    n_pr_tier = len(set(x['pr'][obs_t]))
    rows = []
    for b in range(min(L.shape[0], draws + 1)):
        t = L[b, pr_i].astype(float)
        if t.min() == t.max():
            continue
        r = eng.fit(p, t)
        rows.append(dict(**meta, draw=b, coef=r['coef'], se_model=r.get('se_model'), se_robust=r.get('se_robust'),
                         se_cl_repo=r.get('se_cl_repo'), se_cl_pr=r.get('se_cl_pr'),
                         n=len(x), n_pr_tier=n_pr_tier, n_repo_tier=len(set(x['slug'][obs_t]))))
    print(f"{variant} {engine} {meta} {len(rows)} draws {time.time() - t0:.0f}s", flush=True)
    return rows


def run(a):
    os.makedirs(WD, exist_ok=True)
    ds = pd.read_csv(os.path.join(WD1, 'spec_datasets.csv'))
    u = pr_universe(ds, a.variant)
    pr_index = pd.Series(np.arange(len(u)), index=u['pr'])
    labels = permutations(u, a.draws)
    print(f'{a.variant}: PR universe {len(u):,}; permutations ready', flush=True)
    tiers = a.tiers.split(',')
    tasks = []
    for r in ds.itertuples():
        units = ['line', 'PR-weighted'] if r.unit_kind == 'line' else ['hunk']
        defs = ['syntactic'] + (['structural'] if r.unit_kind == 'line' and bool(r.has_ast) else [])
        for unit in units:
            for dfn in defs:
                for cmp in ['across', 'across_shared', 'within']:
                    for tier in tiers:
                        tasks.append((dict(key=r.key, data=r.data, unit=unit, definition=dfn, comparison=cmp, tier=tier), r.path))
    print(f'{len(tasks)} specifications x {a.draws + 1} fits ({a.engine})', flush=True)
    res = Parallel(n_jobs=a.jobs, verbose=0)(delayed(run_spec)(m, pth, labels, pr_index, a.variant, a.engine, a.draws)
                                             for m, pth in tasks)
    out = pd.DataFrame([row for rr in res for row in rr])
    out.to_csv(os.path.join(WD, f'null_{a.variant}_{a.engine}.csv.gz'), index=False)
    print('done', len(out), flush=True)


Z = 1.959964
CMP = {'across': 'across', 'across_shared': 'across_shared', 'within': 'within'}


def summarize_draws(d, est):
    d = d.copy()
    if est == 'current':
        se = np.where(d['comparison'] == 'within', d['se_cl_repo'], np.where(d['unit'] == 'PR-weighted', d['se_robust'], d['se_model']))
    else:
        se = d[f'se_{est}']
    d['hr'] = np.exp(d['coef'])
    d['L'] = d['coef'] + Z * se < 0
    d['S'] = d['coef'] - Z * se > 0
    g = d.groupby(['tier', 'comparison', 'draw']).agg(n=('hr', 'size'), longer=('L', 'sum'), sooner=('S', 'sum'),
                                                      median_hr=('hr', 'median')).reset_index()
    rows = []
    for (t, cmp), x in g.groupby(['tier', 'comparison']):
        obs = x[x['draw'] == 0]
        if not len(obs):
            continue
        obs = obs.iloc[0]
        null = x[(x['draw'] > 0) & (x['n'] == obs['n'])]
        rows.append(dict(estimator=est, tier=t, comparison=cmp, n_specs=int(obs['n']), draws=len(null),
                         longer=int(obs['longer']), none=int(obs['n'] - obs['longer'] - obs['sooner']), sooner=int(obs['sooner']),
                         median_hr=obs['median_hr'],
                         null_longer_mean=null['longer'].mean(), null_longer_95=np.percentile(null['longer'], 95) if len(null) else np.nan,
                         null_sooner_mean=null['sooner'].mean(), null_sooner_95=np.percentile(null['sooner'], 95) if len(null) else np.nan,
                         p_longer=(1 + (null['longer'] >= obs['longer']).sum()) / (1 + len(null)),
                         p_sooner=(1 + (null['sooner'] >= obs['sooner']).sum()) / (1 + len(null)),
                         null_med_lo=np.percentile(null['median_hr'], 2.5) if len(null) else np.nan,
                         null_med_hi=np.percentile(null['median_hr'], 97.5) if len(null) else np.nan,
                         p_median=(1 + (np.abs(np.log(null['median_hr'])) >= abs(np.log(obs['median_hr']))).sum()) / (1 + len(null))))
    return rows


def summarize():
    files = sorted(glob.glob(os.path.join(WD, 'null_*_*.csv.gz')))
    srcs = [(os.path.basename(f)[5:-7].rsplit('_', 1), f) for f in files]
    srcs.append((('full', 'breslow'), os.path.join(WD1, 'c2_null_draws.csv.gz')))   # round 1 (fastcox_r1)
    rows, npr, draws = [], {}, {}
    for (variant, engine), f in srcs:
        d = pd.read_csv(f)
        draws[(variant, engine)] = d
        for est in ['current', 'model', 'cl_repo', 'cl_pr']:
            for r in summarize_draws(d, est):
                rows.append(dict(variant=variant, engine=engine, **r))
        if 'n_pr_tier' in d:
            h = d[(d['draw'] == 0) & (d['key'] == 'v3all_line') & (d['unit'] == 'line') & (d['comparison'] == 'across')]
            for t, x in h.groupby('tier'):
                npr[(variant, t)] = int(x['n_pr_tier'].iloc[0])
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, 'e1_e4_permutation_summary.csv'), index=False)
    report(res, npr, draws)
    return res


def _cell(res, v, eng, tier, cmp, est, perm=True):
    x = res[(res['variant'] == v) & (res['engine'] == eng) & (res['tier'] == tier) & (res['comparison'] == cmp) & (res['estimator'] == est)]
    if not len(x):
        return 'n/a'
    x = x.iloc[0]
    s = f"{x['longer']}/{x['none']}/{x['sooner']} ({x['median_hr']:.2f})"
    if perm and x['draws'] > 0:
        s += f" p_L={x['p_longer']:.3f} p_S={x['p_sooner']:.3f} p_med={x['p_median']:.3f}"
    return s


def report(res, npr, draws):
    out = ['E1 / E10c. Devin (and human-supervised) verdicts per observation window and cohort',
           'Refits: Efron (src/fastcox_efron_r2.py = R coxph ties="efron"); counts longer / none / sooner with the median HR of the 22',
           'specifications; permutation p (500 PR-level permutations, seed 42): p_L / p_S for the count of longer / sooner verdicts,',
           'p_med two-sided for the median HR. "current" = estimator of the paper (model-based for line/hunk, per-observation robust',
           'for PR-weighted across; repository-clustered within).', '']
    order = ['full', 'c8', 'c8h', 'c10h', 'nopdfme', 'coh270_150', 'coh8m_150', 'c6', 'c6h']
    for tier in ['fully_autonomous', 'human_supervised']:
        out.append(f'== {tier} ==')
        for v in order:
            if not ((res['variant'] == v) & (res['engine'] == 'efron') & (res['tier'] == tier)).any():
                continue
            out.append(f'{LABEL[v]} [{v}], tier PRs {npr.get((v, tier), "?")}:')
            for cmp, est, nm in [('across', 'cl_repo', 'across, clustered'), ('across', 'current', 'across, model-based (paper)'), ('across_shared', 'cl_repo', 'across shared, clustered'),
                                 ('within', 'cl_repo', 'within, clustered')]:
                out.append(f'    {nm:30s} {_cell(res, v, "efron", tier, cmp, est)}')
        out.append('')
    # E4
    out += ['E4. Human-supervised (and Devin) full window: Efron vs Breslow permutation tests', '']
    for tier in ['human_supervised', 'fully_autonomous']:
        for cmp in ['across', 'across_shared', 'within']:
            for est in ['cl_repo', 'current', 'cl_pr']:
                out.append(f'  {tier:17s} {cmp:14s} {est:8s} Efron   {_cell(res, "full", "efron", tier, cmp, est)}')
                out.append(f'  {"":17s} {"":14s} {"":8s} Breslow {_cell(res, "full", "breslow", tier, cmp, est)}')
    # Efron observed counts against the Breslow null
    out += ['', 'Efron observed counts against the Breslow null distribution (round-1 draws):']
    br = draws[('full', 'breslow')]
    ef = draws.get(('full', 'efron'))
    if ef is not None:
        for tier in ['human_supervised', 'fully_autonomous']:
            for cmp in ['across', 'within']:
                for est in ['cl_repo', 'current']:
                    o = [r for r in summarize_draws(ef[ef['draw'] == 0], est) if r['tier'] == tier and r['comparison'] == cmp][0]
                    nb = pd.DataFrame(summarize_draws_null(br, est))
                    nb = nb[(nb['tier'] == tier) & (nb['comparison'] == cmp)]
                    pl = (1 + (nb['longer'] >= o['longer']).sum()) / (1 + len(nb))
                    ps = (1 + (nb['sooner'] >= o['sooner']).sum()) / (1 + len(nb))
                    out.append(f'  {tier:17s} {cmp:8s} {est:8s} Efron observed {o["longer"]}/{o["none"]}/{o["sooner"]}: '
                               f'p_L={pl:.3f} p_S={ps:.3f} vs Breslow null')
    # reconciliation of the within median
    out += ['', 'Reconciliation of the Devin within-repository median HR (full window):']
    for (v, eng), lab in [(('full', 'efron'), 'Efron (= R refits, c1_cox_variance.csv, c5_window_curve.txt)'),
                          (('full', 'breslow'), 'Breslow (fastcox_r1, c2_spec_inference)')]:
        d = draws.get((v, eng))
        if d is None:
            continue
        x = d[(d['draw'] == 0) & (d['tier'] == 'fully_autonomous') & (d['comparison'] == 'within')]
        out.append(f'  {lab:62s} median {np.exp(x["coef"]).median():.4f} over {len(x)} specifications')
    txt = '\n'.join(out) + '\n'
    open(os.path.join(OUT, 'e1_window_curve.txt'), 'w').write(txt)
    print(txt)


def summarize_draws_null(d, est):
    """per-draw counts (draw > 0) per tier x comparison."""
    d = d.copy()
    if est == 'current':
        se = np.where(d['comparison'] == 'within', d['se_cl_repo'], np.where(d['unit'] == 'PR-weighted', d['se_robust'], d['se_model']))
    else:
        se = d[f'se_{est}']
    d['L'] = d['coef'] + Z * se < 0
    d['S'] = d['coef'] - Z * se > 0
    g = d[d['draw'] > 0].groupby(['tier', 'comparison', 'draw']).agg(longer=('L', 'sum'), sooner=('S', 'sum')).reset_index()
    return g.to_dict('records')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'summarize'])
    ap.add_argument('--variant', default='full', choices=list(VARIANTS))
    ap.add_argument('--engine', default='efron', choices=['efron', 'breslow'])
    ap.add_argument('--tiers', default='fully_autonomous')
    ap.add_argument('--draws', type=int, default=500)
    ap.add_argument('--jobs', type=int, default=7)
    a = ap.parse_args()
    if a.mode == 'run':
        run(a)
    else:
        print(summarize().to_string())
