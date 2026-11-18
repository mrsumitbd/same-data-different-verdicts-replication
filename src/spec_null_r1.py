"""
spec_null_r1.py 

Inference for the specification curve (Simonsohn, Simmons & Nelson 2020):
how many specifications would be significant in each direction, and what
median hazard ratio would they give, if tier had no effect?

Null: tier vs human labels are permuted at the PR level (every line / hunk of a
PR keeps its PR's label), with one permutation per draw shared by all datasets
and units:
  across repositories          labels permuted among all PRs of the tier and human
  across, shared repositories  labels permuted within repository
  within repository            labels permuted within repository (strata = repository)
Every specification of the curve (tier x data x unit x definition x
comparison, as in src/specification_curve.py, plus the shared-repository
pooled specification of C1c) is refitted on each draw with src/fastcox_r1.py
(Breslow ties, one binary covariate), with the variance estimators model,
robust (per observation), repository-clustered and PR-clustered.
Draw 0 is the observed data.

Usage: python src/spec_null_r1.py [--draws 500] [--jobs 8]
Input : results_v6/round1/work/spec_datasets.csv (src/export_spec_data_r1.py)
Output: results_v6/round1/work/c2_null_draws.csv.gz (one row per spec x draw)
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
from fastcox_r1 import prepare, fit

WD = 'results_v6/round1/work'
TIERS = ['human_supervised', 'fully_autonomous']
SEED = 42


def pr_universe(ds):
    prs = []
    for r in ds.itertuples():
        x = pd.read_csv(r.path, usecols=['pr', 'slug', 'grp']).drop_duplicates('pr')
        prs.append(x)
    u = pd.concat(prs).drop_duplicates('pr').reset_index(drop=True)
    return u


def permutations(u, draws):
    """labels[tier][scope] : (draws+1, n_pr) bool array of 'is tier' (row 0 observed)."""
    rng = np.random.default_rng(SEED)
    out = {}
    for tier in TIERS:
        elig = u['grp'].isin([tier, 'human']).values
        obs = (u['grp'] == tier).values
        glob = np.zeros((draws + 1, len(u)), bool)
        within = np.zeros((draws + 1, len(u)), bool)
        glob[0] = within[0] = obs
        idx = np.flatnonzero(elig)
        groups = [np.flatnonzero(elig & (u['slug'] == s).values) for s in u.loc[elig, 'slug'].unique()]
        for b in range(1, draws + 1):
            lab = obs[idx].copy()
            rng.shuffle(lab)
            glob[b, idx] = lab
            for g in groups:
                lab = obs[g].copy()
                rng.shuffle(lab)
                within[b, g] = lab
        out[tier] = {'global': glob, 'within': within}
    return out


def run_spec(meta, path, labels, pr_index):
    t0 = time.time()
    x = pd.read_csv(path, usecols=['d', 'e1', 'e2', 'grp', 'slug', 'pr', 'w'])
    x = x[x['grp'].isin([meta['tier'], 'human'])]
    obs_t = (x['grp'] == meta['tier']).values
    if meta['comparison'] != 'across':
        both = pd.Series(obs_t).groupby(x['slug'].values).transform('nunique').values == 2
        x, obs_t = x[both], obs_t[both]
    pr_i = x['pr'].map(pr_index).values
    e = (x['e1'] if meta['definition'] == 'syntactic' else x['e2']).values
    w = x['w'].values if meta['unit'] == 'PR-weighted' else None
    repo = pd.factorize(x['slug'])[0]
    prc = pd.factorize(x['pr'])[0]
    p = prepare(x['d'].values, e, w, repo if meta['comparison'] == 'within' else None,
                {'cl_repo': repo, 'cl_pr': prc})
    scope = 'global' if meta['comparison'] == 'across' else 'within'
    L = labels[meta['tier']][scope]
    rows = []
    for b in range(L.shape[0]):
        t = L[b, pr_i].astype(float)
        if t.min() == t.max():
            continue
        r = fit(p, t)
        rows.append(dict(**meta, draw=b, coef=r['coef'], se_model=r.get('se_model'), se_robust=r.get('se_robust'),
                         se_cl_repo=r.get('se_cl_repo'), se_cl_pr=r.get('se_cl_pr')))
    print(f"{meta} {len(rows)} draws {time.time() - t0:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=500)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    ds = pd.read_csv(os.path.join(WD, 'spec_datasets.csv'))
    u = pr_universe(ds)
    pr_index = pd.Series(np.arange(len(u)), index=u['pr'])
    labels = permutations(u, a.draws)
    print(f'PR universe {len(u):,}; permutations ready', flush=True)
    tasks = []
    for r in ds.itertuples():
        units = ['line', 'PR-weighted'] if r.unit_kind == 'line' else ['hunk']
        defs = ['syntactic'] + (['structural'] if r.unit_kind == 'line' and bool(r.has_ast) else [])
        for unit in units:
            for dfn in defs:
                for cmp in ['across', 'across_shared', 'within']:
                    for tier in TIERS:
                        tasks.append((dict(key=r.key, data=r.data, unit=unit, definition=dfn,
                                           comparison=cmp, tier=tier), r.path))
    print(f'{len(tasks)} specifications x {a.draws + 1} fits', flush=True)
    res = Parallel(n_jobs=a.jobs, verbose=0)(delayed(run_spec)(m, pth, labels, pr_index) for m, pth in tasks)
    out = pd.DataFrame([row for rr in res for row in rr])
    out.to_csv(os.path.join(WD, 'c2_null_draws.csv.gz'), index=False)
    print('done', len(out))


if __name__ == '__main__':
    main()
