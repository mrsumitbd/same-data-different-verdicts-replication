"""
fastcox_r1.py 

Fast Cox model for one binary covariate (numpy), used for the
specification-curve permutation test, where every specification is refitted
hundreds of times.

  - Breslow partial likelihood (ties handled exactly under the Breslow
    approximation), optional case weights, optional strata.
  - Newton-Raphson on the single coefficient.
  - Variance: model-based (inverse information), and the Lin-Wei sandwich from
    the Breslow score residuals: per observation, clustered by an arbitrary
    grouping (repository, PR). Matches R survival::coxph(ties = "breslow")
    with cluster(); validated in validate() below.

Data layout is precomputed once per (dataset, strata) by prepare(); a fit then
only needs the covariate vector, so permutations are cheap.
"""

import numpy as np


class Prepared:
    pass


def prepare(d, e, w=None, strata=None, clusters=None):
    """Sort by (stratum, time); group tied times within strata.
    clusters: dict name -> integer codes (in the original row order)."""
    n = len(d)
    w = np.ones(n) if w is None else np.asarray(w, float)
    s = np.zeros(n, np.int64) if strata is None else np.asarray(strata, np.int64)
    order = np.lexsort((d, s))
    p = Prepared()
    p.order = order
    p.d, p.e, p.w, p.s = d[order], e[order].astype(float), w[order], s[order]
    new_grp = np.ones(n, bool)
    new_grp[1:] = (p.d[1:] != p.d[:-1]) | (p.s[1:] != p.s[:-1])
    p.g = np.cumsum(new_grp) - 1
    p.G = p.g[-1] + 1
    gs = p.s[new_grp]                       # stratum of each group
    p.last_of_stratum = np.r_[gs[1:] != gs[:-1], True]   # group is last in its stratum
    p.first_of_stratum = np.r_[True, gs[1:] != gs[:-1]]
    p.gstrat = gs
    p.D = np.bincount(p.g, p.w * p.e, p.G)  # weighted events per group
    p.clusters = {k: np.asarray(v)[order] for k, v in (clusters or {}).items()}
    return p


def _rev_cumsum_within(v, p):
    """For each group: sum of v over groups with time >= its time in the same stratum."""
    c = np.cumsum(v[::-1])[::-1]            # suffix sums over all groups
    # subtract the suffix sum of the next stratum's first group
    nxt = np.zeros(p.G)
    # suffix sum starting at the first group of the following stratum
    first_idx = np.flatnonzero(p.first_of_stratum)
    stratum_id = np.cumsum(p.first_of_stratum) - 1
    following_start = np.r_[first_idx[1:], p.G]
    tail = np.r_[c, 0.0][following_start]
    return c - tail[stratum_id]


def _cumsum_within(v, p):
    c = np.cumsum(v)
    first_idx = np.flatnonzero(p.first_of_stratum)
    stratum_id = np.cumsum(p.first_of_stratum) - 1
    before = np.r_[0.0, c][first_idx]
    return c - before[stratum_id]


def fit(p, x, max_iter=30, tol=1e-10, variances=('model', 'robust')):
    """x: 0/1 covariate in the ORIGINAL row order. Returns dict(coef, se_<name>...)."""
    x = np.asarray(x, float)[p.order]
    X_g = np.bincount(p.g, p.w * p.e * x, p.G)
    keep = p.D > 0
    beta = 0.0
    for _ in range(max_iter):
        r = p.w * np.exp(beta * x)
        R0 = _rev_cumsum_within(np.bincount(p.g, r, p.G), p)
        R1 = _rev_cumsum_within(np.bincount(p.g, r * x, p.G), p)
        xb = np.where(keep, R1 / np.where(R0 > 0, R0, 1), 0)
        U = np.sum(X_g - p.D * xb)
        I = np.sum(p.D * xb * (1 - xb))
        if I <= 0:
            return dict(coef=np.nan)
        step = U / I
        # step halving if needed is unnecessary for one concave parameter, but cap huge steps
        step = np.clip(step, -5, 5)
        beta += step
        if abs(step) < tol:
            break
    r = p.w * np.exp(beta * x)
    R0 = _rev_cumsum_within(np.bincount(p.g, r, p.G), p)
    R1 = _rev_cumsum_within(np.bincount(p.g, r * x, p.G), p)
    xb = np.where(keep, R1 / np.where(R0 > 0, R0, 1), 0)
    I = np.sum(p.D * xb * (1 - xb))
    out = dict(coef=beta, info=I, se_model=np.sqrt(1 / I))
    if len(variances) > 1 or variances[0] != 'model':
        dl = np.where(keep, p.D / np.where(R0 > 0, R0, 1), 0)
        A = _cumsum_within(dl, p)
        B = _cumsum_within(dl * xb, p)
        ex = np.exp(beta * x)
        Ui = p.w * (p.e * (x - xb[p.g]) - ex * (x * A[p.g] - B[p.g]))
        out['se_robust'] = np.sqrt(np.sum(Ui ** 2)) / I
        for k, c in p.clusters.items():
            out[f'se_{k}'] = np.sqrt(np.sum(np.bincount(c, Ui) ** 2)) / I
    return out


def validate():
    import pandas as pd, subprocess, tempfile, os
    rng = np.random.default_rng(1)
    n = 4000
    s = rng.integers(0, 30, n)
    x = rng.integers(0, 2, n)
    cl = rng.integers(0, 300, n)
    d = np.round(rng.exponential(1 / np.exp(0.3 * x + 0.05 * s)), 1) + 0.1
    e = (rng.random(n) < 0.6).astype(int)
    w = rng.uniform(0.2, 2, n)
    f = tempfile.mktemp(suffix='.csv')
    pd.DataFrame(dict(d=d, e=e, x=x, s=s, cl=cl, w=w)).to_csv(f, index=False)
    r = subprocess.run(['Rscript', '-e', f'''suppressMessages(library(survival)); x<-read.csv("{f}")
a<-coxph(Surv(d,e)~x+strata(s),data=x,weights=w,ties="breslow",cluster=cl); b<-coxph(Surv(d,e)~x,data=x,ties="breslow")
cat(coef(a), sqrt(a$var), sqrt(a$naive.var), coef(b), sqrt(b$var), "\\n")'''], capture_output=True, text=True)
    os.remove(f)
    p = prepare(d, e, w, s, {'cl': cl})
    a = fit(p, x)
    p2 = prepare(d, e)
    b = fit(p2, x)
    print('R  :', r.stdout.strip(), r.stderr[-300:])
    print('py :', a['coef'], a['se_cl'], a['se_model'], b['coef'], b['se_model'])


if __name__ == '__main__':
    validate()
