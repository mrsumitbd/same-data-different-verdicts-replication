"""
fastcox_efron_r2.py 

Efron-ties version of src/fastcox_r1.py (Cox model, one binary covariate, optional
case weights and strata), so that the permutation test of the specification curve
uses the same tie handling as the R refits (survival::coxph, ties = "efron") and as
lifelines.

Weighted Efron as in R coxph: in a tied set with d events (counted unweighted) and
event weight sum D_w, meanwt = D_w / d; for l = 0..d-1, f = l/d,
  denom_l = R0 - f E0,  xbar_l = (R1 - f E1) / denom_l,  dLambda_l = meanwt / denom_l,
where R0, R1 are the weighted risk sums and E0, E1 the same sums over the tied events.
  U = sum_sets [X_set - meanwt * sum_l xbar_l],  I = sum_sets meanwt * sum_l xbar_l (1 - xbar_l).
Score residuals (Efron): an event in the set gets x - mean_l xbar_l and is at risk with
weight (1 - f) at the l-th pseudo-event; every other unit at risk gets full weight.
Sandwich variances (per observation, clustered) from the weighted score residuals,
as coxph(..., cluster=) without small-sample correction. validate() checks against R.
"""
import numpy as np

from fastcox_r1 import prepare as _prepare, _rev_cumsum_within, _cumsum_within


def prepare(d, e, w=None, strata=None, clusters=None):
    p = _prepare(d, e, w, strata, clusters)
    # unweighted event count per tie group, and the expanded pseudo-event index
    p.nd = np.bincount(p.g, p.e, p.G).astype(np.int64)
    gi = np.flatnonzero(p.nd > 0)
    p.ev_groups = gi
    p.l_grp = np.repeat(gi, p.nd[gi])                      # group of each pseudo-event
    starts = np.r_[0, np.cumsum(p.nd[gi])[:-1]]
    p.l_frac = (np.arange(p.l_grp.size) - np.repeat(starts, p.nd[gi])) / np.repeat(p.nd[gi], p.nd[gi])
    p.meanwt = np.where(p.nd > 0, p.D / np.maximum(p.nd, 1), 0.0)
    return p


def _pieces(p, x, beta):
    r = p.w * np.exp(beta * x)
    R0 = _rev_cumsum_within(np.bincount(p.g, r, p.G), p)
    R1 = _rev_cumsum_within(np.bincount(p.g, r * x, p.G), p)
    E0 = np.bincount(p.g, r * p.e, p.G)
    E1 = np.bincount(p.g, r * p.e * x, p.G)
    g, f = p.l_grp, p.l_frac
    den = R0[g] - f * E0[g]
    xb = (R1[g] - f * E1[g]) / den
    return r, den, xb


def fit(p, x, max_iter=30, tol=1e-10, variances=('model', 'robust')):
    x = np.asarray(x, float)[p.order]
    X_g = np.bincount(p.g, p.w * p.e * x, p.G)
    mw = p.meanwt[p.l_grp]
    beta = 0.0
    for _ in range(max_iter):
        r, den, xb = _pieces(p, x, beta)
        U = X_g.sum() - np.sum(mw * xb)
        I = np.sum(mw * xb * (1 - xb))
        if not I > 0:
            return dict(coef=np.nan)
        step = np.clip(U / I, -5, 5)
        beta += step
        if abs(step) < tol:
            break
    r, den, xb = _pieces(p, x, beta)
    I = np.sum(mw * xb * (1 - xb))
    out = dict(coef=beta, info=I, se_model=np.sqrt(1 / I))
    if len(variances) > 1 or variances[0] != 'model':
        G = p.G
        dl = mw / den
        f = p.l_frac
        A = np.bincount(p.l_grp, dl, G)                     # full-weight hazard increments per group
        Bx = np.bincount(p.l_grp, dl * xb, G)
        Af = np.bincount(p.l_grp, (1 - f) * dl, G)          # (1 - f)-weighted, for the set's own events
        Bf = np.bincount(p.l_grp, (1 - f) * dl * xb, G)
        M = np.where(p.nd > 0, np.bincount(p.l_grp, xb, G) / np.maximum(p.nd, 1), 0.0)
        Ac, Bc = _cumsum_within(A, p), _cumsum_within(Bx, p)
        prevA = Ac - A                                      # cumulative up to the previous group (same stratum)
        prevB = Bc - Bx
        gi = p.g
        ev = p.e > 0
        Acum = np.where(ev, prevA[gi] + Af[gi], Ac[gi])
        Bcum = np.where(ev, prevB[gi] + Bf[gi], Bc[gi])
        ex = np.exp(beta * x)
        Ui = p.w * (p.e * (x - M[gi]) - ex * (x * Acum - Bcum))
        out['se_robust'] = np.sqrt(np.sum(Ui ** 2)) / I
        for k, c in p.clusters.items():
            out[f'se_{k}'] = np.sqrt(np.sum(np.bincount(c, Ui) ** 2)) / I
    return out


def validate():
    import os
    import subprocess
    import tempfile
    import pandas as pd
    rng = np.random.default_rng(1)
    n = 4000
    s = rng.integers(0, 30, n)
    x = rng.integers(0, 2, n)
    cl = rng.integers(0, 300, n)
    d = np.round(rng.exponential(1 / np.exp(0.3 * x + 0.05 * s)), 1) + 0.1   # heavy ties
    e = (rng.random(n) < 0.6).astype(int)
    w = rng.uniform(0.2, 2, n)
    f = tempfile.mktemp(suffix='.csv')
    pd.DataFrame(dict(d=d, e=e, x=x, s=s, cl=cl, w=w)).to_csv(f, index=False)
    r = subprocess.run(['Rscript', '-e', f'''suppressMessages(library(survival)); x<-read.csv("{f}")
a<-coxph(Surv(d,e)~x+strata(s),data=x,weights=w,ties="efron",cluster=cl)
b<-coxph(Surv(d,e)~x,data=x,ties="efron")
c<-coxph(Surv(d,e)~x,data=x,weights=w,ties="efron",robust=TRUE)
cat(coef(a), sqrt(a$var), sqrt(a$naive.var), coef(b), sqrt(b$var), coef(c), sqrt(c$var), "\\n")'''],
                       capture_output=True, text=True)
    os.remove(f)
    a = fit(prepare(d, e, w, s, {'cl': cl}), x)
    b = fit(prepare(d, e), x)
    c = fit(prepare(d, e, w), x)
    print('R  :', r.stdout.strip(), r.stderr[-300:])
    print('py :', a['coef'], a['se_cl'], a['se_model'], b['coef'], b['se_model'], c['coef'], c['se_robust'])


if __name__ == '__main__':
    validate()
