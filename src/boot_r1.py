"""
boot_r1.py 

Repository cluster bootstrap of the repository-stratified Cox model, exactly as
in src/analyze_rq1_master.py (stratified_cox_bootstrap / parallel_bootstrap_hrs):
seed 42, 500 resamples, repositories drawn with replacement from all
repositories in the input (in order of first appearance), each drawn copy its
own stratum, lifelines CoxPHFitter (Efron), percentile intervals. Extended to
optional extra covariates (the tier coefficient is reported) and to save every
draw, so other interval levels (97.5%, Bonferroni) and bootstrap SEs can be read
from the saved draws.

boot_within(df, name, covariates=(), n_boot=500, seed=42)
    df columns: duration_days, event, is_tier, slug (+ covariates)
    writes results_v6/round1/boot_draws/<name>.csv (draw, hr) and returns a dict.
    BOOT_ENGINE=R runs the same scheme in R (src/boot_within_r1.R), several times faster.
    p_boot: two-sided bootstrap p, 2 x the smaller share of draws on either side of HR = 1.
"""

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from lifelines import CoxPHFitter

OUT = 'results_v6/round1/boot_draws'


def _draw(strat_input, sampled, cols):
    pieces = []
    for i, repo in enumerate(sampled):
        piece = strat_input[strat_input['slug'] == repo].copy()
        piece['_boot_stratum'] = f'{repo}__draw{i}'
        pieces.append(piece)
    resampled = pd.concat(pieces, ignore_index=True)
    try:
        c = CoxPHFitter()
        c.fit(resampled[['duration_days', 'event', 'is_tier', *cols, '_boot_stratum']],
              duration_col='duration_days', event_col='event', strata=['_boot_stratum'])
        return float(np.exp(c.summary.loc['is_tier', 'coef']))
    except Exception:
        return np.nan


def point_estimate(df, covariates=()):
    c = CoxPHFitter()
    c.fit(df[['duration_days', 'event', 'is_tier', *covariates, 'slug']], 'duration_days', 'event', strata=['slug'])
    s = c.summary
    return float(np.exp(s.loc['is_tier', 'coef'])), s


def boot_within(df, name, covariates=(), n_boot=500, seed=42, jobs=-1):
    cols = list(covariates)
    strat_input = df[['duration_days', 'event', 'is_tier', 'slug', *cols]].dropna().reset_index(drop=True)
    hr, summ = point_estimate(strat_input, cols)
    if os.environ.get('BOOT_ENGINE') == 'R':
        # same scheme in R survival (src/boot_within_r1.R); R's RNG, so not the same resamples
        import subprocess, tempfile
        tmp = tempfile.mkdtemp(prefix='boot_r1_')
        fin, fout = os.path.join(tmp, 'in.csv'), os.path.join(tmp, 'out.csv')
        strat_input.to_csv(fin, index=False)
        subprocess.run(['Rscript', 'src/boot_within_r1.R', fin, fout, str(n_boot), str(seed),
                        str(os.cpu_count() if jobs == -1 else jobs), *cols], check=True)
        hrs = pd.read_csv(fout)['hr'].to_numpy(float)
        os.remove(fin); os.remove(fout); os.rmdir(tmp)
        engine = 'R survival (Efron), R RNG'
    else:
        rng = np.random.default_rng(seed)
        repos = strat_input['slug'].unique()
        draws = [rng.choice(repos, size=len(repos), replace=True) for _ in range(n_boot)]
        with parallel_config(backend='loky', inner_max_num_threads=1):
            hrs = Parallel(n_jobs=jobs)(delayed(_draw)(strat_input, s, cols) for s in draws)
        hrs = np.array(hrs, float)
        engine = 'lifelines (Efron), numpy RNG as analyze_rq1_master.py'
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame({'draw': np.arange(1, n_boot + 1), 'hr': hrs}).to_csv(os.path.join(OUT, f'{name}.csv'), index=False)
    ok = hrs[np.isfinite(hrs)]
    lg = np.log(ok)
    n_contrib = int((strat_input.groupby('slug')['is_tier'].nunique() >= 2).sum())
    p_boot = float(min(1.0, 2 * min((ok < 1).mean(), (ok > 1).mean())))
    return dict(name=name, engine=engine, p_boot=p_boot, hr=hr, n_tier=int(strat_input['is_tier'].sum()), n_contributing_strata=n_contrib,
                n_strata=int(strat_input['slug'].nunique()), boot_valid=len(ok),
                boot_median=float(np.median(ok)),
                lo95=float(np.percentile(ok, 2.5)), hi95=float(np.percentile(ok, 97.5)),
                lo975=float(np.percentile(ok, 1.25)), hi975=float(np.percentile(ok, 98.75)),
                se_log_boot=float(np.std(lg, ddof=1)),
                covariates={k: (float(np.exp(summ.loc[k, 'coef'])), float(summ.loc[k, 'p'])) for k in cols})
