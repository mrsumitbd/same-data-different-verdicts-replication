"""
specification_curve.py

Specification curve ("small multiverse") of the tier-vs-human hazard ratio
over the measurement choices studied in the paper:

  data      v2-tracked capped units   (data/survival_events_line_v4_ast.csv;
                                       line number at HEAD, 20 files / 100 lines caps)
            v3-tracked capped units   (data/survival_events_line_v4units_v5track_ast.csv;
                                       same units, identity tracking)
            v3-tracked all lines      (data/survival_events_line_v5_analysis.csv)
  unit      line-weighted | PR-weighted (each PR counts once) | hunk
            (hunk units: data/survival_events_hunk_v5.csv for v3 all lines,
             data/survival_events_hunk_v3.csv for the v2 miner)
  definition syntactic | AST-structural (lines only)
  comparison unadjusted | repository-stratified (cluster-robust SE by repository)

For each specification and tier: HR, 95% CI, p, and whether the CI excludes 1.
Stratified estimates use the analytic cluster-robust SE (the headline
stratified results in the paper use the 500-resample cluster bootstrap).

Conference born lines (DATA_TAG=v6conf or v6confnv): the curve also starts from
the conference's exact units, with vendored/generated lines kept:
  conference tracking, all lines, with vendored  (conf_* columns of
      data/survival_events_line_v6conf.csv: blame --reverse, death = first later
      commit touching the file; hunk = its lines' first death)
  v3 tracking, all lines, with vendored  (data/survival_events_line_v6conf_analysis.csv,
      data/survival_events_hunk_v6conf.csv)
These two skip the generated-path exclusion; every other dataset applies it (and
with DATA_TAG=v6confnv the vendored lines are already removed). The conference
tracker has no AST status, so it has syntactic specifications only.

Combinations not defined (and therefore absent): v2 tracking on all lines
(the v2 miner only produced capped samples); AST-structural definition for
hunk units; PR weighting for hunk units; v3 tracking for capped hunks.

Usage:
    python src/specification_curve.py
    python src/specification_curve.py --plot-only   # redraw from results/spec_curve.csv
Output:
    results/spec_curve.csv, paper-overfeal-files/visuals/spec_curve.pdf (+ .png)
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.constants import OBSERVATION_CUTOFF
from config.observation_window import apply_observation_window
from config.paths import P
from analyze_rq1_master import add_two_tier_column, is_generated, OUTLIER_PR_KEYS

warnings.filterwarnings('ignore')
TIERS = {'human_supervised': 'Human-supervised', 'fully_autonomous': 'Fully-autonomous (Devin)'}


def prep(df, label, drop_generated=True):
    df = add_two_tier_column(apply_observation_window(df, label=label))
    if drop_generated:
        df = df[~df['unit_id'].apply(is_generated)]
    df = df[~pd.Series(list(zip(df['slug'], df['pr_number'])), index=df.index).isin(OUTLIER_PR_KEYS)]
    return df


def fit(df, tier, event_col, weighted, stratified):
    x = df[df['two_tier'].isin([tier, 'human'])]
    d = pd.DataFrame({'d': x['duration_days'].values, 'e': x[event_col].astype(int).values,
                      't': (x['two_tier'] == tier).astype(int).values, 's': x['slug'].values})
    kw = {}
    if weighted:
        d['w'] = 1.0 / x.groupby(['slug', 'pr_number'])['unit_id'].transform('size').values
        kw['weights_col'] = 'w'
    if stratified:
        d = d[d.groupby('s')['t'].transform('nunique') == 2]
        d['c'] = d['s']
        kw.update(strata=['s'], cluster_col='c')
        cols = ['d', 'e', 't', 's', 'c'] + (['w'] if weighted else [])
    else:
        if weighted:
            kw['robust'] = True
        cols = ['d', 'e', 't'] + (['w'] if weighted else [])
    c = CoxPHFitter().fit(d[cols], 'd', 'e', **kw)
    r = c.summary.loc['t']
    return r['exp(coef)'], r['exp(coef) lower 95%'], r['exp(coef) upper 95%'], r['p']


def conference_units(base):
    """Line and hunk units as tracked by the conference (conf_* columns), all lines."""
    d = pd.read_csv(f'data/survival_events_line_{base}.csv', low_memory=False)
    d['event'] = d['conf_event'].astype(int)
    d['death_date'] = d['conf_death_date']
    bd, dd = pd.to_datetime(d['birth_date']), pd.to_datetime(d['conf_death_date'])
    end = dd.where(d['event'] == 1, pd.Timestamp(OBSERVATION_CUTOFF))
    d['duration_days'] = ((end - bd).dt.total_seconds() / 86400).clip(lower=0.001)
    d['ast_death'] = np.nan
    keys = ['slug', 'pr_number', 'hunk_id']
    h = d.groupby(keys, as_index=False).agg(is_agent=('is_agent', 'first'), agent_tier=('agent_tier', 'first'),
                                            birth_date=('birth_date', 'first'), event=('event', 'max'))
    first = d[d['event'] == 1].groupby(keys)['conf_death_date'].min().rename('death_date')
    h = h.merge(first, on=keys, how='left')
    bd, dd = pd.to_datetime(h['birth_date']), pd.to_datetime(h['death_date'])
    end = dd.where(h['event'] == 1, pd.Timestamp(OBSERVATION_CUTOFF))
    h['duration_days'] = ((end - bd).dt.total_seconds() / 86400).clip(lower=0.001)
    h['unit_id'] = h['hunk_id']
    return d, h


def main():
    data = {
        'v2 tracking, capped': prep(pd.read_csv(P('data/survival_events_line_v4_ast.csv'), low_memory=False), 'v2'),
        'v3 tracking, capped': None,
        'v3 tracking, all lines': prep(pd.read_csv(P('data/survival_events_line_v5_analysis.csv'), low_memory=False), 'v5'),
    }
    v4u = pd.read_csv(P('data/survival_events_line_v4units_v5track_ast.csv'), low_memory=False)
    v4u = v4u[v4u['event_type'] != 'unverified'].copy()
    v4u['event'] = v4u['event'].astype(int)
    data['v3 tracking, capped'] = prep(v4u, 'v3 capped')
    hunks = {'v2 tracking, capped': prep(pd.read_csv(P('data/survival_events_hunk_v3.csv'), low_memory=False), 'hunk v2'),
             'v3 tracking, all lines': prep(pd.read_csv(P('data/survival_events_hunk_v5.csv'), low_memory=False), 'hunk v5')}
    tag = os.environ.get('DATA_TAG', '')
    base = tag[:-2] if tag.endswith('nv') else tag
    if base.startswith('v6conf'):
        cl, ch = conference_units(base)
        data = {'conference tracking, all lines, with vendored': prep(cl, 'conference', drop_generated=False),
                'v3 tracking, all lines, with vendored': prep(pd.read_csv(f'data/survival_events_line_{base}_analysis.csv',
                                                                          low_memory=False), 'v3 with vendored',
                                                              drop_generated=False), **data}
        hunks = {'conference tracking, all lines, with vendored': prep(ch, 'conference hunks', drop_generated=False),
                 'v3 tracking, all lines, with vendored': prep(pd.read_csv(f'data/survival_events_hunk_{base}.csv',
                                                                           low_memory=False), 'hunk with vendored',
                                                               drop_generated=False), **hunks}

    rows = []
    # every specification is an independent Cox fit: run them in parallel (SPEC_JOBS, default 6)
    from joblib import Parallel, delayed
    cols = ['duration_days', 'event', 'event_def2', 'two_tier', 'slug', 'pr_number', 'unit_id']
    tasks = []
    for dname, df in data.items():
        df = df.copy()
        df['event_def2'] = ((df['event'] == 1) & (df['ast_death'] == 1.0)).astype(int)
        df = df[cols]
        for unit in ['line', 'PR-weighted']:
            for defn, col in [('syntactic', 'event'), ('structural', 'event_def2')]:
                if defn == 'structural' and data[dname]['ast_death'].isna().all():
                    continue
                for strat in [False, True]:
                    for tier in TIERS:
                        tasks.append((dict(tier=tier, data=dname, unit=unit, definition=defn,
                                           comparison='within repository' if strat else 'across repositories'),
                                      df, col, unit == 'PR-weighted', strat))
    for dname, h in hunks.items():
        h = h[['duration_days', 'event', 'two_tier', 'slug', 'pr_number', 'unit_id']]
        for strat in [False, True]:
            for tier in TIERS:
                tasks.append((dict(tier=tier, data=dname, unit='hunk', definition='syntactic',
                                   comparison='within repository' if strat else 'across repositories'),
                              h, 'event', False, strat))
    print(f'{len(tasks)} specifications', flush=True)

    def one(meta, df, col, weighted, strat):
        hr, lo, hi, p = fit(df, meta['tier'], col, weighted, strat)
        return dict(**meta, hr=hr, ci_low=lo, ci_high=hi, p=p)

    rows = Parallel(n_jobs=int(os.environ.get('SPEC_JOBS', '6')), verbose=5)(
        delayed(one)(*t) for t in tasks)
    res = pd.DataFrame(rows)
    res['tracking'] = np.where(res['data'].str.startswith('v2'), 'line number (v2)',
                               np.where(res['data'].str.startswith('conference'), 'conference (blame --reverse)',
                                        'identity (v3)'))
    res['caps'] = np.where(res['data'].str.contains('capped'), 'capped',
                           np.where(res['data'].str.contains('with vendored'), 'all lines, with vendored', 'all lines'))
    res['ci_excludes_1'] = (res['ci_high'] < 1) | (res['ci_low'] > 1)
    res['verdict'] = np.where(~res['ci_excludes_1'], 'no difference', np.where(res['hr'] < 1, 'agent retained longer', 'agent modified sooner'))
    res.to_csv(P('results/spec_curve.csv'), index=False)
    print(res.groupby(['tier', 'verdict']).size())
    plot(res)


def plot(res):
    dims = [('tracking', [v for v in ['conference (blame --reverse)', 'line number (v2)', 'identity (v3)']
                          if v in set(res['tracking'])]),
            ('caps', [v for v in ['capped', 'all lines, with vendored', 'all lines'] if v in set(res['caps'])]),
            ('unit', ['line', 'PR-weighted', 'hunk']),
            ('definition', ['syntactic', 'structural']),
            ('comparison', ['across repositories', 'within repository'])]
    nrow = sum(len(v) for _, v in dims)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex='col',
                             gridspec_kw={'height_ratios': [2.2, 2.6], 'hspace': 0.05, 'wspace': 0.15})
    for k, tier in enumerate(TIERS):
        r = res[res['tier'] == tier].sort_values('hr').reset_index(drop=True)
        ax, axg = axes[0, k], axes[1, k]
        col = np.where(~r['ci_excludes_1'], '#999999', np.where(r['hr'] < 1, '#0072B2', '#D55E00'))
        x = np.arange(len(r))
        ax.vlines(x, r['ci_low'], r['ci_high'], color=col, lw=1)
        ax.scatter(x, r['hr'], c=col, s=14, zorder=3)
        ax.axhline(1, color='#333333', lw=0.8, ls='--')
        ax.set_yscale('log')
        ax.set_yticks([0.5, 0.75, 1, 1.5, 2])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(TIERS[tier], fontsize=10, pad=14)
        if k == 0:
            ax.set_ylabel('HR vs human (log scale, 95% CI)')
        y, labels = 0, []
        for dim, vals in dims:
            for v in vals:
                m = (r[dim] == v).values
                axg.scatter(x[m], np.full(m.sum(), y), marker='s', s=10, color='#333333')
                labels.append(v)
                y -= 1
            y -= 0.4
        ticks = []
        y = 0
        for dim, vals in dims:
            for v in vals:
                ticks.append(y)
                y -= 1
            y -= 0.4
        axg.set_yticks(ticks)
        axg.set_yticklabels(labels if k == 0 else [], fontsize=7)
        axg.set_xticks([])
        axg.set_xlabel('specifications, sorted by HR', fontsize=8)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
            axg.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', ls='', color=c, label=l) for c, l in
               [('#0072B2', 'agent code retained longer (CI excludes 1)'),
                ('#D55E00', 'agent code modified sooner (CI excludes 1)'), ('#999999', 'no difference')]]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(P('paper-overfeal-files/visuals/spec_curve.pdf'), bbox_inches='tight')
    fig.savefig(P('paper-overfeal-files/visuals/spec_curve.png'), dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    if '--plot-only' in sys.argv:
        plot(pd.read_csv(P('results/spec_curve.csv')))
    else:
        main()
