"""
plot_round1_figs_r1.py 

G1  specification curve with readable labels ('earlier study's tracker',
    'line number', 'identity'), drawn twice:
      spec_curve_labels.{pdf,png}        the paper's intervals (results_v6/spec_curve.csv:
                                         model-based across repositories, robust for PR
                                         weighting, repository-clustered within)
      spec_curve_clustered.{pdf,png}     every interval clustered by repository
                                         (R refits, results_v6/round1/c1_spec_curve_clustered.csv)
    The variance estimator is stated under the figure.
G2  RQ1 ladder with the value labels in a column to the right of both tiers'
    intervals, so they never overlap a marker; again with the paper's intervals
    and with repository-clustered intervals.
Output: results_v6/round1/visuals/ (never paper-overfeal-files/visuals/)
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
from plot_rq1_ladder import LADDER_CONF

OUT = 'results_v6/round1/visuals'
TIERS = {'human_supervised': 'Human-supervised', 'fully_autonomous': 'Fully-autonomous (Devin)'}
TRACK = {'conference (blame --reverse)': "earlier study's tracker", 'line number (v2)': 'line number',
         'identity (v3)': 'identity'}
CAPS = {'capped': 'sampled (20 files / 100 lines)', 'all lines, with vendored': 'all lines, incl. vendored',
        'all lines': 'all lines, vendored/generated removed'}
UNIT = {'line': 'per line', 'PR-weighted': 'per PR (PR-weighted)', 'hunk': 'per hunk'}
DEFN = {'syntactic': 'syntactic (Def. 1)', 'structural': 'AST-structural (Def. 2)'}
COMP = {'across repositories': 'across repositories', 'within repository': 'within repository'}
NOTE = {'paper': 'Intervals: model-based SE across repositories (robust SE for PR weighting); '
                 'SE clustered by repository within repository.',
        'clustered': 'Intervals: all SEs clustered by repository (sandwich estimator).'}


def spec_curve(res, name, note):
    res = res.copy()
    res['tracking'] = res['tracking'].map(TRACK)
    res['caps'] = res['caps'].map(CAPS)
    dims = [('tracking', ["earlier study's tracker", 'line number', 'identity']),
            ('caps', [CAPS['capped'], CAPS['all lines, with vendored'], CAPS['all lines']]),
            ('unit', list(UNIT)), ('definition', list(DEFN)), ('comparison', list(COMP))]
    pretty = {**UNIT, **DEFN, **COMP}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex='col',
                             gridspec_kw={'height_ratios': [2.2, 2.6], 'hspace': 0.05, 'wspace': 0.12})
    for k, tier in enumerate(TIERS):
        r = res[res['tier'] == tier].sort_values('hr').reset_index(drop=True)
        ax, axg = axes[0, k], axes[1, k]
        exc = (r['ci_high'] < 1) | (r['ci_low'] > 1)
        col = np.where(~exc, '#999999', np.where(r['hr'] < 1, '#0072B2', '#D55E00'))
        x = np.arange(len(r))
        ax.vlines(x, r['ci_low'], r['ci_high'], color=col, lw=1)
        ax.scatter(x, r['hr'], c=col, s=14, zorder=3)
        ax.axhline(1, color='#333333', lw=0.8, ls='--')
        ax.set_yscale('log')
        ax.set_yticks([0.25, 0.5, 1, 2, 4])
        ax.set_ylim(max(0.1, r['ci_low'].min() * 0.9), min(10, r['ci_high'].max() * 1.1))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        n_l, n_s = int(((r['hr'] < 1) & exc).sum()), int(((r['hr'] > 1) & exc).sum())
        ax.set_title(f'{TIERS[tier]}: {n_l} longer / {len(r) - n_l - n_s} none / {n_s} sooner', fontsize=9.5, pad=6)
        if k == 0:
            ax.set_ylabel('HR vs human (log scale, 95% CI)')
        y, labels, ticks = 0, [], []
        for dim, vals in dims:
            for v in vals:
                m = (r[dim] == v).values
                axg.scatter(x[m], np.full(m.sum(), y), marker='s', s=9, color='#333333')
                labels.append(pretty.get(v, v))
                ticks.append(y)
                y -= 1
            y -= 0.5
        axg.set_yticks(ticks)
        axg.set_yticklabels(labels if k == 0 else [], fontsize=7.5)
        axg.set_xticks([])
        axg.set_xlabel('specifications, sorted by hazard ratio', fontsize=8)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
            axg.spines[s].set_visible(False)
    handles = [Line2D([], [], marker='o', ls='', color=c, label=l) for c, l in
               [('#0072B2', 'agent code retained longer (CI excludes 1)'),
                ('#D55E00', 'agent code modified sooner (CI excludes 1)'), ('#999999', 'no difference')]]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, -0.01, note, ha='center', fontsize=8, style='italic')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def ladder(d, name, note):
    d = d[(d['definition'] == 'syntactic') & (d['comparison'] == 'across repositories')]
    style = [('human_supervised', 'Human-supervised', '#0072B2', -0.15, 'o'),
             ('fully_autonomous', 'Fully-autonomous (Devin)', '#D55E00', 0.15, 's')]
    labels = [x[2].replace('Conference tracker', "Earlier study's tracker") for x in LADDER_CONF]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    xmax_all = d['ci_high'].max()
    for i, (data, unit, _) in enumerate(LADDER_CONF):
        rows = {t: d[(d['tier'] == t) & (d['data'] == data) & (d['unit'] == unit)].iloc[0] for t, *_ in style}
        xlab = max(r['ci_high'] for r in rows.values()) * 1.08
        for tier, nm, col, off, mk in style:
            r = rows[tier]
            y = i + off
            ax.plot([r['ci_low'], r['ci_high']], [y, y], color=col, lw=1.4)
            ax.plot(r['hr'], y, mk, color=col, ms=5, label=nm if i == 0 else None)
            ax.text(xlab, y, f"{r['hr']:.2f}", va='center', ha='left', fontsize=7, color=col)
    ax.axvline(1, color='#444444', ls='--', lw=0.8)
    for y in (2.5, 5.5, 8.5, 10.5):
        ax.axhline(y, color='#BBBBBB', lw=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ticks = [0.1, 0.25, 0.5, 1, 2, 4, 8]
    ax.set_xlim(max(0.08, d['ci_low'].min() * 0.85), min(12, xmax_all * 1.6))
    ax.set_xticks([t for t in ticks if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('Hazard ratio vs human code (log scale, 95% CI)\n< 1: retained longer    > 1: modified sooner', fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.text(0.5, -0.02, note, ha='center', fontsize=7.5, style='italic', wrap=True)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    sc = pd.read_csv('results_v6/spec_curve.csv')
    spec_curve(sc, 'spec_curve_labels', NOTE['paper'])
    ladder(sc, 'rq1_ladder_labels', 'Intervals: model-based SE (robust SE for PR weighting), not clustered (as in the paper).')
    p = 'results_v6/round1/c1_spec_curve_clustered.csv'
    if os.path.exists(p):
        c = pd.read_csv(p)
        c = c[c['comparison'] != 'across_shared'].copy()
        c['comparison'] = c['comparison'].map({'across': 'across repositories', 'within': 'within repository'})
        c['ci_low'], c['ci_high'] = c['lo_cl_repo'], c['hi_cl_repo']
        c = c.merge(sc[['tier', 'data', 'unit', 'definition', 'comparison', 'tracking', 'caps']],
                    on=['tier', 'data', 'unit', 'definition', 'comparison'])
        spec_curve(c, 'spec_curve_clustered', NOTE['clustered'])
        ladder(c, 'rq1_ladder_clustered', NOTE['clustered'])
    print('figures in', OUT)


if __name__ == '__main__':
    main()
