"""
plot_figs_r2.py 

(a) RQ1 ladder, repository-clustered intervals (as results_v6/round1/visuals/rq1_ladder_clustered.pdf),
    with left-side block labels that follow the steps of RQ1: the tracker and the lines
    (earlier tracker / identity on all lines incl. vendored; vendored removed; sampled; line number
    on the sampled units), and the unit per row (per line / per PR / per hunk).
(b) Specification curve, repository-clustered intervals (as spec_curve_clustered.pdf), with the
    y-axis extended to 0.08 (ticks from 0.1) so no interval is clipped; an interval that still
    leaves the axis would be drawn to the edge with an arrow.
Inputs: results_v6/round1/c1_spec_curve_clustered.csv (R refits), results_v6/spec_curve.csv (labels).
Output: results_v6/round2/visuals/rq1_ladder_clustered.{pdf,png}, spec_curve_clustered.{pdf,png}
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
from plot_round1_figs_r1 import TIERS, TRACK, CAPS, UNIT, DEFN, COMP, NOTE

OUT = 'results_v6/round2/visuals'
YLO, YHI = 0.08, 4.5

# ladder blocks: (data, block label, step note, units)
BLOCKS = [
    ('conference tracking, all lines, with vendored', "Earlier study's tracker\nall lines incl. vendored", 'start', ['line', 'PR-weighted', 'hunk']),
    ('v3 tracking, all lines, with vendored', 'Identity tracking\nall lines incl. vendored', 'step 1: tracker', ['line', 'PR-weighted', 'hunk']),
    ('v3 tracking, all lines', 'Identity tracking\nvendored/generated removed', 'step 2: vendored removed', ['line', 'PR-weighted', 'hunk']),
    ('v3 tracking, capped', 'Identity tracking\nsampled', 'step 3: sampling', ['line', 'PR-weighted']),
    ('v2 tracking, capped', 'Line-number tracking\nsampled (same units)', 'step 1: tracker', ['line', 'PR-weighted', 'hunk']),
]
ULAB = {'line': 'per line', 'PR-weighted': 'per PR', 'hunk': 'per hunk'}


def load_clustered():
    sc = pd.read_csv('results_v6/spec_curve.csv')
    c = pd.read_csv('results_v6/round1/c1_spec_curve_clustered.csv')
    c = c[c['comparison'] != 'across_shared'].copy()
    c['comparison'] = c['comparison'].map({'across': 'across repositories', 'within': 'within repository'})
    c['ci_low'], c['ci_high'] = c['lo_cl_repo'], c['hi_cl_repo']
    return c.merge(sc[['tier', 'data', 'unit', 'definition', 'comparison', 'tracking', 'caps']],
                   on=['tier', 'data', 'unit', 'definition', 'comparison'])


def ladder(d):
    d = d[(d['definition'] == 'syntactic') & (d['comparison'] == 'across repositories')]
    style = [('human_supervised', 'Human-supervised', '#0072B2', -0.15, 'o'),
             ('fully_autonomous', 'Fully-autonomous (Devin)', '#D55E00', 0.15, 's')]
    rows = [(b, u) for b in BLOCKS for u in b[3]]
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    y, ys, ylabels, spans = 0, [], [], []
    for bi, b in enumerate(BLOCKS):
        y0 = y
        for u in b[3]:
            rr = {t: d[(d['tier'] == t) & (d['data'] == b[0]) & (d['unit'] == u)].iloc[0] for t, *_ in style}
            xlab = max(r['ci_high'] for r in rr.values()) * 1.08
            for tier, nm, col, off, mk in style:
                r = rr[tier]
                ax.plot([r['ci_low'], r['ci_high']], [y + off] * 2, color=col, lw=1.4)
                ax.plot(r['hr'], y + off, mk, color=col, ms=5, label=nm if y == 0 else None)
                ax.text(xlab, y + off, f"{r['hr']:.2f}", va='center', ha='left', fontsize=7, color=col)
            ys.append(y)
            ylabels.append(ULAB[u])
            y += 1
        spans.append((y0, y - 1, b[1], b[2]))
        if bi < len(BLOCKS) - 1:
            ax.axhline(y - 0.5, color='#BBBBBB', lw=0.6)
    ax.axvline(1, color='#444444', ls='--', lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_ylim(y - 0.5, -0.5)
    ax.set_xscale('log')
    ax.set_xlim(0.1, 12)
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 8])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('Hazard ratio vs human code (log scale, 95% CI)\n< 1: retained longer    > 1: modified sooner', fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    # block labels on the left, outside the tick labels, with a bracket
    trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for a, b_, lab, step in spans:
        ax.plot([-0.135, -0.135], [a - 0.3, b_ + 0.3], color='#555555', lw=0.8, transform=trans, clip_on=False)
        ax.text(-0.15, (a + b_) / 2, lab, transform=trans, ha='right', va='center', fontsize=7.8, clip_on=False)
        ax.text(-0.15, (a + b_) / 2 + (0.95 if b_ > a + 1 else 0.75), f'({step})', transform=trans, ha='right', va='center',
                fontsize=6.8, color='#666666', style='italic', clip_on=False)
    ax.legend(fontsize=8, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.text(0.55, -0.02, NOTE['clustered'] + ' Sampled: at most 20 files per PR and 100 lines per file.',
             ha='center', fontsize=7.5, style='italic', wrap=True)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/rq1_ladder_clustered.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def spec_curve(res):
    res = res.copy()
    res['tracking'] = res['tracking'].map(TRACK)
    res['caps'] = res['caps'].map(CAPS)
    dims = [('tracking', ["earlier study's tracker", 'line number', 'identity']),
            ('caps', [CAPS['capped'], CAPS['all lines, with vendored'], CAPS['all lines']]),
            ('unit', list(UNIT)), ('definition', list(DEFN)), ('comparison', list(COMP))]
    pretty = {**UNIT, **DEFN, **COMP}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex='col',
                             gridspec_kw={'height_ratios': [2.4, 2.6], 'hspace': 0.05, 'wspace': 0.12})
    n_clip = 0
    for k, tier in enumerate(TIERS):
        r = res[res['tier'] == tier].sort_values('hr').reset_index(drop=True)
        ax, axg = axes[0, k], axes[1, k]
        exc = (r['ci_high'] < 1) | (r['ci_low'] > 1)
        col = np.where(~exc, '#999999', np.where(r['hr'] < 1, '#0072B2', '#D55E00'))
        x = np.arange(len(r))
        lo, hi = r['ci_low'].clip(lower=YLO), r['ci_high'].clip(upper=YHI)
        ax.vlines(x, lo, hi, color=col, lw=1)
        for i in np.flatnonzero(r['ci_low'] < YLO):
            ax.annotate('', xy=(x[i], YLO), xytext=(x[i], YLO * 1.35), arrowprops=dict(arrowstyle='->', color=col[i], lw=1))
            n_clip += 1
        for i in np.flatnonzero(r['ci_high'] > YHI):
            ax.annotate('', xy=(x[i], YHI), xytext=(x[i], YHI / 1.35), arrowprops=dict(arrowstyle='->', color=col[i], lw=1))
            n_clip += 1
        ax.scatter(x, r['hr'], c=col, s=14, zorder=3)
        ax.axhline(1, color='#333333', lw=0.8, ls='--')
        ax.set_yscale('log')
        ax.set_ylim(YLO, YHI)
        ax.set_yticks([0.1, 0.25, 0.5, 1, 2, 4])
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
    fig.text(0.5, -0.01, NOTE['clustered'], ha='center', fontsize=8, style='italic')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/spec_curve_clustered.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return n_clip


def main():
    os.makedirs(OUT, exist_ok=True)
    c = load_clustered()
    assert len(c) == 88, len(c)
    ladder(c)
    n = spec_curve(c)
    print(f'figures in {OUT}; intervals beyond [{YLO}, {YHI}] drawn with arrows: {n}; '
          f'min lower bound {c["ci_low"].min():.3f}, max upper bound {c["ci_high"].max():.3f}')


if __name__ == '__main__':
    main()
