"""
plot_rq1_ladder.py

RQ1 figure: the "measurement ladder". Hazard ratio of each agent tier against
the human baseline (syntactic definition, comparison across repositories),
as line tracking and the unit of analysis change. Every value is read from
results/spec_curve.csv (src/specification_curve.py); nothing is refitted.

Usage:
    python src/plot_rq1_ladder.py
Output:
    paper-overfeal-files/visuals/rq1_ladder.pdf (+ .png)
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config.paths import P

OUT = os.path.join(ROOT, P('paper-overfeal-files/visuals/rq1_ladder'))

# (data, unit, label), top to bottom
LADDER = [
    ('v2 tracking, capped', 'line', 'Line number, sampled, per line'),
    ('v2 tracking, capped', 'PR-weighted', 'Line number, sampled, per PR'),
    ('v2 tracking, capped', 'hunk', 'Line number, sampled, per hunk'),
    ('v3 tracking, capped', 'line', 'Identity, sampled, per line'),
    ('v3 tracking, capped', 'PR-weighted', 'Identity, sampled, per PR'),
    ('v3 tracking, all lines', 'line', 'Identity, all lines, per line'),
    ('v3 tracking, all lines', 'PR-weighted', 'Identity, all lines, per PR'),
    ('v3 tracking, all lines', 'hunk', 'Identity, all lines, per hunk'),
]
# conference born lines (results with the two 'with vendored' datasets): start from the
# conference's exact units and tracker, then change one measurement choice at a time
LADDER_CONF = [
    ('conference tracking, all lines, with vendored', 'line', 'Conference tracker, all lines incl. vendored, per line'),
    ('conference tracking, all lines, with vendored', 'PR-weighted', 'Conference tracker, all lines incl. vendored, per PR'),
    ('conference tracking, all lines, with vendored', 'hunk', 'Conference tracker, all lines incl. vendored, per hunk'),
    ('v3 tracking, all lines, with vendored', 'line', 'Identity, all lines incl. vendored, per line'),
    ('v3 tracking, all lines, with vendored', 'PR-weighted', 'Identity, all lines incl. vendored, per PR'),
    ('v3 tracking, all lines, with vendored', 'hunk', 'Identity, all lines incl. vendored, per hunk'),
    ('v3 tracking, all lines', 'line', 'Identity, vendored/generated removed, per line'),
    ('v3 tracking, all lines', 'PR-weighted', 'Identity, vendored/generated removed, per PR'),
    ('v3 tracking, all lines', 'hunk', 'Identity, vendored/generated removed, per hunk'),
    ('v3 tracking, capped', 'line', 'Identity, sampled, per line'),
    ('v3 tracking, capped', 'PR-weighted', 'Identity, sampled, per PR'),
    ('v2 tracking, capped', 'line', 'Line number, sampled, per line'),
    ('v2 tracking, capped', 'PR-weighted', 'Line number, sampled, per PR'),
    ('v2 tracking, capped', 'hunk', 'Line number, sampled, per hunk'),
]
TIERS = [('human_supervised', 'Human-supervised', '#0072B2', -0.14, 'o'),
         ('fully_autonomous', 'Fully-autonomous (Devin)', '#D55E00', 0.14, 's')]


def main():
    d = pd.read_csv(os.path.join(ROOT, P('results/spec_curve.csv')))
    d = d[(d['definition'] == 'syntactic') & (d['comparison'] == 'across repositories')]
    conf = d['data'].str.startswith('conference').any()
    ladder = LADDER_CONF if conf else LADDER
    seps = (2.5, 5.5, 8.5, 10.5) if conf else (2.5, 4.5)
    fig, ax = plt.subplots(figsize=(6.4, 3.6 if not conf else 5.4))
    for tier, name, col, off, mk in TIERS:
        for i, (data, unit, _) in enumerate(ladder):
            r = d[(d['tier'] == tier) & (d['data'] == data) & (d['unit'] == unit)]
            assert len(r) == 1, (tier, data, unit)
            r = r.iloc[0]
            y = i + off
            ax.plot([r['ci_low'], r['ci_high']], [y, y], color=col, lw=1.4)
            ax.plot(r['hr'], y, mk, color=col, ms=5, label=name if i == 0 else None)
            ax.text(max(r['ci_high'], r['hr']) * 1.05, y, f"{r['hr']:.2f}", va='center', fontsize=7, color=col)
    ax.axvline(1, color='#444444', ls='--', lw=0.8)
    for y in seps:
        ax.axhline(y, color='#BBBBBB', lw=0.6)
    ax.set_yticks(range(len(ladder)))
    ax.set_yticklabels([x[2] for x in ladder], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xticks([0.25, 0.5, 0.75, 1, 1.25, 1.5, 2] if conf else [0.5, 0.75, 1, 1.25, 1.5, 2])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(min(0.45, d['ci_low'].min() * 0.9), max(2.1, d['ci_high'].max() * 1.1))
    ax.set_xlabel('Hazard ratio vs human code (log scale, 95% CI)\n< 1: retained longer    > 1: modified sooner', fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}.{ext}', dpi=200, bbox_inches='tight')
    print('wrote', OUT + '.pdf')


if __name__ == '__main__':
    main()
