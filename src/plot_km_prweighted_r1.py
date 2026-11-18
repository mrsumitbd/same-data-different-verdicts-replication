"""
plot_km_prweighted_r1.py 

Kaplan-Meier retention curves per tier, line level, identity tracking, all
lines with vendored and generated code removed (Definition 1), with every line
weighted by 1 / lines in its PR so that each PR counts once (the weighting of
the earlier study's primary design). The unweighted curves are drawn faintly
for comparison. Input: the spec-curve export (src/export_spec_data_r1.py).

Output: results_v6/round1/visuals/km_prweighted_line.{pdf,png}, and
        results_v6/round1/c9c_km_prweighted.csv (survival at 30/90/150 days)
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
GROUPS = [('human', 'Human', '#555555'), ('human_supervised', 'Human-supervised agents', '#0072B2'),
          ('fully_autonomous', 'Devin (fully autonomous)', '#D55E00')]


def main():
    x = pd.read_csv('results_v6/round1/work/spec_v3all_line.csv', usecols=['d', 'e1', 'grp', 'w', 'pr'])
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    rows = []
    for g, name, col in GROUPS:
        s = x[x['grp'] == g]
        for weighted in (True, False):
            km = KaplanMeierFitter().fit(s['d'], s['e1'], weights=s['w'] if weighted else None,
                                         label=f'{name} ({s["pr"].nunique():,} PRs)' if weighted else None)
            if weighted:
                km.plot_survival_function(ax=ax, color=col, ci_show=False, lw=1.8)
            else:
                ax.step(km.survival_function_.index, km.survival_function_.iloc[:, 0], where='post',
                        color=col, lw=0.8, ls=':', alpha=0.8)
            for t in (30, 90, 150):
                rows.append(dict(group=g, weighting='PR-weighted' if weighted else 'unweighted', day=t,
                                 retained=float(km.survival_function_at_times(t).iloc[0]),
                                 n_lines=len(s), n_prs=s['pr'].nunique()))
    ax.set_xlim(0, 300)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel('Days since merge')
    ax.set_ylabel('Share of lines not yet modified')
    ax.set_title('Line retention, each PR weighted equally (dotted: every line counts once)', fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs('results_v6/round1/visuals', exist_ok=True)
    fig.savefig('results_v6/round1/visuals/km_prweighted_line.pdf')
    fig.savefig('results_v6/round1/visuals/km_prweighted_line.png', dpi=200)
    pd.DataFrame(rows).to_csv('results_v6/round1/c9c_km_prweighted.csv', index=False)
    print(pd.DataFrame(rows).pivot_table(index=['group', 'weighting'], columns='day', values='retained').round(3))


if __name__ == '__main__':
    main()
