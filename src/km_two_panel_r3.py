"""
km_two_panel_r3.py 

Two-panel Kaplan-Meier retention of all lines by tier (identity tracking, lines
without vendored/generated code, 215-repository cohort):
  left  Definition 1 (syntactic): event = event
  right Definition 2 (AST-structural): event = (ast_death == 1); every other line,
        including modified lines without an AST status, is censored at its
        duration (i.e. at that modification), as in src/analyze_rq1_master.py step 3.
Same inputs and preprocessing as src/plot_rq1.py / src/analyze_rq1_master.py
(DATA_TAG=v6confnv, apply_observation_window). Palette as in plot_rq1.py.

Also reports max follow-up per group, merge (birth) date range per tier, the
data end date, and KM survival at days 30 and 40 under both definitions.

Run from the repository root:
  DATA_TAG=v6confnv RESULTS_DIR=results_v6 python src/km_two_panel_r3.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.observation_window import apply_observation_window
from config.constants import OBSERVATION_CUTOFF, PR_INCLUSION_CUTOFF
from config.paths import P
from plot_rq1 import add_two_tier_column, TIER_COLORS

OUT = 'results_v6/round3'
LABELS = {'human_supervised': 'Human-supervised', 'fully_autonomous': 'Devin', 'human': 'Human baseline'}
ORDER = ['human_supervised', 'fully_autonomous', 'human']

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'axes.spines.top': False, 'axes.spines.right': False,
})


def main():
    path = P('data/survival_events_line_v5_analysis.csv')
    print('input:', path)
    line = apply_observation_window(pd.read_csv(path), label='line')
    line = add_two_tier_column(line)
    assert len(line) == 644177, len(line)
    line['event_def2'] = (line['ast_death'] == 1.0).astype(int)
    line['birth_dt'] = pd.to_datetime(line['birth_date'])

    rep = ['Task 1: Kaplan-Meier, two panels (Def 1 syntactic, Def 2 AST-structural)',
           f'input: {path} ({len(line):,} lines after observation window)',
           f'data end (administrative censoring): {OBSERVATION_CUTOFF}; merge inclusion cutoff: < {PR_INCLUSION_CUTOFF}', '']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    surv_rows = []
    for ax, (ev, title) in zip(axes, [('event', 'Definition 1 (syntactic)'),
                                      ('event_def2', 'Definition 2 (AST-structural)')]):
        for tier in ORDER:
            sub = line[line['two_tier'] == tier]
            kmf = KaplanMeierFitter()
            kmf.fit(sub['duration_days'], sub[ev], label=LABELS[tier])
            kmf.plot_survival_function(ax=ax, color=TIER_COLORS[tier], ci_show=True, ci_alpha=0.15, linewidth=1.8)
            for d in (30, 40):
                s = float(kmf.survival_function_at_times(d).iloc[0])
                ci = kmf.confidence_interval_survival_function_
                # CI at day d: last row with index <= d
                row = ci[ci.index <= d].iloc[-1]
                surv_rows.append(dict(definition=ev, tier=LABELS[tier], day=d, survival=s,
                                      ci_lo=float(row.iloc[0]), ci_hi=float(row.iloc[1])))
        ax.set_xlim(0, 365)
        ax.set_ylim(0.6, 1.0)
        ax.set_xlabel('Days since merge')
        ax.text(0.98, 0.97, title, transform=ax.transAxes, ha='right', va='top', fontsize=12)
        ax.grid(True, alpha=0.25, linestyle=':')
        ax.legend(loc='lower left', framealpha=0.9)
    axes[0].set_ylabel('Retention probability')
    axes[1].get_legend().remove()
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/rq1_km_def1_def2_line.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)

    # follow-up and merge dates
    rep.append('Follow-up and merge (birth) dates per group:')
    rep.append(f'{"group":18s} {"lines":>8s} {"max follow-up (d)":>18s} {"earliest merge":>20s} {"latest merge":>20s} '
               f'{"min follow-up (d)":>18s} {"max event day":>14s}')
    for tier in ORDER:
        sub = line[line['two_tier'] == tier]
        rep.append(f'{LABELS[tier]:18s} {len(sub):8,d} {sub["duration_days"].max():18.1f} '
                   f'{str(sub["birth_dt"].min()):>20s} {str(sub["birth_dt"].max()):>20s} '
                   f'{(OBSERVATION_CUTOFF - sub["birth_dt"].max()).total_seconds() / 86400:18.1f} '
                   f'{sub.loc[sub["event"] == 1, "duration_days"].max():14.1f}')
    rep.append('Max follow-up = data end - earliest merge of the group (censored lines run to the data end).')
    rep.append('')
    rep.append('Lines still at risk (duration >= d) per group:')
    days = [150, 200, 245, 250, 265, 280, 300, 340]
    rep.append(f'{"group":18s} ' + ' '.join(f'{"d" + str(d):>8s}' for d in days))
    for tier in ORDER:
        sub = line[line['two_tier'] == tier]
        rep.append(f'{LABELS[tier]:18s} ' + ' '.join(f'{int((sub["duration_days"] >= d).sum()):8,d}' for d in days))
    rep.append('')
    # PR-level date distribution
    prs = line.drop_duplicates(['slug', 'pr_number'])
    rep.append('Merge-month distribution of PRs per group (PR counts):')
    tab = pd.crosstab(prs['birth_dt'].dt.to_period('M'), prs['two_tier'].map(LABELS))
    rep.append(tab[[LABELS[t] for t in ORDER]].to_string())
    rep.append('')
    sv = pd.DataFrame(surv_rows)
    sv['definition'] = sv['definition'].map({'event': 'Def 1', 'event_def2': 'Def 2'})
    rep.append('KM survival (retention probability) at days 30 and 40, 95% CI:')
    for _, r in sv.iterrows():
        rep.append(f'  {r.definition}  {r.tier:18s} day {r.day:2d}: {r.survival:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]')
    sv.to_csv(f'{OUT}/km_survival_day30_40.csv', index=False)
    txt = '\n'.join(rep) + '\n'
    open(f'{OUT}/km_two_panel.txt', 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
