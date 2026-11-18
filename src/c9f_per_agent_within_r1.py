"""
c9f_per_agent_within_r1.py 

Per-agent within-repository line HRs against human code (Definition 1,
identity tracking, all lines; repository-stratified Cox, 500-resample
repository cluster bootstrap, seed 42; src/boot_r1.py), for GitHub Copilot,
OpenAI Codex, Cursor, Claude Code and Devin. Each agent is compared with the
human lines of all repositories (repositories without that agent contribute
nothing to the stratified fit, as in the tier models).

Usage: DATA_TAG=v6confnv RESULTS_DIR=results_v6 python src/c9f_per_agent_within_r1.py
Output: results_v6/round1/c9f_per_agent_within.csv / .txt
"""

import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
from config.observation_window import apply_observation_window
from config.paths import P
from analyze_rq1_master import add_two_tier_column
from boot_r1 import boot_within

AGENTS = ['GitHub Copilot', 'OpenAI Codex', 'Cursor', 'Claude Code', 'Devin']


def main():
    line = add_two_tier_column(apply_observation_window(
        pd.read_csv(P('data/survival_events_line_v5_analysis.csv'), low_memory=False), label='line'))
    h = line[line['two_tier'] == 'human'][['duration_days', 'event', 'slug']]
    rows, out = [], ['C9f. Per-agent within-repository HR vs human (line, Definition 1, 500-resample repository bootstrap)', '']
    for ag in AGENTS:
        a = line[line['is_agent'] & (line['agent_name'] == ag)][['duration_days', 'event', 'slug', 'pr_number']]
        df = pd.concat([a.drop(columns='pr_number').assign(is_tier=1), h.assign(is_tier=0)], ignore_index=True)
        r = boot_within(df, f'c9f_within_{ag.replace(" ", "_")}')
        r.update(agent=ag, n_prs=int(a.groupby(['slug', 'pr_number']).ngroups))
        r.pop('covariates')
        print(json.dumps(r), flush=True)
        rows.append(r)
        sig = 'excludes 1' if (r['hi95'] < 1 or r['lo95'] > 1) else 'includes 1'
        out.append(f'{ag:15s} lines {r["n_tier"]:>7,} PRs {r["n_prs"]:>4} repos with agent+human {r["n_contributing_strata"]:>3}  '
                   f'HR {r["hr"]:.3f} boot95 [{r["lo95"]:.3f}, {r["hi95"]:.3f}] ({sig})')
    pd.DataFrame(rows).to_csv('results_v6/round1/c9f_per_agent_within.csv', index=False)
    open('results_v6/round1/c9f_per_agent_within.txt', 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
