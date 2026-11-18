"""
per_agent_boot_r2.py 

Per-agent within-repository line HRs against human code (Table rq3_per_agent; round-1 C9f),
re-run with the SAME engine, seed and data order as the headline within-repository bootstrap
(src/analyze_rq1_master.py step 4 and src/c4_bootstrap_r1.py): lifelines CoxPHFitter (Efron),
numpy default_rng(42), 500 repository resamples, percentile 95% CIs (src/boot_r1.py with
BOOT_ENGINE unset). Round-1 C9f used the R engine (R RNG), which is why its Devin row
(1.059 [0.476, 2.138]) differed from the headline (1.059 [0.409, 2.076]).
Data and comparison exactly as src/c9f_per_agent_within_r1.py: agent lines (identity tracking,
all lines, Definition 1) against the human lines of all repositories.

Usage: DATA_TAG=v6confnv RESULTS_DIR=results_v6 python src/per_agent_boot_r2.py
Output: results_v6/round2/e3_per_agent_within.csv / .txt, draws in results_v6/round2/boot_draws/
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
os.environ.pop('BOOT_ENGINE', None)
from config.observation_window import apply_observation_window
from config.paths import P
from analyze_rq1_master import add_two_tier_column
import boot_r1

boot_r1.OUT = 'results_v6/round2/boot_draws'
AGENTS = ['Devin', 'GitHub Copilot', 'OpenAI Codex', 'Cursor', 'Claude Code']   # Devin first: reproduction check


def main():
    line = add_two_tier_column(apply_observation_window(
        pd.read_csv(P('data/survival_events_line_v5_analysis.csv'), low_memory=False), label='line'))
    h = line[line['two_tier'] == 'human'][['duration_days', 'event', 'slug']]
    fa = line[line['two_tier'] == 'fully_autonomous']
    dv = line[line['is_agent'] & (line['agent_name'] == 'Devin')]
    print('Devin rows identical to the fully-autonomous tier rows (same order):', fa.index.equals(dv.index), flush=True)
    rows = []
    out = ['E3. Per-agent within-repository HR vs human (line, Definition 1, identity tracking, all lines)',
           'Engine: lifelines CoxPHFitter (Efron), numpy default_rng(42), 500 repository resamples, percentile CIs',
           '(same as the headline within-repository bootstrap, src/analyze_rq1_master.py step 4).', '']
    for ag in AGENTS:
        a = line[line['is_agent'] & (line['agent_name'] == ag)][['duration_days', 'event', 'slug', 'pr_number']]
        df = pd.concat([a.drop(columns='pr_number').assign(is_tier=1), h.assign(is_tier=0)], ignore_index=True)
        r = boot_r1.boot_within(df, f'e3_within_{ag.replace(" ", "_")}')
        r.update(agent=ag, n_prs=int(a.groupby(['slug', 'pr_number']).ngroups))
        r.pop('covariates')
        print(json.dumps(r), flush=True)
        rows.append(r)
        sig = 'excludes 1' if (r['hi95'] < 1 or r['lo95'] > 1) else 'includes 1'
        out.append(f'{ag:15s} lines {r["n_tier"]:>7,} PRs {r["n_prs"]:>4} repos with agent+human {r["n_contributing_strata"]:>3}  '
                   f'HR {r["hr"]:.3f} boot95 [{r["lo95"]:.3f}, {r["hi95"]:.3f}] ({sig}); valid draws {r["boot_valid"]}, p_boot {r["p_boot"]:.3f}')
        pd.DataFrame(rows).to_csv('results_v6/round2/e3_per_agent_within.csv', index=False)
    open('results_v6/round2/e3_per_agent_within.txt', 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
