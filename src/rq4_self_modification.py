"""
rq4_self_modification.py

When an agent modifies agent-attributed code, is it the agent that wrote
the code, and how soon after the merge (RQ4)? Covers the two agent
modifiers that dominate SECTION B of rq4_results.txt: Copilot on
human-supervised code and devin-ai-integration[bot] on fully-autonomous
code.

One row per modification commit, with the birth tier from
the majority vote of analyze_rq4.get_birth_tier_map (observation window
applied, outlier PR and generated files dropped), the same basis as
SECTION B of rq4_results.txt. A commit modifies
"lines written by the same agent" when any line it modifies was born in a
PR of that agent; its lag is the shortest time since merge among the
lines it modifies.

Usage:
    python src/rq4_self_modification.py [--line-data data/survival_events_line_v5_analysis.csv]
Output:
    results/rq4_self_modification.txt
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.observation_window import apply_observation_window
from config.paths import P
from analyze_rq4 import get_birth_tier_map, is_generated, OUTLIER_PR_KEYS

CASES = [('human_supervised', 'Copilot', 'GitHub Copilot'),
         ('fully_autonomous', 'devin-ai-integration[bot]', 'Devin')]


def main(line_data: str, commits: str, modifier_authors: str, output: str) -> None:
    lines = pd.read_csv(line_data, usecols=[
        'slug', 'pr_number', 'is_agent', 'agent_tier', 'agent_name', 'unit_id', 'death_sha',
        'event', 'duration_days', 'birth_date', 'death_date'], low_memory=False)
    lines = apply_observation_window(lines, label='line').dropna(subset=['death_sha'])
    tier_map = get_birth_tier_map(lines)
    lines = lines[~pd.Series(list(zip(lines['slug'], lines['pr_number'])), index=lines.index).isin(OUTLIER_PR_KEYS)]
    lines = lines[~lines['unit_id'].apply(is_generated)]
    per_commit = (lines.rename(columns={'death_sha': 'sha'}).groupby(['slug', 'sha'])
                  .agg(agents=('agent_name', lambda s: set(s.dropna())), lag=('duration_days', 'min'))
                  .reset_index())
    mods = pd.read_csv(commits)[['slug', 'sha']].drop_duplicates()
    authors = pd.read_csv(modifier_authors)[['slug', 'sha', 'author_login']].drop_duplicates(['slug', 'sha'])
    df = mods.merge(authors, on=['slug', 'sha']).merge(tier_map, on=['slug', 'sha']).merge(per_commit, on=['slug', 'sha'])

    out = ['one row per modification commit, birth tier by majority vote (as in rq4_results.txt Section B); '
           'lag = shortest time since merge among the modified lines']
    for tier, login, agent in CASES:
        g = df[(df['birth_tier'] == tier) & (df['author_login'] == login)]
        same = g['agents'].apply(lambda a: agent in a)
        out += ['', f'{tier} code modified by {login}: {len(g)} commits',
                f'  commits modifying lines written by the same agent: {int(same.sum())}',
                f'  median lag (days): {g["lag"].median():.2f}; within 1 day: {int((g["lag"] <= 1).sum())}; '
                f'within 7 days: {int((g["lag"] <= 7).sum())}']
    text = '\n'.join(out) + '\n'
    print(text)
    Path(output).write_text(text)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--line-data', default=P('data/survival_events_line_v5_analysis.csv'))
    ap.add_argument('--commits', default=P('data/death_commit_messages_v5.csv'))
    ap.add_argument('--modifier-authors', default=P('results/modifier_authors.csv'))
    ap.add_argument('--output', default=P('results/rq4_self_modification.txt'))
    a = ap.parse_args()
    main(a.line_data, a.commits, a.modifier_authors, a.output)
