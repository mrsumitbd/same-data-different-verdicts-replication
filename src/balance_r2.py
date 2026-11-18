"""
balance_r2.py 

(a) Per-agent intent: modification commits whose majority birth agent (agent_name of the lines a
    commit modifies; same majority rule as the tier map) is GitHub Copilot, OpenAI Codex, Cursor,
    Claude Code (or Devin), against commits whose majority birth group is human. Shares and
    chi-square p under keyword rules, each LLM and the majority vote (+ the other combinations),
    with labels built as in analyze_intent_aggregation.run_full (618 human labels; src/rq5_r2.py).
(b) Human-supervised concentration (as results_v6/rq1_largest_prs.txt does for Devin): share of the
    tier's lines in its five largest PRs and in its largest repository, and the modification rate of
    those lines against the rest.
(c) Six-month window row (>= 180 days of follow-up) for human-supervised code: from
    src/spec_null_r2.py variants c6 / c6h (R-equivalent Efron refits) -> see e1_window_curve.txt.

Usage: DATA_TAG=v6confnv RESULTS_DIR=results_v6 python src/balance_r2.py
Output: results_v6/round2/e10_balance.txt (+ e10a_per_agent_intent.csv)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
from config.observation_window import apply_observation_window
from analyze_rq4 import is_generated, OUTLIER_PR_KEYS, add_two_tier_column
import rq5_r2 as R

OUT = 'results_v6/round2'
AGENTS = ['GitHub Copilot', 'OpenAI Codex', 'Cursor', 'Claude Code', 'Devin']


def part_a(out):
    lines, tier_map = R.load_lines()
    # majority birth agent per commit, same tie rule as get_birth_tier_map (value_counts order)
    agent_map = lines.groupby(['slug', 'sha'])['agent_name'].agg(lambda s: s.value_counts().index[0]).rename('birth_agent').reset_index()
    c = tier_map.merge(agent_map, on=['slug', 'sha'], how='inner')
    corpus, labels = R.intent_corpus()
    d = c.merge(corpus, on=['slug', 'sha'], how='left')
    out += ['(a) Intent of modifications by the agent that wrote the modified code (majority birth agent of the commit) vs human code',
            '    (majority birth group human). Shares corrective / perfective / adaptive / preventive (%), chi-square p (Cramer V).',
            f'    commits: ' + ', '.join(f'{k} {v}' for k, v in d['birth_agent'].value_counts().items()) +
            f'; agreement of the majority agent with the majority tier: Human <-> human for '
            f'{int(((d["birth_agent"] == "Human") == (d["birth_tier"] == "human")).sum()):,} of {len(d):,}', '']
    rows = []
    d['grp'] = np.where(d['birth_tier'] == 'human', 'human', d['birth_agent'])
    for ag in AGENTS:
        x = d[(d['grp'] == ag) | (d['grp'] == 'human')].copy()
        x['who'] = np.where(x['grp'] == ag, 'agent', 'human')
        out.append(f'  {ag} (n = {int((x["who"] == "agent").sum())}) vs human (n = {int((x["who"] == "human").sum())}):')
        for col, name in R.APPROACHES:
            sh, n, p, v = R.intent_test(x, col, 'who', labels)
            out.append(f'    {name:30s} {ag[:14]:14s} ' + '/'.join(f'{sh["agent"][l]:.1f}' for l in labels) +
                       f'  human ' + '/'.join(f'{sh["human"][l]:.1f}' for l in labels) + f'  p={p:.3g} (V={v:.3f}; n={n["agent"]}/{n["human"]})')
            r = dict(agent=ag, approach=col, n_agent=n['agent'], n_human=n['human'], p=p, V=v)
            for l in labels:
                r[f'agent_{l}'] = sh['agent'][l]
                r[f'human_{l}'] = sh['human'][l]
            rows.append(r)
        out.append('')
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'e10a_per_agent_intent.csv'), index=False)


def part_b(out):
    l = pd.read_csv('data/survival_events_line_v6confnv_analysis.csv', low_memory=False,
                    usecols=['slug', 'pr_number', 'is_agent', 'agent_tier', 'agent_name', 'event', 'duration_days', 'birth_date', 'death_date'])
    l = add_two_tier_column(apply_observation_window(l, label='line'))
    out += ['(b) Concentration of each tier (all lines of the analysis data, window applied; as rq1_largest_prs.txt)']
    for tier in ['human_supervised', 'fully_autonomous']:
        x = l[l['two_tier'] == tier].copy()
        x['pr'] = x['slug'] + '#' + x['pr_number'].astype(str)
        n = len(x)
        by_pr = x.groupby('pr').agg(n=('event', 'size'), mr=('event', 'mean')).sort_values('n', ascending=False)
        top5 = by_pr.index[:5]
        in5 = x['pr'].isin(top5)
        by_repo = x.groupby('slug').agg(n=('event', 'size'), mr=('event', 'mean')).sort_values('n', ascending=False)
        top_repo = by_repo.index[0]
        inr = x['slug'] == top_repo
        out.append(f'  {tier}: {n:,} lines in {by_pr.shape[0]} PRs and {by_repo.shape[0]} repositories')
        for pr, r in by_pr.head(5).iterrows():
            ag = x.loc[x['pr'] == pr, 'agent_name'].iloc[0]
            out.append(f'    {pr}: {int(r.n):,} lines ({100 * r.n / n:.1f}%), {100 * r.mr:.1f}% modified ({ag})')
        out.append(f'    five largest PRs: {100 * in5.mean():.1f}% of lines, {100 * x.loc[in5, "event"].mean():.1f}% modified; '
                   f'other PRs: {100 * x.loc[~in5, "event"].mean():.1f}% modified')
        out.append(f'    largest repository {top_repo}: {int(inr.sum()):,} lines ({100 * inr.mean():.1f}%), {100 * x.loc[inr, "event"].mean():.1f}% modified; '
                   f'other repositories: {100 * x.loc[~inr, "event"].mean():.1f}% modified; '
                   f'next largest: ' + ', '.join(f'{s} {100 * r.n / n:.1f}%' for s, r in by_repo.iloc[1:4].iterrows()))
        out.append(f'    largest PR share {100 * by_pr.iloc[0].n / n:.1f}%; PRs needed for half the lines: '
                   f'{int((by_pr["n"].cumsum() < n / 2).sum()) + 1}')
    out.append('')


def main():
    out = ['E10. Balance checks', '']
    part_b(out)
    part_a(out)
    out += ['(c) Six-month window: see e1_window_curve.txt (variants c6, c6h; src/spec_null_r2.py).']
    txt = '\n'.join(out) + '\n'
    open(os.path.join(OUT, 'e10_balance.txt'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
