"""
build_conf_births_v6.py

Births for the identity tracker (measure_retention_v5.py --phase track) taken
from the mining of the earlier study instead of the journal's own PR selection,
so that the journal measures exactly the conference's born lines.

Input (earlier_study/tables/, written by scripts/build_tables.py):
  lines.parquet  one row per added source line of a merged PR (net diff
                 base..tip, line numbers in the tip commit = birth_sha)
  prs.parquet    per PR: match_method, pr_id
Kept: lines the conference tracked (event not null; drops its 364
death_commit_not_found lines), i.e. 854,139 lines, 2,802 PRs, 215 repositories.
Vendored/generated lines are kept and flagged (column vendored, the
conference's regex), as are blank lines (column blank).

Per line, as in the v5 births: path, line (number in birth_sha), birth_content
(read from the clone), hunk_id / hunk_size (the conference hunk: a contiguous
run of added lines in one -U0 diff hunk), slug, pr_number, is_agent,
agent_name, agent_tier, birth_sha, birth_date (naive UTC), merge_type,
match_reason (conference match_method), unit_id. The conference's own
tracking is carried along as conf_* columns for the agreement check.

Output: data/survival_events_line_v6conf_partial/<slug>.births.csv
(resumable: existing files are skipped). Then:
    V5_COHORT=conf215births python src/measure_retention_v5.py --phase track
    V5_COHORT=conf215births python src/measure_retention_v5.py --phase combine

Usage:
    python src/build_conf_births_v6.py [--slugs ...]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.agent_taxonomy import AGENT_TO_TIER
from measure_retention_line_v3 import REPO_DIR, git

TABLES = os.path.join(ROOT, 'earlier_study', 'tables')
OUT = os.path.join(ROOT, 'data', 'survival_events_line_v6conf_partial')
AGENT_NAMES = {'Copilot': 'GitHub Copilot', 'OpenAI_Codex': 'OpenAI Codex', 'Claude_Code': 'Claude Code',
               'Devin': 'Devin', 'Cursor': 'Cursor', 'Human': 'Human'}
WORKERS = 6


def build(slug: str, df: pd.DataFrame) -> dict:
    repo = os.path.join(REPO_DIR, slug.replace('/', '__') + '.git')
    if not os.path.isdir(repo):
        return {'slug': slug, 'error': 'no clone'}
    content, missing = [], 0
    parents = {}
    for (sha, path), g in df.groupby(['birth_sha', 'file'], sort=False):
        text = git(repo, 'show', f'{sha}:{path}', check=False)
        lines = text.split('\n')
        for i, n in zip(g.index, g['line']):
            if 0 < n <= len(lines):
                content.append((i, lines[n - 1]))
            else:
                content.append((i, None))
                missing += 1
        if sha not in parents:
            parents[sha] = len(git(repo, 'rev-list', '--parents', '-n', '1', sha).split()) - 1
    c = pd.Series(dict(content))
    out = pd.DataFrame({
        'path': df['file'], 'line': df['line'], 'birth_content': c.reindex(df.index),
        'hunk_id': df['file'] + '::' + df['hunk'].astype(str),
        'slug': slug, 'pr_number': df['pr_number'], 'is_agent': df['is_agent'].astype(bool),
        'agent_name': df['agent_name'].map(AGENT_NAMES),
        'birth_sha': df['birth_sha'],
        'birth_date': df['birth_date'].dt.tz_convert('UTC').dt.tz_localize(None).map(lambda d: d.isoformat()),
        'merge_type': df['birth_sha'].map(lambda s: 'merge' if parents[s] > 1 else 'squash/rebase'),
        'match_reason': df['match_method'], 'pr_id': df['pr_id'],
        'vendored': df['vendored'], 'blank': df['blank'],
        'conf_event': df['event'], 'conf_kind': df['kind'], 'conf_last_seen_sha': df['last_seen_sha'],
        'conf_death_sha': df['death_sha'],
        'conf_death_date': df['death_date'].dt.tz_convert('UTC').dt.tz_localize(None).map(
            lambda d: d.isoformat() if pd.notna(d) else None),
    })
    out['agent_tier'] = [AGENT_TO_TIER.get(a, 'unknown') if ia else 'human'
                         for a, ia in zip(out['agent_name'], out['is_agent'])]
    out['hunk_size'] = out.groupby(['pr_number', 'hunk_id'])['line'].transform('size')
    out['unit_id'] = out['path'] + '::' + out['line'].astype(str)
    out.to_csv(os.path.join(OUT, f'{slug.replace("/", "__")}.births.csv'), index=False)
    return {'slug': slug, 'prs': out['pr_number'].nunique(), 'units': len(out), 'content_missing': missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slugs', nargs='*')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lines = pd.read_parquet(os.path.join(TABLES, 'lines.parquet'))
    prs = pd.read_parquet(os.path.join(TABLES, 'prs.parquet'), columns=['slug', 'pr_number', 'match_method', 'pr_id'])
    lines = lines[lines['event'].notna()]
    lines = lines.merge(prs, on=['slug', 'pr_number'], how='left')
    print(f'conference lines tracked: {len(lines):,}; PRs {lines[["slug", "pr_number"]].drop_duplicates().shape[0]:,}; '
          f'repos {lines["slug"].nunique()}', flush=True)
    slugs = args.slugs or sorted(lines['slug'].unique())
    todo = [s for s in slugs if not os.path.exists(os.path.join(OUT, f'{s.replace("/", "__")}.births.csv'))]
    print(f'{len(todo)} repos to build', flush=True)
    stats = []
    with ProcessPoolExecutor(WORKERS) as ex:
        fut = {ex.submit(build, s, lines[lines['slug'] == s].reset_index(drop=True)): s for s in todo}
        for i, f in enumerate(as_completed(fut), 1):
            try:
                st = f.result()
            except Exception as e:
                st = {'slug': fut[f], 'error': f'{type(e).__name__}: {e}'[:300]}
            stats.append(st)
            print(f'[{i}/{len(todo)}] {st}', flush=True)
    old = os.path.join(OUT, 'build_stats.csv')
    prev = pd.read_csv(old) if os.path.exists(old) else pd.DataFrame()
    pd.concat([prev, pd.DataFrame(stats)]).to_csv(old, index=False)


if __name__ == '__main__':
    main()
