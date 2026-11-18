"""
measure_position_v6.py

Position-based (v2) line tracking on the conference cohort's born lines, so
that the tracking-method comparison and the specification curve can be built
on the same units as the identity tracker (v6conf).

Units: the births in data/survival_events_line_v6conf_partial/ (conference
born lines, src/build_conf_births_v6.py), sampled with the v2 caps: per PR
the first MAX_FILES_PER_PR source files in path order (v2's `git show
--name-only` order, which is empty for true merges), and per file at most MAX_LINES_PER_FILE lines drawn at random
(v2 drew them unseeded; here the draw is seeded per PR and file).

Tracking: measure_retention_line_v2.check_line_survival, unchanged (blame of
the line NUMBER at HEAD, pickaxe search for the death commit, file deletion
via ls-files), run in a shared, index-only clone of data/temp_repos/<slug>.git
whose HEAD is the default branch at clone time (as v2's fresh clone).

Output (v4 schema, + conference flags):
  data/survival_events_line_pos_v6conf_partial/<slug>.csv  (resumable)
  data/survival_events_line_pos_v6conf.csv                 (--combine)

Hunk level (POS_GRANULARITY=hunk): the v2 hunk tracker
(measure_retention_hunk_v2.check_hunk_survival, unchanged: blame of the hunk's
line range at HEAD, earliest modifying commit) on every conference hunk, no caps
(as v2's hunk miner). A conference hunk is a contiguous run of added lines in
one -U0 diff hunk (start = its first line, size = its lines). Output (v3 hunk
schema): data/survival_events_hunk_pos_v6conf_partial/, data/survival_events_hunk_pos_v6conf.csv.

Usage:
    python src/measure_position_v6.py
    python src/measure_position_v6.py --combine
    POS_GRANULARITY=hunk python src/measure_position_v6.py [--combine]
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.constants import MAX_FILES_PER_PR, MAX_LINES_PER_FILE
from measure_retention_line_v2 import check_line_survival
from measure_retention_hunk_v2 import Hunk, check_hunk_survival
from measure_retention_line_v3 import REPO_DIR, git

BIRTHS = os.path.join(ROOT, 'data', 'survival_events_line_v6conf_partial')
GRANULARITY = os.environ.get('POS_GRANULARITY', 'line')
PARTIAL = os.path.join(ROOT, 'data', f'survival_events_{GRANULARITY}_pos_v6conf_partial')
OUTPUT = os.path.join(ROOT, 'data', f'survival_events_{GRANULARITY}_pos_v6conf.csv')
SEED = 42
WORKERS = 6


def sample(repo: str, b: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for (pr, sha), g in b.groupby(['pr_number', 'birth_sha']):
        # v2 took the first files of `git show --name-only` (path order); that lists no files
        # for a true merge commit, so the PR's own files are taken in path order instead
        files = sorted(set(g['path']))[:MAX_FILES_PER_PR]
        for f in files:
            lines = sorted(g.loc[g['path'] == f, 'line'].tolist())
            if len(lines) > MAX_LINES_PER_FILE:
                lines = sorted(random.Random(f'{SEED}:{pr}:{f}').sample(lines, MAX_LINES_PER_FILE))
            keep.append(g[(g['path'] == f) & g['line'].isin(lines)])
    return pd.concat(keep) if keep else b.iloc[:0]


def hunks(work: str, slug: str, b: pd.DataFrame, head: str) -> list:
    rows = []
    for (pr, sha), g in b.groupby(['pr_number', 'birth_sha']):
        hs = g.groupby(['path', 'hunk_id'], as_index=False).agg(
            start=('line', 'min'), end=('line', 'max'), size=('line', 'size'), first=('line', 'first'),
            vendored=('vendored', 'first'))
        r0 = g.iloc[0]
        bd = datetime.fromisoformat(r0.birth_date)
        for h in hs.itertuples():
            idx = int(str(h.hunk_id).rsplit('::', 1)[1])
            hk = Hunk(filepath=h.path, start_line=int(h.start), line_count=int(h.size), hunk_index=idx)
            event, duration, dsha, ddate, lm, ls = check_hunk_survival(work, hk, sha, bd)
            rows.append({'slug': slug, 'pr_number': pr, 'is_agent': r0.is_agent, 'agent_name': r0.agent_name,
                         'agent_tier': r0.agent_tier, 'unit_id': hk.unit_id, 'event': event, 'duration_days': duration,
                         'birth_sha': sha, 'birth_date': r0.birth_date, 'death_sha': dsha,
                         'death_date': ddate.isoformat() if ddate else None, 'lines_modified': lm,
                         'lines_survived': ls, 'modification_rate': lm / hk.line_count,
                         'pr_hunk_count': len(hs), 'pr_file_count': hs['path'].nunique(), 'granularity': 'hunk',
                         'hunk_start': hk.start_line, 'hunk_end': hk.end_line, 'hunk_size': hk.line_count,
                         'contiguous': int(h.end - h.start + 1 == h.size), 'hunk_id': h.hunk_id,
                         'vendored': h.vendored, 'head_sha': head})
    return rows


def run(slug: str) -> dict:
    t0 = time.time()
    tag = slug.replace('/', '__')
    bf = os.path.join(BIRTHS, f'{tag}.births.csv')
    b = pd.read_csv(bf, keep_default_na=False, na_values=['']) if os.path.getsize(bf) > 1 else pd.DataFrame()
    if not len(b):
        pd.DataFrame().to_csv(os.path.join(PARTIAL, f'{tag}.csv'), index=False)
        return {'slug': slug, 'units': 0}
    bare = os.path.join(REPO_DIR, f'{tag}.git')
    work = tempfile.mkdtemp(prefix='pos_')
    try:
        subprocess.run(['git', 'clone', '--quiet', '--shared', '--no-checkout', bare, work], check=True,
                       capture_output=True)
        subprocess.run(['git', '-C', work, 'read-tree', 'HEAD'], check=True, capture_output=True)
        head = git(work, 'rev-parse', 'HEAD').strip()
        if GRANULARITY == 'hunk':
            rows = hunks(work, slug, b, head)
            pd.DataFrame(rows).to_csv(os.path.join(PARTIAL, f'{tag}.csv'), index=False)
            return {'slug': slug, 'births': len(b), 'units': len(rows), 'seconds': round(time.time() - t0, 1)}
        s = sample(work, b)
        rows = []
        for r in s.itertuples():
            bd = datetime.fromisoformat(r.birth_date)
            event, duration, dsha, ddate, bc, dc, ted, mag = check_line_survival(work, r.path, int(r.line), r.birth_sha, bd)
            rows.append({'slug': slug, 'pr_number': r.pr_number, 'is_agent': r.is_agent, 'agent_name': r.agent_name,
                         'agent_tier': r.agent_tier, 'unit_id': r.unit_id, 'event': event, 'duration_days': duration,
                         'birth_sha': r.birth_sha, 'birth_date': r.birth_date, 'death_sha': dsha,
                         'death_date': ddate.isoformat() if ddate else None, 'birth_content': bc,
                         'death_content': dc, 'token_edit_distance': ted, 'death_magnitude': mag,
                         'granularity': 'line', 'vendored': r.vendored, 'blank': r.blank, 'head_sha': head})
        pd.DataFrame(rows).to_csv(os.path.join(PARTIAL, f'{tag}.csv'), index=False)
        return {'slug': slug, 'births': len(b), 'units': len(rows), 'seconds': round(time.time() - t0, 1)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slugs', nargs='*')
    ap.add_argument('--combine', action='store_true')
    args = ap.parse_args()
    os.makedirs(PARTIAL, exist_ok=True)
    if args.combine:
        frames = [pd.read_csv(os.path.join(PARTIAL, f)) for f in sorted(os.listdir(PARTIAL))
                  if f.endswith('.csv') and 'stats' not in f and os.path.getsize(os.path.join(PARTIAL, f)) > 1]
        df = pd.concat(frames, ignore_index=True).drop_duplicates(['slug', 'pr_number', 'unit_id'])
        df.to_csv(OUTPUT, index=False)
        print('wrote', OUTPUT, len(df))
        return
    have = sorted(f[:-len('.births.csv')] for f in os.listdir(BIRTHS) if f.endswith('.births.csv'))
    slugs = args.slugs or [t.replace('__', '/', 1) for t in have]
    todo = [s for s in slugs if not os.path.exists(os.path.join(PARTIAL, f'{s.replace("/", "__")}.csv'))]
    print(f'position tracking: {len(todo)} repos to do', flush=True)
    stats = []
    with ProcessPoolExecutor(WORKERS) as ex:
        fut = {ex.submit(run, s): s for s in todo}
        for i, f in enumerate(as_completed(fut), 1):
            try:
                st = f.result()
            except Exception as e:
                st = {'slug': fut[f], 'error': f'{type(e).__name__}: {e}'[:300]}
            stats.append(st)
            print(f'[{i}/{len(todo)}] {st}', flush=True)
    old = os.path.join(PARTIAL, 'run_stats.csv')
    prev = pd.read_csv(old) if os.path.exists(old) else pd.DataFrame()
    pd.concat([prev, pd.DataFrame(stats)]).to_csv(old, index=False)


if __name__ == '__main__':
    main()
