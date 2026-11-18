"""
measure_retention_v5.py

Line-level retention, mined end to end with identity-based tracking.
Replaces measure_retention_line_v2.py (v4 data) and the separate hunk miner.

Phase "births" (per repository):
  - PRs: AIDev agent and human PRs in data/aidev_filtered_v1.csv repositories,
    merge commit found as in v2 (find_merge_commit: AIDev timeline merge SHA on
    the default branch, else the default-branch commit within 60 s of merged_at).
  - Birth lines: every line ADDED by the merge commit relative to its FIRST
    parent (git diff -M -U0 <first parent> <merge>), which covers squash,
    rebase and true merge commits alike. No file or line caps (v2 sampled at
    most 20 files per PR and 100 lines per file, and dropped true merges).
  - Kept: files with a source extension (config/constants.py), not in a
    generated or vendored path (the analysis exclusion list).
  - Each birth line carries its hunk id: a hunk is a maximal run of
    consecutive added lines in one file of the birth diff.

Phase "track" (per repository): identity tracking as in
measure_retention_line_v3.py (reverse blame from birth to the end snapshot,
first-parent, -M -C; the death commit verified in its diff; event_type
content_modified / line_deleted / file_deleted / unverified / censored;
whitespace_only flag).

Per-repository checkpoints in data/survival_events_line_v5_partial/
(<slug>.births.csv, <slug>.csv); finished repositories are skipped on resume.
The births phase also writes <slug>.pr_status.csv: one row per candidate PR with
the merge commit found (or the skip reason) and its number of birth lines.

Cohort (V5_COHORT in the environment):
  journal (default)  repositories in data/aidev_filtered_v1.csv,
                     checkpoints in data/survival_events_line_v5_partial/,
                     output data/survival_events_line_v5.csv
  conf215            repositories in data/cohort_conf215_slugs.csv (every slug
                     of the conference cohort, earlier_study/prs_cohort256.csv),
                     checkpoints in data/survival_events_line_v6_partial/,
                     output data/survival_events_line_v6.csv
  conf215births      the conference's own born lines (src/build_conf_births_v6.py
                     writes the births), checkpoints in
                     data/survival_events_line_v6conf_partial/, output
                     data/survival_events_line_v6conf.csv; track and combine only
The PR selection rules are the same for journal and conf215; the conference PR
list is applied later, in derive_v5_datasets.py --cohort-prs.

Usage:
    python src/measure_retention_v5.py --phase births
    python src/measure_retention_v5.py --phase track
    python src/measure_retention_v5.py --phase combine
    V5_COHORT=conf215 python src/measure_retention_v5.py --phase births
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.constants import SOURCE_EXTENSIONS, PR_INCLUSION_CUTOFF
from config.agent_taxonomy import AGENT_TO_TIER
from measure_retention_line_v2 import find_merge_commit, identify_agent, normalize_slug
from measure_retention_line_v3 import (clone, end_snapshot, commit_dates, reverse_blame,
                                       commit_diff, classify, git, EVENT_TYPES, OBSERVATION_CUTOFF)

GENERATED_PATH_PATTERNS = [
    r'/dist/', r'/third_party/', r'/vendor/', r'/generated/',
    r'_pb2\.py$', r'\.min\.js$', r'immutable_entry',
    r'_generated\.', r'zz_generated\.',
]
# cohort -> (slug list, checkpoint dir, output), all under data/
COHORTS = {
    'journal': ('aidev_filtered_v1.csv', 'survival_events_line_v5_partial', 'survival_events_line_v5.csv'),
    'conf215': ('cohort_conf215_slugs.csv', 'survival_events_line_v6_partial', 'survival_events_line_v6.csv'),
    'conf215births': ('cohort_conf215_slugs.csv', 'survival_events_line_v6conf_partial', 'survival_events_line_v6conf.csv'),
}
COHORT = os.environ.get('V5_COHORT', 'journal')
SLUG_FILE, _partial, _output = COHORTS[COHORT]
# V5_PARTIAL_DIR / V5_OUTPUT let the same tracker run on another set of births
# (e.g. the old v4 units, for the tracking-method comparison).
PARTIAL = Path(os.environ.get('V5_PARTIAL_DIR', Path(ROOT) / 'data' / _partial))
OUTPUT = Path(os.environ.get('V5_OUTPUT', Path(ROOT) / 'data' / _output))
WORKERS = 6
HUNK_RE = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@')


def keep_file(path: str) -> bool:
    p = '/' + path
    return path.endswith(SOURCE_EXTENSIONS) and not any(re.search(x, p) for x in GENERATED_PATH_PATTERNS)


def added_lines(repo: str, merge: str) -> list:
    """(path, new line number, content, hunk id) for lines added vs the first parent."""
    out = git(repo, 'diff', '-M', '-U0', '--no-color', '--no-ext-diff', f'{merge}^1', merge, check=False)
    rows, path, n = [], None, None
    run_start = prev = None
    for line in out.splitlines():
        if line.startswith('+++ '):
            path = None if line == '+++ /dev/null' else line[6:]
            path = path if path and keep_file(path) else None
            prev = None
        elif line.startswith('@@'):
            m = HUNK_RE.match(line)
            n = int(m.group(1))
            prev = None
        elif path and line.startswith('+') and not line.startswith('+++'):
            if prev is None or n != prev + 1:
                run_start = n
            rows.append([path, n, line[1:], f'{path}::{run_start}'])
            prev = n
            n += 1
        elif path and line.startswith(' '):
            n += 1
    # hunk id = path::first-last line of the run
    df = pd.DataFrame(rows, columns=['path', 'line', 'birth_content', 'run'])
    if len(df):
        last = df.groupby('run')['line'].transform('max')
        df['hunk_id'] = df['run'] + '-' + last.astype(str)
        df['hunk_size'] = df.groupby('run')['line'].transform('size')
    return df.drop(columns='run') if len(df) else df


def births(slug: str, prs: pd.DataFrame, token: str) -> dict:
    t0 = time.time()
    repo = clone(slug, token)
    frames, skipped, status = [], {}, []
    for r in prs.itertuples():
        sha, date, reason = find_merge_commit(repo, r.merged_at, r.timeline_merge_sha)
        st = {'pr_id': r.id, 'pr_number': r.number, 'is_agent': r.is_agent, 'birth_sha': sha,
              'match_reason': reason, 'skip': None, 'units': 0}
        status.append(st)
        if not sha:
            st['skip'] = reason
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        if date >= PR_INCLUSION_CUTOFF:
            st['skip'] = 'merged on or after the inclusion cutoff'
            skipped['merged on or after the inclusion cutoff'] = skipped.get('merged on or after the inclusion cutoff', 0) + 1
            continue
        parents = len(git(repo, 'rev-list', '--parents', '-n', '1', sha).split()) - 1
        df = added_lines(repo, sha)
        if not len(df):
            st['skip'] = 'no added source lines'
            skipped['no added source lines'] = skipped.get('no added source lines', 0) + 1
            continue
        st['units'] = len(df)
        agent = identify_agent(pd.Series(r._asdict())) if r.is_agent else 'Human'
        df = df.assign(slug=slug, pr_number=r.number, is_agent=r.is_agent, agent_name=agent,
                       agent_tier=AGENT_TO_TIER.get(agent, 'unknown') if r.is_agent else 'human',
                       birth_sha=sha, birth_date=date.isoformat(), merge_type='merge' if parents > 1 else 'squash/rebase',
                       match_reason=reason)
        df['unit_id'] = df['path'] + '::' + df['line'].astype(str)
        frames.append(df)
    out = pd.concat(frames) if frames else pd.DataFrame()
    pd.DataFrame(status).to_csv(PARTIAL / f'{slug.replace("/", "__")}.pr_status.csv', index=False)
    out.to_csv(PARTIAL / f'{slug.replace("/", "__")}.births.csv', index=False)
    return {'slug': slug, 'prs_in': len(prs), 'prs_with_units': out['pr_number'].nunique() if len(out) else 0,
            'units': len(out), 'seconds': round(time.time() - t0, 1), **{f'skip: {k}': v for k, v in skipped.items()}}


def ranges_of(lines: list) -> list:
    out, start, prev = [], lines[0], lines[0]
    for n in lines[1:]:
        if n != prev + 1:
            out.append((start, prev))
            start = n
        prev = n
    out.append((start, prev))
    return out


def parse_porcelain(out: str) -> dict:
    """line in the blamed revision -> (commit, line there, path there)."""
    res, cur = {}, None
    for line in out.splitlines():
        m = re.match(r'^([0-9a-f]{40}) (\d+) (\d+)', line)
        if m:
            cur = [m.group(1), int(m.group(2)), int(m.group(3)), None]
        elif line.startswith('filename ') and cur:
            cur[3] = line[len('filename '):]
        elif line.startswith('\t') and cur:
            res[cur[2]] = (cur[0], cur[1], cur[3])
            cur = None
    return res


def track(slug: str, token: str, remove_clone: bool) -> dict:
    t0 = time.time()
    bf = PARTIAL / f'{slug.replace("/", "__")}.births.csv'
    # a repository with no birth lines has an empty births file (no header)
    b = pd.read_csv(bf, keep_default_na=False, na_values=['']) if bf.stat().st_size > 1 else pd.DataFrame()
    if not len(b):
        pd.DataFrame().to_csv(PARTIAL / f'{slug.replace("/", "__")}.csv', index=False)
        return {'slug': slug, 'units': 0, 'seconds': 0}
    repo = clone(slug, token)
    end = end_snapshot(repo)
    chain, dates = commit_dates(repo, end)
    pos = {s: i for i, s in enumerate(chain)}
    diffs, res = {}, []

    def blank():
        return {'event_type': 'unverified', 'death_sha': None, 'death_date': None, 'whitespace_only': None,
                'death_content': None, 'path_at_death': None, 'line_at_death': None,
                'last_alive_sha': None, 'start_sha': None, 'note': None}

    def follow(start, path, lines):
        """Track lines (numbered in `start`, which is on the first-parent chain) to the end snapshot."""
        alive = reverse_blame(repo, start, end, path, lines)
        out = {}
        for line in lines:
            o = blank()
            o['start_sha'] = start
            hit = alive.get(line)
            if hit is None:
                o['note'] = 'line not reported by reverse blame'
            else:
                last, lineno, lpath = hit
                o['path_at_death'], o['line_at_death'], o['last_alive_sha'] = lpath, lineno, last
                if last == end:
                    o['event_type'] = 'censored'
                elif last in pos and pos[last] + 1 < len(chain):
                    death = chain[pos[last] + 1]
                    if (last, death) not in diffs:
                        diffs[(last, death)] = commit_diff(repo, last, death)
                    et, ws, dc = classify(diffs[(last, death)], lpath, lineno)
                    o.update(event_type=et, death_sha=death, death_date=dates[death].isoformat(),
                             whitespace_only=ws, death_content=dc)
                else:
                    o['note'] = 'last-alive commit not on the first-parent chain'
            out[line] = o
        return out

    integration = {}
    for (birth, path), g in b.groupby(['birth_sha', 'path']):
        lines = sorted(set(g['line'].tolist()))
        if birth in pos:
            for line, o in follow(birth, path, lines).items():
                res.append((birth, path, line, o))
            continue
        # Birth on a side branch merged into the main line later: map each line to
        # the first main-line commit that contains the birth, then track from there.
        if birth not in integration:
            # oldest main-line commit that contains the birth (containment is monotonic along the chain)
            def contains(i):
                return subprocess.run(['git', '-C', repo, 'merge-base', '--is-ancestor', birth, chain[i]],
                                      capture_output=True).returncode == 0
            lo, hi = 0, len(chain) - 1
            if not contains(hi):
                integration[birth] = None
            else:
                while lo < hi:
                    mid = (lo + hi) // 2
                    if contains(mid):
                        hi = mid
                    else:
                        lo = mid + 1
                integration[birth] = chain[lo]
        x = integration[birth]
        if x is None:
            for line in lines:
                o = blank(); o['note'] = 'birth not reachable from the end snapshot'
                res.append((birth, path, line, o))
            continue
        args = ['blame', '--reverse', f'{birth}..{x}', '-M', '-C', '--line-porcelain']
        for a, z in ranges_of(lines):
            args += ['-L', f'{a},{z}']
        mapped = parse_porcelain(git(repo, *args, '--', path, check=False))
        by_path = {}
        for line in lines:
            hit = mapped.get(line)
            if hit is None or hit[0] != x:
                o = blank()
                o['note'] = 'changed before reaching the main line' if hit else 'line not reported by reverse blame'
                res.append((birth, path, line, o))
            else:
                by_path.setdefault(hit[2], []).append((line, hit[1]))
        for xpath, pairs in by_path.items():
            tracked = follow(x, xpath, [n for _, n in pairs])
            for line, n in pairs:
                res.append((birth, path, line, tracked[n]))
    t = pd.DataFrame([{'birth_sha': a, 'path': p, 'line': l, **o} for a, p, l, o in res])
    t = t.drop_duplicates(['birth_sha', 'path', 'line'])
    df = b.merge(t, on=['birth_sha', 'path', 'line'], how='left')
    df['event'] = df['event_type'].isin(EVENT_TYPES).astype('Int64')
    df.loc[df['event_type'] == 'unverified', 'event'] = pd.NA
    bd, dd = pd.to_datetime(df['birth_date']), pd.to_datetime(df['death_date'])
    df['duration_days'] = ((dd.where(df['event'] == 1, OBSERVATION_CUTOFF) - bd).dt.total_seconds() / 86400).clip(lower=0.001)
    df['end_sha'] = end
    df['granularity'] = 'line'
    df.to_csv(PARTIAL / f'{slug.replace("/", "__")}.csv', index=False)
    if remove_clone:
        shutil.rmtree(repo, ignore_errors=True)
    return {'slug': slug, 'units': len(df), 'seconds': round(time.time() - t0, 1),
            'unverified': int((df['event_type'] == 'unverified').sum())}


def load_prs() -> pd.DataFrame:
    slugs = set(pd.read_csv(os.path.join(ROOT, 'data', SLUG_FILE))['slug'])
    a = pd.read_parquet(os.path.join(ROOT, 'data', 'all_pull_request.parquet'))
    h = pd.read_parquet(os.path.join(ROOT, 'data', 'human_pull_request.parquet'))
    for d, flag in ((a, True), (h, False)):
        d['slug'] = d['repo_url'].apply(normalize_slug)
        d['is_agent'] = flag
    prs = pd.concat([a, h], ignore_index=True)
    prs = prs[prs['slug'].isin(slugs)]
    tl = pd.read_parquet(os.path.join(ROOT, 'data', 'pr_timeline.parquet'), columns=['pr_id', 'event', 'commit_id'])
    sha = tl[tl['event'] == 'merged'].dropna(subset=['commit_id']).drop_duplicates('pr_id').set_index('pr_id')['commit_id']
    prs['timeline_merge_sha'] = prs['id'].map(sha)
    return prs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['births', 'track', 'combine'], required=True)
    ap.add_argument('--slugs', nargs='*')
    ap.add_argument('--remove-clones', action='store_true')
    args = ap.parse_args()
    from dotenv import dotenv_values
    token = (dotenv_values(os.path.join(ROOT, '.env')).get('GITHUB_TOKEN') or '').strip()
    PARTIAL.mkdir(parents=True, exist_ok=True)
    print(f'cohort {COHORT}: slugs from data/{SLUG_FILE}, checkpoints in {PARTIAL}', flush=True)
    prs = load_prs()
    slugs = args.slugs or sorted(prs['slug'].unique())
    if args.phase == 'births':
        if COHORT == 'conf215births':
            sys.exit('conf215births: births come from src/build_conf_births_v6.py, not from this miner')
        todo = [s for s in slugs if not (PARTIAL / f'{s.replace("/", "__")}.births.csv').exists()]
        print(f'births: {len(slugs)} repos, {len(todo)} to do', flush=True)
        stats = []
        with ProcessPoolExecutor(WORKERS) as ex:
            fut = {ex.submit(births, s, prs[prs['slug'] == s], token): s for s in todo}
            for i, f in enumerate(as_completed(fut), 1):
                try:
                    st = f.result()
                except Exception as e:
                    st = {'slug': fut[f], 'error': f'{type(e).__name__}: {e}'[:300]}
                stats.append(st)
                print(f'[{i}/{len(todo)}] {st}', flush=True)
        old = PARTIAL / 'births_stats.csv'
        prev = pd.read_csv(old) if old.exists() else pd.DataFrame()
        pd.concat([prev, pd.DataFrame(stats)]).to_csv(old, index=False)
    elif args.phase == 'track':
        todo = [s for s in slugs if (PARTIAL / f'{s.replace("/", "__")}.births.csv').exists()
                and not (PARTIAL / f'{s.replace("/", "__")}.csv').exists()]
        print(f'track: {len(todo)} repos to do', flush=True)
        stats = []
        with ProcessPoolExecutor(WORKERS) as ex:
            fut = {ex.submit(track, s, token, args.remove_clones): s for s in todo}
            for i, f in enumerate(as_completed(fut), 1):
                try:
                    st = f.result()
                except Exception as e:
                    st = {'slug': fut[f], 'error': f'{type(e).__name__}: {e}'[:300], 'tb': traceback.format_exc()[-300:]}
                stats.append(st)
                print(f'[{i}/{len(todo)}] {st}', flush=True)
        old = PARTIAL / 'track_stats.csv'
        prev = pd.read_csv(old) if old.exists() else pd.DataFrame()
        pd.concat([prev, pd.DataFrame(stats)]).to_csv(old, index=False)
    else:
        frames = [pd.read_csv(p) for p in sorted(PARTIAL.glob('*.csv'))
                  if not p.name.endswith(('.births.csv', '.pr_status.csv')) and 'stats' not in p.name
                  and p.stat().st_size > 1]
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(OUTPUT, index=False)
        print('wrote', OUTPUT, len(df))


if __name__ == '__main__':
    main()
