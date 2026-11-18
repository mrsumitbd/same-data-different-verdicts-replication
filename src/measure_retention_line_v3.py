"""
measure_retention_line_v3.py

Line-level retention tracked by line IDENTITY (v2 tracked line numbers).

What v2 got wrong (src/measure_retention_line_v2.py, check_line_survival):
  - survival was checked with `git blame -L <birth line number> HEAD`, so any
    insertion above the line made it look modified;
  - when pickaxe found no removal, the death was set to the first later commit
    that touched the file, whether or not it changed the line;
  - a file path missing at HEAD counted as a deletion, so renames and moves
    "killed" every line in the file;
  - HEAD was the clone-time HEAD, not the observation cutoff.

v3:
  - Birth units are taken unchanged from the v4 line data (same PRs, merge
    commits, files and line numbers), so v2 and v3 are compared like for like.
  - End snapshot: the last first-parent commit of the default branch at or
    before the observation cutoff (config/constants.py).
  - Each file's birth lines are tracked forward with
        git blame --reverse <birth>..<end> --first-parent -M -C --line-porcelain -L ...
    which reports, for every birth line, the last first-parent commit in which
    the line still existed (following renames and moves). If that is the end
    snapshot, the line survived (censored). Otherwise the modifying commit is
    the next first-parent commit, and v3 verifies in its diff that the line's
    position is inside a changed hunk.
  - event_type: content_modified (the hunk replaces the line), line_deleted
    (the hunk removes it and adds nothing), file_deleted (file removed with no
    rename target), unverified (the next commit does not visibly change the
    line; NOT counted as a modification), censored.
  - whitespace_only: the removed line equals an added line in the same hunk
    after removing whitespace.

Usage:
    python src/measure_retention_line_v3.py --slugs go-vikunja/vikunja agentops-ai/agentops
Output:
    data/survival_events_line_v5.csv (default; one row per birth unit)
"""

import argparse
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config.constants import OBSERVATION_CUTOFF

REPO_DIR = os.path.join(ROOT, 'data', 'temp_repos')
EVENT_TYPES = ('content_modified', 'line_deleted', 'file_deleted')
HUNK_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


def git(repo: str, *args, check: bool = True) -> str:
    r = subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True, errors='replace')
    if check and r.returncode != 0:
        raise RuntimeError(f'git {" ".join(args[:3])}: {r.stderr.strip()[:300]}')
    return r.stdout


def clone(slug: str, token: str) -> str:
    path = os.path.join(REPO_DIR, slug.replace('/', '__') + '.git')
    if os.path.isdir(path):
        return path
    os.makedirs(REPO_DIR, exist_ok=True)
    url = f'https://github.com/{slug}.git'
    env = dict(os.environ)
    if token:
        # pass the token as a header so it never appears in the remote URL or logs
        import base64
        b = base64.b64encode(f'x-access-token:{token}'.encode()).decode()
        cmd = ['git', '-c', f'http.extraheader=AUTHORIZATION: basic {b}', 'clone', '--bare', '--quiet', url, path]
    else:
        cmd = ['git', 'clone', '--bare', '--quiet', url, path]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f'clone failed: {slug}: {r.stderr.strip().splitlines()[-1:]}')
    return path


def end_snapshot(repo: str) -> str:
    cutoff = OBSERVATION_CUTOFF.strftime('%Y-%m-%d %H:%M:%S +0000')   # explicit UTC; git reads bare dates as local time
    return git(repo, 'rev-list', '-1', '--first-parent', f'--before={cutoff}', 'HEAD').strip()


def commit_dates(repo: str, end: str) -> tuple:
    """First-parent chain up to `end` (oldest first) and committer dates."""
    out = git(repo, 'log', '--first-parent', '--format=%H %ct', end)
    chain, dates = [], {}
    for line in reversed(out.splitlines()):
        sha, ts = line.split()
        chain.append(sha)
        dates[sha] = datetime.utcfromtimestamp(int(ts))
    return chain, dates


def ranges(lines: list) -> list:
    lines = sorted(set(lines))
    out, start, prev = [], lines[0], lines[0]
    for n in lines[1:]:
        if n != prev + 1:
            out.append((start, prev))
            start = n
        prev = n
    out.append((start, prev))
    return out


def reverse_blame(repo: str, birth: str, end: str, path: str, lines: list) -> dict:
    """birth line number -> (last-alive commit, line number there, path there)."""
    args = ['blame', '--reverse', f'{birth}..{end}', '--first-parent', '-M', '-C', '--line-porcelain']
    for a, b in ranges(lines):
        args += ['-L', f'{a},{b}']
    out = git(repo, *args, '--', path, check=False)
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


def commit_diff(repo: str, parent: str, child: str) -> dict:
    """old path -> {'new': new path or None, 'hunks': [(a, b, removed, added)]}."""
    out = git(repo, 'diff', '-M', '-U0', '--no-color', '--no-ext-diff', parent, child, check=False)
    files, cur, hunk = {}, None, None
    old = new = None
    for line in out.splitlines():
        if line.startswith('diff --git '):
            cur, hunk, old, new = None, None, None, None
        elif line.startswith('--- '):
            old = None if line == '--- /dev/null' else line[6:]
        elif line.startswith('+++ '):
            new = None if line == '+++ /dev/null' else line[6:]
            if old is not None:
                cur = files.setdefault(old, {'new': new, 'hunks': []})
        elif line.startswith('rename from ') or line.startswith('rename to '):
            pass
        elif cur is not None and line.startswith('@@'):
            m = HUNK_RE.match(line)
            a, b = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            hunk = [a, b, [], []]
            cur['hunks'].append(hunk)
        elif hunk is not None and line.startswith('-'):
            hunk[2].append(line[1:])
        elif hunk is not None and line.startswith('+'):
            hunk[3].append(line[1:])
    # pure renames without content change have no ---/+++ lines; record them
    for m in re.finditer(r'^diff --git a/(.+) b/(.+)\n(?:similarity index 100%\n)rename from', out, re.M):
        files.setdefault(m.group(1), {'new': m.group(2), 'hunks': []})
    return files


def classify(diff: dict, path: str, lineno: int) -> tuple:
    """(event_type, whitespace_only, death_content) for one line in one commit diff."""
    f = diff.get(path)
    if f is None:
        return 'unverified', None, None
    if f['new'] is None:
        return 'file_deleted', None, None
    for a, b, removed, added in f['hunks']:
        if b > 0 and a <= lineno <= a + b - 1:
            old_line = removed[lineno - a] if lineno - a < len(removed) else ''
            if not added:
                return 'line_deleted', None, None
            squash = lambda s: re.sub(r'\s+', '', s)
            ws = any(squash(x) == squash(old_line) for x in added)
            k = lineno - a
            return 'content_modified', ws, added[min(k, len(added) - 1)]
    return 'unverified', None, None


def process(slug: str, units: pd.DataFrame, token: str) -> tuple:
    t0 = time.time()
    repo = clone(slug, token)
    t_clone = time.time() - t0
    end = end_snapshot(repo)
    chain, dates = commit_dates(repo, end)
    pos = {s: i for i, s in enumerate(chain)}
    diffs = {}
    rows = []
    units = units.copy()
    units['path'] = units['unit_id'].str.rsplit('::', n=1).str[0]
    units['line'] = units['unit_id'].str.rsplit('::', n=1).str[1].astype(int)
    for (birth, path), g in units.groupby(['birth_sha', 'path']):
        alive = reverse_blame(repo, birth, end, path, g['line'].tolist())
        for r in g.itertuples():
            out = {'event_type': 'unverified', 'death_sha': None, 'death_date': None, 'whitespace_only': None,
                   'death_content': None, 'last_alive_sha': None, 'path_at_death': None}
            hit = alive.get(r.line)
            if hit is None:
                out['note'] = 'line not reported by reverse blame'
            else:
                last, lineno, lpath = hit
                out['last_alive_sha'], out['path_at_death'] = last, lpath
                if last == end or last.startswith(end):
                    out['event_type'] = 'censored'
                elif last in pos and pos[last] + 1 < len(chain):
                    death = chain[pos[last] + 1]
                    if (last, death) not in diffs:
                        diffs[(last, death)] = commit_diff(repo, last, death)
                    et, ws, dc = classify(diffs[(last, death)], lpath, lineno)
                    out.update(event_type=et, death_sha=death, death_date=dates[death].isoformat(),
                               whitespace_only=ws, death_content=dc)
                else:
                    out['note'] = 'last-alive commit not on the first-parent chain'
            rows.append({**r._asdict(), **out})
    df = pd.DataFrame(rows).drop(columns=['Index'], errors='ignore')
    birth = pd.to_datetime(df['birth_date'])
    death = pd.to_datetime(df['death_date'])
    df['event'] = df['event_type'].isin(EVENT_TYPES).astype(int)
    df.loc[df['event_type'] == 'unverified', 'event'] = pd.NA
    df['duration_days'] = ((death.where(df['event'] == 1, OBSERVATION_CUTOFF) - birth).dt.total_seconds() / 86400).clip(lower=0.001)
    parents = {s: len(git(repo, 'rev-list', '--parents', '-n', '1', s).split()) - 1 for s in df['birth_sha'].unique()}
    df['birth_is_merge'] = df['birth_sha'].map(lambda s: parents[s] > 1)
    df['end_sha'] = end
    return df, {'slug': slug, 'units': len(df), 'clone_s': round(t_clone, 1),
                'track_s': round(time.time() - t0 - t_clone, 1), 'end_sha': end,
                'birth_prs': df['pr_number'].nunique(),
                'merge_commit_prs': df.drop_duplicates('pr_number')['birth_is_merge'].sum()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slugs', nargs='+', required=True)
    ap.add_argument('--line-data', default='data/survival_events_line_v4_ast.csv')
    ap.add_argument('--output', default='data/survival_events_line_v5.csv')
    ap.add_argument('--stats', default='data/survival_events_line_v5_runtime.csv')
    args = ap.parse_args()
    from dotenv import dotenv_values
    token = (dotenv_values(os.path.join(ROOT, '.env')).get('GITHUB_TOKEN') or '').strip()
    keep = ['slug', 'pr_number', 'is_agent', 'agent_name', 'agent_tier', 'unit_id', 'birth_sha',
            'birth_date', 'birth_content']
    v4 = pd.read_csv(args.line_data, usecols=keep)
    outs, stats = [], []
    for slug in args.slugs:
        print(f'[{slug}] tracking {int((v4["slug"] == slug).sum()):,} units', flush=True)
        df, st = process(slug, v4[v4['slug'] == slug], token)
        print(f'  {st}', flush=True)
        outs.append(df)
        stats.append(st)
    pd.concat(outs).to_csv(args.output, index=False)
    pd.DataFrame(stats).to_csv(args.stats, index=False)
    print('wrote', args.output)


if __name__ == '__main__':
    main()
