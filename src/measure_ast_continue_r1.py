"""
measure_ast_continue_r1.py 

Definition 2 (AST-structural) without censoring at leaf-only edits.

The current Definition 2 ends a line's follow-up at its first modification:
a structural first modification is an event, a leaf-only one (ast_death == 0)
censors the line there. Here a line whose first modification is a leaf-only
content change is followed further: its new version in the modifying commit is
located from that commit's diff (same hunk offset rule as the tracker's
classify(), measure_retention_line_v3.py), tracked onwards with the same
identity tracker (reverse blame from that commit to the end snapshot,
first-parent, -M -C, death verified in the diff), and its next modification is
classified with the same AST rule (measure_ast_v5.classify). This repeats until
the first structural modification (event), the end snapshot (censored at the
cutoff), or a modification that cannot be followed further:
  - leaf-only line deletion (the line is gone; censored there, as now),
  - modification with unknown AST status (censored there, as Definition 2 treats it now),
  - unverified / not reported by reverse blame (censored there).

Input : data/survival_events_line_v6confnv_ast.csv (DATA_TAG-independent paths given explicitly)
Output: results_v6/round1/c3_ast_continued.csv  (one row per continued line:
        slug, pr_number, unit_id, birth_sha, hops, final_status, final_event,
        final_date, final_ast_node, final_reason)
Usage : python src/measure_ast_continue_r1.py [--workers 7] [--slugs a/b ...]
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
from measure_retention_line_v3 import (git, end_snapshot, commit_dates, reverse_blame, classify as classify_event,
                                       REPO_DIR)
from measure_ast_survival import get_language_for_file
import measure_ast_v5
from measure_ast_v5 import renamed_to

HUNK = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
MAX_HOPS = 25


def diff_hunks(repo, parent, child):
    """old path -> {'new': new path, 'hunks': [(a, b, c, d, removed, added)]} (new-side start kept)."""
    out = git(repo, 'diff', '-M', '-U0', '--no-color', '--no-ext-diff', parent, child, check=False)
    files, cur, hunk, old = {}, None, None, None
    for line in out.splitlines():
        if line.startswith('diff --git '):
            cur, hunk, old = None, None, None
        elif line.startswith('--- '):
            old = None if line == '--- /dev/null' else line[6:]
        elif line.startswith('+++ '):
            new = None if line == '+++ /dev/null' else line[6:]
            if old is not None:
                cur = files.setdefault(old, {'new': new, 'hunks': []})
        elif cur is not None and line.startswith('@@'):
            m = HUNK.match(line)
            a, b = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            c, d = int(m.group(3)), int(m.group(4) if m.group(4) is not None else 1)
            hunk = [a, b, c, d, [], []]
            cur['hunks'].append(hunk)
        elif hunk is not None and line.startswith('-'):
            hunk[4].append(line[1:])
        elif hunk is not None and line.startswith('+'):
            hunk[5].append(line[1:])
    return files


def new_position(files, path, lineno):
    """(new path, new line number) of a content-modified line, or None."""
    f = files.get(path)
    if f is None or f['new'] is None:
        return None
    for a, b, c, d, removed, added in f['hunks']:
        if b > 0 and a <= lineno <= a + b - 1:
            if not added:
                return None
            return f['new'], c + min(lineno - a, len(added) - 1)
    return None


def process(slug, rows):
    t0 = time.time()
    repo = os.path.join(REPO_DIR, slug.replace('/', '__') + '.git')
    end = end_snapshot(repo)
    chain, dates = commit_dates(repo, end)
    pos = {s: i for i, s in enumerate(chain)}
    diffs, files, out = {}, {}, []

    def show(sha, path):
        k = (sha, path)
        if k not in files:
            r = git(repo, 'show', f'{sha}:{path}', check=False)
            files[k] = r if r else None
        return files[k]

    def hunks(a, b):
        if (a, b) not in diffs:
            diffs[(a, b)] = diff_hunks(repo, a, b)
        return diffs[(a, b)]

    # state per line: (key, last_alive, path, line, death_sha)
    state = [dict(key=r, last=r.last_alive_sha, path=r.path_at_death, line=int(r.line_at_death),
                  death=r.death_sha, hops=0) for r in rows.itertuples()]
    active = state
    for hop in range(MAX_HOPS):
        nxt = {}
        for s in active:
            p = new_position(hunks(s['last'], s['death']), s['path'], s['line'])
            if p is None:
                s.update(status='censored', reason='new position not found in diff', date=dates[s['death']])
                continue
            s['start'], s['spath'], s['sline'] = s['death'], p[0], p[1]
            nxt.setdefault((s['start'], s['spath']), []).append(s)
        if not nxt:
            break
        cont = []
        for (start, spath), group in nxt.items():
            alive = reverse_blame(repo, start, end, spath, sorted({g['sline'] for g in group}))
            for s in group:
                s['hops'] += 1
                hit = alive.get(s['sline'])
                if hit is None:
                    s.update(status='censored', reason='not reported by reverse blame', date=dates[start])
                    continue
                last, lineno, lpath = hit
                if last == end:
                    s.update(status='censored', reason='reached end snapshot', date=None)
                    continue
                if last not in pos or pos[last] + 1 >= len(chain):
                    s.update(status='censored', reason='last-alive not on first-parent chain', date=dates[start])
                    continue
                death = chain[pos[last] + 1]
                et, _, _ = classify_event({k: {'new': v['new'], 'hunks': [(h[0], h[1], h[4], h[5]) for h in v['hunks']]}
                                           for k, v in hunks(last, death).items()}, lpath, lineno)
                if et == 'unverified':
                    s.update(status='censored', reason='next modification unverified', date=dates[death])
                    continue
                if et == 'file_deleted':
                    s.update(status='event', reason='file_deleted', date=dates[death], ast=1)
                    continue
                lang, why = get_language_for_file(lpath)
                before = show(last, lpath)
                after = show(death, lpath)
                if after is None:
                    new = renamed_to(repo, last, death, lpath)
                    after = show(death, new) if new else None
                a = None
                if lang is not None and before is not None and after is not None:
                    a, _, _ = measure_ast_v5.classify(before, after, int(lineno), lang)
                if a == 1:
                    s.update(status='event', reason=et, date=dates[death], ast=1)
                elif a is None:
                    s.update(status='censored', reason='next modification AST unknown', date=dates[death])
                elif et == 'line_deleted':
                    s.update(status='censored', reason='leaf-only deletion', date=dates[death])
                else:
                    s.update(last=last, path=lpath, line=lineno, death=death)
                    cont.append(s)
        active = cont
        if not active:
            break
    for s in active:
        s.update(status='censored', reason='max hops', date=dates[s['death']])
    for s in state:
        r = s['key']
        out.append(dict(slug=slug, pr_number=r.pr_number, unit_id=r.unit_id, birth_sha=r.birth_sha,
                        hops=s['hops'], final_status=s.get('status'), final_reason=s.get('reason'),
                        final_date=None if s.get('date') is None else s['date'].isoformat()))
    return pd.DataFrame(out), slug, time.time() - t0, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/survival_events_line_v6confnv_ast.csv')
    ap.add_argument('--output', default='results_v6/round1/c3_ast_continued.csv')
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--slugs', nargs='*')
    a = ap.parse_args()
    df = pd.read_csv(a.input, low_memory=False,
                     usecols=['slug', 'pr_number', 'unit_id', 'birth_sha', 'event', 'event_type', 'ast_death',
                              'last_alive_sha', 'path_at_death', 'line_at_death', 'death_sha'])
    todo = df[(df['event'] == 1) & (df['event_type'] == 'content_modified') & (df['ast_death'] == 0)]
    if a.slugs:
        todo = todo[todo['slug'].isin(a.slugs)]
    print(f'{len(todo):,} leaf-only content modifications in {todo["slug"].nunique()} repos', flush=True)
    parts = []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        fut = [ex.submit(process, s, g) for s, g in todo.groupby('slug')]
        for i, f in enumerate(as_completed(fut), 1):
            res, slug, secs, n = f.result()
            parts.append(res)
            print(f'  [{i}/{len(fut)}] {slug}: {n} lines, {secs:.0f}s (elapsed {time.time() - t0:.0f}s)', flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(a.output, index=False)
    print(out['final_status'].value_counts().to_dict(), out['final_reason'].value_counts().to_dict())
    print(out['hops'].describe())


if __name__ == '__main__':
    main()
