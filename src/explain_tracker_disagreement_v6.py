"""
explain_tracker_disagreement_v6.py

Why do lines the conference tracker calls modified survive under the identity
tracker? For a random sample of such lines (conf_kind == 'died', identity
event == 0), re-run `git blame --reverse <birth>..<end>` on the line, adding the
identity tracker's options one at a time, and record the first that makes the
line reach the end snapshot:

  same end snapshot?   conference cutoff commit == identity end commit
  plain                the conference call (no options); should not survive
  --first-parent       line only changed on a side branch / merge parent
  -M                   line moved within the file
  -C                   line moved or copied from another file (incl. renames)
  none                 survives only with all options plus the identity
                       tracker's integration/rename handling (or not at all)

Also per line: whether the identity tracker reports a different path at the end
(file renamed/moved), and the conference death commit's diff for the file
(whitespace-only change of the line, file touched without changing the line).

Output: results_v6/tracker_disagreement_breakdown.txt and a per-line CSV.

Usage:
    python src/explain_tracker_disagreement_v6.py [--n 1000]
"""

import argparse
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from measure_retention_line_v3 import REPO_DIR, git

SEED = 42
STEPS = [('plain', []), ('--first-parent', ['--first-parent']), ('-M', ['--first-parent', '-M']),
         ('-C', ['--first-parent', '-M', '-C'])]


def last_seen(repo, birth, end, path, line, opts):
    out = git(repo, 'blame', '--reverse', *opts, '--porcelain', '-L', f'{line},{line}',
              f'{birth}..{end}', '--', path, check=False)
    m = re.match(r'^([0-9a-f]{40}) \d+ (\d+)', out)
    return m.group(1) if m else None


def conf_death_touch(repo, death, path, content):
    """How the conference death commit changes the file: line changed / whitespace only / not in diff."""
    if not isinstance(death, str):
        return 'no death commit'
    out = git(repo, 'show', '--format=', '-U0', '--first-parent', '-m', death, '--', path, check=False)
    if not out.strip():
        return 'file not changed by death commit (vs first parent)'
    c = content.strip() if isinstance(content, str) else ''
    removed = [l[1:] for l in out.splitlines() if l.startswith('-') and not l.startswith('---')]
    added = [l[1:] for l in out.splitlines() if l.startswith('+') and not l.startswith('+++')]
    if c and any(r.strip() == c for r in removed):
        if any(re.sub(r'\s', '', a) == re.sub(r'\s', '', c) for a in added):
            return 'line removed and re-added (whitespace/move within diff)'
        return 'line removed by death commit'
    return 'file touched, line not removed'


def explain(args):
    slug, rows = args
    repo = os.path.join(REPO_DIR, slug.replace('/', '__') + '.git')
    res = []
    for r in rows.itertuples():
        conf_end = r.conf_cutoff
        rec = {'slug': slug, 'pr_number': r.pr_number, 'unit_id': r.unit_id, 'vendored': r.vendored,
               'same_end': r.end_sha == conf_end, 'renamed_at_end': r.path_at_death != r.path,
               'birth_on_end_chain': None}
        first = 'none'
        for name, opts in STEPS:
            ls = last_seen(repo, r.birth_sha, r.end_sha, r.path, int(r.line), opts)
            rec[f'ls_{name}'] = ls
            if ls == r.end_sha and first == 'none':
                first = name
        rec['first_option_surviving'] = first
        rec['conf_death_commit'] = conf_death_touch(repo, r.conf_death_sha, r.path, r.birth_content)
        res.append(rec)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partial', default=os.path.join(ROOT, 'data', 'survival_events_line_v6conf_partial'))
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--output', default=os.path.join(ROOT, 'results_v6', 'tracker_disagreement_breakdown.txt'))
    a = ap.parse_args()
    cols = ['slug', 'pr_number', 'unit_id', 'path', 'line', 'birth_sha', 'birth_content', 'vendored', 'event',
            'conf_kind', 'conf_last_seen_sha', 'conf_death_sha', 'path_at_death', 'end_sha']
    files = [f for f in glob.glob(os.path.join(a.partial, '*.csv'))
             if not f.endswith(('.births.csv', '.pr_status.csv')) and 'stats' not in f and os.path.getsize(f) > 1]
    d = pd.concat([pd.read_csv(f, usecols=cols, low_memory=False) for f in files], ignore_index=True)
    # conference cutoff commit per repo = last_seen of its censored lines
    cut = d[d['conf_kind'] == 'censored'].groupby('slug')['conf_last_seen_sha'].agg(lambda s: s.mode().iat[0])
    d['conf_cutoff'] = d['slug'].map(cut)
    dis = d[(d['conf_kind'] == 'died') & (d['event'] == 0)]
    s = dis.sample(min(a.n, len(dis)), random_state=SEED)
    with ProcessPoolExecutor(6) as ex:
        parts = list(ex.map(explain, list(s.groupby('slug'))))
    r = pd.DataFrame([x for p in parts for x in p])
    reason = {'--first-parent': 'changed only off the first-parent chain (side branch)',
              '-M': 'moved within the file', '-C': 'moved/copied from another file'}
    r['reason'] = [reason.get(f) or ('file renamed/moved (identity tracker follows the rename)' if ren else 'other')
                   for f, ren in zip(r['first_option_surviving'], r['renamed_at_end'])]
    r.to_csv(a.output.replace('.txt', '.csv'), index=False)
    out = [f'Lines conference=modified, identity=censored: {len(dis):,} of {len(d):,} '
           f'({dis["vendored"].astype(bool).mean() * 100:.1f}% vendored). Sample: {len(r):,} (seed {SEED}).',
           f'\nsame end snapshot as the conference cutoff: {r["same_end"].mean() * 100:.1f}%',
           f'identity tracker reports a different path at the end (file renamed/moved): {r["renamed_at_end"].mean() * 100:.1f}%',
           '\nfirst blame --reverse option under which the line reaches the end snapshot:']
    out.append((r['first_option_surviving'].value_counts(normalize=True) * 100).round(1).to_string())
    out.append('\nreason (first option that makes the line survive; else rename at end):')
    out.append((r['reason'].value_counts(normalize=True) * 100).round(1).to_string())
    out.append('\nconference death commit vs the line (first-parent diff of the file):')
    out.append((r['conf_death_commit'].value_counts(normalize=True) * 100).round(1).to_string())
    out.append('\ncross-tab (rows first option, columns death-commit check), %:')
    out.append((pd.crosstab(r['first_option_surviving'], r['conf_death_commit'], normalize='all') * 100).round(1).to_string())
    for v in (False, True):
        x = r[r['vendored'].astype(bool) == v]
        out.append(f'\nvendored={v} (n={len(x)}): ' + ', '.join(
            f'{k} {p:.1f}%' for k, p in (x['first_option_surviving'].value_counts(normalize=True) * 100).items()))
    text = '\n'.join(out)
    open(a.output, 'w').write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
