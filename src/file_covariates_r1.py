"""
file_covariates_r1.py 

File-level covariates at merge for every (repository, birth commit, file) of the
primary line data, from the local bare clones (data/temp_repos):
  file_age_days      days from the first first-parent commit that touched the
                     path to the birth (merge) commit; 0 if the PR created it
  commits_90d        first-parent commits touching the path in the 90 days
                     before the birth commit (birth commit excluded)
  created_by_pr      the path does not exist in the birth commit's first parent
  test_file          path looks like a test (test/tests/__tests__/spec dirs,
                     test_*.py, *_test.*, *.test.*, *.spec.*, *Test.java ...)
History is the first-parent chain up to the end snapshot (as the tracker),
paths without rename following.

Output: results_v6/round1/work/file_covariates.csv
"""

import os
import re
import subprocess
import sys
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
from measure_retention_line_v3 import end_snapshot, REPO_DIR

TEST_RE = re.compile(r'(^|/)(tests?|__tests__|spec|specs|testing|e2e)/|(^|/)test_[^/]*$|_test\.[^/]+$|'
                     r'\.(test|spec)\.[^/]+$|(Test|Tests|Spec)\.(java|kt|cs|swift|scala|m|mm)$|_spec\.rb$',
                     re.I)


def git(repo, *a):
    return subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True, errors='replace').stdout


def process(slug, items):
    repo = os.path.join(REPO_DIR, slug.replace('/', '__') + '.git')
    end = end_snapshot(repo)
    touches, t = {}, None
    for line in git(repo, 'log', '--first-parent', '--no-renames', '--format=\x01%ct', '--name-only', end).splitlines():
        if line.startswith('\x01'):
            t = int(line[1:])
        elif line:
            touches.setdefault(line, []).append(t)
    for v in touches.values():
        v.sort()
    out, trees = [], {}
    for birth, path in items:
        bt = int(git(repo, 'show', '-s', '--format=%ct', birth).strip() or 0)
        if birth not in trees:
            trees[birth] = set(git(repo, 'ls-tree', '-r', '--name-only', f'{birth}^1').splitlines())
        created = path not in trees[birth]
        ts = touches.get(path, [])
        before = ts[:bisect_left(ts, bt)]
        age = 0.0 if created or not before else (bt - before[0]) / 86400
        n90 = sum(1 for x in before if x >= bt - 90 * 86400)
        out.append(dict(slug=slug, birth_sha=birth, path=path, file_age_days=age, commits_90d=n90,
                        created_by_pr=int(created), test_file=int(bool(TEST_RE.search(path)))))
    return out


def main():
    d = pd.read_csv('data/survival_events_line_v6confnv_analysis.csv', usecols=['slug', 'birth_sha', 'unit_id'],
                    low_memory=False)
    d['path'] = d['unit_id'].str.rsplit('::', n=1).str[0]
    g = d.drop_duplicates(['slug', 'birth_sha', 'path'])
    rows = []
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        fut = [ex.submit(process, s, list(zip(x['birth_sha'], x['path']))) for s, x in g.groupby('slug')]
        for i, f in enumerate(as_completed(fut), 1):
            rows += f.result()
            if i % 20 == 0:
                print(f'{i}/{len(fut)} repos', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv('results_v6/round1/work/file_covariates.csv', index=False)
    print(out.describe().T)


if __name__ == '__main__':
    main()
