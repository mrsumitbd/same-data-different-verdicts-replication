"""
make_ast_validation_sheet.py

Draws the sample for the manual validation of the AST-structural
modification rule (Definition 2) and writes a blind annotation sheet.

Population: modified line-level units (event == 1) with an AST status
(ast_death in {0, 1}) after the observation window, i.e. the units on
which Definition 2 decides "leaf-only" (0) or "structural" (1).

Sample size: Cochran's formula for a proportion at 95% confidence,
5% margin of error, worst case p = 0.5, with the finite-population
correction for the actual population size.

Stratification: proportional to tier (human-supervised, fully-autonomous,
human) x rule label (leaf-only, structural), largest-remainder rounding,
seed 42.

Context comes from the local clone of each repository (data/temp_repos):
the file's diff between the last commit that still contains the line and
the modifying commit (5 lines of context per hunk), with the modified line
located at its exact position there (line_at_death in the tracking
output) and marked ">>". Long diffs and deleted files are cut to the
part around the line. The hunk header gives git's enclosing function or class.
If a clone is missing, the cached diff (results/diff_cache_full_corpus.csv)
is searched for the line's text instead. No API calls are made.

Outputs:
  results/ast_validation_sheet.xlsx (+ .csv twin)   blind sheet for labelling
  results/ast_validation_key.csv                    rule labels, kept apart
  results/ast_validation_sample.txt                 n computation, strata, language mix

Usage:
    python src/make_ast_validation_sheet.py
"""

import argparse
import gc
import math
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.observation_window import apply_observation_window
from analyze_rq1_master import add_two_tier_column
import make_annotation_sheet as mas

SEED = 42
Z95 = 1.959963984540054
LABELS = ['leaf-only', 'structural', 'unclear']
CONTEXT = 5
COLUMNS = ['id', 'repo', 'file', 'language', 'tier', 'birth_commit_url', 'modifying_commit_url',
           'line_before', 'line_after', 'enclosing_node', 'hunk_header', 'diff_context',
           'human_label', 'notes']
COL_WIDTHS = [8, 26, 36, 11, 16, 30, 30, 45, 45, 18, 30, 80, 14, 30]
LANG = {'py': 'Python', 'ts': 'TypeScript', 'tsx': 'TypeScript', 'js': 'JavaScript', 'jsx': 'JavaScript',
        'mjs': 'JavaScript', 'rs': 'Rust', 'go': 'Go', 'cs': 'C#', 'rb': 'Ruby', 'php': 'PHP', 'c': 'C',
        'h': 'C', 'cpp': 'C++', 'cc': 'C++', 'hpp': 'C++', 'kt': 'Kotlin', 'java': 'Java',
        'swift': 'Swift', 'scala': 'Scala', 'sh': 'Shell', 'lua': 'Lua'}


def sample_size(N: int, e: float = 0.05) -> tuple:
    n0 = Z95 ** 2 * 0.25 / e ** 2
    return n0, n0 / (1 + (n0 - 1) / N), math.ceil(n0 / (1 + (n0 - 1) / N))


def allocate(sizes: pd.Series, n: int) -> pd.Series:
    """Largest-remainder proportional allocation summing to n."""
    exact = sizes / sizes.sum() * n
    alloc = np.floor(exact).astype(int)
    for k in (exact - alloc).sort_values(ascending=False).index[: n - alloc.sum()]:
        alloc[k] += 1
    return alloc


def language(path: str) -> str:
    m = re.search(r'\.([A-Za-z0-9]+)$', path)
    return LANG.get(m.group(1).lower(), m.group(1).lower()) if m else 'other'


def file_section(diff_text: str, path: str) -> str:
    """Return the part of a cached diff that belongs to one file."""
    body = diff_text.split('--- DIFF ---', 1)[-1]
    parts = re.split(r'(?m)^--- (.+?) ---$', body)
    for i in range(1, len(parts) - 1, 2):
        if parts[i].strip() == path:
            return parts[i + 1]
    return ''


def hunk_context(section: str, before: str, after: str) -> tuple:
    """Find the hunk that removes `before` (or adds `after`); return (header, context)."""
    hunks = re.split(r'(?m)^(@@[^\n]*@@[^\n]*)$', section)
    b, a = str(before).strip(), str(after).strip()
    for i in range(1, len(hunks) - 1, 2):
        lines = hunks[i + 1].strip('\n').split('\n')
        hit = next((k for k, l in enumerate(lines) if l[:1] == '-' and b and l[1:].strip() == b), None)
        if hit is None:
            hit = next((k for k, l in enumerate(lines) if l[:1] == '+' and a and l[1:].strip() == a), None)
        if hit is not None:
            lo, hi = max(0, hit - CONTEXT), min(len(lines), hit + CONTEXT + 1)
            header = re.sub(r'^@@[^@]*@@\s*', '', hunks[i]).strip()
            return header, '\n'.join(lines[lo:hi])
    return '', ''


REPO_DIR = os.path.join(ROOT, 'data', 'temp_repos')
FULL_DIFF_CHARS = 12000
HUNK_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@(.*)$')


def git_context(slug: str, last: str, death: str, path: str, lineno: int) -> tuple:
    """(header, context) for the line at `lineno` of `path` in `last`, as changed by `death`.

    The context is this file's whole diff in the modifying commit (5 lines of context per
    hunk), with the modified line marked ">>", so that changes elsewhere in the enclosing
    unit are visible. Long diffs (and deleted files) are cut to the hunk that holds the line,
    or to FULL_DIFF_CHARS around it.
    """
    repo = os.path.join(REPO_DIR, slug.replace('/', '__') + '.git')
    if not os.path.isdir(repo) or not isinstance(last, str) or pd.isna(lineno):
        return '', ''
    out = subprocess.run(['git', '-C', repo, 'diff', '-M', f'-U{CONTEXT}', '--no-color', '--no-ext-diff',
                          last, death, '--', path], capture_output=True, text=True, errors='replace').stdout
    if not out:
        return '', ''
    deleted = bool(re.search(r'(?m)^deleted file mode', out))
    if deleted:
        # a pathspec hides the target of a rename, so the diff shows a deletion; look for the new path
        status = subprocess.run(['git', '-C', repo, 'diff', '-M', '--name-status', last, death],
                                capture_output=True, text=True, errors='replace').stdout
        new = [l.split('\t')[2] for l in status.splitlines()
               if l.startswith('R') and l.split('\t')[1:2] == [path]]
        if new:
            out = subprocess.run(['git', '-C', repo, 'diff', '-M', f'-U{CONTEXT}', '--no-color', '--no-ext-diff',
                                  last, death, '--', path, new[0]], capture_output=True, text=True,
                                 errors='replace').stdout
            deleted = False
    lines, lineno = out.split('\n'), int(lineno)
    hunks, target = [], None
    for i, l in enumerate(lines):
        m = HUNK_RE.match(l)
        if not m:
            continue
        body, k = [], i + 1
        while k < len(lines) and not lines[k].startswith(('@@', 'diff --git')):
            body.append(lines[k])
            k += 1
        cur, marked = int(m.group(1)), []
        for b in body:
            hit = b[:1] in ('-', ' ') and cur == lineno
            if hit:
                target = (len(hunks), len(marked), m.group(3).strip())
            marked.append(('>> ' if hit else '   ') + b)
            if b[:1] in ('-', ' '):
                cur += 1
        hunks.append((l, marked))
    if target is None:
        return '', ''
    h, t, header = target
    full = '\n'.join(hh + '\n' + '\n'.join(body) for hh, body in hunks)
    if deleted or len(full) > FULL_DIFF_CHARS:
        body = hunks[h][1]
        lo, hi = max(0, t - 15), min(len(body), t + 16)
        note = '[file deleted in the modifying commit]' if deleted else '[long diff: only the part around the line is shown; see modifying_commit_url]'
        return header, note + '\n' + hunks[h][0] + '\n' + '\n'.join(body[lo:hi])
    return header, full


def run(line_data: str, track_data: str, diffs: str, out_dir: str) -> None:
    d = add_two_tier_column(apply_observation_window(pd.read_csv(line_data, low_memory=False), label='line'))
    pop = d[(d['event'] == 1) & d['ast_death'].notna()].copy()
    pop['rule_label'] = np.where(pop['ast_death'] == 1, 'structural', 'leaf-only')
    pop['file'] = pop['unit_id'].str.split('::').str[0]
    pop['language'] = pop['file'].map(language)

    N = len(pop)
    n0, n_exact, n = sample_size(N)
    strata = pop.groupby(['two_tier', 'rule_label']).size()
    alloc = allocate(strata, n)
    parts = [pop[(pop['two_tier'] == t) & (pop['rule_label'] == l)].sample(k, random_state=SEED)
             for (t, l), k in alloc.items()]
    s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    s['id'] = [f'A{i + 1:03d}' for i in range(len(s))]
    lang_pop = pop['language'].value_counts(normalize=True) * 100
    del d, pop  # free memory before starting git subprocesses
    gc.collect()
    pos = pd.read_csv(track_data, usecols=['slug', 'unit_id', 'birth_sha', 'last_alive_sha', 'line_at_death'],
                      low_memory=False).drop_duplicates(['slug', 'unit_id', 'birth_sha'])
    # the tracking output repeats a few units; the analysis table (line_data) has none
    s = s.merge(pos, on=['slug', 'unit_id', 'birth_sha'], how='left', validate='one_to_one')

    cache = None
    headers, contexts, n_exact = [], [], 0
    for r in s.itertuples():
        header, ctx = git_context(r.slug, r.last_alive_sha, r.death_sha, r.path_at_death, r.line_at_death)
        if r.Index % 50 == 0:
            print(f'context {r.Index}/{len(s)}', flush=True)
        if ctx:
            n_exact += 1
            headers.append(header)
            contexts.append(ctx)
            continue
        if cache is None:  # only needed when a clone is missing
            cache = pd.read_csv(diffs).drop_duplicates(['slug', 'sha']).set_index(['slug', 'sha'])['diff_text']
        diff_text = cache.get((r.slug, r.death_sha), '')
        section = file_section(str(diff_text), r.file)
        header, ctx = hunk_context(section, r.birth_content, r.death_content)
        if not ctx and section:
            # line not matched verbatim: show the whole change to this file
            ctx = '[line not matched; full change to this file]\n' + section.strip()[:4000]
        elif not ctx:
            ctx = '[file not in the cached diff (large commit); see modifying_commit_url]'
        headers.append(header)
        contexts.append(ctx)
    tier_name = {'human_supervised': 'human-supervised', 'fully_autonomous': 'fully-autonomous', 'human': 'human'}
    rows = pd.DataFrame({
        'id': s['id'], 'repo': s['slug'], 'file': s['file'], 'language': s['language'],
        'tier': s['two_tier'].map(tier_name),
        'birth_commit_url': 'https://github.com/' + s['slug'] + '/commit/' + s['birth_sha'],
        'modifying_commit_url': 'https://github.com/' + s['slug'] + '/commit/' + s['death_sha'],
        'line_before': s['birth_content'].fillna('[empty line]').map(mas.cap),
        'line_after': s['death_content'].fillna('[line removed or empty]').map(mas.cap),
        'enclosing_node': s['ast_enclosing_node'].fillna(''),
        'hunk_header': headers, 'diff_context': [mas.cap(c) for c in contexts],
        'human_label': '', 'notes': '',
    })[COLUMNS]

    out = Path(out_dir)
    mas.COLUMNS, mas.COL_WIDTHS, mas.LABELS = COLUMNS, COL_WIDTHS, LABELS
    write_xlsx(rows, out / 'ast_validation_sheet.xlsx')
    mas.write_csv(rows, out / 'ast_validation_sheet.csv')
    s[['id', 'slug', 'unit_id', 'birth_sha', 'death_sha', 'two_tier', 'rule_label']].to_csv(
        out / 'ast_validation_key.csv', index=False)

    lang_s = s['language'].value_counts(normalize=True) * 100
    missing = [l for l, v in lang_pop.items() if v >= 5 and l not in lang_s.index]
    lines = ['=' * 70, 'AST validation sample', '=' * 70,
             f'Population: modified line units with an AST status, window applied: N = {N:,}',
             f'Cochran (95%, e = 0.05, p = 0.5): n0 = {n0:.2f}; with finite-population correction n = {n_exact:.2f} -> {n}',
             '', 'Strata (tier x rule label): population -> sample']
    for (t, l), k in alloc.items():
        lines.append(f'  {t:17s} {l:10s} {strata[(t, l)]:6,d} ({strata[(t, l)] / N * 100:5.1f}%) -> {k}')
    lines.append(f'  total {alloc.sum()}')
    lines += ['', 'Language mix (% of population vs % of sample):']
    for l in lang_pop.index[:12]:
        lines.append(f'  {l:12s} {lang_pop[l]:5.1f}%  {lang_s.get(l, 0):5.1f}%')
    lines.append(f'Languages with >= 5% of the population missing from the sample: {missing or "none"}')
    lines.append(f'Context: exact position {n_exact}, of which deleted files {sum(1 for c in contexts if c.startswith("[file deleted"))} and long diffs cut {sum(1 for c in contexts if c.startswith("[long diff"))}; '
                 f'file shown in full {sum(1 for c in contexts if c.startswith("[line not matched"))}, '
                 f'file absent from cached diff {sum(1 for c in contexts if c.startswith("[file not in"))}')
    text = '\n'.join(lines) + '\n'
    (out / 'ast_validation_sample.txt').write_text(text)
    print(text)


def write_xlsx(rows: pd.DataFrame, path: Path) -> None:
    """Reuse the intent sheet writer with this sheet's columns and labels."""
    mas.write_xlsx(rows, path)
    # the shared writer hard-codes the intent error message
    with zipfile.ZipFile(path) as z:
        files = {n: z.read(n).decode('utf-8') for n in z.namelist()}
    files['xl/worksheets/sheet1.xml'] = files['xl/worksheets/sheet1.xml'].replace(
        'Choose corrective, perfective, adaptive or preventive.', 'Choose leaf-only, structural or unclear.')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for n, c in files.items():
            z.writestr(n, c)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--line-data', default='data/survival_events_line_v5_analysis.csv')
    parser.add_argument('--track-data', default='data/survival_events_line_v5_ast.csv')
    parser.add_argument('--diffs', default='results/diff_cache_full_corpus.csv')
    parser.add_argument('--out-dir', default='results')
    args = parser.parse_args()
    run(args.line_data, args.track_data, args.diffs, args.out_dir)
