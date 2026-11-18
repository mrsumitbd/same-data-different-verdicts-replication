"""
measure_ast_v5.py

Definition 2 (AST-structural) on the v5 retention data.

For each verified modification, the line's enclosing named AST node is
compared just before and just after the modifying commit:
  before = the file at the last commit in which the line existed
           (last_alive_sha, path_at_death, line_at_death)
  after  = the same file at the modifying commit (death_sha), following a
           rename made by that commit.
The classification rule is unchanged (measure_ast_survival.classify_ast_death):
ast_death = 1 if the node's ordered structural signature changes or the node
disappears, 0 if only leaf tokens change, null if it cannot be classified.

The v4 step compared the file at the birth commit with the file at the death
commit, which mixes in every intermediate change and ignores renames.

event_type handling:
  content_modified, line_deleted -> classified as above
  file_deleted                   -> ast_death = 1 (the enclosing node is gone)
  censored, unverified           -> ast_death = null

Usage:
    python src/measure_ast_v5.py
Input / output:
    data/survival_events_line_v5.csv -> data/survival_events_line_v5_ast.csv
--reuse <earlier output>: repositories present there are not re-classified;
their AST columns are copied by (slug, pr_number, unit_id, birth_sha). Use only
when those repositories' tracking is unchanged.
"""

import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import measure_ast_survival
from measure_ast_survival import classify_ast_death, get_language_for_file
from measure_retention_line_v3 import git, REPO_DIR

WORKERS = 6
CHUNK = 5000   # rows per task; large repositories are split so they use several workers


class _CachingParser:
    """tree-sitter parser whose parse() reuses the tree for identical file content
    (trees are only read, so results are unchanged). Rows of one file version
    otherwise re-parse the whole file once per line."""
    def __init__(self, parser):
        self.parser, self.cache = parser, {}

    def parse(self, data):
        if data not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[data] = self.parser.parse(data)
        return self.cache[data]


_parsers = {}
_orig_get_parser = measure_ast_survival.get_parser


def _get_parser(language):
    if language not in _parsers:
        _parsers[language] = _CachingParser(_orig_get_parser(language))
    return _parsers[language]


measure_ast_survival.get_parser = _get_parser
_memo = {}


def classify(before: str, after: str, line: int, lang: str):
    """classify_ast_death, memoized on (file before, file after, enclosing node of the line):
    the rule depends only on those, so all lines of one node in one file version share
    the result (a bundled file otherwise repeats the whole-tree comparison per line)."""
    try:
        tree = _get_parser(lang).parse(before.encode('utf-8'))
        node, _ = measure_ast_survival.find_enclosing_node(tree, line, lang)
    except Exception:
        node = None
    if node is None:
        return classify_ast_death(before, after, line, lang)
    k = (hash(before), hash(after), lang, node.start_byte, node.end_byte, node.type)
    if k not in _memo:
        if len(_memo) > 200000:
            _memo.clear()
        _memo[k] = classify_ast_death(before, after, line, lang)
    return _memo[k]


def renamed_to(repo: str, parent: str, child: str, path: str):
    out = git(repo, 'diff', '--name-status', '-M', parent, child, check=False)
    for line in out.splitlines():
        parts = line.split('\t')
        if parts[0].startswith('R') and len(parts) == 3 and parts[1] == path:
            return parts[2]
    return None


def process(slug: str, rows: pd.DataFrame) -> pd.DataFrame:
    repo = os.path.join(REPO_DIR, slug.replace('/', '__') + '.git')
    cache, out = {}, []

    def show(sha, path):
        k = (sha, path)
        if k not in cache:
            r = git(repo, 'show', f'{sha}:{path}', check=False)
            cache[k] = r if r else None
        return cache[k]

    for r in rows.itertuples():
        res = {'idx': r.Index, 'ast_death': None, 'ast_enclosing_node': None, 'ast_exclusion_reason': None}
        if r.event_type == 'file_deleted':
            res['ast_death'] = 1
        else:
            lang, why = get_language_for_file(r.path_at_death)
            if lang is None:
                res['ast_exclusion_reason'] = why
            else:
                before = show(r.last_alive_sha, r.path_at_death)
                after = show(r.death_sha, r.path_at_death)
                if after is None:
                    new = renamed_to(repo, r.last_alive_sha, r.death_sha, r.path_at_death)
                    after = show(r.death_sha, new) if new else None
                if before is None:
                    res['ast_exclusion_reason'] = 'fetch_failed_birth'
                elif after is None:
                    res['ast_exclusion_reason'] = 'fetch_failed_death'
                else:
                    a, node, why = classify(before, after, int(r.line_at_death), lang)
                    res.update(ast_death=a, ast_enclosing_node=node, ast_exclusion_reason=why)
        out.append(res)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/survival_events_line_v5.csv')
    ap.add_argument('--output', default='data/survival_events_line_v5_ast.csv')
    ap.add_argument('--reuse', default=None)
    args = ap.parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    todo = df[df['event_type'].isin(['content_modified', 'line_deleted', 'file_deleted'])]
    AST = ['ast_death', 'ast_enclosing_node', 'ast_exclusion_reason']
    KEY = ['slug', 'pr_number', 'unit_id', 'birth_sha']
    prev = None
    if args.reuse:
        prev = pd.read_csv(args.reuse, usecols=KEY + AST, low_memory=False).drop_duplicates(KEY)
        todo = todo[~todo['slug'].isin(set(prev['slug']))]
        print(f'reusing AST status for {prev["slug"].nunique()} repos from {args.reuse}', flush=True)
    print(f'units {len(df):,}; modifications to classify {len(todo):,}', flush=True)
    parts = []
    with ProcessPoolExecutor(WORKERS) as ex:
        tasks = []
        for s, g in todo.groupby('slug'):
            g = g.sort_values(['last_alive_sha', 'path_at_death', 'death_sha'])
            tasks += [(s, g.iloc[i:i + CHUNK]) for i in range(0, len(g), CHUNK)]
        fut = {ex.submit(process, s, g): s for s, g in tasks}
        for i, f in enumerate(as_completed(fut), 1):
            parts.append(f.result())
            if i % 10 == 0:
                print(f'  {i}/{len(fut)} tasks', flush=True)
    res = pd.concat(parts).set_index('idx') if parts else pd.DataFrame(columns=AST)
    for c in AST:
        df[c] = res[c].reindex(df.index)
    if prev is not None:
        old = df[KEY].merge(prev, on=KEY, how='left')
        old.index = df.index
        mask = df['slug'].isin(set(prev['slug']))
        for c in AST:
            df.loc[mask, c] = old.loc[mask, c]
    df.to_csv(args.output, index=False)
    ev = df[df['event'] == 1]
    print('ast_death among modifications:', ev['ast_death'].value_counts(dropna=False).to_dict())
    print('exclusion reasons:', ev['ast_exclusion_reason'].value_counts().head(10).to_dict())
    print('wrote', args.output)


if __name__ == '__main__':
    main()
