"""
make_ast_relabel_sheet.py

Re-labelling sheet for AST validation rows whose context was wrong.

The first version of the sheets built each row's context with
`git diff -M <last> <death> -- <path>`. A pathspec hides the target of a
rename, so 12 renamed files were shown as deleted. git_context now follows
the rename. This script rebuilds the context of exactly those rows (main
and top-up sheets; rows shown as deleted whose file was in fact renamed)
and writes a blind sheet for re-labelling. The labelled sheets are not
modified; score_ast_validation.py uses the re-labels in their place.

Output: results/ast_validation_relabel_sheet.csv (+ .xlsx)

Usage:
    python src/make_ast_relabel_sheet.py
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import make_annotation_sheet as mas
from make_ast_validation_sheet import COLUMNS, COL_WIDTHS, LABELS, REPO_DIR, git_context, write_xlsx

RES = Path(ROOT) / 'results'


def main():
    parts = []
    for sheet, key in (('ast_validation_sheet.csv', 'ast_validation_key.csv'),
                       ('ast_validation_topup_sheet.csv', 'ast_validation_topup_key.csv')):
        s = pd.read_csv(RES / sheet, encoding='utf-8-sig', keep_default_na=False)
        k = pd.read_csv(RES / key)[['id', 'slug', 'unit_id', 'birth_sha']]
        parts.append(s[s['diff_context'].str.startswith('[file deleted')].merge(k, on='id'))
    d = pd.concat(parts, ignore_index=True)
    track = pd.read_csv(os.path.join(ROOT, 'data', 'survival_events_line_v5_ast.csv'),
                        usecols=['slug', 'unit_id', 'birth_sha', 'last_alive_sha', 'line_at_death', 'path_at_death',
                                 'death_sha'], low_memory=False).drop_duplicates(['slug', 'unit_id', 'birth_sha'])
    d = d.merge(track, on=['slug', 'unit_id', 'birth_sha'], how='left', validate='one_to_one')
    keep = []
    for i, r in d.iterrows():
        header, ctx = git_context(r['slug'], r['last_alive_sha'], r['death_sha'], r['path_at_death'], r['line_at_death'])
        if ctx and not ctx.startswith('[file deleted'):
            d.at[i, 'hunk_header'], d.at[i, 'diff_context'] = header, mas.cap(ctx)
            keep.append(i)
    rows = d.loc[keep, COLUMNS].copy()
    rows['human_label'], rows['notes'] = '', ''
    mas.COLUMNS, mas.COL_WIDTHS, mas.LABELS = COLUMNS, COL_WIDTHS, LABELS
    write_xlsx(rows, RES / 'ast_validation_relabel_sheet.xlsx')
    mas.write_csv(rows, RES / 'ast_validation_relabel_sheet.csv')
    print(f'rows shown as deleted: {len(d)}; renamed, rebuilt for re-labelling: {len(rows)}')
    print(', '.join(rows['id']))


if __name__ == '__main__':
    main()
