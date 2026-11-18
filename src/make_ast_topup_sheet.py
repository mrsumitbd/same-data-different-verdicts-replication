"""
make_ast_topup_sheet.py

Top-up sample for the AST validation (Definition 2), to estimate how often
the rule's leaf-only label is right, overall and per tier.

The main sample (make_ast_validation_sheet.py) is proportional to tier x
rule label, so it holds only a few leaf-only rows. This script adds
leaf-only rows per tier until each tier has LEAF_PER_TIER leaf-only rows
across both samples, plus DECOYS_PER_TIER structural rows per tier so
that the annotator cannot infer the rule's label from the batch. Units
already in the main sample are excluded. Rows are shuffled (seed 42).

Population-level accuracy and kappa come from the main sample only; the
top-up rows are used for the per-label (leaf-only) precision.

Outputs:
  results/ast_validation_topup_sheet.xlsx (+ .csv twin)   blind sheet for labelling
  results/ast_validation_topup_key.csv                    rule labels, kept apart
  results/ast_validation_topup.txt                        counts

Usage:
    python src/make_ast_topup_sheet.py
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.observation_window import apply_observation_window
from analyze_rq1_master import add_two_tier_column
import make_annotation_sheet as mas
from make_ast_validation_sheet import COLUMNS, COL_WIDTHS, LABELS, SEED, git_context, language, write_xlsx

LEAF_PER_TIER = 32
DECOYS_PER_TIER = 10
TIERS = ['human_supervised', 'fully_autonomous', 'human']
TIER_NAME = {'human_supervised': 'human-supervised', 'fully_autonomous': 'fully-autonomous', 'human': 'human'}


def run(line_data: str, track_data: str, main_key: str, out_dir: str) -> None:
    d = add_two_tier_column(apply_observation_window(pd.read_csv(line_data, low_memory=False), label='line'))
    pop = d[(d['event'] == 1) & d['ast_death'].notna()].copy()
    del d
    gc.collect()
    pop['rule_label'] = np.where(pop['ast_death'] == 1, 'structural', 'leaf-only')
    pop['file'] = pop['unit_id'].str.split('::').str[0]
    pop['language'] = pop['file'].map(language)

    main = pd.read_csv(main_key)
    taken = set(zip(main['slug'], main['unit_id'], main['birth_sha']))
    pop = pop[[k not in taken for k in zip(pop['slug'], pop['unit_id'], pop['birth_sha'])]]

    have = main[main['rule_label'] == 'leaf-only'].groupby('two_tier').size()
    plan, parts = [], []
    for t in TIERS:
        k_leaf = max(0, LEAF_PER_TIER - int(have.get(t, 0)))
        for label, k in (('leaf-only', k_leaf), ('structural', DECOYS_PER_TIER)):
            g = pop[(pop['two_tier'] == t) & (pop['rule_label'] == label)]
            k = min(k, len(g))
            parts.append(g.sample(k, random_state=SEED))
            plan.append((t, label, len(g), k))
    s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    del pop
    gc.collect()
    s['id'] = [f'T{i + 1:03d}' for i in range(len(s))]
    pos = pd.read_csv(track_data, usecols=['slug', 'unit_id', 'birth_sha', 'last_alive_sha', 'line_at_death'],
                      low_memory=False).drop_duplicates(['slug', 'unit_id', 'birth_sha'])
    s = s.merge(pos, on=['slug', 'unit_id', 'birth_sha'], how='left', validate='one_to_one')

    headers, contexts = [], []
    for r in s.itertuples():
        header, ctx = git_context(r.slug, r.last_alive_sha, r.death_sha, r.path_at_death, r.line_at_death)
        headers.append(header)
        contexts.append(ctx or '[context not found; see modifying_commit_url]')
    rows = pd.DataFrame({
        'id': s['id'], 'repo': s['slug'], 'file': s['file'], 'language': s['language'],
        'tier': s['two_tier'].map(TIER_NAME),
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
    write_xlsx(rows, out / 'ast_validation_topup_sheet.xlsx')
    mas.write_csv(rows, out / 'ast_validation_topup_sheet.csv')
    s[['id', 'slug', 'unit_id', 'birth_sha', 'death_sha', 'two_tier', 'rule_label']].to_csv(
        out / 'ast_validation_topup_key.csv', index=False)

    lines = ['=' * 70, 'AST validation top-up sample (leaf-only precision)', '=' * 70,
             f'Target: {LEAF_PER_TIER} leaf-only rows per tier across main + top-up; '
             f'{DECOYS_PER_TIER} structural decoys per tier; main-sample units excluded; seed {SEED}',
             '', 'tier x rule label: available (not in main sample) -> drawn']
    for t, label, avail, k in plan:
        lines.append(f'  {t:17s} {label:10s} {avail:7,d} -> {k}')
    lines.append(f'  total {len(s)}')
    lines.append(f'Context: found {sum(1 for c in contexts if not c.startswith("[context not found"))} of {len(s)}; '
                 f'deleted files {sum(1 for c in contexts if c.startswith("[file deleted"))}; '
                 f'long diffs cut {sum(1 for c in contexts if c.startswith("[long diff"))}')
    text = '\n'.join(lines) + '\n'
    (out / 'ast_validation_topup.txt').write_text(text)
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--line-data', default='data/survival_events_line_v5_analysis.csv')
    parser.add_argument('--track-data', default='data/survival_events_line_v5_ast.csv')
    parser.add_argument('--main-key', default='results/ast_validation_key.csv')
    parser.add_argument('--out-dir', default='results')
    args = parser.parse_args()
    run(args.line_data, args.track_data, args.main_key, args.out_dir)
