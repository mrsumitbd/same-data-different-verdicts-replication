"""
make_second_rater_sheets.py

Builds the second-rater packages for the manual annotation tasks of the journal
paper. For each task it writes, to results_v6/second_rater/<task>/:

  <task>_sheet.xlsx         blank sheet for the second rater (sheet `sample`
                            with a drop-down for the label, sheet `labels` with
                            the allowed labels)
  <task>_KEY_INTERNAL.csv   rater-1 label and sampling stratum per row
                            (never sent to the second rater)
  <task>_guide.md           copied from docs/second_rater/
  evidence/                 tracker task only: evidence pages of the sampled lines

Sampling rule (all tasks): at most 100 items, stratified by the rater-1 label
(and by tier where the task has tiers), taking the smallest, most
straightforward items of each stratum (shortest diff, shortest history), with
ties broken by a seeded shuffle. Rows are then shuffled (seed 42) and given new
ids, so neither the order nor the id reveals the stratum.

Tasks:
  ast            leaf-only vs structural (483 labeled modifications)
  intent         primary intent of modification commits (618 labeled commits)
  tracker        tracker ground truth (300 labeled lines; lines moved to
                 another file are left out because they need a second history)
  selfmod        agent self-modifications (all 40 labeled commits)
  human_baseline 100 of the 200 human-baseline PRs, labeled blind by both raters

Usage: python src/make_second_rater_sheets.py [--tasks ast intent ...]
Scored by src/score_second_rater.py.
"""

import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results_v6' / 'second_rater'
GUIDES = ROOT / 'docs' / 'second_rater'
SEED = 42
N = 100


def smallest(df, n, size_col, seed=SEED):
    """n rows with the smallest size_col; ties broken by a seeded shuffle."""
    d = df.sample(frac=1, random_state=seed)
    return d.sort_values(size_col, kind='stable').head(n)


def allocate(counts: dict, total: int) -> dict:
    """Split total across cells as evenly as their sizes allow."""
    alloc = {k: 0 for k in counts}
    left = total
    while left > 0:
        open_cells = [k for k in counts if alloc[k] < counts[k]]
        if not open_cells:
            break
        for k in sorted(open_cells, key=lambda k: (alloc[k], counts[k])):
            if left == 0:
                break
            alloc[k] += 1
            left -= 1
    return alloc


def stratified(df, strata, n_per_group: dict, group_col, size_col):
    """Within each value of group_col, spread its quota evenly over strata."""
    parts = []
    for g, quota in n_per_group.items():
        sub = df[df[group_col] == g]
        counts = sub.groupby(strata).size().to_dict()
        for cell, k in allocate(counts, quota).items():
            cell = cell if isinstance(cell, tuple) else (cell,)
            mask = np.ones(len(sub), bool)
            for col, v in zip(strata, cell):
                mask &= (sub[col] == v).to_numpy()
            parts.append(smallest(sub[mask], k, size_col))
    return pd.concat(parts)


def finish(sample, prefix):
    s = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    s.insert(0, 'rater2_id', [f'{prefix}-{i:03d}' for i in range(1, len(s) + 1)])
    return s


def clean(v):
    if pd.isna(v):
        return ''
    v = str(v)
    if v[:1] in ('=', '+', '-', '@'):
        v = ' ' + v
    return v[:32000]


def write_sheet(path, df, label_col, labels, widths=None):
    wb = Workbook()
    ws = wb.active
    ws.title = 'sample'
    cols = list(df.columns)
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in df.itertuples(index=False):
        ws.append([clean(v) for v in row])
    li = cols.index(label_col) + 1
    letter = ws.cell(row=1, column=li).column_letter
    dv = DataValidation(type='list', formula1='"' + ','.join(labels) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f'{letter}2:{letter}{len(df) + 1}')
    for i, col in enumerate(cols, 1):
        w = (widths or {}).get(col, 18)
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w
    for r in ws.iter_rows(min_row=2):
        for c in r:
            c.alignment = Alignment(wrap_text=True, vertical='top')
    ws.freeze_panes = 'B2'
    lab = wb.create_sheet('labels')
    lab.append(['allowed labels'])
    for x in labels:
        lab.append([x])
    wb.save(path)


def out_dir(task):
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    g = GUIDES / f'{task}_guide.md'
    if g.exists():
        shutil.copy(g, d / f'{task}_guide.md')
    return d


# ---------------------------------------------------------------- AST
def task_ast():
    res = ROOT / 'results'
    parts = []
    for sheet, key in [('ast_validation_sheet.csv', 'ast_validation_key.csv'),
                       ('ast_validation_topup_sheet.csv', 'ast_validation_topup_key.csv')]:
        s = pd.read_csv(res / sheet, encoding='utf-8-sig', keep_default_na=False)
        k = pd.read_csv(res / key)
        parts.append(s.merge(k[['id', 'two_tier', 'rule_label']], on='id', validate='one_to_one'))
    d = pd.concat(parts, ignore_index=True)
    rl = pd.read_csv(res / 'ast_validation_relabel_sheet.csv', encoding='utf-8-sig', keep_default_na=False)
    for r in rl.itertuples():   # final labels: the re-labeled rows on corrected context
        i = d.index[d['id'] == r.id]
        if len(i) and str(r.human_label).strip():
            d.loc[i[0], ['human_label', 'diff_context']] = [r.human_label, r.diff_context]
    d['rater1_label'] = d['human_label'].str.strip().str.lower()
    assert set(d['rater1_label']) == {'leaf-only', 'structural'}, set(d['rater1_label'])
    d['size'] = d['diff_context'].str.len()
    # Kappa needs both labels: half leaf-only, half structural, each spread over the tiers.
    s = stratified(d, ['two_tier'], {'leaf-only': N // 2, 'structural': N - N // 2}, 'rater1_label', 'size')
    s = finish(s, 'AST')
    o = out_dir('ast')
    cols = ['rater2_id', 'repo', 'file', 'language', 'modifying_commit_url', 'line_before', 'line_after',
            'enclosing_node', 'hunk_header', 'diff_context']
    sheet = s[cols].copy()
    sheet['label'] = ''
    sheet['notes'] = ''
    write_sheet(o / 'ast_sheet.xlsx', sheet, 'label', ['leaf-only', 'structural', 'unclear'],
                {'diff_context': 80, 'line_before': 40, 'line_after': 40, 'file': 30, 'notes': 30})
    s[['rater2_id', 'id', 'two_tier', 'rater1_label', 'rule_label', 'size']].to_csv(o / 'ast_KEY_INTERNAL.csv', index=False)
    return s, ['rater1_label', 'two_tier']


# ---------------------------------------------------------------- intent
def task_intent():
    g = pd.read_csv(ROOT / 'results_v6/labels618/gold.csv')
    smp = pd.read_csv(ROOT / 'results_v6/labels618/sample.csv')
    d = g.merge(smp[['sample_id', 'slug', 'sha', 'tier']], on='sample_id', validate='one_to_one')
    sheets = pd.concat([pd.read_excel(ROOT / p, dtype=str) for p in
                        ['results/intent_annotation_sheet.xlsx', 'results/intent_annotation_topup.xlsx',
                         'results_v6/intent-top-up-v6-labeled-sheet-v001-v256.xlsx']], ignore_index=True)
    sheets = sheets[['sample_id', 'commit_url', 'commit_message', 'files_changed', 'diff_length']]
    d = d.merge(sheets, on='sample_id', how='left', validate='one_to_one')
    assert d['commit_url'].notna().all(), d[d['commit_url'].isna()]['sample_id'].tolist()
    d['rater1_label'] = d['human_label'].str.strip().str.lower()
    d['size'] = pd.to_numeric(d['diff_length'], errors='coerce').fillna(10 ** 9)
    d = d[d['size'] > 0]   # an empty diff cannot be judged from the code
    labels = ['corrective', 'perfective', 'adaptive', 'preventive']
    assert set(d['rater1_label']) <= set(labels)
    s = stratified(d, ['tier'], {lab: N // 4 for lab in labels}, 'rater1_label', 'size')
    s = finish(s, 'INT')
    o = out_dir('intent')
    sheet = s[['rater2_id', 'commit_url', 'commit_message', 'files_changed', 'diff_length']].copy()
    sheet['label'] = ''
    sheet['notes'] = ''
    write_sheet(o / 'intent_sheet.xlsx', sheet, 'label', labels,
                {'commit_url': 40, 'commit_message': 60, 'files_changed': 50, 'notes': 30})
    s[['rater2_id', 'sample_id', 'set', 'slug', 'sha', 'tier', 'rater1_label', 'size']].to_csv(
        o / 'intent_KEY_INTERNAL.csv', index=False)
    return s, ['rater1_label', 'tier']


# ---------------------------------------------------------------- tracker
def task_tracker():
    D = ROOT / 'results_v6/round1/tracker_gt'
    lab = pd.read_excel(D / 'tracker_groundtruth_sheet_labelled.xlsx', dtype=str)
    key = pd.read_csv(D / 'tracker_groundtruth_key.csv', dtype=str)
    d = lab.merge(key[['line_id', 'cell', 'agent_tier']], on='line_id', validate='one_to_one')
    d['status'] = d['status'].str.strip().str.lower()
    d['n'] = d['n_commits_listed'].astype(int)
    # Straightforward lines: one history to read (no move to another file), something to judge
    # (at least one listed commit), and rater 1 sure of the label.
    d = d[(d['status'].isin(['modified', 'retained'])) & (d['n'] >= 1) & (d['confident'].str.lower() == 'yes')]

    def first(x):
        try:
            return str(int(float(x)))
        except (TypeError, ValueError):
            return ''
    d['rater1_label'] = np.where(d['status'] == 'modified', 'modified@' + d['first_change_commit_no'].map(first),
                                 'retained')
    s = stratified(d, ['cell'], {'modified': N // 2, 'retained': N - N // 2}, 'status', 'n')
    s = finish(s, 'TRK')
    o = out_dir('tracker')
    ev = o / 'evidence'
    ev.mkdir(exist_ok=True)   # same seed gives the same ids, so pages are overwritten
    for r in s.itertuples():
        html = (D / r.evidence).read_text(encoding='utf-8')
        html = re.sub(rf'\b{r.line_id}\b', r.rater2_id, html)
        (ev / f'{r.rater2_id}.html').write_text(html, encoding='utf-8')
    sheet = s[['rater2_id', 'repository', 'file_at_merge', 'line_number', 'line_text', 'n_commits_listed']].copy()
    sheet['evidence'] = 'evidence/' + s['rater2_id'] + '.html'
    for c in ['status', 'first_change_commit_no', 'kind', 'notes']:
        sheet[c] = ''
    write_sheet(o / 'tracker_sheet.xlsx', sheet, 'status', ['modified', 'retained', 'moved to other file', 'unclear'],
                {'file_at_merge': 40, 'line_text': 60, 'evidence': 22, 'notes': 30})
    # second drop-down for kind
    from openpyxl import load_workbook
    wb = load_workbook(o / 'tracker_sheet.xlsx')
    ws = wb['sample']
    cols = [c.value for c in ws[1]]
    letter = ws.cell(row=1, column=cols.index('kind') + 1).column_letter
    dv = DataValidation(type='list', allow_blank=True,
                        formula1='"content changed,whitespace only,line deleted,file deleted,moved and changed"')
    ws.add_data_validation(dv)
    dv.add(f'{letter}2:{letter}{len(s) + 1}')
    ei = cols.index('evidence') + 1
    for r in range(2, len(s) + 2):   # clickable link to the evidence page
        c = ws.cell(row=r, column=ei)
        c.hyperlink = c.value
        c.style = 'Hyperlink'
    wb.save(o / 'tracker_sheet.xlsx')
    s[['rater2_id', 'line_id', 'cell', 'agent_tier', 'status', 'first_change_commit_no', 'rater1_label', 'n']].to_csv(
        o / 'tracker_KEY_INTERNAL.csv', index=False)
    return s, ['status', 'cell']


# ---------------------------------------------------------------- self-modification
def task_selfmod():
    D = ROOT / 'results_v6/round2'
    parts = []
    for agent in ['copilot', 'devin']:
        x = pd.read_excel(D / f'{agent}_selfmod_sample_labelled.xlsx', dtype=str)
        x['agent'] = agent
        parts.append(x)
    d = pd.concat(parts, ignore_index=True)
    d['rater1_label'] = d['label'].str.strip().str.lower()
    labels = ['same task', 'next item of a series', 'later maintenance']
    assert set(d['rater1_label']) <= set(labels), set(d['rater1_label'])
    s = finish(d, 'SM')
    o = out_dir('selfmod')
    sheet = s[['rater2_id', 'repo', 'sha', 'commit message', 'modifying-commit URL', 'original PR URL',
               'original PR title and description']].copy()
    sheet['label'] = ''
    sheet['notes'] = ''
    write_sheet(o / 'selfmod_sheet.xlsx', sheet, 'label', labels,
                {'commit message': 60, 'original PR title and description': 60, 'modifying-commit URL': 40,
                 'original PR URL': 40, 'notes': 30})
    s[['rater2_id', 'agent', 'repo', 'sha', 'rater1_label']].to_csv(o / 'selfmod_KEY_INTERNAL.csv', index=False)
    return s, ['rater1_label', 'agent']


# ---------------------------------------------------------------- human baseline
def task_human_baseline():
    """100 of the 200 human-baseline PRs, labeled blind by BOTH raters (the earlier
    verdicts were mostly set by rules, so they are not used as rater-1 labels).
    Strata are the earlier screening outcomes: no AI signal (set to human by rule),
    bot account (by username), empty description (by rule), and the 26 PRs judged by
    hand, all of which are kept."""
    d = pd.read_csv(ROOT / 'data/attribution_validation.csv')
    notes = d['notes'].fillna('')
    d['stratum'] = np.select(
        [notes.str.startswith('Auto: bot'), notes.str.startswith('Auto: empty'), d['signal_tier'] == 'CLEAN'],
        ['bot account', 'empty description', 'no AI signal'], 'judged by hand')
    d['size'] = d['body'].fillna('').str.len()
    quota = {'judged by hand': 26, 'bot account': 15, 'empty description': 15}
    quota['no AI signal'] = N - sum(quota.values())
    parts = [smallest(d[d['stratum'] == k], q, 'size') for k, q in quota.items()]
    s = finish(pd.concat(parts), 'HB')
    o = out_dir('human_baseline')
    sheet = s[['rater2_id', 'github_url', 'title', 'user', 'body']].copy()
    sheet['label'] = ''
    sheet['notes'] = ''
    write_sheet(o / 'human_baseline_sheet.xlsx', sheet, 'label', ['human', 'bot', 'ai-attributed', 'cannot assess'],
                {'github_url': 40, 'title': 40, 'body': 80, 'notes': 30})
    s['earlier_verdict'] = s['manual_review_verdict']
    s[['rater2_id', 'slug', 'number', 'signal_tier', 'stratum', 'earlier_verdict', 'size']].to_csv(
        o / 'human_baseline_KEY_INTERNAL.csv', index=False)
    return s, ['stratum', 'signal_tier']


TASKS = {'ast': task_ast, 'intent': task_intent, 'tracker': task_tracker, 'selfmod': task_selfmod,
         'human_baseline': task_human_baseline}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=['ast', 'intent', 'tracker', 'selfmod'])
    a = ap.parse_args()
    log = []
    for t in a.tasks:
        s, by = TASKS[t]()
        log.append(f'== {t}: {len(s)} rows')
        log.append(pd.crosstab(s[by[0]], s[by[1]], margins=True).to_string())
    txt = '\n'.join(log)
    (OUT / 'sample_log.txt').write_text(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
