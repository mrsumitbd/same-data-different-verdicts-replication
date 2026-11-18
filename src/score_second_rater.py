"""
score_second_rater.py

Inter-rater agreement for the second-rater tasks built by
src/make_second_rater_sheets.py. For each task whose filled sheet exists
(results_v6/second_rater/<task>/<task>_sheet_labelled.xlsx), joins the second
rater's labels to the rater-1 key on rater2_id (for human_baseline, to the first
author's filled copy of the same sheet, <task>_sheet_rater1.xlsx) and reports Cohen's kappa with a
95% bootstrap interval (2,000 resamples, seed 42), the raw agreement, and the
confusion table. Writes results_v6/second_rater/kappa_results.md and one
<task>_disagreements.csv per task for the discussion round.

Tracker task: the compared label is `retained` or `modified@<first commit>`
(a different first commit counts as a disagreement); kappa on the status alone
is reported next to it.

Usage: python src/score_second_rater.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = Path(os.environ.get('SECOND_RATER_DIR', ROOT / 'results_v6' / 'second_rater'))
SEED, B = 42, 2000
TASKS = ['human_baseline', 'ast', 'intent', 'tracker', 'selfmod']


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def boot(a, b):
    rng = np.random.default_rng(SEED)
    a, b = np.asarray(a), np.asarray(b)
    ks = [kappa(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(B))]
    return np.percentile(ks, [2.5, 97.5])


def norm(x):
    return x.fillna('').astype(str).str.strip().str.lower()


def first_no(x):
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return ''


def load(task):
    sheet = D / task / f'{task}_sheet_labelled.xlsx'
    if not sheet.exists():
        return None
    r2 = pd.read_excel(sheet, sheet_name='sample', dtype=str)
    key = pd.read_csv(D / task / f'{task}_KEY_INTERNAL.csv', dtype=str)
    if task == 'human_baseline':   # both raters labeled the blank sheet; rater 1 is the first author
        r1 = D / task / f'{task}_sheet_rater1.xlsx'
        if not r1.exists():
            return None
        r1 = pd.read_excel(r1, sheet_name='sample', dtype=str)[['rater2_id', 'label']]
        key = key.merge(r1.rename(columns={'label': 'rater1_label'}), on='rater2_id', validate='one_to_one')
    key = key.rename(columns={c: f'r1_{c}' for c in key.columns if c in r2.columns and c != 'rater2_id'})
    d = key.merge(r2, on='rater2_id', how='left', validate='one_to_one')
    if task == 'tracker':
        st = norm(d['status'])
        d['rater2_label'] = np.where(st == 'modified', 'modified@' + d['first_change_commit_no'].map(first_no), st)
    else:
        d['rater2_label'] = norm(d['label'])
    d['rater1_label'] = norm(d['rater1_label'])
    return d


def main():
    out = ['# Second-rater agreement', '',
           f'Cohen\'s kappa with a 95% bootstrap interval ({B:,} resamples, seed {SEED}).', '']
    for t in TASKS:
        d = load(t)
        if d is None:
            out.append(f'## {t}\n\nNo filled sheet yet (`{t}/{t}_sheet_labelled.xlsx`).\n')
            continue
        missing = (d['rater2_label'] == '').sum()
        x = d[d['rater2_label'] != '']
        k = kappa(x['rater1_label'], x['rater2_label'])
        lo, hi = boot(x['rater1_label'].to_numpy(), x['rater2_label'].to_numpy())
        agree = (x['rater1_label'] == x['rater2_label']).sum()
        out += [f'## {t}', '', f'- Rows: {len(d)} ({missing} without a rater-2 label, left out)',
                f'- Raw agreement: {agree} of {len(x)} ({agree / len(x):.1%})',
                f'- Cohen\'s kappa: {k:.3f} [{lo:.3f}, {hi:.3f}]']
        if t == 'tracker':
            s1 = np.where(x['rater1_label'].str.startswith('modified'), 'modified', x['rater1_label'])
            s2 = np.where(x['rater2_label'].str.startswith('modified'), 'modified', x['rater2_label'])
            ks = kappa(s1, s2)
            los, his = boot(s1, s2)
            out.append(f'- Cohen\'s kappa on the status alone (modified or retained): {ks:.3f} [{los:.3f}, {his:.3f}]')
            x = x.assign(rater1_label=s1, rater2_label=s2)
        out += ['', '```', pd.crosstab(x['rater1_label'], x['rater2_label'], rownames=['rater 1'],
                                       colnames=['rater 2']).to_string(), '```', '']
        dis = d[d['rater1_label'] != d['rater2_label']]
        dis.to_csv(D / t / f'{t}_disagreements.csv', index=False)
    txt = '\n'.join(out)
    (D / 'kappa_results.md').write_text(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
