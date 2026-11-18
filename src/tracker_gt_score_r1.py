"""Score the three trackers against the labeled ground truth.

Reads the filled sheet (results_v6/round1/tracker_gt/tracker_groundtruth_sheet.xlsx),
the key and the sampling frame, and reports for each tracker:
  - precision of its "modified" claims and of its "retained" claims,
  - recall of truly modified lines,
  - overall accuracy (modified vs retained in the window),
  - for lines that both the tracker and the label call modified, how often the
    tracker names the same first-parent commit.
Population estimates weight each line by (eligible lines in its cell) / (sampled
lines in its cell); 95% intervals come from a stratified bootstrap (B = 2000,
seed 42). Sample counts are reported as well.

Two conventions for lines that are moved without change:
  main:   a move is not a modification (the line is followed to its new place),
          as in Definition 1 and the identity tracker;
  strict: a move counts as a modification at the commit that moves the line.
Lines labeled "unclear" are left out and counted. Lines labeled "moved to other
file" are scored under the main convention from the follow-up sheet
(tracker_groundtruth_moves_sheet.xlsx, built by src/tracker_gt_moves_r1.py);
until a follow-up status is filled, they are scored only under the strict one.

Run: python3 src/tracker_gt_score_r1.py  -> results_v6/round1/tracker_gt/tracker_groundtruth_scores.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'results_v6' / 'round1' / 'tracker_gt'
TRACKERS = {'identity': 'identity tracking', 'earlier': "earlier study's tracker",
            'line_number_tracker': 'line-number tracking'}
COMMIT_COL = {'identity': 'identity_commit_no', 'earlier': 'earlier_commit_no',
              'line_number_tracker': 'line_number_commit_no'}
SHA_COL = {'identity': 'identity_sha', 'earlier': 'earlier_sha', 'line_number_tracker': 'line_number_sha'}
B, SEED = 2000, 42


def num(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def truth(row, convention):
    st = str(row['status']).strip().lower()
    moved_no = num(row['moved_in_commit_no'])
    if st == 'unclear' or st in ('', 'nan'):
        return None, None
    if convention == 'strict' and moved_no is not None:
        first = num(row['first_change_commit_no'])
        if st != 'modified' or first is None or moved_no <= first:
            return 'modified', moved_no
    if st == 'moved to other file':
        if convention == 'strict':
            return 'modified', moved_no
        fst = str(row.get('f_status', '')).strip().lower()
        if fst == 'modified':
            f = num(row.get('f_first_change_commit_no'))
            return 'modified', (f'F{f}' if f is not None else None)
        if fst == 'retained':
            return 'retained', None
        return None, None
    if st == 'modified':
        return 'modified', num(row['first_change_commit_no'])
    if st == 'retained':
        return 'retained', None
    return None, None


def metrics(df, w, t):
    tm, cm = df['truth'] == 'modified', df[t] == 'modified'
    def ratio(a, b):
        return float((w * a).sum() / (w * b).sum()) if (w * b).sum() > 0 else np.nan
    return dict(prec_mod=ratio(tm & cm, cm), prec_ret=ratio(~tm & ~cm, ~cm),
                recall_mod=ratio(tm & cm, tm), accuracy=ratio(tm == cm, pd.Series(True, index=df.index)))


def score(lab, frame, convention, out):
    t = lab.apply(lambda r: truth(r, convention), axis=1, result_type='expand')
    lab = lab.assign(truth=t[0], truth_commit=t[1])
    n_unclear = lab['truth'].isna().sum()
    df = lab[lab['truth'].notna()].copy()
    rate = frame.set_index('cell')
    df['w'] = df['cell'].map(rate['eligible_population_est']) / df['cell'].map(df['cell'].value_counts())
    out.append(f'\n=== Convention: {convention} (n = {len(df)} labeled lines; {n_unclear} unclear or not scorable) ===')
    rng = np.random.default_rng(SEED)
    groups = [g.index.to_numpy() for _, g in df.groupby('cell')]
    for t, name in TRACKERS.items():
        est = metrics(df, df['w'], t)
        boots = []
        for _ in range(B):
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            bd = df.loc[idx]
            boots.append(metrics(bd, bd['w'], t))
        bt = pd.DataFrame(boots)
        out.append(f'\n{name}:')
        for m, lab_m in (('prec_mod', 'precision of "modified"'), ('prec_ret', 'precision of "retained"'),
                         ('recall_mod', 'recall of modified lines'), ('accuracy', 'accuracy')):
            lo, hi = np.nanpercentile(bt[m], [2.5, 97.5])
            out.append(f'  {lab_m:28s} {est[m]:.3f}  [{lo:.3f}, {hi:.3f}]  (population-weighted)')
        cm = df[t] == 'modified'
        tm = df['truth'] == 'modified'
        out.append(f'  sample: claims modified {cm.sum()}, of which truly modified {(cm & tm).sum()}; '
                   f'claims retained {(~cm).sum()}, of which truly retained {(~cm & ~tm).sum()}')
        both = df[cm & tm]
        def same_commit(r):
            tc = r['truth_commit']
            if isinstance(tc, str) and tc.startswith('F'):   # first change after a cross-file move
                fs = str(r.get('f_shas', '') or '').split()
                k = int(tc[1:])
                return 0 < k <= len(fs) and str(r[SHA_COL[t]]) == fs[k - 1]
            return num(r[COMMIT_COL[t]]) is not None and num(r[COMMIT_COL[t]]) == num(tc)
        same = sum(same_commit(r) for _, r in both.iterrows())
        out.append(f'  same first commit when both say modified: {same} of {len(both)}')


def main():
    import os
    lab = pd.read_excel(D / os.environ.get('TGT_SHEET', 'tracker_groundtruth_sheet_labelled.xlsx'), sheet_name=0, dtype=str)
    key = pd.read_csv(D / 'tracker_groundtruth_key.csv', dtype=str)
    frame = pd.read_csv(D / 'sampling_frame.csv', dtype={'cell': str})
    lab = lab.merge(key, on='line_id', how='left')
    mv = D / 'tracker_groundtruth_moves_sheet.xlsx'
    if mv.exists():   # follow-up labels for lines moved to another file (main convention)
        f = pd.read_excel(mv, dtype=str)[['line_id', 'status', 'first_change_commit_no']]
        f = f.rename(columns={'status': 'f_status', 'first_change_commit_no': 'f_first_change_commit_no'})
        c = pd.read_csv(D / 'tracker_groundtruth_moves_commits.csv', dtype=str).rename(
            columns={'followup_commit_shas': 'f_shas'})
        lab = lab.merge(f, on='line_id', how='left').merge(c, on='line_id', how='left')
    out = [f'Tracker ground truth: {len(lab)} lines in the sheet, '
           f'{(lab["status"].fillna("").str.strip() != "").sum()} labeled.',
           'Cells (identity, earlier tracker, line-number; 1 = modified in the window):',
           frame.to_string(index=False)]
    for conv in ('main', 'strict'):
        score(lab, frame, conv, out)
    txt = '\n'.join(out)
    (D / 'tracker_groundtruth_scores.txt').write_text(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
