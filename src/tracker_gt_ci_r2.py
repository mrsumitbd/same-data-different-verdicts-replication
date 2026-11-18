"""
tracker_gt_ci_r2.py 

Precision / recall of the three trackers against the 299 labeled lines (main
convention: a move to another file is followed), with
  - population-weighted point estimates (stratum weights = eligible lines in the cell / sampled lines),
  - 95% CIs from a stratified bootstrap that resamples the labeled lines WITHIN each stratum
    (cell of the three trackers' claims), 2,000 resamples, seed 42, stratum weights fixed
    (same scheme as src/tracker_gt_score_r1.py, re-run here),
  - unweighted sample counts: correct "modified" claims / "modified" claims, and
    found / truly modified (labeled) lines,
  - as a check on cells whose 50 labeled lines all agree (the bootstrap cannot vary there):
    a stratified Bayesian interval, Jeffreys Beta(k + 0.5, n - k + 0.5) per cell for the share
    truly modified, combined with the fixed weights (20,000 draws, seed 42).
Strict convention (moves count as modifications) is reported too.

Usage: python src/tracker_gt_ci_r2.py -> results_v6/round2/e2_tracker_ci.txt (+ .csv)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import tracker_gt_score_r1 as T

OUT = 'results_v6/round2'
B, SEED, NB = 2000, 42, 20000


def load():
    D = T.D
    lab = pd.read_excel(D / 'tracker_groundtruth_sheet_labelled.xlsx', sheet_name=0, dtype=str)
    key = pd.read_csv(D / 'tracker_groundtruth_key.csv', dtype=str)
    frame = pd.read_csv(D / 'sampling_frame.csv', dtype={'cell': str})
    lab = lab.merge(key, on='line_id', how='left')
    f = pd.read_excel(D / 'tracker_groundtruth_moves_sheet.xlsx', dtype=str)[['line_id', 'status', 'first_change_commit_no']]
    f = f.rename(columns={'status': 'f_status', 'first_change_commit_no': 'f_first_change_commit_no'})
    c = pd.read_csv(D / 'tracker_groundtruth_moves_commits.csv', dtype=str).rename(columns={'followup_commit_shas': 'f_shas'})
    lab = lab.merge(f, on='line_id', how='left').merge(c, on='line_id', how='left')
    return lab, frame


def main():
    lab0, frame = load()
    rows, out = [], ['E2. Tracker ground truth: precision / recall with stratified-bootstrap 95% CIs',
                     f'Stratified bootstrap: lines resampled within each cell, B = {B}, seed {SEED}, cell weights fixed.',
                     'Bayesian check: Jeffreys Beta per cell for the share truly modified, fixed weights, 20,000 draws, seed 42.', '']
    for conv in ('main', 'strict'):
        t = lab0.apply(lambda r: T.truth(r, conv), axis=1, result_type='expand')
        lab = lab0.assign(truth=t[0])
        df = lab[lab['truth'].notna()].copy()
        cellw = frame.set_index('cell')['eligible_population_est']
        df['w'] = df['cell'].map(cellw) / df['cell'].map(df['cell'].value_counts())
        out.append(f'=== Convention: {conv} (n = {len(df)} labeled lines) ===')
        rng = np.random.default_rng(SEED)
        groups = [g.index.to_numpy() for _, g in df.groupby('cell')]
        # per-cell counts for the Bayesian check
        cells = df.groupby('cell').agg(n=('truth', 'size'), k=('truth', lambda s: (s == 'modified').sum()))
        cells['W'] = cells.index.map(cellw)
        brng = np.random.default_rng(SEED)
        pmod = np.stack([brng.beta(k + 0.5, n - k + 0.5, NB) for k, n in zip(cells['k'], cells['n'])], axis=1)
        for tr, name in T.TRACKERS.items():
            est = T.metrics(df, df['w'], tr)
            boots = []
            for _ in range(B):
                idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
                bd = df.loc[idx]
                boots.append(T.metrics(bd, bd['w'], tr))
            bt = pd.DataFrame(boots)
            # cell-level claims of this tracker (constant within a cell)
            claim = df.groupby('cell')[tr].agg(lambda s: (s == 'modified').mean())
            assert set(claim.unique()) <= {0.0, 1.0}
            cm = claim.reindex(cells.index).values.astype(bool)
            W = cells['W'].values
            tp = (pmod[:, cm] * W[cm]).sum(1)
            bayes = dict(prec_mod=tp / W[cm].sum(), recall_mod=tp / (pmod * W).sum(1))
            tm, cmm = df['truth'] == 'modified', df[tr] == 'modified'
            cnt = dict(prec_mod=(int((tm & cmm).sum()), int(cmm.sum())), recall_mod=(int((tm & cmm).sum()), int(tm.sum())),
                       prec_ret=(int((~tm & ~cmm).sum()), int((~cmm).sum())), accuracy=(int((tm == cmm).sum()), len(df)))
            out.append(f'\n{name}:')
            for m, lab_m in (('prec_mod', 'precision of "modified"'), ('recall_mod', 'recall of modified lines'),
                             ('prec_ret', 'precision of "retained"'), ('accuracy', 'accuracy')):
                lo, hi = np.nanpercentile(bt[m], [2.5, 97.5])
                a, b = cnt[m]
                s = f'  {lab_m:26s} {est[m]:.3f} [{lo:.3f}, {hi:.3f}]  unweighted {a}/{b} = {a / b:.3f}'
                r = dict(convention=conv, tracker=name, metric=m, estimate=est[m], boot_lo=lo, boot_hi=hi, count=a, of=b,
                         unweighted=a / b)
                if m in bayes:
                    blo, bhi = np.percentile(bayes[m], [2.5, 97.5])
                    s += f'  (Jeffreys check [{blo:.3f}, {bhi:.3f}])'
                    r.update(bayes_lo=blo, bayes_hi=bhi)
                out.append(s)
                rows.append(r)
        out.append('\nPer-cell labeled counts (k truly modified of n):')
        out.append(cells.assign(W=cells['W'].astype(int)).to_string())
        out.append('')
    out.append('Note: the bootstrap cannot vary in a cell whose labeled lines all agree (k = 0 or k = n), so its intervals are\n'
               'narrow when the cells that carry most weight are unanimous; the Jeffreys check gives such cells their\n'
               'binomial uncertainty.')
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'e2_tracker_ci.csv'), index=False)
    txt = '\n'.join(out) + '\n'
    open(os.path.join(OUT, 'e2_tracker_ci.txt'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
