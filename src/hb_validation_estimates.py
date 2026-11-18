"""
hb_validation_estimates.py

Shares of bots, AI-attributed PRs, and PRs that cannot be assessed among the 200
human-baseline PRs of data/attribution_validation.csv, from the 100 PRs that
both raters labeled (results_v6/second_rater/human_baseline/). After discussion
the raters agreed on every PR, so the final label is the first author's.

The 100 PRs were drawn per screening stratum (all 26 judged by hand, 15 of the
35 bot accounts, 15 of the 30 empty descriptions, 44 of the 109 PRs with no AI
signal), so each stratum's share is weighted by its size in the 200. The 95%
interval comes from a two-stage bootstrap (2,000 resamples, seed 42): the 200
PRs are resampled, which varies the stratum sizes, and the labeled PRs are
resampled within each stratum.

Usage: python src/hb_validation_estimates.py [--out <file>]
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('REPO_ROOT', Path(__file__).resolve().parent.parent))
D = ROOT / 'results_v6' / 'second_rater' / 'human_baseline'
SEED, B = 42, 2000
LABELS = ['human', 'bot', 'ai-attributed', 'cannot assess']


def strata_of_200():
    d = pd.read_csv(ROOT / 'data/attribution_validation.csv')
    notes = d['notes'].fillna('')
    return np.select([notes.str.startswith('Auto: bot'), notes.str.startswith('Auto: empty'),
                      d['signal_tier'] == 'CLEAN'],
                     ['bot account', 'empty description', 'no AI signal'], 'judged by hand')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(D / 'hb_estimates.txt'))
    a = ap.parse_args()
    key = pd.read_csv(D / 'human_baseline_KEY_INTERNAL.csv')
    r1 = pd.read_excel(D / 'human_baseline_sheet_rater1.xlsx', sheet_name='sample', dtype=str)
    lab = key.merge(r1[['rater2_id', 'label']], on='rater2_id', validate='one_to_one')
    lab['label'] = lab['label'].str.strip().str.lower()
    pop = pd.Series(strata_of_200())
    N = pop.value_counts().to_dict()
    total = len(pop)
    strata = sorted(N)
    by = {h: lab.loc[lab['stratum'] == h, 'label'].to_numpy() for h in strata}

    def estimate(sizes, samples):
        return {L: sum(sizes[h] * np.mean(samples[h] == L) for h in strata) / total for L in LABELS}

    point = estimate(N, by)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(B):
        draw = rng.choice(pop.to_numpy(), size=total, replace=True)
        sizes = {h: int((draw == h).sum()) for h in strata}
        samples = {h: rng.choice(by[h], size=len(by[h]), replace=True) for h in strata}
        boots.append(estimate(sizes, samples))
    out = [f'Human-baseline validation: {total} PRs, {len(lab)} labeled by both raters (final = first author).',
           'Stratum sizes in the 200 and labels in the double-coded sample:']
    out.append(pd.crosstab(lab['stratum'], lab['label']).assign(N_in_200=pd.Series(N)).to_string())
    out.append('')
    out.append('Weighted share of the 200 [95% two-stage bootstrap interval] and expected count:')
    for L in LABELS:
        v = np.array([b[L] for b in boots])
        lo, hi = np.percentile(v, [2.5, 97.5])
        out.append(f'  {L:15s} {100 * point[L]:5.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]  ~{point[L] * total:.1f} PRs')
    txt = '\n'.join(out)
    Path(a.out).write_text(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
