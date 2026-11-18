"""
run_no_outlier_r1.py 

Runs an existing analysis script with the stale outlier exclusion
OUTLIER_PR_KEYS = {('microsoft/typespec', 7115)} removed (the PR has 46 lines in
the conference cohort; the 3xIQR rule that flagged it applied to its 15,546
hunks in the first mining). The scripts are
not edited: every module that defines the constant is loaded from its source
with that one line replaced by an empty set.

Run with RESULTS_DIR pointing to a round-1 folder so nothing in results_v6/ is
overwritten, e.g.
    DATA_TAG=v6confnv RESULTS_DIR=results_v6/round1/no_typespec_outlier \
        python src/run_no_outlier_r1.py src/check_sampling_caps.py
Extra modes (no script of its own):
    --hunk-size    the hunk-size strata of analyze_rq1_master.py (its hunk section)
    --spec-hunks   the hunk-unit specifications of specification_curve.py (lifelines, as the paper)
"""

import os
import runpy
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)
os.chdir(ROOT)
LINE = "OUTLIER_PR_KEYS = {('microsoft/typespec', 7115)}"
PATCHED = ['analyze_rq1_master', 'analyze_rq2', 'analyze_rq4', 'specification_curve']


def load_patched(name):
    path = os.path.join(SRC, name + '.py')
    src = open(path).read()
    if LINE in src:
        src = src.replace(LINE, 'OUTLIER_PR_KEYS = set()  # round-1: stale typespec#7115 exclusion removed')
    mod = types.ModuleType(name)
    mod.__file__ = path
    sys.modules[name] = mod
    exec(compile(src, path, 'exec'), mod.__dict__)
    return mod


def main():
    for n in PATCHED:
        load_patched(n)
    assert not sys.modules['analyze_rq1_master'].OUTLIER_PR_KEYS
    mode = sys.argv[1]
    if mode == '--hunk-size':
        hunk_size()
    elif mode == '--spec-hunks':
        spec_hunks()
    else:
        src = open(mode).read()
        src = src.replace(LINE, 'OUTLIER_PR_KEYS = set()  # round-1')
        sys.argv = [mode] + sys.argv[2:]
        g = {'__name__': '__main__', '__file__': os.path.abspath(mode)}
        exec(compile(src, mode, 'exec'), g)


def hunk_size():
    import numpy as np
    import pandas as pd
    from lifelines import CoxPHFitter
    from config.observation_window import apply_observation_window
    from config.paths import P
    m = sys.modules['analyze_rq1_master']
    hunk = m.add_two_tier_column(m.preprocess_hunk(apply_observation_window(
        pd.read_csv(P('data/survival_events_hunk_v5.csv')), label='hunk')))
    hh = hunk[hunk['two_tier'] == 'human']
    rows = []
    for tier in m.TIERS:
        th = hunk[hunk['two_tier'] == tier]
        for lab, lo, hi in [('single_line', 1, 1), ('small_block', 2, 10), ('large_block', 11, 10 ** 9)]:
            a = th[(th['hunk_size'] >= lo) & (th['hunk_size'] <= hi)]
            h = hh[(hh['hunk_size'] >= lo) & (hh['hunk_size'] <= hi)]
            d = pd.concat([a[['duration_days', 'event']].assign(is_tier=1),
                           h[['duration_days', 'event']].assign(is_tier=0)]).dropna()
            s = CoxPHFitter().fit(d, 'duration_days', 'event').summary.loc['is_tier']
            rows.append(dict(tier=tier, size_group=lab, n_agent=len(a), n_human=len(h), hr=s['exp(coef)'],
                             ci_low=s['exp(coef) lower 95%'], ci_high=s['exp(coef) upper 95%'], p=s['p']))
    out = pd.DataFrame(rows)
    path = P('results/rq1_hunk_size_no_outlier.csv')
    out.to_csv(path, index=False)
    print(f'hunks after window/generated filter: {len(hunk):,}; human {len(hh):,}')
    print(out.to_string())


def spec_hunks():
    import pandas as pd
    from config.paths import P
    sc = sys.modules['specification_curve']
    tag = os.environ.get('DATA_TAG', '')
    base = tag[:-2] if tag.endswith('nv') else tag
    _, ch = sc.conference_units(base)
    hunks = {'conference tracking, all lines, with vendored': sc.prep(ch, 'conference hunks', drop_generated=False),
             'v3 tracking, all lines, with vendored': sc.prep(pd.read_csv(f'data/survival_events_hunk_{base}.csv',
                                                                          low_memory=False), 'hv', drop_generated=False),
             'v2 tracking, capped': sc.prep(pd.read_csv(P('data/survival_events_hunk_v3.csv'), low_memory=False), 'h2'),
             'v3 tracking, all lines': sc.prep(pd.read_csv(P('data/survival_events_hunk_v5.csv'), low_memory=False), 'h5')}
    rows = []
    for dname, h in hunks.items():
        for strat in (False, True):
            for tier in sc.TIERS:
                hr, lo, hi, p = sc.fit(h, tier, 'event', False, strat)
                rows.append(dict(tier=tier, data=dname, unit='hunk', definition='syntactic',
                                 comparison='within repository' if strat else 'across repositories',
                                 hr=hr, ci_low=lo, ci_high=hi, p=p, n_typespec_7115=int(
                                     ((h['slug'] == 'microsoft/typespec') & (h['pr_number'] == 7115)).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(P('results/spec_curve_hunks_no_outlier.csv'), index=False)
    print(out.to_string())


if __name__ == '__main__':
    main()
