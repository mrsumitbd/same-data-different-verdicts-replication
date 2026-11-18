"""
figs_r6.py  (figures for RQ2 and the Synthesis)

Figure A (RQ2): hazard ratio of each tier against human code under Definition 1
(syntactic), Definition 2 (AST-structural) and Definition 2 followed past leaf-only
edits, across repositories (standard errors clustered by repository) and within
repositories (500-resample cluster bootstrap), identity tracking, all lines.
Inputs: results_v6/round1/c1_cox_variance.csv (across, Def 1 and Def 2),
        results_v6/rq1_def2_stratified_results.txt (within, Def 1 and Def 2),
        results_v6/round2/e5_def2_continued.txt (Def 2 followed past leaf-only edits).

Figure B (Synthesis): null distribution of the verdict counts of the specification
curve under 500 PR-level label permutations (src/spec_null_r2.py, Efron ties), with
the observed count and its permutation p-value.
Inputs: results_v6/round2/work/null_<variant>_efron.csv.gz, the per-permutation
        draws written by src/spec_null_r2.py (stage `permutations` of reproduce.sh).
        NULL_DIR overrides their location. The counts and p-values are checked against
        results_v6/round2/e1_e4_permutation_summary.csv.

Output (results_v6/round6 unless OUT is set): figA_rq2_definitions.{pdf,png},
        figB_permutation_null.{pdf,png}, figs_r6_values.txt (every plotted number).
Usage (from the repository root): python src/figs_r6.py [A] [B]
"""
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
os.chdir(ROOT)
OUT = os.path.abspath(os.environ.get('OUT', 'results_v6/round6'))
NULL_DIR = os.path.abspath(os.environ.get('NULL_DIR', 'results_v6/round2/work'))
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, 'src'))

COL = {'Human-supervised': '#0072B2', 'Devin': '#D55E00', 'Human baseline': '#009E73'}
RED = '#C0392B'
WIDTH = 5.5  # acmsmall text width, inches
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
LOG = []


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- Figure A
def rq2_values():
    c = pd.read_csv('results_v6/round1/c1_cox_variance.csv')
    c = c[(c['data'] == 'v3 tracking, all lines') & (c['unit'] == 'line') & (c['comparison'] == 'across')]
    rows = []
    for tier, lab in [('human_supervised', 'Human-supervised'), ('fully_autonomous', 'Devin')]:
        for dfn, dl in [('syntactic', 'def1'), ('structural', 'def2')]:
            r = c[(c['tier'] == tier) & (c['definition'] == dfn)].iloc[0]
            rows.append(dict(tier=lab, scope='across', defn=dl, hr=r['hr'], lo=r['lo_cl_repo'], hi=r['hi_cl_repo'],
                             src='results_v6/round1/c1_cox_variance.csv (v3all_line, line, across, cl_repo)'))
    # within: analytic HR with the bootstrap 95% CI, as in the paper
    txt = open('results_v6/rq1_def2_stratified_results.txt').read()
    for tier, lab in [('human_supervised', 'Human-supervised'), ('fully_autonomous', 'Devin')]:
        block = txt.split(f'TIER: {tier}')[1].split('TIER:')[0]
        for d, dl in [('1', 'def1'), ('2', 'def2')]:
            b = block.split(f'[Def {d} x Stratified + Bootstrap]')[1]
            hr = float(re.search(r'HR_analytic=([\d.]+)', b).group(1))
            lo, hi = map(float, re.search(r'95% CI=\[([\d.]+), ([\d.]+)\]', b).groups())
            rows.append(dict(tier=lab, scope='within', defn=dl, hr=hr, lo=lo, hi=hi,
                             src='results_v6/rq1_def2_stratified_results.txt (stratified + bootstrap)'))
    e5 = open('results_v6/round2/e5_def2_continued.txt').read()
    for line in e5.splitlines():
        m = re.match(r'\s+continued past leaf-only\s+(\w+)\s+HR ([\d.]+) (?:repo-cl|boot95) \[([\d.]+), ([\d.]+)\]', line)
        if m:
            tier = 'Human-supervised' if m.group(1) == 'human_supervised' else 'Devin'
            scope = 'across' if 'repo-cl' in line else 'within'
            rows.append(dict(tier=tier, scope=scope, defn='def2c', hr=float(m.group(2)), lo=float(m.group(3)),
                             hi=float(m.group(4)), src='results_v6/round2/e5_def2_continued.txt'))
    d = pd.DataFrame(rows)
    assert len(d) == 12, d
    return d


def fig_a():
    d = rq2_values()
    # paper values, Table tab:rq2_def_comparison (3 decimals)
    paper = {('Human-supervised', 'across', 'def1'): (0.888, 0.612, 1.289), ('Devin', 'across', 'def1'): (0.896, 0.425, 1.889),
             ('Human-supervised', 'across', 'def2'): (0.919, 0.625, 1.350), ('Devin', 'across', 'def2'): (0.847, 0.357, 2.011),
             ('Human-supervised', 'across', 'def2c'): (0.906, 0.617, 1.328), ('Devin', 'across', 'def2c'): (0.841, 0.350, 2.017),
             ('Human-supervised', 'within', 'def1'): (0.807, 0.544, 1.032), ('Devin', 'within', 'def1'): (1.059, 0.409, 2.076),
             ('Human-supervised', 'within', 'def2'): (0.717, 0.477, 0.921), ('Devin', 'within', 'def2'): (1.155, 0.519, 2.176),
             ('Human-supervised', 'within', 'def2c'): (0.707, 0.469, 0.905), ('Devin', 'within', 'def2c'): (1.103, 0.472, 2.196)}
    LOG.append('Figure A: tier, scope, definition, HR [95% CI] (file) | paper Table tab:rq2_def_comparison | match')
    for r in d.itertuples():
        p = paper[(r.tier, r.scope, r.defn)]
        ok = all(abs(round(a, 3) - b) < 1e-9 for a, b in zip((r.hr, r.lo, r.hi), p))
        LOG.append(f'  {r.tier:17s} {r.scope:7s} {r.defn:6s} {r.hr:.4f} [{r.lo:.4f}, {r.hi:.4f}] | '
                   f'{p[0]:.3f} [{p[1]:.3f}, {p[2]:.3f}] | {"ok" if ok else "MISMATCH"}  ({r.src})')
        assert ok, (r, p)

    rows = [('Human-supervised', 'across'), ('Human-supervised', 'within'), ('Devin', 'across'), ('Devin', 'within')]
    ylab = {'across': 'Across repositories\n(clustered SEs)', 'within': 'Within repositories\n(cluster bootstrap)'}
    off = {'def1': 0.22, 'def2': 0.0, 'def2c': -0.22}
    mk = {'def1': dict(marker='o', mfc='white', ms=5.5, mew=1.2),
          'def2': dict(marker='o', ms=5.5, mew=1.2),
          'def2c': dict(marker='^', ms=4.5, mew=0.8, alpha=0.75)}
    ys = {rows[0]: 4.6, rows[1]: 3.6, rows[2]: 1.6, rows[3]: 0.6}
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
    for key, y in ys.items():
        col = COL[key[0]]
        x = d[(d['tier'] == key[0]) & (d['scope'] == key[1])].set_index('defn')
        ax.plot([x.loc['def1', 'hr'], x.loc['def2', 'hr']], [y + off['def1'], y + off['def2']], color=col, lw=0.6, alpha=0.6)
        for dfn in ['def1', 'def2', 'def2c']:
            r = x.loc[dfn]
            lw = 0.8 if dfn == 'def2c' else 1.3
            ax.plot([r['lo'], r['hi']], [y + off[dfn]] * 2, color=col, lw=lw, alpha=0.75 if dfn == 'def2c' else 1)
            ax.plot(r['hr'], y + off[dfn], ls='', color=col, mec=col, **mk[dfn])
            sig = r['hi'] < 1 or r['lo'] > 1
            ax.text(3.15, y + off[dfn], f"{r['hr']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]", va='center', ha='left',
                    fontsize=6.8, color=col, fontweight='bold' if sig else 'normal')
    ax.axvline(1, color='#444444', ls='--', lw=0.8)
    ax.axhline(2.6, color='#BBBBBB', lw=0.6)
    ax.set_xscale('log')
    ax.set_xlim(0.3, 3.1)
    ax.set_xticks([0.3, 0.5, 0.7, 1, 1.5, 2, 3])
    ax.set_xticklabels(['0.3', '0.5', '0.7', '1', '1.5', '2', '3'])
    ax.minorticks_off()
    ax.set_ylim(0.15, 5.55)
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels([ylab[k[1]] for k in ys])
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    for tier, y in [('Human-supervised', 5.2), ('Devin', 2.2)]:
        ax.text(0.302, y, 'Human-supervised' if tier != 'Devin' else 'Fully-autonomous (Devin)', color=COL[tier],
                fontweight='bold', va='center', fontsize=8)
    ax.text(0.97, 5.42, '← agent code retained longer', ha='right', va='center', fontsize=7, color='grey')
    ax.text(1.03, 5.42, 'modified sooner →', ha='left', va='center', fontsize=7, color='grey')
    ax.text(3.15, 5.42, 'HR [95% CI]', ha='left', va='center', fontsize=6.8, color='grey')
    ax.set_xlabel('Hazard ratio against human code (log scale, 95% CI)')
    h = [Line2D([], [], color='#333333', ls='-', lw=1.3, marker='o', mfc='white', ms=5.5, label='Definition 1 (syntactic)'),
         Line2D([], [], color='#333333', ls='-', lw=1.3, marker='o', ms=5.5, label='Definition 2 (AST-structural)'),
         Line2D([], [], color='#333333', ls='-', lw=0.8, marker='^', ms=4.5, alpha=0.75,
                label='Definition 2, followed past leaf-only edits')]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.45, -0.17), ncol=3, frameon=False, fontsize=7,
              handlelength=1.6, columnspacing=1.2)
    save(fig, 'figA_rq2_definitions')


# ---------------------------------------------------------------- Figure B
EST_SE = {'cl_repo': lambda d: d['se_cl_repo'],
          'current': lambda d: np.where(d['comparison'] == 'within', d['se_cl_repo'],
                                        np.where(d['unit'] == 'PR-weighted', d['se_robust'], d['se_model']))}
Z = 1.959964

# (variant, tier, comparison, estimator, direction, label, paper text: observed / null mean / p)
FULL = [('full', 'human_supervised', 'across', 'cl_repo', 'longer', 'Human-supervised\nacross, clustered\n“retained longer”'),
        ('full', 'human_supervised', 'within', 'cl_repo', 'longer', 'Human-supervised\nwithin, clustered\n“retained longer”'),
        ('full', 'fully_autonomous', 'across', 'current', 'sooner', 'Devin\nacross, model-based\n“modified sooner”'),
        ('full', 'fully_autonomous', 'across', 'cl_repo', 'sooner', 'Devin\nacross, clustered\n“modified sooner”')]
WIN = [('full', 'Full\n(≥ 5 mo)'), ('c8', '≥ 8 mo'), ('c8h', '≥ 8 mo,\ncens. 240 d'),
       ('c10h', '≥ 10 mo,\ncens. 300 d'), ('nopdfme', 'Full, w/o\npdfme #1059'),
       ('coh270_150', 'w/o the\n142 PRs,\ncens. 150 d')]
WINDOWS = [(v, 'fully_autonomous', 'across', 'cl_repo', 'sooner', lab) for v, lab in WIN]
# numbers reported in the paper (observed, null mean, p)
PAPER = {('full', 'human_supervised', 'across', 'cl_repo', 'longer'): (13, 1.8, 0.008),
         ('full', 'human_supervised', 'within', 'cl_repo', 'longer'): (11, 1.4, 0.008),
         ('full', 'fully_autonomous', 'across', 'current', 'sooner'): (15, 6.9, 0.086),
         ('full', 'fully_autonomous', 'across', 'cl_repo', 'sooner'): (3, 1.6, 0.35),
         ('c8', 'fully_autonomous', 'across', 'cl_repo', 'sooner'): (8, None, 0.046)}


def counts(variant, tier, cmp, est):
    f = os.path.join(NULL_DIR, f'null_{variant}_efron.csv.gz')
    d = pd.read_csv(f)
    d = d[(d['tier'] == tier) & (d['comparison'] == cmp)].copy()
    se = EST_SE[est](d)
    d['L'] = d['coef'] + Z * se < 0
    d['S'] = d['coef'] - Z * se > 0
    g = d.groupby('draw').agg(n=('coef', 'size'), longer=('L', 'sum'), sooner=('S', 'sum'))
    n_obs = g.loc[0, 'n']
    null = g[(g.index > 0) & (g['n'] == n_obs)]
    return g.loc[0], null


def bean(ax, x, vals, obs, p, col, seed):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    grid = np.linspace(-0.4, min(vals.max() + 0.4, 22.4), 300)
    bw = max(0.6, 1.06 * vals.std() * len(vals) ** -0.2)
    dens = np.exp(-0.5 * ((grid[:, None] - vals[None, :]) / bw) ** 2).sum(1)
    dens = dens / dens.max() * 0.36
    ax.fill_betweenx(grid, x - dens, x + dens, color=col, alpha=0.18, lw=0)
    ax.plot(x - dens, grid, color=col, lw=0.6, alpha=0.6)
    ax.plot(x + dens, grid, color=col, lw=0.6, alpha=0.6)
    jit = rng.uniform(-1, 1, len(vals)) * np.interp(vals, grid, dens) * 0.9
    ax.scatter(x + jit, vals + rng.uniform(-0.18, 0.18, len(vals)), s=1.6, color=col, alpha=0.35, lw=0)
    q95 = np.percentile(vals, 95)
    ax.plot([x - 0.3, x + 0.3], [q95, q95], color='#555555', lw=0.7, ls=':')
    ax.plot([x - 0.3, x + 0.3], [obs, obs], color=RED, lw=1.0)
    ax.plot(x, obs, marker='D', color=RED, ms=5, ls='')
    ax.text(x, 23.6, f'obs. {int(obs)}\np = {p:.3f}', color=RED, fontsize=6.6, va='bottom', ha='center', linespacing=1.05)


def fig_b():
    summ = pd.read_csv('results_v6/round2/e1_e4_permutation_summary.csv')
    summ = summ[summ['engine'] == 'efron']
    LOG.append('')
    LOG.append('Figure B: variant, tier, comparison, estimator, direction: observed, null mean, null 95th pct, p '
               '(from null_draws) | e1_e4_permutation_summary.csv | paper')
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.8), gridspec_kw=dict(height_ratios=[1, 1], hspace=0.8))
    for ax, spec, title in [(axes[0], FULL, '(a) Full observation window'),
                            (axes[1], WINDOWS, '(b) Devin, across repositories, clustered: “modified sooner” by observation window')]:
        for i, (v, tier, cmp, est, dr, lab) in enumerate(spec):
            obs, null = counts(v, tier, cmp, est)
            o, vals = int(obs[dr]), null[dr].values
            p = (1 + (vals >= o).sum()) / (1 + len(vals))
            s = summ[(summ['variant'] == v) & (summ['tier'] == tier) & (summ['comparison'] == cmp) & (summ['estimator'] == est)].iloc[0]
            ok = (o == s[dr]) and abs(vals.mean() - s[f'null_{dr}_mean']) < 1e-9 and abs(p - s[f'p_{dr}']) < 1e-9 \
                and len(vals) == s['draws']
            pp = PAPER.get((v, tier, cmp, est, dr))
            ptxt = f'{pp[0]} / {pp[1]} / {pp[2]}' if pp else '-'
            LOG.append(f'  {v:10s} {tier:17s} {cmp:7s} {est:8s} {dr:6s}: {o:2d}, {vals.mean():.3f}, {np.percentile(vals, 95):.1f}, '
                       f'p={p:.3f} (draws {len(vals)}) | {"ok" if ok else "MISMATCH"} | {ptxt}')
            col = COL['Devin'] if tier == 'fully_autonomous' else COL['Human-supervised']
            bean(ax, i, vals, o, p, col, seed=42 + i)
        ax.set_xticks(range(len(spec)))
        ax.set_xticklabels([s_[-1] for s_ in spec], fontsize=7)
        ax.set_xlim(-0.55, len(spec) - 0.45)
        ax.set_ylabel('Significant verdicts\n(of 22 specifications)')
        ax.set_title(title, fontsize=8, loc='left', pad=24)
        ax.tick_params(axis='x', length=0)
    axes[0].set_ylim(-0.8, 23)
    axes[1].set_ylim(-0.8, 23)
    h = [Line2D([], [], marker='o', ls='', color='#888888', ms=3, alpha=0.6, label='Permuted labels (500 draws)'),
         Line2D([], [], ls=':', color='#555555', lw=0.8, label='95th percentile of the null'),
         Line2D([], [], marker='D', color=RED, ms=5, lw=1.0, label='Observed count, permutation p')]
    axes[0].legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 1.3), ncol=3, frameon=False, fontsize=7)
    save(fig, 'figB_permutation_null')


if __name__ == '__main__':
    which = sys.argv[1:] or ['A', 'B']
    if 'A' in which:
        fig_a()
    if 'B' in which:
        fig_b()
    open(os.path.join(OUT, 'figs_r6_values.txt'), 'w').write('\n'.join(LOG) + '\n')
    print('\n'.join(LOG))
