"""
forest_r6.py  (RQ3 forest plot: within-repository hazard ratio of each agent against
human code, with cluster-bootstrap CIs, read from tab:rq3_per_agent of
results_v6/paper_tables.tex; nothing is recomputed. Cursor and Claude Code, which share
few repositories with human code, are set apart in a shaded band; a CI beyond either
axis limit is drawn as an arrow; significant values are printed in bold.)

Run from the repository root:
  python src/forest_r6.py
"""
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.environ.get('OUT', 'results_v6/round6')
FEW = ['Cursor', 'Claude Code']
XMIN, XMAX = 0.1, 4.5
TEX = open('results_v6/paper_tables.tex').read()
COL = {'Human-supervised': '#0072B2', 'Devin': '#D55E00', 'Human baseline': '#009E73'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
log = []


def table(label):
    i = TEX.index(r'\label{' + label + '}')
    j = TEX.index(r'\end{tabular}', i)
    body = TEX[i:j]
    body = body[body.index(r'\midrule'):]
    rows = []
    for ln in body.split(r'\\'):
        ln = ln.replace(r'\midrule', '').replace(r'\bottomrule', '').strip()
        if not ln or ln.startswith(r'\multicolumn'):
            continue
        rows.append([c.strip() for c in ln.split('&')])
    return rows


CI = re.compile(r'([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]')


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)


# ── 2. forest plot ──────────────────────────────────────────────────
def forest():
    items = []   # (section, label, estimator, hr, lo, hi, colour)
    nrepo = {}
    for r in table('tab:rq3_per_agent'):
        nrepo[r[0]] = int(r[3].replace(',', ''))
        hr, lo, hi = map(float, CI.search(r[4]).groups())
        items.append(('Per agent, syntactic', r[0], 'Within (cluster bootstrap)', hr, lo, hi,
                      COL['Devin'] if r[0] == 'Devin' else COL['Human-supervised']))
    log.append('Values read from results_v6/paper_tables.tex (tab:rq3_per_agent):')
    for it in items:
        log.append(f'  {it[0]:24s} {it[1]:18s} {it[2]:28s} {it[3]:.3f} [{it[4]:.3f}, {it[5]:.3f}]')

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y, yt, ylab, sec_pos = 0, [], [], []
    prev = None
    few_y = []
    for sec, lab, est, hr, lo, hi, c in items:
        if sec != prev:
            y += 0.8
            sec_pos.append((y, sec))
            y += 0.7
            prev = sec
        if lab in FEW and not few_y:
            y += 0.9                      # room for the group label
        if lab not in FEW and few_y and len(few_y) == len(FEW):
            y += 0.4
        if lab in FEW:
            few_y.append(y)
        mk = 'o' if est.startswith('Across') else 's'
        fc = c if est.startswith('Across') else 'white'
        a_lo, a_hi = max(lo, XMIN), min(hi, XMAX)
        ax.plot([a_lo, a_hi], [y, y], color=c, lw=1.6)
        for beyond, end, d in ((lo < XMIN, XMIN, -1), (hi > XMAX, XMAX, 1)):
            if beyond:
                log.append(f'  arrow: {lab} CI {"lower" if d < 0 else "upper"} end beyond axis')
                ax.annotate('', xy=(end, y), xytext=(end / 1.25 if d > 0 else end * 1.25, y),
                            arrowprops=dict(arrowstyle='->', color=c, lw=1.6))
        ax.plot(hr, y, marker=mk, ms=6.5, mfc=fc, mec=c, mew=1.6, ls='none')
        ax.text(4.6, y, f'{hr:.3f} [{lo:.3f}, {hi:.3f}]', va='center', fontsize=8.5,
                fontweight='bold' if (hi < 1 or lo > 1) else 'normal')
        yt.append(y)
        ylab.append(f'{lab}, {"across" if est.startswith("Across") else "within"}' if sec.startswith('Tier') else lab)
        y += 1
    for yy, s in sec_pos:
        ax.text(0.105, yy, s, fontweight='bold', va='center', fontsize=9.5)
    if few_y:
        top, bot = few_y[0] - 1.05, few_y[-1] + 0.5
        ax.axhspan(top, bot, color='0.93', zorder=0, lw=0)
        for yy in (top, bot):
            ax.axhline(yy, color='0.55', lw=0.8, ls='-')
        lbl = f'Few shared repositories (n = {nrepo[FEW[0]]} and {nrepo[FEW[1]]})'
        ax.text(0.105, top + 0.45, lbl, va='center', fontsize=8.5, style='italic', color='0.3')
        log.append(f'  few-shared block: {FEW} repositories {[nrepo[f] for f in FEW]}')
    ax.axvline(1, color='grey', ls='--', lw=1)
    ax.set_xscale('log')
    ax.set_xlim(XMIN, XMAX)
    ticks = [0.1, 0.2, 0.5, 1, 2, 4]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticks(yt)
    ax.set_yticklabels(ylab)
    ax.set_ylim(y, 0.2)
    ax.set_xlabel('Hazard ratio against human code (log scale)')
    ax.text(0.12, y - 0.1, '← retained longer', fontsize=8.5, color='grey', va='bottom')
    ax.text(3.9, y - 0.1, 'modified sooner →', fontsize=8.5, color='grey', ha='right', va='bottom')
    save(fig, 'rq3_forest')


if __name__ == '__main__':
    forest()
    open(f'{OUT}/forest_r6_values.txt', 'w').write('\n'.join(log) + '\n')
    print('\n'.join(log))
