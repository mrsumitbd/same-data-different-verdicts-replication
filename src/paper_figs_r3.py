"""
paper_figs_r3.py 

Figures drawn from the values printed in the paper's tables (parsed from
results_v6/paper_tables.tex, the tables of the paper; nothing is recomputed):
  3. RQ4 stacked bars: tab:rq4_intent rows Keyword, Majority vote, Human labels.
  4. RQ5: tab:modifier_variants (agent share by unit and window).

Run from the repository root:
  python src/paper_figs_r3.py
"""
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = 'results_v6/round3'
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


# ── 3. RQ4 stacked bars ─────────────────────────────────────────────
def intent():
    rows = {r[0]: r for r in table('tab:rq4_intent')}
    use = [('Keyword rules', [k for k in rows if k.startswith('Keyword')][0]),
           ('Majority vote', [k for k in rows if k.startswith('Majority')][0]),
           ('Human labels', [k for k in rows if k.startswith('Human labels')][0])]
    groups = ['Human-supervised', 'Devin', 'Human baseline']
    cats = ['Corrective', 'Perfective', 'Adaptive', 'Preventive', 'Unclassified (keyword)']
    ccol = ['#D55E00', '#0072B2', '#009E73', '#CC79A7', '#BBBBBB']
    data = {}
    log.append('Task 3 shares read from results_v6/paper_tables.tex (tab:rq4_intent), C/P/A/Prev (5th value = keyword unclassified; majority vote shares are of the 5,263 decided commits, residual is rounding):')
    for name, key in use:
        for g, cell in zip(groups, rows[key][2:5]):
            v = [float(x) for x in cell.split('/')]
            rem = round(100 - sum(v), 1)
            v.append(rem if abs(rem) >= 0.5 else 0.0)   # majority vote: shares of decided commits; residual = rounding
            data[(name, g)] = v
            log.append(f'  {name:14s} {g:18s} ' + ' / '.join(f'{x:.1f}' for x in v))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs, xl = [], []
    x = 0
    for name, _ in use:
        ax.text(0, x - 0.05, name, ha='left', va='center', fontweight='bold', fontsize=9)
        x += 0.8
        for g in groups:
            v = data[(name, g)]
            left = 0
            for k, (c, col) in enumerate(zip(cats, ccol)):
                ax.barh(x, v[k], left=left, color=col, edgecolor='white', lw=0.6, height=0.75,
                        label=c if len(xs) == 0 else None)
                if v[k] >= 4:
                    ax.text(left + v[k] / 2, x, f'{v[k]:.1f}', ha='center', va='center', fontsize=7.5,
                            color='white' if k < 4 else 'black')
                left += v[k]
            xs.append(x)
            xl.append(g)
            x += 1
        x += 0.3
    ax.set_yticks(xs)
    ax.set_yticklabels(xl, fontsize=8.5)
    ax.set_ylim(x - 0.6, -0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Share of modification commits (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=5, frameon=False, fontsize=8.5,
              handlelength=1.2, columnspacing=1.0)
    save(fig, 'rq4_intent_stacked')


# ── 4. RQ5 agent share by unit and window ───────────────────────────
def rq5():
    rows = table('tab:modifier_variants')
    groups = ['Human-supervised', 'Devin', 'Human baseline']
    pat = re.compile(r'([\d.]+)(?:\s*\(([\d.]+)\))?')
    recs = []
    for r in rows:
        vals = []
        for cell in r[2:5]:
            m = pat.search(cell)
            vals.append((float(m.group(1)), float(m.group(2)) if m.group(2) else None))
        recs.append((r[0], r[1], vals))
    log.append('\nTask 4 values read from results_v6/paper_tables.tex (tab:modifier_variants), agent share (self share):')
    for u, w, v in recs:
        log.append(f'  {u:24s} {w:8s} ' + '  '.join(f'{a:.1f} ({b if b is not None else "-"})' for a, b in v))
    units = ['Commit', 'Line', 'Follow-up PR']
    windows = ['7 days', '30 days', 'Full']
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, u in zip(axes, units):
        sub = {w: v for uu, w, v in recs if uu == u}
        xs = np.arange(len(windows))
        bw = 0.26
        for gi, g in enumerate(groups):
            tot = [sub[w][gi][0] for w in windows]
            slf = [sub[w][gi][1] for w in windows]
            pos = xs + (gi - 1) * bw
            ax.bar(pos, tot, bw, color=COL[g], alpha=0.35 if gi < 2 else 0.9, edgecolor=COL[g], lw=1,
                   label=f'{g}, any agent' if gi < 2 else f'{g}, any agent')
            if gi < 2:
                ax.bar(pos, slf, bw, color=COL[g], edgecolor=COL[g], lw=1, label=f'{g}, authoring agent')
            for p, t in zip(pos, tot):
                ax.text(p, t + 0.5, f'{t:.1f}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(windows)
        ax.set_title(u, fontsize=10)
        ax.set_ylim(0, 33)
        ax.grid(axis='y', alpha=0.25, ls=':')
    axes[0].set_ylabel('Modifications made by agents (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, 'rq5_agent_share')


if __name__ == '__main__':
    intent()
    rq5()
    open(f'{OUT}/paper_figs_values.txt', 'w').write('\n'.join(log) + '\n')
    print('\n'.join(log))
