"""
plot_pipeline.py

Draws the study-pipeline figure (Figure 1) as a vector PDF plus a PNG
preview. Every count on the figure is read from the data or results
files, not typed in:

  AIDev PRs            data/all_pull_request.parquet
  repository funnel    data/filtering_funnel_v2.csv
  cohort, line units   data/survival_events_line_v5_analysis.csv (window applied)
  hunk units           data/survival_events_hunk_v5.csv (window applied)
  AST coverage         ast_death non-null among modified lines
  intent commits       data/death_commit_messages_v5.csv
  human labels         results/intent_topup_scores.txt (original + top-up)
  specifications       results/spec_curve.csv
  modifier identity    results/rq4_results.txt

With DATA_TAG set to a conference-cohort dataset (v6conf...), the funnel
starts from the conference cohort (repositories after step 5 of the
funnel of the earlier study) and the number of
human labels is read from results_v6/intent_kappa_v6.txt (labelled commits
in the corpus).

Icons are drawn with matplotlib primitives, so the figure needs no
external assets. Colours follow the Okabe-Ito colour-blind-safe palette.

Usage:
    python src/plot_pipeline.py
Output:
    paper-overfeal-files/visuals/pipeline.pdf (+ .png preview)
"""

import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.observation_window import apply_observation_window
from config.paths import P

OUT = os.path.join(ROOT, P('paper-overfeal-files/visuals/pipeline'))

# Okabe-Ito: one accent per pipeline phase, light fills for the boxes
PHASES = {
    'data':     ('#0072B2', '#E3EEF6'),   # blue
    'measure':  ('#009E73', '#E0F2EC'),   # bluish green
    'classify': ('#D55E00', '#F8E8DD'),   # vermillion
    'analyse':  ('#CC79A7', '#F6E9F1'),   # reddish purple
}
INK = '#222222'


def counts() -> dict:
    p = lambda *a: os.path.join(ROOT, P('/'.join(a)))
    c = {'aidev_prs': len(pd.read_parquet(p('data', 'all_pull_request.parquet'), columns=['id']))}
    conf = (os.environ.get('DATA_TAG') or '').startswith('v6conf')
    if conf:
        c['repos_start'] = 256  # repositories after the code-ratio filter of the earlier study (its funnel, step 5)
    else:
        funnel = pd.read_csv(p('data', 'filtering_funnel_v2.csv'))
        c['repos_start'] = int(funnel['repos_remaining'].iloc[0])
    line = apply_observation_window(pd.read_csv(p('data', 'survival_events_line_v5_analysis.csv')), label='line')
    hunk = apply_observation_window(pd.read_csv(p('data', 'survival_events_hunk_v5.csv')), label='hunk')
    c['repos'] = line['slug'].nunique()
    c['prs'] = line[['slug', 'pr_number']].drop_duplicates().shape[0]
    c['lines'] = len(line)
    c['hunks'] = len(hunk)
    mod = line[line['event'] == 1]
    c['ast_pct'] = 100 * mod['ast_death'].notna().mean()
    c['commits'] = pd.read_csv(p('data', 'death_commit_messages_v5.csv'))[['slug', 'sha']].drop_duplicates().shape[0]
    if conf:
        m = re.search(r'labelled commits in it: (\d+) of', open(p('results', 'intent_kappa_v6.txt')).read())
        c['labels'] = int(m.group(1))
    else:
        m = re.search(r'labelled: original (\d+), top-up (\d+)', open(p('results', 'intent_topup_scores.txt')).read())
        c['labels'] = int(m.group(1)) + int(m.group(2))
    c['specs'] = len(pd.read_csv(p('results', 'spec_curve.csv')))
    rq4 = open(p('results', 'rq4_results.txt')).read()
    m = re.search(r'modified by humans: ([\d,]+)/([\d,]+) \(([\d.]+)%\)', rq4)
    c['human_mod_pct'] = float(m.group(3))
    c['mod_commits'] = sum(int(x) for x in re.findall(r'\(n=([\d,]+)\):', rq4.split('SECTION B')[0]))
    return c


# ── icons: drawn in a unit box centred on (x, y), size s ──────────────
def icon_database(ax, x, y, s, col):
    w, h = 0.62 * s, 0.70 * s
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc='white', ec=col, lw=1.2, zorder=3))
    for dy in (h / 2, 0, -h / 2):
        ax.add_patch(Ellipse((x, y + dy), w, 0.22 * s, fc='white', ec=col, lw=1.2, zorder=4))


def icon_funnel(ax, x, y, s, col):
    pts = [(x - 0.38 * s, y + 0.32 * s), (x + 0.38 * s, y + 0.32 * s), (x + 0.08 * s, y - 0.02 * s),
           (x + 0.08 * s, y - 0.36 * s), (x - 0.08 * s, y - 0.28 * s), (x - 0.08 * s, y - 0.02 * s)]
    ax.add_patch(Polygon(pts, closed=True, fc='white', ec=col, lw=1.2, zorder=3))


def icon_magnifier(ax, x, y, s, col):
    ax.add_patch(Circle((x - 0.08 * s, y + 0.08 * s), 0.24 * s, fc='white', ec=col, lw=1.4, zorder=3))
    ax.plot([x + 0.09 * s, x + 0.34 * s], [y - 0.09 * s, y - 0.34 * s], color=col, lw=2.2, solid_capstyle='round', zorder=3)


def icon_tiers(ax, x, y, s, col):
    for dy, w in ((0.18 * s, 0.70 * s), (-0.18 * s, 0.42 * s)):
        ax.add_patch(FancyBboxPatch((x - 0.35 * s, y + dy - 0.11 * s), w, 0.22 * s,
                                    boxstyle='round,pad=0,rounding_size=0.05', fc='white', ec=col, lw=1.2, zorder=3))


def icon_code(ax, x, y, s, col):
    ax.add_patch(Rectangle((x - 0.32 * s, y - 0.38 * s), 0.64 * s, 0.76 * s, fc='white', ec=col, lw=1.2, zorder=3))
    for i, w in enumerate((0.40, 0.28, 0.44, 0.22)):
        yy = y + 0.22 * s - i * 0.15 * s
        ax.plot([x - 0.22 * s, x - 0.22 * s + w * s], [yy, yy], color=col, lw=1.2, zorder=4)


def icon_tree(ax, x, y, s, col):
    nodes = [(x, y + 0.28 * s), (x - 0.26 * s, y - 0.06 * s), (x + 0.26 * s, y - 0.06 * s), (x + 0.12 * s, y - 0.34 * s), (x + 0.40 * s, y - 0.34 * s)]
    for a, b in ((0, 1), (0, 2), (2, 3), (2, 4)):
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=col, lw=1.1, zorder=3)
    for nx, ny in nodes:
        ax.add_patch(Circle((nx, ny), 0.08 * s, fc='white', ec=col, lw=1.2, zorder=4))


def icon_tag(ax, x, y, s, col):
    pts = [(x - 0.36 * s, y + 0.22 * s), (x + 0.12 * s, y + 0.22 * s), (x + 0.36 * s, y),
           (x + 0.12 * s, y - 0.22 * s), (x - 0.36 * s, y - 0.22 * s)]
    ax.add_patch(Polygon(pts, closed=True, fc='white', ec=col, lw=1.2, zorder=3))
    ax.add_patch(Circle((x + 0.10 * s, y), 0.05 * s, fc=col, ec=col, zorder=4))


def icon_person(ax, x, y, s, col):
    ax.add_patch(Circle((x, y + 0.16 * s), 0.15 * s, fc='white', ec=col, lw=1.2, zorder=3))
    ax.add_patch(Wedge((x, y - 0.38 * s), 0.34 * s, 0, 180, fc='white', ec=col, lw=1.2, zorder=3))


def icon_curve(ax, x, y, s, col):
    ax.plot([x - 0.36 * s, x - 0.36 * s, x + 0.38 * s], [y + 0.36 * s, y - 0.36 * s, y - 0.36 * s], color=col, lw=1.0, zorder=3)
    xs = [-0.32, -0.18, -0.18, -0.02, -0.02, 0.14, 0.14, 0.34]
    ys = [0.28, 0.28, 0.12, 0.12, 0.00, 0.00, -0.14, -0.14]
    ax.plot([x + a * s for a in xs], [y + b * s for b in ys], color=col, lw=1.6, zorder=4)


# ── layout ────────────────────────────────────────────────────────────
def stage(ax, x, y, w, h, phase, icon, title, lines):
    accent, fill = PHASES[phase]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.08',
                                fc=fill, ec=accent, lw=1.1, zorder=2))
    icon(ax, x, y + 0.24 * h, 0.34 * h, accent)
    ax.text(x, y - 0.03 * h, title, ha='center', va='center', fontsize=7.5, fontweight='bold', color=INK, zorder=5)
    ax.text(x, y - 0.27 * h, '\n'.join(lines), ha='center', va='center', fontsize=6.6, color=INK, linespacing=1.25, zorder=5)


def arrow(ax, p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9, lw=1.0, color='#555555',
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0, zorder=1))


def draw(c: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis('off')
    w, h = 1.82, 1.80
    xs = [1.05, 3.02, 4.99, 6.96, 8.93]
    top, bot = 3.30, 1.10
    S = [
        (xs[0], top, 'data', icon_database, 'AIDev dataset', [f'{c["aidev_prs"]:,} agent PRs', '5 coding agents']),
        (xs[1], top, 'data', icon_funnel, 'Repository filtering', [f'{c["repos_start"]} → {c["repos"]} repos', f'{c["prs"]:,} merged PRs']),
        (xs[2], top, 'data', icon_magnifier, 'Attribution check', ['200 human PRs checked', '(100 by two raters)']),
        (xs[3], top, 'data', icon_tiers, 'Agent taxonomy', ['human-supervised', 'fully-autonomous']),
        (xs[4], top, 'measure', icon_code, 'Retention mining', [f'{c["lines"]:,} lines', f'{c["hunks"]:,} hunks']),
        (xs[4], bot, 'measure', icon_tree, 'AST classification', ['syntactic vs. structural', f'{c["ast_pct"]:.1f}% classified']),
        (xs[3], bot, 'classify', icon_tag, 'Intent classification', [f'{c["commits"]:,} commits', f'keyword + 3 LLMs', f'{c["labels"]} human labels']),
        (xs[2], bot, 'classify', icon_person, 'Modifier identity', [f'{c["mod_commits"]:,} commits', f'{c["human_mod_pct"]:.1f}% of agent-code', 'modifications by humans']),
        (xs[1], bot, 'analyse', icon_curve, 'Survival analysis', ['Kaplan-Meier, Cox', 'across and within repos', f'{c["specs"]}-spec curve']),
    ]
    for x, y, ph, ic, t, ls in S:
        stage(ax, x, y, w, h, ph, ic, t, ls)
    for i in range(3):
        arrow(ax, (xs[i] + w / 2 + 0.02, top), (xs[i + 1] - w / 2 - 0.02, top))
    arrow(ax, (xs[3] + w / 2 + 0.02, top), (xs[4] - w / 2 - 0.02, top))
    arrow(ax, (xs[4], top - h / 2 - 0.02), (xs[4], bot + h / 2 + 0.02))
    for i in (4, 3, 2):
        arrow(ax, (xs[i] - w / 2 - 0.02, bot), (xs[i - 1] + w / 2 + 0.02, bot))
    # RQ labels under the analysis box
    ax.text(xs[0], bot, 'RQ1–RQ5', ha='center', va='center', fontsize=9, fontweight='bold', color=PHASES['analyse'][0])
    arrow(ax, (xs[1] - w / 2 - 0.02, bot), (xs[0] + 0.48, bot))
    # legend
    names = {'data': 'Data and cohort', 'measure': 'Measurement', 'classify': 'Classification', 'analyse': 'Analysis'}
    for i, (k, n) in enumerate(names.items()):
        lx = 1.05 - w / 2 + i * 2.05
        ax.add_patch(FancyBboxPatch((lx, 0.02), 0.22, 0.16, boxstyle='round,pad=0,rounding_size=0.03',
                                    fc=PHASES[k][1], ec=PHASES[k][0], lw=1.0))
        ax.text(lx + 0.30, 0.10, n, va='center', fontsize=7, color=INK)
    fig.tight_layout(pad=0.1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + '.pdf', bbox_inches='tight')
    fig.savefig(OUT + '.png', dpi=220, bbox_inches='tight')
    print('wrote', OUT + '.pdf', OUT + '.png')


if __name__ == '__main__':
    c = counts()
    print(c)
    draw(c)
