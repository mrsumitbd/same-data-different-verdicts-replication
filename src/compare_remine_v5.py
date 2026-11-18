"""
compare_remine_v5.py

Side-by-side comparison of the headline numbers before (v4 data: line-number
tracking, capped sampling) and after (v5 data: identity tracking, all added
lines, merge commits included) the re-mine, with every conclusion that
changes direction or significance flagged.

Usage:
    python src/compare_remine_v5.py --old <dir with v4 results> --new results
Output:
    results/remine_v3_comparison.md
"""

import argparse
import io
import re
from pathlib import Path

import pandas as pd


def section(path: Path, name: str) -> pd.DataFrame:
    text = path.read_text()
    block = text.split(f'SECTION: {name}\n')[1].split('=' * 70)[1]
    return pd.read_csv(io.StringIO(block.strip()), sep=r'\s+')


def sig(p: float, alpha: float = 0.05) -> str:
    return 'sig' if p < alpha else 'NS'


def rq1_rows(old: Path, new: Path) -> list:
    rows = []
    for sec, label_col in [('step1', 'label'), ('step2', 'label'), ('step3', 'label')]:
        o, n = section(old, sec).set_index(label_col), section(new, sec).set_index(label_col)
        for lab in n.index:
            if lab not in o.index:
                continue
            a, b = o.loc[lab], n.loc[lab]
            rows.append({'quantity': f'{sec} {lab}: N agent / human',
                         'old': f'{int(a.n_agent):,} / {int(a.n_human):,}', 'new': f'{int(b.n_agent):,} / {int(b.n_human):,}', 'flag': ''})
            rows.append({'quantity': f'{sec} {lab}: modification rate agent / human (%)',
                         'old': f'{a.mod_rate_agent} / {a.mod_rate_human}', 'new': f'{b.mod_rate_agent} / {b.mod_rate_human}', 'flag': ''})
            rows.append({'quantity': f'{sec} {lab}: median days agent / human',
                         'old': f'{a.median_agent} / {a.median_human}', 'new': f'{b.median_agent} / {b.median_human}', 'flag': ''})
            flag = ''
            if (a.hr < 1) != (b.hr < 1):
                flag = 'DIRECTION'
            elif sig(a.p_cox) != sig(b.p_cox):
                flag = 'SIGNIFICANCE'
            rows.append({'quantity': f'{sec} {lab}: unadjusted HR (p)', 'old': f'{a.hr:.3f} ({a.p_cox:.2g})',
                         'new': f'{b.hr:.3f} ({b.p_cox:.2g})', 'flag': flag})
    o, n = section(old, 'step4').set_index('label'), section(new, 'step4').set_index('label')
    for lab in n.index:
        a, b = o.loc[lab], n.loc[lab]
        def verdict(r):
            return 'sig' if (r.boot_ci_high < 1 or r.boot_ci_low > 1) else 'NS'
        flag = 'DIRECTION' if (a.hr_analytic < 1) != (b.hr_analytic < 1) else ('SIGNIFICANCE' if verdict(a) != verdict(b) else '')
        rows.append({'quantity': f'step4 {lab}: stratified HR [bootstrap CI] (p); strata',
                     'old': f'{a.hr_analytic:.3f} [{a.boot_ci_low:.3f}, {a.boot_ci_high:.3f}] ({a.p_analytic:.2g}); {int(a.n_contributing_strata)}/{int(a.n_total_strata)}',
                     'new': f'{b.hr_analytic:.3f} [{b.boot_ci_low:.3f}, {b.boot_ci_high:.3f}] ({b.p_analytic:.2g}); {int(b.n_contributing_strata)}/{int(b.n_total_strata)}',
                     'flag': flag})
    o, n = section(old, 'hunk_size'), section(new, 'hunk_size')
    m = o.merge(n, on=['tier', 'size_group'], suffixes=('_o', '_n'))
    for r in m.itertuples():
        flag = 'DIRECTION' if ((r.hr_o < 1) != (r.hr_n < 1) and (r.p_o < .05 or r.p_n < .05)) else \
               ('SIGNIFICANCE' if sig(r.p_o) != sig(r.p_n) else '')
        rows.append({'quantity': f'hunk {r.tier} {r.size_group}: HR (p); n agent',
                     'old': f'{r.hr_o:.3f} ({r.p_o:.2g}); {int(r.n_agent_o):,}', 'new': f'{r.hr_n:.3f} ({r.p_n:.2g}); {int(r.n_agent_n):,}',
                     'flag': flag})
    return rows


def def2_strat(path: Path) -> dict:
    t = path.read_text()
    out = {}
    for tier in ['human_supervised', 'fully_autonomous']:
        blk = t.split(f'TIER: {tier}')[1]
        m = re.search(r'Def 2 x Stratified.*?HR_analytic=([\d.]+), p=([\d.e+-]+).*?95% CI=\[([\d.]+), ([\d.]+)\]', blk, re.S)
        out[tier] = tuple(float(x) for x in m.groups())
    return out


def rq3(path: Path) -> dict:
    t = path.read_text()
    g = lambda pat: re.search(pat, t, re.S).group(1)
    return {'universe': g(r'human_supervised: \d+ / (\d+) repos'),
            'HS repos': g(r'human_supervised: (\d+ / \d+ repos \([\d.]+%\))'),
            'FA repos': g(r'fully_autonomous: (\d+ / \d+ repos \([\d.]+%\))'),
            'human repos': g(r'  human: (\d+ / \d+ repos \([\d.]+%\))'),
            'Jaccard': g(r'Jaccard=([\d.]+)'),
            'one agent tier': g(r'exactly one agent tier: (\d+ / \d+ \([\d.]+%\))'),
            'FA contributors p': g(r'fully_autonomous: n_with.*?contributors\s*:.*?p=([\d.]+)') if re.search(r'fully_autonomous: n_with', t) else ''}


def rq4(path: Path) -> dict:
    t = path.read_text()
    out = {}
    for tier in ['human_supervised', 'fully_autonomous', 'human']:
        m = re.search(rf'  {tier} \(n=([\d,]+)\):\n    human: ([\d,]+) \(([\d.]+)%\)\n    agent: ([\d,]+) \(([\d.]+)%\)', t)
        out[tier] = f'n={m.group(1)}, human {m.group(3)}%, agent {m.group(5)}%' if m else ''
    m = re.search(r'modified by humans: ([\d,]+)/([\d,]+) \(([\d.]+)%\)', t)
    out['agent code modified by humans'] = f'{m.group(1)}/{m.group(2)} ({m.group(3)}%)' if m else ''
    m = re.search(r'fully_autonomous modified by agent \(n=(\d+)\):\n\s+devin-ai-integration\[bot\]: (\d+) \(([\d.]+)%\)', t)
    out['Devin share of agent mods of FA code'] = f'{m.group(2)}/{m.group(1)} ({m.group(3)}%)' if m else ''
    return out


def intent(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r'^(\w+)\s+k=([\d.]+)\s+HS ([\d./u]+)\s+FA ([\d./u]+)\s+Hu ([\d./u]+)\s+\| HS p=([\d.e+-]+) V=([\d.]+)\s+\| FA p=([\d.e+-]+) V=([\d.]+)', line)
        if m:
            rows.append({'approach': m.group(1), 'HS shares': m.group(3), 'FA shares': m.group(4),
                         'HS p': float(m.group(6)), 'FA p': float(m.group(8))})
    return pd.DataFrame(rows).set_index('approach')


def md_table(rows: list) -> str:
    df = pd.DataFrame(rows)
    out = ['| Quantity | Old (v4) | New (v5) | Changed conclusion |', '|---|---|---|---|']
    for r in df.itertuples():
        out.append(f'| {r.quantity} | {r.old} | {r.new} | {r.flag} |')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', default='results')
    ap.add_argument('--output', default='results/remine_v3_comparison.md')
    a = ap.parse_args()
    old, new = Path(a.old), Path(a.new)
    md = ['# Re-mine comparison: v4 (line-number tracking) vs v5 (identity tracking)', '',
          'Old: data/survival_events_line_v4_ast.csv and survival_events_hunk_v3.csv (capped sampling, '
          'v2 miner). New: data/survival_events_line_v5_analysis.csv and survival_events_hunk_v5.csv '
          '(src/measure_retention_v5.py: every added source line, merge commits included, lines tracked by '
          'identity with reverse blame, deaths verified in the diff). "DIRECTION" = the hazard ratio '
          'crosses 1; "SIGNIFICANCE" = a significant result becomes non-significant or vice versa '
          '(p < 0.05; stratified models judged by the bootstrap CI).', '']
    md += ['## RQ1 retention (line level unless stated)', '', md_table(rq1_rows(old / 'rq1_master_results.txt', new / 'rq1_master_results.txt')), '']
    rows = []
    o, n = def2_strat(old / 'rq1_def2_stratified_results.txt'), def2_strat(new / 'rq1_def2_stratified_results.txt')
    for t in o:
        a_, b_ = o[t], n[t]
        va = 'sig' if a_[3] < 1 or a_[2] > 1 else 'NS'
        vb = 'sig' if b_[3] < 1 or b_[2] > 1 else 'NS'
        flag = 'DIRECTION' if (a_[0] < 1) != (b_[0] < 1) else ('SIGNIFICANCE' if va != vb else '')
        rows.append({'quantity': f'Def2 x stratified {t}: HR [bootstrap CI] (p)',
                     'old': f'{a_[0]:.3f} [{a_[2]:.3f}, {a_[3]:.3f}] ({a_[1]:.2g})',
                     'new': f'{b_[0]:.3f} [{b_[2]:.3f}, {b_[3]:.3f}] ({b_[1]:.2g})', 'flag': flag})
    md += ['## Definition 2 under repository stratification', '', md_table(rows), '']
    o, n = rq3(old / 'rq3_results.txt'), rq3(new / 'rq3_results.txt')
    md += ['## RQ3 repository coverage', '', md_table([{'quantity': k, 'old': o[k], 'new': n[k], 'flag': ''} for k in n]), '']
    if (new / 'rq4_results.txt').exists():
        o, n = rq4(old / 'rq4_results.txt'), rq4(new / 'rq4_results.txt')
        md += ['## RQ4 modifier identity', '', md_table([{'quantity': k, 'old': o[k], 'new': n[k], 'flag': ''} for k in n]), '']
    if (new / 'intent_aggregation_full_corpus.txt').exists():
        o, n = intent(old / 'intent_aggregation_full_corpus.txt'), intent(new / 'intent_aggregation_full_corpus.txt')
        rows = []
        for ap_ in n.index:
            if ap_ not in o.index:
                continue
            for tier in ['HS', 'FA']:
                flag = 'SIGNIFICANCE' if sig(o.loc[ap_, f'{tier} p']) != sig(n.loc[ap_, f'{tier} p']) else ''
                rows.append({'quantity': f'intent {ap_}: {tier} vs human p (shares C/P/A/Pv)',
                             'old': f'{o.loc[ap_, f"{tier} p"]:.2g} ({o.loc[ap_, f"{tier} shares"]})',
                             'new': f'{n.loc[ap_, f"{tier} p"]:.2g} ({n.loc[ap_, f"{tier} shares"]})', 'flag': flag})
        md += ['## RQ2 intent verdicts (full corpus)', '', md_table(rows), '']
    Path(a.output).write_text('\n'.join(md) + '\n')
    print('wrote', a.output)


if __name__ == '__main__':
    main()
