"""
score_ast_validation.py

Scores the manual AST validation (Definition 2) against the rule's labels.

Inputs (labelled by the first author, blind to the rule's label):
  results/ast_validation_sheet.csv        main sample, proportional to tier x rule label
  results/ast_validation_topup_sheet.csv  top-up: extra leaf-only rows + structural decoys
and their keys (results/ast_validation_key.csv, results/ast_validation_topup_key.csv).

Reported:
  A. Main sample (population estimate): accuracy and Cohen's kappa with
     cluster-free bootstrap 95% CIs (2,000 resamples, seed 42), confusion
     matrix, per tier.
  B. Precision of each rule label with Wilson 95% CIs: leaf-only from all
     leaf-only rows (main + top-up), structural from the main sample; per tier
     and pooled (tiers weighted by their population counts).
  C. Leaf-only share of modified lines per tier, as labelled by the rule and
     corrected with the tier's precisions:
       true leaf share = p_rule_leaf * prec_leaf + (1 - p_rule_leaf) * (1 - prec_struct)
  D. Accuracy by kind of enclosing unit (main sample).
  E. Rows where the human label differs from the rule, with the notes.

Re-labels: rows whose context first showed a renamed file as deleted were
re-labelled on a corrected context (results/ast_validation_relabel_sheet.csv,
from make_ast_relabel_sheet.py); those labels replace the first ones.

Output: results/ast_validation_scores.txt

Usage:
    python src/score_ast_validation.py
"""

import math
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
SEED = 42
B = 2000
Z = 1.959963984540054
TIERS = ['human_supervised', 'fully_autonomous', 'human']


def load(sheet: str, key: str, sample: str) -> pd.DataFrame:
    s = pd.read_csv(os.path.join(RES, sheet), encoding='utf-8-sig', keep_default_na=False)
    k = pd.read_csv(os.path.join(RES, key))
    d = s[['id', 'human_label', 'notes', 'enclosing_node', 'diff_context']].merge(k[['id', 'slug', 'two_tier', 'rule_label']], on='id',
                                                   how='inner', validate='one_to_one')
    assert len(d) == len(s) == len(k), (sheet, len(s), len(k), len(d))
    d['human_label'] = d['human_label'].str.strip().str.lower()
    d['sample'] = sample
    return d


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in np.union1d(a, b))
    return (po - pe) / (1 - pe) if pe < 1 else float('nan')


def boot_ci(a: np.ndarray, b: np.ndarray, f) -> tuple:
    rng = np.random.default_rng(SEED)
    n = len(a)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        vals.append(f(a[i], b[i]))
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def wilson(k: int, n: int) -> tuple:
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    den = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def main():
    main_s = load('ast_validation_sheet.csv', 'ast_validation_key.csv', 'main')
    top = load('ast_validation_topup_sheet.csv', 'ast_validation_topup_key.csv', 'topup')
    relabel_note = 'none'
    rp = os.path.join(RES, 'ast_validation_relabel_sheet.csv')
    if os.path.exists(rp):
        rl = pd.read_csv(rp, encoding='utf-8-sig', keep_default_na=False)
        rl['human_label'] = rl['human_label'].str.strip().str.lower()
        if (rl['human_label'] != '').all():
            changed = 0
            for d in (main_s, top):
                for r in rl.itertuples():
                    i = d.index[d['id'] == r.id]
                    if len(i):
                        changed += int(d.loc[i[0], 'human_label'] != r.human_label)
                        d.loc[i[0], ['human_label', 'notes', 'diff_context']] = [r.human_label, r.notes, r.diff_context]
            relabel_note = f'{len(rl)} rows re-labelled on corrected context ({changed} labels changed)'
        else:
            relabel_note = f'{len(rl)} rows pending re-labelling; first labels used'
    pop = {}
    for line in open(os.path.join(RES, 'ast_validation_sample.txt')):
        p = line.split()
        if len(p) >= 3 and p[0] in TIERS and p[1] in ('leaf-only', 'structural'):
            pop[(p[0], p[1])] = int(p[2].replace(',', ''))

    out = ['=' * 70, 'AST validation: human labels vs the Definition 2 rule', '=' * 70,
           f'labelled: main {len(main_s)}, top-up {len(top)}; human labels main '
           f'{main_s["human_label"].value_counts().to_dict()}, top-up {top["human_label"].value_counts().to_dict()}',
           f're-labels: {relabel_note}', '']

    # A. main sample
    m = main_s[main_s['human_label'].isin(['leaf-only', 'structural'])]
    a, b = m['rule_label'].to_numpy(), m['human_label'].to_numpy()
    acc, k = np.mean(a == b), kappa(a, b)
    acc_ci = boot_ci(a, b, lambda x, y: np.mean(x == y))
    k_ci = boot_ci(a, b, kappa)
    out += ['A. Main sample (proportional; population estimate)',
            f'  n = {len(m)} (unclear excluded: {len(main_s) - len(m)})',
            f'  accuracy {acc * 100:.1f}% [{acc_ci[0] * 100:.1f}, {acc_ci[1] * 100:.1f}]',
            f'  Cohen kappa {k:.3f} [{k_ci[0]:.3f}, {k_ci[1]:.3f}]  (bootstrap, {B} resamples, seed {SEED})',
            '  confusion (rows rule, columns human):',
            '    ' + pd.crosstab(m['rule_label'], m['human_label']).to_string().replace('\n', '\n    ')]
    for t in TIERS:
        x = m[m['two_tier'] == t]
        out.append(f'  {t:17s} n={len(x):3d} accuracy {np.mean(x.rule_label == x.human_label) * 100:5.1f}%')
    out.append('')

    # B. precision per rule label
    allrows = pd.concat([main_s, top])
    allrows = allrows[allrows['human_label'].isin(['leaf-only', 'structural'])]
    out.append('B. Precision of the rule labels (Wilson 95% CI)')
    prec = {}
    for label, rows in (('leaf-only', allrows[allrows['rule_label'] == 'leaf-only']),
                        ('structural', m[m['rule_label'] == 'structural'])):
        src = 'main + top-up' if label == 'leaf-only' else 'main only'
        pooled, wsum = 0.0, 0
        for t in TIERS:
            x = rows[rows['two_tier'] == t]
            kk, n = int((x['human_label'] == label).sum()), len(x)
            lo, hi = wilson(kk, n)
            prec[(t, label)] = kk / n if n else float('nan')
            out.append(f'  rule {label:10s} {t:17s} {kk:3d}/{n:3d} = {kk / n * 100:5.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]  ({src})')
            w = pop.get((t, label), 0)
            pooled += w * prec[(t, label)]
            wsum += w
        kk, n = int((rows['human_label'] == label).sum()), len(rows)
        lo, hi = wilson(kk, n)
        out.append(f'  rule {label:10s} {"all, unweighted":17s} {kk:3d}/{n:3d} = {kk / n * 100:5.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]')
        out.append(f'  rule {label:10s} {"all, pop.-weighted":17s} {pooled / wsum * 100:5.1f}%')
    out.append('')

    # C. corrected leaf-only share
    out.append('C. Leaf-only share of modified lines with an AST status, rule vs corrected')
    for t in TIERS:
        nl, ns = pop[(t, 'leaf-only')], pop[(t, 'structural')]
        p = nl / (nl + ns)
        corr = p * prec[(t, 'leaf-only')] + (1 - p) * (1 - prec[(t, 'structural')])
        out.append(f'  {t:17s} rule {p * 100:5.1f}%  corrected {corr * 100:5.1f}%')
    out.append('')

    # D. accuracy by kind of enclosing unit
    anon = {'arrow_function', 'closure_expression', 'lambda_literal', 'func_literal', 'function_expression',
            'lambda', 'anonymous_function'}
    def kind(r):
        if r['diff_context'].startswith('[file deleted'):
            return 'file deleted'
        if r['enclosing_node'] in anon:
            return 'anonymous function'
        if r['enclosing_node'].endswith(('definition', 'declaration', 'item')) and r['enclosing_node'] not in (
                'lexical_declaration', 'use_declaration', 'import_declaration', 'variable_declaration',
                'const_declaration', 'var_declaration'):
            return 'named function, method or class'
        return 'module-level statement'
    m = m.assign(kind=m.apply(kind, axis=1))
    out.append('D. Main sample accuracy by kind of enclosing unit')
    for kd, g in m.groupby('kind'):
        out.append(f'  {kd:32s} n={len(g):3d} accuracy {np.mean(g.rule_label == g.human_label) * 100:5.1f}%')
    nd = m[m['kind'] != 'file deleted']
    a2, b2 = nd['rule_label'].to_numpy(), nd['human_label'].to_numpy()
    kci = boot_ci(a2, b2, kappa)
    out.append(f'  excluding deleted files: n={len(nd)} accuracy {np.mean(a2 == b2) * 100:.1f}% '
               f'kappa {kappa(a2, b2):.3f} [{kci[0]:.3f}, {kci[1]:.3f}]')
    out.append('')

    # E. disagreements
    dis = allrows[allrows['rule_label'] != allrows['human_label']]
    out.append(f'E. Disagreements ({len(dis)} rows; main {int((dis["sample"] == "main").sum())}, top-up {int((dis["sample"] == "topup").sum())})')
    for r in dis.sort_values(['rule_label', 'id']).itertuples():
        out.append(f'  {r.id} {r.two_tier:17s} rule {r.rule_label:10s} human {r.human_label:10s} {r.slug}: {str(r.notes)[:200]}')

    text = '\n'.join(out) + '\n'
    open(os.path.join(RES, 'ast_validation_scores.txt'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
