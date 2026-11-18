"""
compute_intent_kappa.py

Agreement of the automatic intent classifiers with the human gold labels on
the 385-commit stratified validation sample.

Gold labels: human_label in the annotation sheet (built by
make_annotation_sheet.py, labelled per docs/annotation-guide.md), joined on
sample_id. Accepted formats: .xlsx (as saved by Excel/Numbers) or .csv.

Classifiers scored against gold:
  - keyword   data/death_commit_messages.csv (intent), via slug+sha;
              cross-checked with keyword_label in the sample file
  - deepseek, gpt_oss, minimax
              results/intent_llm_<model>_t0.csv, written by
              classify_intent_full_corpus.py (temperature 0), each reported
              as its own classifier
Each is reported overall, per two-tier group (tier from the sample file) and
excluding the rows the annotator flagged in `notes`, with bootstrap 95% CIs.

The keyword classifier emits 'unknown' for messages that match no rule.
Its kappa is reported both over all commits ('unknown' counts as a
disagreement) and over the commits it labelled.

Section D repeats the RQ2 comparison on the validation sample itself: the
per-tier intent shares under the human labels and under each classifier,
chi-square + Cramer's V of each agent tier vs the human-authored baseline
(as in analyze_rq2.py), and each classifier's gap from the human labels.

The script stops if any gold label is missing or invalid; --partial scores
only the rows labelled so far. --second-annotator adds inter-annotator kappa
and lists disagreements.

Outputs:
  results/intent_kappa_results.txt

Usage:
    python src/compute_intent_kappa.py \
        --gold results/intent_annotation_sheet.xlsx \
        --sample results/intent_validation_sample_full.csv \
        --intent-kw data/death_commit_messages.csv \
        --output results/intent_kappa_results.txt
"""

import argparse
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import P

INTENT_LABELS = ['corrective', 'perfective', 'adaptive', 'preventive']
HUMAN_SUPERVISED_TIERS = {'pair_programmer', 'batch_generator'}
CLASSIFIERS = ['keyword']
LLM_RUNS = {
    'deepseek': P('results/intent_llm_deepseek_t0.csv'),
    'gpt_oss': P('results/intent_llm_gpt_oss_t0.csv'),
    'minimax': P('results/intent_llm_minimax_t0.csv'),
}
N_BOOT = 2000
SEED = 42
_NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def read_xlsx(path: str) -> pd.DataFrame:
    """Read the first sheet of an .xlsx with the standard library."""
    z = zipfile.ZipFile(path)
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', _NS):
            shared.append(''.join(t.text or '' for t in si.iter(f'{{{_NS["m"]}}}t')))
    sheet = sorted(n for n in z.namelist() if n.startswith('xl/worksheets/sheet'))[0]
    rows = []
    for r in ET.fromstring(z.read(sheet)).iter(f'{{{_NS["m"]}}}row'):
        row = {}
        for c in r.findall('m:c', _NS):
            col = re.match(r'[A-Z]+', c.get('r')).group()
            v, inline = c.find('m:v', _NS), c.find('m:is', _NS)
            if c.get('t') == 's' and v is not None:
                row[col] = shared[int(v.text)]
            elif inline is not None:
                row[col] = ''.join(t.text or '' for t in inline.iter(f'{{{_NS["m"]}}}t'))
            else:
                row[col] = v.text if v is not None else None
        rows.append(row)
    header = rows[0]
    return pd.DataFrame(rows[1:]).rename(columns=header)[list(header.values())]


def read_sheet(path: str) -> pd.DataFrame:
    return read_xlsx(path) if path.endswith('.xlsx') else pd.read_csv(path)


def read_labels(path: str, partial: bool, who: str) -> pd.Series:
    """human_label by sample_id; stop on missing/invalid labels unless partial."""
    sheet = read_sheet(path)
    labels = sheet.set_index(sheet['sample_id'].str.strip())['human_label']
    labels = labels.fillna('').astype(str).str.strip().str.lower()
    missing = labels == ''
    invalid = ~missing & ~labels.isin(INTENT_LABELS)
    if invalid.any():
        sys.exit(f'{who}: {invalid.sum()} invalid labels, e.g. '
                 f'{labels[invalid].head(5).to_dict()} (allowed: {INTENT_LABELS})')
    if missing.any() and not partial:
        sys.exit(f'{who}: {missing.sum()} of {len(labels)} rows have no human_label '
                 f'(e.g. {list(labels[missing].index[:5])}); finish labelling or pass --partial')
    if missing.all():
        sys.exit(f'{who}: no labels yet')
    return labels[~missing]


def agreement(gold: pd.Series, pred: pd.Series) -> str:
    """Cohen's kappa with a bootstrap 95% CI, and raw agreement."""
    labels = INTENT_LABELS + ['unknown', 'missing']
    g, p = gold.to_numpy(), pred.to_numpy()
    kappa = cohen_kappa_score(g, p, labels=labels)
    rng = np.random.default_rng(SEED)
    boot = [cohen_kappa_score(g[i], p[i], labels=labels)
            for i in (rng.integers(0, len(g), len(g)) for _ in range(N_BOOT))]
    return (f'kappa={kappa:.3f} [{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]'
            f'  agreement={(g == p).mean() * 100:.1f}%  (n={len(g)})')


def tier_test(df: pd.DataFrame, col: str, tier: str) -> str:
    """Chi-square + Cramer's V of one tier vs the human-authored baseline."""
    sub = df[df['two_tier'].isin([tier, 'human'])]
    table = pd.crosstab(sub['two_tier'], sub[col])
    chi2, p, _, _ = chi2_contingency(table)
    v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
    return f'chi2={chi2:.2f} p={p:.3f} V={v:.3f}'


def run(gold_path: str, sample: str, intent_kw: str,
        output: str, partial: bool, second: str, corpus: str = None) -> None:

    print('Loading data...')
    gold = read_labels(gold_path, partial, 'gold')
    sheet = read_sheet(gold_path)
    noted = set(sheet.loc[sheet['notes'].fillna('').astype(str).str.strip() != '', 'sample_id'])
    samp = pd.read_csv(sample).set_index('sample_id')
    kw = pd.read_csv(intent_kw)

    df = samp.loc[gold.index, ['slug', 'sha', 'tier', 'keyword_label']].assign(gold=gold)
    df['two_tier'] = df['tier'].map(
        lambda t: 'human' if t == 'human' else
        'human_supervised' if t in HUMAN_SUPERVISED_TIERS else 'fully_autonomous')
    df = df.reset_index().merge(
        kw[['slug', 'sha', 'intent']].rename(columns={'intent': 'keyword'}),
        on=['slug', 'sha'], how='left')
    classifiers = list(CLASSIFIERS)
    runs = {}
    for col, path in LLM_RUNS.items():
        try:
            runs[col] = pd.read_csv(path)
        except FileNotFoundError:
            continue
        df = df.merge(runs[col][['slug', 'sha', 'label']].rename(columns={'label': col}),
                      on=['slug', 'sha'], how='left')
        classifiers.append(col)
    for col in classifiers:
        df[col] = df[col].fillna('missing').astype(str).str.strip().str.lower()
    df['noted'] = df['sample_id'].isin(noted)

    out_lines = []

    # ── SECTION A: Kappa vs gold ───────────────────────────────────────
    out_lines.append('\n' + '='*70)
    out_lines.append("SECTION A: Cohen's kappa vs human gold labels (primary intent)")
    out_lines.append('='*70)
    out_lines.append(f'\nGold labels: {gold_path}  ({len(df)} of {len(samp)} commits labelled'
                     + (', PARTIAL' if len(df) < len(samp) else '') + ')')
    out_lines.append('Gold distribution: ' + ', '.join(
        f'{k}={v}' for k, v in df['gold'].value_counts().items()))
    out_lines.append('By two-tier group: ' + ', '.join(
        f'{k}={v}' for k, v in df['two_tier'].value_counts().items()))

    out_lines.append('')
    for col in classifiers:
        out_lines.append(f'  {col:17s} {agreement(df["gold"], df[col])}')
    labelled = df['keyword'].isin(INTENT_LABELS)
    out_lines.append(f'  {"keyword (labelled)":17s} '
                     f'{agreement(df.loc[labelled, "gold"], df.loc[labelled, "keyword"])}'
                     f'  [{(~labelled).sum()} unknown dropped]')

    out_lines.append('\nBy two-tier group:')
    for tier in ['human_supervised', 'fully_autonomous', 'human']:
        sub = df[df['two_tier'] == tier]
        if len(sub) == 0:
            continue
        out_lines.append(f'  {tier} (n={len(sub)}):')
        for col in classifiers:
            out_lines.append(f'    {col:17s} {agreement(sub["gold"], sub[col])}')

    clear = df[~df['noted']]
    out_lines.append(f'\nExcluding the {df["noted"].sum()} rows flagged in notes (n={len(clear)}):')
    for col in classifiers:
        out_lines.append(f'  {col:17s} {agreement(clear["gold"], clear[col])}')

    if corpus and os.path.exists(corpus):
        cur = pd.read_csv(corpus)[['slug', 'sha']].drop_duplicates()
        inc = df.merge(cur.assign(_in=1), on=['slug', 'sha'], how='left')['_in'].notna().values
        sub = df[inc]
        out_lines.append(f'\nRestricted to the {len(sub)} sample commits that are modification commits '
                         f'in {corpus} (the rest are not modification commits in the current corpus):')
        for col in classifiers:
            out_lines.append(f'  {col:17s} {agreement(sub["gold"], sub[col])}')

    # ── SECTION B: Provenance checks ───────────────────────────────────
    out_lines.append('\n' + '='*70)
    out_lines.append('SECTION B: Label provenance checks')
    out_lines.append('='*70)
    out_lines.append(f"\nkeyword (death_commit_messages) == sample keyword_label: "
                     f"{(df['keyword'] == df['keyword_label']).sum()}/{len(df)}")
    for col, run in runs.items():
        sample_run = run.merge(df[['slug', 'sha']], on=['slug', 'sha'])
        content = sample_run['raw_content'].fillna('').astype(str).str.strip()
        out_lines.append(
            f"{col}: {len(sample_run)} calls; finish_reason=length {(sample_run['finish_reason'] == 'length').sum()}, "
            f"empty {(content == '').sum()}, unparseable {(sample_run['label'] == 'unparseable').sum()}, "
            f"api_error {(sample_run['label'] == 'api_error').sum()}, "
            f"retried at {sample_run['max_tokens'].max():.0f} tokens {(sample_run['max_tokens'] > sample_run['max_tokens'].min()).sum()}; "
            f"tokens in {sample_run['prompt_tokens'].sum():,.0f} / out {sample_run['completion_tokens'].sum():,.0f} "
            f"(median out {sample_run['completion_tokens'].median():.0f})")
    for col in classifiers:
        n_missing = (df[col] == 'missing').sum()
        if n_missing:
            out_lines.append(f'{col}: {n_missing} commits without a label')
    for col in runs:
        out_lines.append(f"gold identical to {col}: {(df['gold'] == df[col]).sum()}/{len(df)} "
                         f"(should be well below 100% for an independent annotation)")

    # ── SECTION D: RQ2 comparison on the validation sample ────────────
    out_lines.append('\n' + '='*70)
    out_lines.append('SECTION D: RQ2 comparison on the validation sample (same commits)')
    out_lines.append('Intent shares (%) per tier; gap = classifier share - human-label share')
    out_lines.append('='*70)
    methods = ['gold'] + classifiers
    cats = INTENT_LABELS + ['unknown']
    for tier in ['human_supervised', 'fully_autonomous', 'human']:
        sub = df[df['two_tier'] == tier]
        out_lines.append(f'\n  {tier} (n={len(sub)}):')
        out_lines.append('    ' + f'{"method":17s}' + ''.join(f'{c:>12s}' for c in cats))
        gold_share = sub['gold'].value_counts(normalize=True) * 100
        for col in methods:
            share = sub[col].value_counts(normalize=True) * 100
            out_lines.append('    ' + f'{col:17s}' + ''.join(
                f'{share.get(c, 0):12.1f}' for c in cats))
            if col != 'gold':
                out_lines.append('    ' + f'{"  gap vs human":17s}' + ''.join(
                    f'{share.get(c, 0) - gold_share.get(c, 0):+12.1f}' for c in cats))
    out_lines.append('\n  Tier vs human-authored baseline (chi-square, Cramer\'s V):')
    for col in methods:
        out_lines.append(f'    {col:17s} HS: {tier_test(df, col, "human_supervised")}   '
                         f'FA: {tier_test(df, col, "fully_autonomous")}')

    # ── SECTION C: Inter-annotator agreement ───────────────────────────
    if second:
        other = read_labels(second, partial, 'second annotator')
        both = df.set_index('sample_id')['gold'].to_frame().join(
            other.rename('second'), how='inner')
        out_lines.append('\n' + '='*70)
        out_lines.append('SECTION C: Inter-annotator agreement')
        out_lines.append('='*70)
        out_lines.append(f'\nSecond annotator: {second}')
        out_lines.append(f'  {agreement(both["gold"], both["second"])}')
        diff = both[both['gold'] != both['second']]
        out_lines.append(f'  Disagreements ({len(diff)}): ' +
                         ', '.join(f'{i} ({r.gold}/{r.second})' for i, r in diff.iterrows()))

    result_text = '\n'.join(out_lines)
    print(result_text)
    with open(output, 'w') as f:
        f.write(result_text)
    print(f'\nResults written to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold',         default=P('results/intent_annotation_sheet.xlsx'))
    parser.add_argument('--sample',       default=P('results/intent_validation_sample_full.csv'))
    parser.add_argument('--intent-kw',    default=P('data/death_commit_messages.csv'))
    parser.add_argument('--output',       default=P('results/intent_kappa_results.txt'))
    parser.add_argument('--corpus',       default=P('data/death_commit_messages_v5.csv'),
                        help='current modification-commit table; kappa is also reported on the sample commits it contains')
    parser.add_argument('--partial', action='store_true',
                        help='score only the rows labelled so far')
    parser.add_argument('--second-annotator', default=None,
                        help="second annotator's sheet, for inter-annotator kappa")
    args = parser.parse_args()
    run(args.gold, args.sample, args.intent_kw,
        args.output, args.partial, args.second_annotator, args.corpus)
