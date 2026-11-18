"""
analyze_intent_aggregation.py

How the RQ2 intent conclusion depends on the classifier and on how LLM
labels are combined, measured against the human labels on the 385-commit
validation sample.

Conditions (each one is a full set of labels for the sample):
  1. Individual classifiers: keyword, deepseek, gpt_oss, minimax (the LLMs
     at temperature 0).
  2. Unweighted majority vote over the 3 LLMs (mv3). Ties: '_undecided'
     leaves them undecided;
     '_tiebest' gives them to the highest-kappa model among the tied voters,
     with kappa fitted on the training folds only.
  3. Kappa-weighted vote (wv3): each model's weight is its kappa vs the
     human labels on the training folds (stratified 10-fold CV, seed 42);
     every commit is scored by weights that never saw its label. The
     '_insample' version fits weights on all 385 and is optimistic by
     construction; it is reported only to show the gap.
  4. Dawid-Skene (ds3, ds3kw): EM latent-class aggregation using no
     human labels; ds3kw adds keyword as a fourth judge.
  5. Misclassification-corrected tier shares: for each classifier, invert
     its confusion matrix P(pred | true) estimated against the human labels
     to recover per-tier intent prevalence, with bootstrap CIs (commits
     resampled, matrix re-estimated each time). Negative solutions are
     clipped to 0 and renormalised (flagged). On this sample the matrix is
     estimated from the same commits, so corrected shares sit close to the
     human shares by construction; the method matters for the full corpus.

Per condition: kappa vs human (bootstrap 95% CI) overall, per two-tier group
and excluding the rows the annotator flagged in `notes`; coverage (share not
undecided); per-tier intent shares; chi-square + Cramer's V for HS vs human
and FA vs human; gap from the human-label shares; and whether the
conclusion matches the human labels on levels (all gaps within
LEVEL_TOLERANCE_PP), direction (sign of tier - human for each intent) and
significance (p < 0.05 for HS and FA).

Full corpus (--full-corpus): the same approaches on every
modification commit, tier-mapped through the line data exactly as in
analyze_rq2.py (observation window applied, majority birth tier per
(slug, death_sha)). Reports per approach: per-tier intent shares and
chi-square/V of HS and FA vs human (RQ2); misclassification-corrected tier
shares with the confusion matrix from the 385 human labels (bootstrap over
both the 385 and the corpus commits); and the RQ4 contrast of intent by
modifier type (human vs agent modifier). Weighted-vote weights and the
majority-vote tie order come from kappa on all 385 human labels.

Outputs:
  results/intent_aggregation_results.txt (+ .csv)
  results/intent_aggregation_full_corpus.txt (+ .csv) with --full-corpus

Usage:
    python src/analyze_intent_aggregation.py \
        --gold results/intent_annotation_sheet.xlsx \
        --output results/intent_aggregation_results.txt
    python src/analyze_intent_aggregation.py --full-corpus \
        --line-data data/survival_events_line_v5_analysis.csv \
        --corpus-kw data/death_commit_messages_v5.csv \
        --modifier-authors results/modifier_authors.csv \
        --output results/intent_aggregation_full_corpus.txt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import P
from compute_intent_kappa import read_labels, read_sheet  # noqa: E402

INTENT_LABELS = ['corrective', 'perfective', 'adaptive', 'preventive']
KAPPA_LABELS = INTENT_LABELS + ['unknown', 'undecided', 'missing']
HUMAN_SUPERVISED_TIERS = {'pair_programmer', 'batch_generator'}
TIERS = ['human_supervised', 'fully_autonomous', 'human']
NEW_LLMS = ['deepseek', 'gpt_oss', 'minimax']
LLM_RUNS = {
    'deepseek': P('results/intent_llm_deepseek_t0.csv'),
    'gpt_oss': P('results/intent_llm_gpt_oss_t0.csv'),
    'minimax': P('results/intent_llm_minimax_t0.csv'),
}
FULL_RUNS = {m: P(f'results/intent_{m}_full_corpus_t0.csv') for m in NEW_LLMS}
SEED = 42
N_FOLDS = 10
N_BOOT = 1000
LEVEL_TOLERANCE_PP = 5.0
ALPHA = 0.05


# ── Data ──────────────────────────────────────────────────────────────────

def load(gold_path: str, sample: str, intent_kw: str) -> pd.DataFrame:
    gold = read_labels(gold_path, partial=False, who='gold')
    sheet = read_sheet(gold_path)
    noted = set(sheet.loc[sheet['notes'].fillna('').astype(str).str.strip() != '', 'sample_id'])
    samp = pd.read_csv(sample).set_index('sample_id')
    df = samp.loc[gold.index, ['slug', 'sha', 'tier']].assign(gold=gold).reset_index()
    df['two_tier'] = df['tier'].map(
        lambda t: 'human' if t == 'human' else
        'human_supervised' if t in HUMAN_SUPERVISED_TIERS else 'fully_autonomous')
    df['noted'] = df['sample_id'].isin(noted)

    kw = pd.read_csv(intent_kw)[['slug', 'sha', 'intent']].rename(columns={'intent': 'keyword'})
    df = df.merge(kw, on=['slug', 'sha'], how='left')
    for col, path in LLM_RUNS.items():
        run = pd.read_csv(path)[['slug', 'sha', 'label']]
        df = df.merge(run.rename(columns={'label': col}), on=['slug', 'sha'], how='left')
    for col in ['keyword'] + NEW_LLMS:
        df[col] = df[col].fillna('missing').astype(str).str.strip().str.lower()
        df.loc[~df[col].isin(INTENT_LABELS + ['unknown']), col] = 'missing'
    return df


def kappa(gold, pred) -> float:
    return cohen_kappa_score(np.asarray(gold), np.asarray(pred), labels=KAPPA_LABELS)


# ── Aggregation ───────────────────────────────────────────────────────────

def vote(votes: pd.DataFrame, weights: dict, tie_order: list) -> pd.Series:
    """Weighted plurality per row. Ties go to the first model in tie_order
    that voted for one of the tied labels, or 'undecided' if tie_order is
    empty."""
    out = []
    for _, row in votes.iterrows():
        score = {}
        for model, label in row.items():
            if label in INTENT_LABELS:
                score[label] = score.get(label, 0.0) + weights[model]
        if not score:
            out.append('undecided')
            continue
        best = max(score.values())
        top = [lab for lab, s in score.items() if np.isclose(s, best)]
        if len(top) == 1:
            out.append(top[0])
            continue
        winner = next((row[m] for m in tie_order if row[m] in top), 'undecided')
        out.append(winner)
    return pd.Series(out, index=votes.index)


def cv_vote(df: pd.DataFrame, models: list, weighted: bool, tie_best: bool) -> pd.Series:
    """Out-of-fold predictions: kappa weights / tie order fitted on training folds."""
    pred = pd.Series('undecided', index=df.index)
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for train, test in folds.split(df, df['gold']):
        tr = df.iloc[train]
        k = {m: max(kappa(tr['gold'], tr[m]), 0.0) for m in models}
        weights = k if weighted else {m: 1.0 for m in models}
        order = sorted(models, key=lambda m: -k[m]) if (tie_best or weighted) else []
        pred.iloc[test] = vote(df.iloc[test][models], weights, order).values
    return pred


def insample_vote(df: pd.DataFrame, models: list) -> pd.Series:
    k = {m: max(kappa(df['gold'], df[m]), 0.0) for m in models}
    return vote(df[models], k, sorted(models, key=lambda m: -k[m]))


def dawid_skene(df: pd.DataFrame, judges: list, n_iter: int = 200, tol: float = 1e-8) -> pd.Series:
    """Dawid-Skene EM. True classes = INTENT_LABELS; responses may include
    'unknown'/'missing'. Initialised from majority vote; Laplace smoothing."""
    classes = INTENT_LABELS
    responses = sorted(set(np.concatenate([df[j].unique() for j in judges])))
    K, R, N = len(classes), len(responses), len(df)
    resp_idx = {r: i for i, r in enumerate(responses)}
    obs = np.stack([df[j].map(resp_idx).to_numpy() for j in judges], axis=1)  # N x J
    T = np.full((N, K), 1e-3)
    for n in range(N):
        for r in obs[n]:
            lab = responses[r]
            if lab in classes:
                T[n, classes.index(lab)] += 1
    T /= T.sum(1, keepdims=True)
    prev_ll = -np.inf
    for _ in range(n_iter):
        prior = T.mean(0)
        conf = np.full((len(judges), K, R), 0.01)
        for jj in range(len(judges)):
            for r in range(R):
                conf[jj, :, r] += T[obs[:, jj] == r].sum(0)
        conf /= conf.sum(2, keepdims=True)
        log_t = np.log(prior)[None, :].repeat(N, 0)
        for jj in range(len(judges)):
            log_t += np.log(conf[jj][:, obs[:, jj]]).T
        ll = np.logaddexp.reduce(log_t, axis=1).sum()
        T = np.exp(log_t - log_t.max(1, keepdims=True))
        T /= T.sum(1, keepdims=True)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    return pd.Series(np.array(classes)[T.argmax(1)], index=df.index)


def _test_dawid_skene() -> None:
    """Sanity check: three noisy judges of a known truth; DS should recover it
    better than the noisiest judge."""
    rng = np.random.default_rng(SEED)
    truth = rng.choice(INTENT_LABELS, 600, p=[0.25, 0.25, 0.4, 0.1])
    def noisy(acc):
        flip = rng.random(600) > acc
        return np.where(flip, rng.choice(INTENT_LABELS, 600), truth)
    t = pd.DataFrame({'a': noisy(0.85), 'b': noisy(0.75), 'c': noisy(0.55)})
    ds = dawid_skene(t, ['a', 'b', 'c'])
    assert (ds.to_numpy() == truth).mean() > (t['c'].to_numpy() == truth).mean() + 0.1


# ── Evaluation ────────────────────────────────────────────────────────────

def kappa_ci(gold: pd.Series, pred: pd.Series) -> tuple:
    g, p = gold.to_numpy(), pred.to_numpy()
    rng = np.random.default_rng(SEED)
    boot = [kappa(g[i], p[i]) for i in (rng.integers(0, len(g), len(g)) for _ in range(N_BOOT))]
    return kappa(g, p), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def shares(labels: pd.Series) -> pd.Series:
    """Intent shares (%) among decided labels."""
    decided = labels[labels.isin(INTENT_LABELS + ['unknown'])]
    return decided.value_counts(normalize=True).reindex(INTENT_LABELS + ['unknown'], fill_value=0) * 100


def tier_chi2(df: pd.DataFrame, col: str, tier: str) -> tuple:
    sub = df[df['two_tier'].isin([tier, 'human']) & df[col].isin(INTENT_LABELS + ['unknown'])]
    table = pd.crosstab(sub['two_tier'], sub[col])
    table = table.loc[:, table.sum() > 0]
    chi2, p, _, _ = chi2_contingency(table)
    return chi2, p, np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))


def corrected_shares(df: pd.DataFrame, col: str) -> tuple:
    """Per-tier prevalence after inverting P(pred | true); bootstrap CIs."""
    responses = INTENT_LABELS + ['unknown', 'missing']

    def estimate(d: pd.DataFrame):
        conf = pd.crosstab(d['gold'], d[col]).reindex(
            index=INTENT_LABELS, columns=responses, fill_value=0).to_numpy(float) + 0.5
        conf /= conf.sum(1, keepdims=True)
        out, clipped = {}, False
        for tier in TIERS:
            q = d.loc[d['two_tier'] == tier, col].value_counts(normalize=True).reindex(
                responses, fill_value=0).to_numpy()
            p, *_ = np.linalg.lstsq(conf.T, q, rcond=None)
            if (p < 0).any():
                clipped = True
                p = np.clip(p, 0, None)
            out[tier] = p / p.sum() * 100
        return out, clipped

    point, clipped = estimate(df)
    rng = np.random.default_rng(SEED)
    boots = {t: [] for t in TIERS}
    n_clip = 0
    for _ in range(N_BOOT):
        b, c = estimate(df.iloc[rng.integers(0, len(df), len(df))])
        n_clip += c
        for t in TIERS:
            boots[t].append(b[t])
    ci = {t: (np.percentile(boots[t], 2.5, axis=0), np.percentile(boots[t], 97.5, axis=0))
          for t in TIERS}
    return point, ci, clipped, n_clip


def evaluate(df: pd.DataFrame, col: str, gold_ref: dict) -> dict:
    k, lo, hi = kappa_ci(df['gold'], df[col])
    row = {'condition': col, 'kappa': k, 'kappa_lo': lo, 'kappa_hi': hi,
           'coverage_pct': df[col].isin(INTENT_LABELS).mean() * 100,
           'n_undecided': int((df[col] == 'undecided').sum())}
    clear = df[~df['noted']]
    row['kappa_excl_noted'] = kappa(clear['gold'], clear[col])
    for tier in TIERS:
        sub = df[df['two_tier'] == tier]
        row[f'kappa_{tier}'] = kappa(sub['gold'], sub[col])
        s = shares(sub[col])
        for lab in INTENT_LABELS + ['unknown']:
            row[f'{tier}_{lab}_pct'] = s[lab]
            row[f'{tier}_{lab}_gap_pp'] = s[lab] - gold_ref['shares'][tier][lab]
    for tier, key in [('human_supervised', 'HS'), ('fully_autonomous', 'FA')]:
        chi2, p, v = tier_chi2(df, col, tier)
        row[f'{key}_chi2'], row[f'{key}_p'], row[f'{key}_V'] = chi2, p, v
    gaps = [abs(row[f'{t}_{lab}_gap_pp']) for t in TIERS for lab in INTENT_LABELS + ['unknown']]
    row['levels_match'] = max(gaps) <= LEVEL_TOLERANCE_PP
    signs = []
    for tier in ['human_supervised', 'fully_autonomous']:
        for lab in INTENT_LABELS:
            d = row[f'{tier}_{lab}_pct'] - row[f'human_{lab}_pct']
            signs.append(np.sign(d) == gold_ref['signs'][(tier, lab)])
    row['direction_match_of_8'] = int(sum(signs))
    row['significance_match'] = all(
        (row[f'{k}_p'] < ALPHA) == (gold_ref[f'{k}_p'] < ALPHA) for k in ['HS', 'FA'])
    return row


def run(gold_path: str, sample: str, intent_kw: str, output: str) -> None:
    _test_dawid_skene()
    print('Loading data...')
    df = load(gold_path, sample, intent_kw)

    # Conditions
    df['mv3_undecided'] = vote(df[NEW_LLMS], {m: 1.0 for m in NEW_LLMS}, [])
    df['mv3_tiebest'] = cv_vote(df, NEW_LLMS, weighted=False, tie_best=True)
    df['wv3_cv'] = cv_vote(df, NEW_LLMS, weighted=True, tie_best=True)
    df['wv3_insample'] = insample_vote(df, NEW_LLMS)
    df['ds3'] = dawid_skene(df, NEW_LLMS)
    df['ds3kw'] = dawid_skene(df, NEW_LLMS + ['keyword'])
    conditions = (['keyword'] + NEW_LLMS +
                  ['mv3_undecided', 'mv3_tiebest', 'wv3_cv', 'wv3_insample', 'ds3', 'ds3kw'])

    # Human-label reference
    ref = {'shares': {t: shares(df.loc[df['two_tier'] == t, 'gold']) for t in TIERS}}
    ref['signs'] = {(t, lab): np.sign(ref['shares'][t][lab] - ref['shares']['human'][lab])
                    for t in ['human_supervised', 'fully_autonomous'] for lab in INTENT_LABELS}
    for tier, key in [('human_supervised', 'HS'), ('fully_autonomous', 'FA')]:
        _, ref[f'{key}_p'], _ = tier_chi2(df, 'gold', tier)
    rows = [evaluate(df, 'gold', ref)] + [evaluate(df, c, ref) for c in conditions]
    table = pd.DataFrame(rows)
    table.to_csv(Path(output).with_suffix('.csv'), index=False)

    out = ['=' * 70, 'SECTION A: Agreement with human labels (n=%d; %d flagged in notes)'
           % (len(df), df['noted'].sum()), '=' * 70,
           f'{"condition":17s} {"kappa [95% CI]":24s} {"excl.notes":>10s} {"HS":>6s} {"FA":>6s} '
           f'{"human":>6s} {"coverage":>9s}']
    for r in rows[1:]:
        out.append(f"{r['condition']:17s} {r['kappa']:.3f} [{r['kappa_lo']:.3f}, {r['kappa_hi']:.3f}]"
                   f"   {r['kappa_excl_noted']:10.3f} {r['kappa_human_supervised']:6.3f} "
                   f"{r['kappa_fully_autonomous']:6.3f} {r['kappa_human']:6.3f} "
                   f"{r['coverage_pct']:8.1f}%" + (f"  ({r['n_undecided']} undecided)" if r['n_undecided'] else ''))
    out.append('  *_insample: weights fitted on the same labels they are scored on (optimistic).')

    out += ['', '=' * 70, f'SECTION B: RQ2 conclusion per condition (same {len(df)} commits)', '=' * 70,
            'Shares (%) corrective/perfective/adaptive/preventive[/unknown] per tier; '
            'chi-square vs human-authored baseline', '']
    for r in rows:
        line = f"{r['condition']:17s}"
        for t, short in zip(TIERS, ['HS', 'FA', 'Hu']):
            vals = '/'.join(f"{r[f'{t}_{lab}_pct']:.0f}" for lab in INTENT_LABELS)
            unk = r[f'{t}_unknown_pct']
            line += f"  {short} {vals}" + (f"/{unk:.0f}u" if unk else '')
        line += (f"  | HS p={r['HS_p']:.3f} V={r['HS_V']:.2f}  FA p={r['FA_p']:.3f} V={r['FA_V']:.2f}")
        if r['condition'] != 'gold':
            line += (f"  | levels {'ok' if r['levels_match'] else 'DIFF'}"
                     f", direction {r['direction_match_of_8']}/8"
                     f", signif {'ok' if r['significance_match'] else 'DIFF'}")
        out.append(line)
    out.append(f'  levels ok = every per-tier share within {LEVEL_TOLERANCE_PP:.0f} pp of the human labels; '
               'direction = sign of (tier - human) per intent, 8 comparisons; '
               'signif = same p<0.05 verdict for HS and FA.')

    out += ['', '=' * 70, 'SECTION C: Misclassification-corrected tier shares (bootstrap 95% CI)',
            f'Confusion matrix estimated on the same {len(df)} commits: corrected shares are close',
            'to the human shares by construction here; the correction is meant for the full corpus.',
            '=' * 70]
    for col in ['keyword'] + NEW_LLMS:
        point, ci, clipped, n_clip = corrected_shares(df, col)
        out.append(f'\n  {col}' + (f'  [clipped in point estimate]' if clipped else '') +
                   f'  [{n_clip}/{N_BOOT} bootstrap draws clipped]')
        for t in TIERS:
            out.append(f'    {t:17s} ' + '  '.join(
                f'{lab[:4]} {point[t][i]:5.1f} [{ci[t][0][i]:5.1f}, {ci[t][1][i]:5.1f}]'
                for i, lab in enumerate(INTENT_LABELS)))
    out.append('\n  human labels      ' + '  '.join(
        f'{t}: ' + '/'.join(f'{ref["shares"][t][lab]:.1f}' for lab in INTENT_LABELS) for t in TIERS))

    text = '\n'.join(out)
    print(text)
    Path(output).write_text(text)
    print(f'\nResults written to {output} and {Path(output).with_suffix(".csv")}')


# ── Full corpus ───────────────────────────────────────────────────────────

def tier_map_from_line(line_data: str) -> pd.DataFrame:
    """(slug, sha) -> birth tier, as in analyze_rq2.get_majority_tier."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config.observation_window import apply_observation_window
    from analyze_rq2 import get_majority_tier
    line = pd.read_csv(line_data, usecols=[
        'slug', 'pr_number', 'is_agent', 'agent_tier', 'unit_id', 'death_sha',
        'event', 'duration_days', 'birth_date', 'death_date'])
    line = apply_observation_window(line, label='line').dropna(subset=['death_sha'])
    return get_majority_tier(line)


def corrected_full(sample: pd.DataFrame, corpus: pd.DataFrame, col: str) -> tuple:
    """Corpus tier shares corrected with P(pred | true) from the 385 human labels."""
    responses = INTENT_LABELS + ['unknown', 'missing', 'undecided']

    def estimate(smp, crp):
        conf = pd.crosstab(smp['gold'], smp[col]).reindex(
            index=INTENT_LABELS, columns=responses, fill_value=0).to_numpy(float) + 0.5
        conf /= conf.sum(1, keepdims=True)
        out, clipped = {}, False
        for tier in TIERS:
            q = crp.loc[crp['two_tier'] == tier, col].value_counts(normalize=True).reindex(
                responses, fill_value=0).to_numpy()
            p, *_ = np.linalg.lstsq(conf.T, q, rcond=None)
            if (p < 0).any():
                clipped, p = True, np.clip(p, 0, None)
            out[tier] = p / p.sum() * 100
        return out, clipped

    point, clipped = estimate(sample, corpus)
    rng = np.random.default_rng(SEED)
    boots, n_clip = {t: [] for t in TIERS}, 0
    for _ in range(N_BOOT):
        b, c = estimate(sample.iloc[rng.integers(0, len(sample), len(sample))],
                        corpus.iloc[rng.integers(0, len(corpus), len(corpus))])
        n_clip += c
        for t in TIERS:
            boots[t].append(b[t])
    ci = {t: (np.percentile(boots[t], 2.5, axis=0), np.percentile(boots[t], 97.5, axis=0))
          for t in TIERS}
    return point, ci, clipped, n_clip


def modifier_test(df: pd.DataFrame, col: str) -> tuple:
    """RQ4: intent by modifier type (agent vs human modifier)."""
    sub = df[df['modifier_type'].isin(['agent', 'human']) & df[col].isin(INTENT_LABELS + ['unknown'])]
    table = pd.crosstab(sub['modifier_type'], sub[col])
    table = table.loc[:, table.sum() > 0]
    chi2, p, _, _ = chi2_contingency(table)
    shares_by = {mt: sub.loc[sub['modifier_type'] == mt, col].value_counts(normalize=True)
                 .reindex(INTENT_LABELS + ['unknown'], fill_value=0) * 100 for mt in ['human', 'agent']}
    return chi2, p, np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))), shares_by, len(sub)


def run_full(gold_path: str, sample: str, intent_kw: str, corpus_kw: str,
             line_data: str, modifier_authors: str, output: str) -> None:
    _test_dawid_skene()
    print('Loading data...')
    smp = load(gold_path, sample, intent_kw)
    k = {m: max(kappa(smp['gold'], smp[m]), 0.0) for m in NEW_LLMS}
    order = sorted(NEW_LLMS, key=lambda m: -k[m])

    kw = pd.read_csv(corpus_kw).drop_duplicates(['slug', 'sha'])
    corpus = kw[['slug', 'sha', 'intent']].rename(columns={'intent': 'keyword'})
    for m, path in FULL_RUNS.items():
        run = pd.read_csv(path).drop_duplicates(['slug', 'sha'], keep='last')
        corpus = corpus.merge(run[['slug', 'sha', 'label']].rename(columns={'label': m}),
                              on=['slug', 'sha'], how='left')
    for col in ['keyword'] + NEW_LLMS:
        corpus[col] = corpus[col].fillna('missing').astype(str).str.strip().str.lower()
        corpus.loc[~corpus[col].isin(INTENT_LABELS + ['unknown']), col] = 'missing'
    n_corpus = len(corpus)

    corpus['mv3_undecided'] = vote(corpus[NEW_LLMS], {m: 1.0 for m in NEW_LLMS}, [])
    corpus['mv3_tiebest'] = vote(corpus[NEW_LLMS], {m: 1.0 for m in NEW_LLMS}, order)
    corpus['wv3'] = vote(corpus[NEW_LLMS], k, order)
    corpus['ds3'] = dawid_skene(corpus, NEW_LLMS)
    corpus['ds3kw'] = dawid_skene(corpus, NEW_LLMS + ['keyword'])
    # same aggregations on the 385, for the correction matrices
    smp['mv3_undecided'] = vote(smp[NEW_LLMS], {m: 1.0 for m in NEW_LLMS}, [])
    smp['mv3_tiebest'] = vote(smp[NEW_LLMS], {m: 1.0 for m in NEW_LLMS}, order)
    smp['wv3'] = vote(smp[NEW_LLMS], k, order)
    # DS on the sample itself: not every validation commit is in the corpus
    smp['ds3'] = dawid_skene(smp, NEW_LLMS)
    smp['ds3kw'] = dawid_skene(smp, NEW_LLMS + ['keyword'])
    approaches = ['keyword'] + NEW_LLMS + ['mv3_undecided', 'mv3_tiebest', 'wv3', 'ds3', 'ds3kw']

    tiers = tier_map_from_line(line_data)
    corpus = corpus.merge(tiers, on=['slug', 'sha'], how='inner')
    authors = pd.read_csv(modifier_authors)[['slug', 'sha', 'modifier_type']].drop_duplicates(['slug', 'sha'])
    corpus = corpus.merge(authors, on=['slug', 'sha'], how='left')

    rows, out = [], ['=' * 70,
                     f'FULL CORPUS: {n_corpus:,} modification commits; {len(corpus):,} mapped to a tier '
                     f'via {line_data} (window applied)',
                     'By tier: ' + ', '.join(f'{t}={n:,}' for t, n in corpus['two_tier'].value_counts().items()),
                     f'Weights / tie order from kappa on the {len(smp)} human labels: ' +
                     ', '.join(f'{m}={k[m]:.3f}' for m in order), '=' * 70, '',
                     'SECTION A: RQ2 per approach. Shares (%) corrective/perfective/adaptive/preventive[/unknown]',
                     '']
    for col in approaches:
        r = {'approach': col, 'kappa_385': kappa(smp['gold'], smp[col]),
             'n_undecided': int((corpus[col] == 'undecided').sum()),
             'n_missing': int((corpus[col] == 'missing').sum())}
        line = f'{col:14s} k={r["kappa_385"]:.3f}'
        for t, short in zip(TIERS, ['HS', 'FA', 'Hu']):
            sh = shares(corpus.loc[corpus['two_tier'] == t, col])
            for lab in INTENT_LABELS + ['unknown']:
                r[f'{t}_{lab}_pct'] = sh[lab]
            line += f'  {short} ' + '/'.join(f'{sh[lab]:.1f}' for lab in INTENT_LABELS) + \
                    (f'/{sh["unknown"]:.1f}u' if sh['unknown'] else '')
        for tier, key in [('human_supervised', 'HS'), ('fully_autonomous', 'FA')]:
            chi2, p, v = tier_chi2(corpus, col, tier)
            r[f'{key}_chi2'], r[f'{key}_p'], r[f'{key}_V'] = chi2, p, v
            line += f'  | {key} p={p:.2e} V={v:.3f}'
        if r['n_undecided'] or r['n_missing']:
            line += f'  [{r["n_undecided"]} undecided, {r["n_missing"]} missing]'
        chi2, p, v, by, n = modifier_test(corpus, col)
        r.update({'RQ4_chi2': chi2, 'RQ4_p': p, 'RQ4_V': v, 'RQ4_n': n})
        for mt in ['human', 'agent']:
            for lab in INTENT_LABELS:
                r[f'RQ4_{mt}_{lab}_pct'] = by[mt][lab]
        rows.append(r)
        out.append(line)

    out += ['', 'SECTION B: RQ4 per approach. Intent shares (%) by modifier type '
            '(corrective/perfective/adaptive/preventive); chi-square agent vs human modifier', '']
    for r in rows:
        out.append(f'{r["approach"]:14s} human-modified ' +
                   '/'.join(f'{r[f"RQ4_human_{lab}_pct"]:.1f}' for lab in INTENT_LABELS) +
                   '  agent-modified ' + '/'.join(f'{r[f"RQ4_agent_{lab}_pct"]:.1f}' for lab in INTENT_LABELS) +
                   f'  | p={r["RQ4_p"]:.2e} V={r["RQ4_V"]:.3f} (n={r["RQ4_n"]:,})')

    out += ['', f'SECTION C: Misclassification-corrected tier shares (confusion from the {len(smp)} human labels; '
            f'bootstrap 95% CI over the {len(smp)} and the corpus)', '']
    for col in NEW_LLMS + ['mv3_tiebest', 'wv3', 'ds3']:
        point, ci, clipped, n_clip = corrected_full(smp, corpus, col)
        out.append(f'  {col}' + ('  [clipped]' if clipped else '') + f'  [{n_clip}/{N_BOOT} draws clipped]')
        for t in TIERS:
            out.append(f'    {t:17s} ' + '  '.join(
                f'{lab[:4]} {point[t][i]:5.1f} [{ci[t][0][i]:5.1f}, {ci[t][1][i]:5.1f}]'
                for i, lab in enumerate(INTENT_LABELS)))

    pd.DataFrame(rows).to_csv(Path(output).with_suffix('.csv'), index=False)
    text = '\n'.join(out)
    print(text)
    Path(output).write_text(text)
    print(f'\nResults written to {output} and {Path(output).with_suffix(".csv")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold',         default=P('results/intent_annotation_sheet.xlsx'))
    parser.add_argument('--sample',       default=P('results/intent_validation_sample_full.csv'))
    parser.add_argument('--intent-kw',    default=P('data/death_commit_messages.csv'),
                        help='keyword labels for the 385 validation commits')
    parser.add_argument('--corpus-kw',    default=P('data/death_commit_messages_v5.csv'),
                        help='modification commits + keyword labels for the full corpus')
    parser.add_argument('--output',       default=None)
    parser.add_argument('--full-corpus', action='store_true')
    parser.add_argument('--line-data',    default=P('data/survival_events_line_v5_analysis.csv'))
    parser.add_argument('--modifier-authors', default=P('results/modifier_authors.csv'))
    parser.add_argument('--sample-llm-dir', default=None,
                        help='read the validation commits\' LLM labels from <dir>/intent_llm_<m>_t0.csv '
                             '(e.g. the combined labels of src/build_intent_labels_v6.py)')
    args = parser.parse_args()
    if args.sample_llm_dir:
        LLM_RUNS.update({m: os.path.join(args.sample_llm_dir, f'intent_llm_{m}_t0.csv') for m in NEW_LLMS})
    if args.full_corpus:
        run_full(args.gold, args.sample, args.intent_kw, args.corpus_kw,
                 args.line_data, args.modifier_authors,
                 args.output or P('results/intent_aggregation_full_corpus.txt'))
    else:
        run(args.gold, args.sample, args.intent_kw,
            args.output or P('results/intent_aggregation_results.txt'))
