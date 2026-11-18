"""
Intent classifiers vs the two human raters, and
agreement among the three LLMs as a signal.

Labels for the 618 human-labelled corpus commits come from results_v6/labels618/
(gold = first author; LLM labels identical to results_v6/intent_<m>_full_corpus_t0.csv,
checked: 789/789 equal per model; keyword = full-corpus keyword rules).
Combined approaches are computed on the 618 exactly as analyze_intent_aggregation.py
does for its 618-label run (mv3_undecided, mv3_tiebest (10-fold CV tie order),
wv3_cv, ds3, ds3kw), then restricted to the second rater's 100 commits.

Task 1: Cohen kappa (analyze_intent_aggregation.kappa, fixed label set) with
95% percentile bootstrap CI, 2,000 resamples of the 100 commits, seed 42,
against (a) first author and (b) second rater. Message-only LLM variants from
results_v6/round1/c7_message_only_<m>.csv.

Task 2: Fleiss kappa among the three LLMs on the 618 (bootstrap 2,000, seed 42);
agreement patterns and accuracy vs the human label (Wilson 95% CI);
unweighted and stratum-weighted (old/new, weight N_h/n_h as in intent_kappa_v6.py).

Run from repo root: python results_v6/round4/r4_intent_kappa.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import analyze_intent_aggregation as A  # noqa: E402

OUT = 'results_v6/round4'
L = 'results_v6/labels618'
MODELS = A.NEW_LLMS
B = 2000
SEED = 42


def clean(s):
    s = s.fillna('missing').astype(str).str.strip().str.lower()
    return s.where(s.isin(A.INTENT_LABELS + ['unknown']), 'missing')


def build618():
    gold = pd.read_csv(f'{L}/gold.csv')
    samp = pd.read_csv(f'{L}/sample.csv')
    df = samp.merge(gold[['sample_id', 'human_label']], on='sample_id').rename(columns={'human_label': 'gold'})
    df['gold'] = df['gold'].str.strip().str.lower()
    assert len(df) == 618 and df['gold'].isin(A.INTENT_LABELS).all()
    kw = pd.read_csv(f'{L}/keyword.csv').drop_duplicates(['slug', 'sha']).rename(columns={'intent': 'keyword'})
    df = df.merge(kw[['slug', 'sha', 'keyword']], on=['slug', 'sha'], how='left')
    df['keyword'] = clean(df['keyword'])
    for m in MODELS:
        run = pd.read_csv(f'results_v6/intent_{m}_full_corpus_t0.csv')[['slug', 'sha', 'label']]
        run = run.drop_duplicates(['slug', 'sha'], keep='last').rename(columns={'label': m})
        df = df.merge(run, on=['slug', 'sha'], how='left')
        df[m] = clean(df[m])
    df['mv3_undecided'] = A.vote(df[MODELS], {m: 1.0 for m in MODELS}, [])
    df['mv3_tiebest'] = A.cv_vote(df, MODELS, weighted=False, tie_best=True)
    df['wv3_cv'] = A.cv_vote(df, MODELS, weighted=True, tie_best=True)
    df['ds3'] = A.dawid_skene(df, MODELS)
    df['ds3kw'] = A.dawid_skene(df, MODELS + ['keyword'])
    for m in MODELS:
        mo = pd.read_csv(f'results_v6/round1/c7_message_only_{m}.csv')[['slug', 'sha', 'label']]
        mo = mo.drop_duplicates(['slug', 'sha'], keep='last').rename(columns={'label': f'{m}_msgonly'})
        df = df.merge(mo, on=['slug', 'sha'], how='left')
        df[f'{m}_msgonly'] = clean(df[f'{m}_msgonly'])
    return df


def kappa_boot(g, p, idx_draws):
    g, p = np.asarray(g), np.asarray(p)
    k = A.kappa(g, p)
    bs = [A.kappa(g[i], p[i]) for i in idx_draws]
    return k, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fleiss(labels):
    """labels: (n, raters) array of category strings."""
    cats = sorted(set(labels.ravel()))
    n, r = labels.shape
    cnt = np.stack([(labels == c).sum(1) for c in cats], 1).astype(float)
    p_j = cnt.sum(0) / (n * r)
    P_i = ((cnt ** 2).sum(1) - r) / (r * (r - 1))
    Pbar, Pe = P_i.mean(), (p_j ** 2).sum()
    return (Pbar - Pe) / (1 - Pe)


def wilson(k, n, z=1.959964):
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def main():
    df = build618()
    out = []
    # sanity: reproduce the paper's 618 kappas
    out.append('Sanity check on the 618 (should match results_v6/intent_aggregation_results_618.txt / intent_kappa_v6.txt):')
    out.append('  ' + ', '.join(f'{c} {A.kappa(df["gold"], df[c]):.3f}' for c in
                                ['keyword'] + MODELS + ['mv3_undecided', 'mv3_tiebest', 'wv3_cv', 'ds3', 'ds3kw']))

    # ---------------- Task 1 ----------------
    # The second rater's labels are not part of the replication package; Task 1 runs only where they exist.
    if not os.path.exists('results_v6/second_rater/intent/intent_KEY_INTERNAL.csv'):
        out += ['', 'TASK 1 skipped: the second rater\'s labels are not distributed. '
                'Reported values: results_v6/round4/task1_kappa_second_rater_100.csv']
    else:
        key = pd.read_csv('results_v6/second_rater/intent/intent_KEY_INTERNAL.csv')
        sheet = pd.read_excel('results_v6/second_rater/intent/intent_sheet_labelled.xlsx')
        sheet['rater2'] = sheet['label'].astype(str).str.strip().str.lower()
        s = key.merge(sheet[['rater2_id', 'rater2']], on='rater2_id', validate='1:1')
        s = s.merge(df, on=['slug', 'sha'], how='left', validate='1:1')
        assert len(s) == 100 and s['gold'].notna().all() and (s['gold'] == s['rater1_label']).all()
        assert s['rater2'].isin(A.INTENT_LABELS).all()
        rng = np.random.default_rng(SEED)
        draws = [rng.integers(0, len(s), len(s)) for _ in range(B)]
        rows = []
        k, lo, hi = kappa_boot(s['rater1_label'], s['rater2'], draws)
        rows.append(dict(classifier='first author (reference)', vs_first_author=np.nan, fa_lo=np.nan, fa_hi=np.nan,
                         vs_second_rater=k, sr_lo=lo, sr_hi=hi))
        names = {'keyword': 'keyword rules', 'deepseek': 'DeepSeek-V4.1-Flash', 'gpt_oss': 'gpt-oss-120b',
                 'minimax': 'MiniMax-M3.1-Flash', 'mv3_undecided': 'majority vote (ties undecided)',
                 'mv3_tiebest': 'majority vote (CV tie-break)', 'wv3_cv': 'kappa-weighted vote (CV)',
                 'ds3': 'Dawid-Skene (3 LLMs)', 'ds3kw': 'Dawid-Skene (3 LLMs + keyword)',
                 'deepseek_msgonly': 'DeepSeek message-only', 'gpt_oss_msgonly': 'gpt-oss message-only',
                 'minimax_msgonly': 'MiniMax message-only'}
        for c, nm in names.items():
            k1, l1, h1 = kappa_boot(s['rater1_label'], s[c], draws)
            k2, l2, h2 = kappa_boot(s['rater2'], s[c], draws)
            rows.append(dict(classifier=nm, column=c, vs_first_author=k1, fa_lo=l1, fa_hi=h1,
                             vs_second_rater=k2, sr_lo=l2, sr_hi=h2,
                             n_undecided_or_missing=int(s[c].isin(['undecided', 'missing']).sum()),
                             n_unknown=int((s[c] == 'unknown').sum())))
        t1 = pd.DataFrame(rows)
        t1.to_csv(f'{OUT}/task1_kappa_second_rater_100.csv', index=False)
        out += ['', '=' * 90,
                f'TASK 1. Cohen kappa on the second rater\'s 100 intent commits (bootstrap 95% CI, {B} resamples, seed {SEED};',
                'same 2,000 index draws for every row). Sets: ' + ', '.join(f'{k} {v}' for k, v in s['set_x'].value_counts().items()),
                '=' * 90,
                f'{"classifier":34s} {"vs first author":>24s} {"vs second rater":>24s}']
        for r in rows:
            a = '' if pd.isna(r['vs_first_author']) else f'{r["vs_first_author"]:.3f} [{r["fa_lo"]:.3f}, {r["fa_hi"]:.3f}]'
            b = f'{r["vs_second_rater"]:.3f} [{r["sr_lo"]:.3f}, {r["sr_hi"]:.3f}]'
            out.append(f'{r["classifier"]:34s} {a:>24s} {b:>24s}')
        out.append(f'keyword "unknown" on {int((s["keyword"] == "unknown").sum())} of 100; message-only unparseable: '
                   + ', '.join(f'{m} {int((s[m + "_msgonly"] == "missing").sum())}' for m in MODELS)
                   + f'; mv3_undecided ties: {int((s["mv3_undecided"] == "undecided").sum())}')
        out.append('raw % agreement: first author vs second rater '
                   f'{(s["rater1_label"] == s["rater2"]).mean() * 100:.0f}%; ' + ', '.join(
                       f'{c} vs FA {(s[c] == s["rater1_label"]).mean() * 100:.0f}% / vs SR {(s[c] == s["rater2"]).mean() * 100:.0f}%'
                       for c in MODELS + ['mv3_tiebest']))
        # both raters agree subset
        agree = s[s['rater1_label'] == s['rater2']]
        out.append(f'On the {len(agree)} commits where both humans agree, accuracy: ' + ', '.join(
            f'{c} {(agree[c] == agree["rater2"]).mean() * 100:.1f}%' for c in ['keyword'] + MODELS + ['mv3_tiebest', 'ds3']))

    # ---------------- Task 2 ----------------
    journal = set(pd.read_csv('data/aidev_filtered_v1.csv')['slug'])
    corpus = pd.read_csv('data/death_commit_messages_v6confnv.csv')[['slug', 'sha']].drop_duplicates()
    corpus['stratum'] = np.where(corpus['slug'].isin(journal), 'old', 'new')
    Nh = corpus['stratum'].value_counts()
    df['stratum'] = np.where(df['slug'].isin(journal), 'old', 'new')
    nh = df['stratum'].value_counts()
    df['w'] = df['stratum'].map(lambda h: Nh[h] / nh[h])
    lab = df[MODELS].to_numpy()
    fk = fleiss(lab)
    rng = np.random.default_rng(SEED)
    fb = [fleiss(lab[rng.integers(0, len(df), len(df))]) for _ in range(B)]
    nuniq = df[MODELS].nunique(axis=1)
    df['pattern'] = nuniq.map({1: 'all three agree', 2: 'two agree', 3: 'all differ'})

    def maj(r):
        vc = pd.Series(r).value_counts()
        return vc.index[0] if vc.iloc[0] >= 2 else None
    df['agreed_label'] = [maj(r) for r in lab]
    # pairwise Cohen kappa between models
    pair = []
    for i, a_ in enumerate(MODELS):
        for b_ in MODELS[i + 1:]:
            pair.append(f'{a_}-{b_} {A.kappa(df[a_], df[b_]):.3f}')
    out += ['', '=' * 90,
            'TASK 2. Agreement among the three LLMs on the 618 human-labelled corpus commits',
            '=' * 90,
            f'Fleiss kappa (3 LLMs; categories = labels they emit: {sorted(set(lab.ravel()))}) '
            f'{fk:.3f} [95% CI {np.percentile(fb, 2.5):.3f}, {np.percentile(fb, 97.5):.3f}] ({B} bootstrap resamples, seed {SEED})',
            'pairwise Cohen kappa: ' + ', '.join(pair),
            f'strata: ' + ', '.join(f'{h} n={nh[h]} of N={Nh[h]} (w={Nh[h] / nh[h]:.2f})' for h in ['old', 'new']), '',
            f'{"pattern":16s} {"n":>4s} {"share % (unw)":>14s} {"share % (wtd)":>13s}   accuracy vs human: '
            f'{"label used":28s} {"unweighted [Wilson 95%]":>26s} {"weighted":>9s}']
    rows2 = []
    for pat in ['all three agree', 'two agree', 'all differ']:
        g = df[df['pattern'] == pat]
        sh_u = len(g) / len(df) * 100
        sh_w = g['w'].sum() / df['w'].sum() * 100
        if pat == 'all differ':
            evals = [('MiniMax (best single model)', g['minimax']), ('DeepSeek', g['deepseek']), ('gpt-oss', g['gpt_oss']),
                     ('any of the three is right', None)]
        else:
            evals = [('the agreed (majority) label', g['agreed_label'])]
            if pat == 'two agree':
                evals.append(('the dissenting model\'s label', pd.Series(
                    [next(x for x in r if list(r).count(x) == 1) for r in g[MODELS].to_numpy()], index=g.index)))
        for nm, pred in evals:
            hit = (g[MODELS].eq(g['gold'], axis=0).any(axis=1)) if pred is None else (pred == g['gold'])
            p, lo, hi = wilson(int(hit.sum()), len(g))
            pw = (g['w'] * hit).sum() / g['w'].sum()
            rows2.append(dict(pattern=pat, n=len(g), share_unweighted_pct=sh_u, share_weighted_pct=sh_w,
                              label_used=nm, correct=int(hit.sum()), acc=p, acc_lo=lo, acc_hi=hi, acc_weighted=pw))
            out.append(f'{pat:16s} {len(g):4d} {sh_u:14.1f} {sh_w:13.1f}   {"":19s}{nm:28s} '
                       f'{p * 100:5.1f} [{lo * 100:4.1f}, {hi * 100:4.1f}] ({int(hit.sum())}/{len(g)}) {pw * 100:6.1f}')
    # overall majority accuracy for reference
    hit = df['mv3_tiebest'] == df['gold']
    p, lo, hi = wilson(int(hit.sum()), len(df))
    out.append(f'{"all 618":16s} {len(df):4d} {100:14.1f} {100:13.1f}   {"":19s}{"majority vote (CV tie-break)":28s} '
               f'{p * 100:5.1f} [{lo * 100:4.1f}, {hi * 100:4.1f}] ({int(hit.sum())}/618) {(df["w"] * hit).sum() / df["w"].sum() * 100:6.1f}')
    # kappa of majority within agreement strata
    for pat in ['all three agree', 'two agree']:
        g = df[df['pattern'] == pat]
        out.append(f'Cohen kappa of the agreed label vs human, {pat} (n={len(g)}): {A.kappa(g["gold"], g["agreed_label"]):.3f}')
    out.append('Human-label distribution by pattern (%): ' + '; '.join(
        f'{pat}: ' + ', '.join(f'{l[:4]} {(df[df.pattern == pat]["gold"] == l).mean() * 100:.0f}' for l in A.INTENT_LABELS)
        for pat in ['all three agree', 'two agree', 'all differ']))
    pd.DataFrame(rows2).to_csv(f'{OUT}/task2_llm_agreement_618.csv', index=False)
    text = '\n'.join(out) + '\n'
    open(f'{OUT}/task1_task2_intent.txt', 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
