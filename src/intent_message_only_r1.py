"""
intent_message_only_r1.py 

How much of the LLM classifiers' agreement with the human labels comes from the
diff? Re-classifies the 618 human-labelled commits (results_v6/labels618/) from
the commit message alone: the prompt of src/classify_intent_full_corpus.py with
its "Diff:" block (file list and patch) removed, everything else identical
(Together AI, temperature 0, same max-token / retry rules, same label parser).

Models: MiniMax-M3.1-Flash (highest kappa on the 618 with full input), plus
DeepSeek-V4.1-Flash and gpt-oss-120b for completeness.

Kappa (Cohen, the paper's kappa_ci: 1,000 bootstrap resamples of the commits,
seed 42) against the human labels, next to the same model's full-input kappa
on the same commits and the keyword rules.

Usage: python src/intent_message_only_r1.py [--models minimax deepseek gpt_oss] [--concurrency 8]
Output: results_v6/round1/c7_message_only_<model>.csv (one row per commit, with
        call timestamps and tokens), results_v6/round1/c7_message_only.txt
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
import classify_intent_full_corpus as C
import analyze_intent_aggregation as A

L = 'results_v6/labels618'
OUT = 'results_v6/round1'
DIFF_BLOCK = 'Diff:\n{diff_text}\n\n'
assert DIFF_BLOCK in C.TAXONOMY_PROMPT
MESSAGE_ONLY_PROMPT = C.TAXONOMY_PROMPT.replace(DIFF_BLOCK, '')


def classify(client, model, msg):
    t0 = datetime.now(timezone.utc)
    r = C.classify_one(client, model, 0.0, msg, '')
    r['called_at_utc'] = t0.isoformat()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['minimax', 'deepseek', 'gpt_oss'])
    ap.add_argument('--concurrency', type=int, default=8)
    a = ap.parse_args()
    C.TAXONOMY_PROMPT = MESSAGE_ONLY_PROMPT          # classify_one formats this module-level prompt
    assert '{diff_text}' not in C.TAXONOMY_PROMPT and '{commit_message}' in C.TAXONOMY_PROMPT
    from together import Together
    client = Together(api_key=os.getenv('TOGETHER_AI_API'))

    gold = A.read_labels(f'{L}/gold.csv', partial=False, who='gold')
    samp = pd.read_csv(f'{L}/sample.csv').set_index('sample_id').loc[gold.index].reset_index()
    samp['gold'] = gold.values
    msgs = pd.read_csv('data/death_commit_messages_v6confnv.csv').drop_duplicates(['slug', 'sha'])
    samp = samp.merge(msgs[['slug', 'sha', 'message']], on=['slug', 'sha'], how='left')
    assert samp['message'].notna().all(), 'missing messages'
    print(f'{len(samp)} labelled commits', flush=True)

    rows_out = ['=' * 78, 'C7. Message-only LLM classification of the 618 labelled commits (no diff, no file list)',
                f'prompt: classify_intent_full_corpus.TAXONOMY_PROMPT without the "Diff:" block; temperature 0; Together AI',
                '=' * 78, '']
    kw = pd.read_csv(f'{L}/keyword.csv').drop_duplicates(['slug', 'sha']).rename(columns={'intent': 'keyword'})
    base = samp.merge(kw[['slug', 'sha', 'keyword']], on=['slug', 'sha'], how='left')
    base['keyword'] = base['keyword'].fillna('missing').str.lower()
    k, lo, hi = A.kappa_ci(base['gold'], base['keyword'])
    rows_out.append(f'keyword rules (full corpus labels)           kappa {k:.3f} [{lo:.3f}, {hi:.3f}]')
    for m in a.models:
        model = C.MODELS[m]
        path = os.path.join(OUT, f'c7_message_only_{m}.csv')
        if os.path.exists(path):
            res = pd.read_csv(path)
        else:
            t0 = time.time()
            with ThreadPoolExecutor(a.concurrency) as ex:
                rs = list(ex.map(lambda msg: classify(client, model, msg), samp['message'].tolist()))
            res = pd.concat([samp[['sample_id', 'slug', 'sha', 'gold']], pd.DataFrame(rs)], axis=1)
            res['model'] = model
            res.to_csv(path, index=False)
            print(f'{m}: {len(res)} calls in {time.time() - t0:.0f}s', flush=True)
        pin, pout = C.PRICES[model]
        cost = (res['prompt_tokens'].fillna(0).sum() * pin + res['completion_tokens'].fillna(0).sum() * pout) / 1e6
        full = pd.read_csv(f'{L}/intent_llm_{m}_t0.csv').drop_duplicates(['slug', 'sha'])
        both = res.merge(full.rename(columns={'label': 'full'}), on=['slug', 'sha'], how='left')
        for c in ['label', 'full']:
            both[c] = both[c].fillna('missing').astype(str).str.lower()
            both.loc[~both[c].isin(A.INTENT_LABELS + ['unknown']), c] = 'missing'
        km, lom, him = A.kappa_ci(both['gold'], both['label'])
        kf, lof, hif = A.kappa_ci(both['gold'], both['full'])
        agree = (both['label'] == both['full']).mean() * 100
        rows_out.append(f'{model:45s} message only kappa {km:.3f} [{lom:.3f}, {him:.3f}]   '
                        f'full input kappa {kf:.3f} [{lof:.3f}, {hif:.3f}]   '
                        f'same label as full input {agree:.1f}%   unparseable/api_error '
                        f'{int(both["label"].eq("missing").sum())}')
        rows_out.append(f'{"":45s} tokens in {int(res["prompt_tokens"].sum()):,} out {int(res["completion_tokens"].sum()):,}; '
                        f'cost ${cost:.3f} (Together list price {pin}/{pout} USD per 1M); '
                        f'calls {res["called_at_utc"].min()} to {res["called_at_utc"].max()} UTC')
    text = '\n'.join(rows_out) + '\n'
    open(os.path.join(OUT, 'c7_message_only.txt'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
