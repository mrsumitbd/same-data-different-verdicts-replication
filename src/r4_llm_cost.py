"""
Tokens, cost and wall time of the three full-corpus LLM intent runs.

Tokens: prompt_tokens / completion_tokens per row of results_v6/intent_<m>_full_corpus_t0.csv
(one row per commit; the cache accumulated over several runs). Cost at
classify_intent_full_corpus.PRICES (Together list price, Oct 2026).
Reported for all cached rows and for the 5,303 commits of the analysed corpus
(data/death_commit_messages_v6confnv.csv).

Wall time: only from the run logs that exist (logs/run_v5_llm.log + logs/v5_llm_<m>.log,
2026-10-29; logs/v6conf_llm.log + logs/v6conf_llm_<m>.log, 2026-11-03). The first
~4,064 rows per model come from earlier runs with no log in the repo; their wall time
is unknown. Times are local US Eastern as printed in the logs; the end of the last
job of each chain is the next job's header or the log file's modification time.
"""
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from classify_intent_full_corpus import MODELS, PRICES  # noqa: E402

OUT = 'results_v6/round4'
corpus = pd.read_csv('data/death_commit_messages_v6confnv.csv')[['slug', 'sha']].drop_duplicates()
rows, out = [], ['TASK 3. Tokens and cost of the full-corpus LLM intent runs (from the cached outputs)', '']
for m, model in MODELS.items():
    f = pd.read_csv(f'results_v6/intent_{m}_full_corpus_t0.csv').drop_duplicates(['slug', 'sha'], keep='last')
    pin, pout = PRICES[model]
    for scope, d in [('all cached rows', f), ('analysed corpus', f.merge(corpus, on=['slug', 'sha']))]:
        ti, to = d['prompt_tokens'].sum(), d['completion_tokens'].sum()
        rows.append(dict(model=model, scope=scope, commits=len(d), prompt_tokens=int(ti), completion_tokens=int(to),
                         cost_usd=(ti * pin + to * pout) / 1e6, price_in=pin, price_out=pout,
                         mean_prompt_tokens=ti / len(d), mean_completion_tokens=to / len(d)))
t = pd.DataFrame(rows)
t.to_csv(f'{OUT}/task3_llm_tokens_cost.csv', index=False)
for scope, g in t.groupby('scope', sort=False):
    out.append(f'{scope}:')
    for _, r in g.iterrows():
        out.append(f'  {r.model:34s} {r.commits:5d} commits  in {r.prompt_tokens:>11,}  out {r.completion_tokens:>9,}  '
                   f'${r.cost_usd:6.2f}  (mean {r.mean_prompt_tokens:,.0f} in / {r.mean_completion_tokens:,.0f} out per commit; '
                   f'{r.price_in}/{r.price_out} USD per 1M)')
    out.append(f'  {"total":34s} {"":5s}          in {g.prompt_tokens.sum():>11,}  out {g.completion_tokens.sum():>9,}  '
               f'${g.cost_usd.sum():6.2f}')
out += ['', 'Wall time (only runs with a log; concurrency 4, sequential models; US Eastern):',
        '  2026-10-29 v5 top-up (1,226-1,227 commits per model): deepseek 12:10:24-12:42:49 (32 min), '
        'gpt-oss 12:42:49-12:59:39 (17 min), minimax 12:59:39-13:25:30 (26 min, end = log mtime); 75 min total',
        '  2026-11-03 v6conf (2,396-2,397 commits per model): deepseek 17:07:00-18:01:13 (54 min), '
        'gpt-oss 18:01:13-18:24:14 (23 min), minimax 18:24:14-19:24:13 (60 min, end = log mtime); '
        'deepseek retry of 4 commits ~19:24; 137 min total',
        '  The first ~4,064 cached commits per model come from earlier runs with no log in the repository: wall time unknown.',
        '  Logged cost (run logs / RUN_LOG.md): 2026-10-29 deepseek $5.11, gpt-oss $2.33, minimax $2.48; '
        '2026-11-03 deepseek $11.69 (+$0.03 retry), gpt-oss $5.35, minimax $5.67.',
        '  Per-commit API timestamps are not stored in the full-corpus CSVs (only in the round-1 message-only reruns).']
text = '\n'.join(out) + '\n'
open(f'{OUT}/task3_llm_tokens_cost.txt', 'w').write(text)
print(text)
