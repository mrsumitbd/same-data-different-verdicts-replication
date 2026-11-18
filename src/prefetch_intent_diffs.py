"""
prefetch_intent_diffs.py

Fills the diff cache used by classify_intent_full_corpus.py for every commit of
a modification-commit corpus, with the classifier's own fetch_commit_diff (same
GitHub "Get a commit" format), so that several LLM runs can start without each
fetching the same diffs. Writes every 100 fetches; failed fetches (empty text)
are not cached; waits for the GitHub rate-limit reset when fewer than 50 calls
remain.

Usage:
    python src/prefetch_intent_diffs.py --intent-data data/death_commit_messages_v6conf.csv \
        --diff-cache results_v6/diff_cache_full_corpus.csv
"""

import argparse
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_intent_full_corpus import fetch_commit_diff, flush_diff_cache


def wait_for_quota(token: str) -> None:
    r = requests.get('https://api.github.com/rate_limit', headers={'Authorization': f'token {token}'}, timeout=30)
    core = r.json()['resources']['core']
    if core['remaining'] < 50:
        wait = max(0, core['reset'] - time.time()) + 10
        print(f'  rate limit: {core["remaining"]} left, waiting {wait / 60:.1f} min', flush=True)
        time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--intent-data', required=True)
    ap.add_argument('--diff-cache', required=True)
    a = ap.parse_args()
    load_dotenv()
    token = os.getenv('GITHUB_TOKEN')
    corpus = pd.read_csv(a.intent_data)[['slug', 'sha']].drop_duplicates()
    have = set()
    if os.path.exists(a.diff_cache):
        c = pd.read_csv(a.diff_cache, usecols=['slug', 'sha', 'diff_text'])
        c = c[c['diff_text'].fillna('').str.len() > 0]
        have = set(zip(c['slug'], c['sha']))
    todo = [(s, h) for s, h in zip(corpus['slug'], corpus['sha']) if (s, h) not in have]
    print(f'corpus {len(corpus):,}; cached {len(corpus) - len(todo):,}; to fetch {len(todo):,}', flush=True)
    buf, failed = [], 0
    for i, (s, h) in enumerate(todo, 1):
        if i % 100 == 1:
            wait_for_quota(token)
        d = fetch_commit_diff(s, h, token)
        if d:
            buf.append([s, h, d])
        else:
            failed += 1
        if len(buf) >= 100 or i == len(todo):
            if buf:
                flush_diff_cache(buf, a.diff_cache)
            buf = []
            print(f'  [{i}/{len(todo)}] fetched, failed {failed}', flush=True)
        time.sleep(0.1)
    print(f'done; failed {failed}', flush=True)


if __name__ == '__main__':
    main()
