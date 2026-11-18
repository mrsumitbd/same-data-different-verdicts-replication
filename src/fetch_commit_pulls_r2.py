"""
fetch_commit_pulls_r2.py 

For every modification commit of the v6confnv corpus (data/death_commit_messages_v6confnv.csv,
5,303 (slug, sha) pairs) fetches from the GitHub API:
  - GET /repos/{slug}/commits/{sha}/pulls  -> the PRs that contain the commit (E6, follow-up PR unit)
  - GET /repos/{slug}/commits/{sha}         -> author login, committer login, full message (E7)
Token: GITHUB_TOKEN from .env, as src/fetch_modifier_authors.py.
Responses are cached (resume-safe, saved every 50 commits):
  results_v6/round2/commit_pulls_cache.json  {"slug@sha": [{"number", "user", "merged_at", "created_at", "state", "base"}...]}
  results_v6/round2/commit_meta_cache.json   {"slug@sha": {"author_login", "committer_login", "author_name", "author_email",
                                                           "committer_name", "committer_email", "message"}}
A key with value null means the API returned 404/422 (commit or repo gone).
Rate limits: waits for the reset when X-RateLimit-Remaining < 20.

Usage: DATA_TAG=v6confnv RESULTS_DIR=results_v6 python src/fetch_commit_pulls_r2.py [--only pulls|meta]
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
load_dotenv()
API = 'https://api.github.com'
OUT = 'results_v6/round2'
PULLS = os.path.join(OUT, 'commit_pulls_cache.json')
META = os.path.join(OUT, 'commit_meta_cache.json')
S = requests.Session()
S.headers.update({'Authorization': f"token {os.environ['GITHUB_TOKEN']}", 'Accept': 'application/vnd.github+json'})
NCALLS = 0


def get(url):
    global NCALLS
    for attempt in range(6):
        try:
            r = S.get(url, timeout=30)
            NCALLS += 1
        except Exception as e:
            print('  network error', e, flush=True)
            time.sleep(5 * (attempt + 1))
            continue
        rem = int(r.headers.get('X-RateLimit-Remaining', 100))
        if rem < 20:
            reset = int(r.headers.get('X-RateLimit-Reset', time.time() + 60))
            wait = max(0, reset - time.time()) + 5
            print(f'  rate limit: {rem} left, sleeping {wait:.0f}s', flush=True)
            time.sleep(wait)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (404, 409, 422):
            return None
        if r.status_code in (403, 429):
            reset = int(r.headers.get('X-RateLimit-Reset', time.time() + 60))
            wait = max(30, reset - time.time() + 5) if rem == 0 else 60 * (attempt + 1)
            print(f'  {r.status_code}, sleeping {wait:.0f}s', flush=True)
            time.sleep(wait)
            continue
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'failed: {url}')


def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def save(obj, p):
    tmp = p + '.tmp'
    json.dump(obj, open(tmp, 'w'))
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['pulls', 'meta'])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    c = pd.read_csv('data/death_commit_messages_v6confnv.csv', usecols=['slug', 'sha']).drop_duplicates()
    pulls, meta = load(PULLS), load(META)
    todo = [(s, h) for s, h in zip(c['slug'], c['sha'])
            if (a.only != 'meta' and f'{s}@{h}' not in pulls) or (a.only != 'pulls' and f'{s}@{h}' not in meta)]
    print(f'{len(c):,} commits; cached pulls {len(pulls):,}, meta {len(meta):,}; to fetch {len(todo):,}', flush=True)
    t0 = time.time()
    for i, (s, h) in enumerate(todo, 1):
        k = f'{s}@{h}'
        if a.only != 'meta' and k not in pulls:
            js = get(f'{API}/repos/{s}/commits/{h}/pulls?per_page=100')
            pulls[k] = None if js is None else [
                dict(number=p['number'], user=(p.get('user') or {}).get('login'), merged_at=p.get('merged_at'),
                     created_at=p.get('created_at'), state=p.get('state'), base=(p.get('base') or {}).get('ref'))
                for p in js]
        if a.only != 'pulls' and k not in meta:
            js = get(f'{API}/repos/{s}/commits/{h}')
            meta[k] = None if js is None else dict(
                author_login=(js.get('author') or {}).get('login'), committer_login=(js.get('committer') or {}).get('login'),
                author_name=js['commit']['author'].get('name'), author_email=js['commit']['author'].get('email'),
                committer_name=js['commit']['committer'].get('name'), committer_email=js['commit']['committer'].get('email'),
                message=js['commit'].get('message'))
        if i % 50 == 0 or i == len(todo):
            save(pulls, PULLS)
            save(meta, META)
            print(f'  [{i}/{len(todo)}] {NCALLS} calls, {(time.time() - t0) / 60:.1f} min', flush=True)
    print('done', NCALLS, 'calls', flush=True)


if __name__ == '__main__':
    main()
