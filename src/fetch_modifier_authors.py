"""
fetch_modifier_authors.py

Fetches the author login for each unique (slug, sha) death commit
via the GitHub API. Saves results to a small author lookup file.

We only extract the author field -- no diff content -- so each
API response is tiny and fast. The full diff is already cached
separately in diff_cache_full_corpus.csv.

Saves incrementally every 100 fetches. Resume-safe: skips any
(slug, sha) pair already in the output file.

Usage:
    python src/fetch_modifier_authors.py \
        --intent-data data/death_commit_messages.csv \
        --output results/modifier_authors.csv

    # re-derive modifier_type after a rule change, without API calls
    python src/fetch_modifier_authors.py --reclassify
"""

import argparse
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.agent_logins import is_agent_login
from config.paths import P

load_dotenv()

GITHUB_API_BASE = "https://api.github.com"
MAX_RETRIES = 3
RETRY_BACKOFF = 5

def classify_author(author_login: str) -> str:
    """
    Classify a commit author as 'agent' or 'human' (rules in
    config/agent_logins.py). Returns 'unknown' if author_login is empty.
    """
    if not isinstance(author_login, str) or not author_login:
        return "unknown"
    return "agent" if is_agent_login(author_login) else "human"


def reclassify(output: str) -> None:
    """Re-derive modifier_type from the cached author_login (no API calls)."""
    df = pd.read_csv(output)
    old = df["modifier_type"].copy()
    df["modifier_type"] = df["author_login"].apply(classify_author)
    changed = df[old != df["modifier_type"]]
    print(f"Reclassified {len(df):,} rows; {len(changed)} changed:")
    for (login, a, b), n in (changed.assign(old=old[changed.index])
                             .groupby(["author_login", "old", "modifier_type"]).size().items()):
        print(f"  {login}: {a} -> {b} ({n})")
    df.to_csv(output, index=False)


def fetch_author(slug: str, sha: str, github_token: str) -> str:
    """Fetch only the author login for a commit. Returns empty string on failure."""
    url = f"{GITHUB_API_BASE}/repos/{slug}/commits/{sha}"
    headers = {"Authorization": f"token {github_token}"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("author") and data["author"].get("login"):
                    return data["author"]["login"]
                if data.get("commit", {}).get("author", {}).get("name"):
                    return data["commit"]["author"]["name"]
                return ""
            elif resp.status_code == 404:
                return ""
            elif resp.status_code == 403:
                time.sleep(RETRY_BACKOFF * (attempt + 2))
            else:
                time.sleep(RETRY_BACKOFF)
        except Exception:
            time.sleep(RETRY_BACKOFF)
    return ""


def run(intent_data: str, output: str) -> None:
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise RuntimeError("GITHUB_TOKEN not found in .env")

    intent = pd.read_csv(intent_data)
    commits = intent[["slug", "sha"]].drop_duplicates(
        subset=["slug", "sha"]
    ).reset_index(drop=True)
    print(f"Unique (slug, sha) commits: {len(commits):,}")

    # Resume: skip already-fetched pairs
    done_keys = set()
    if os.path.exists(output):
        existing = pd.read_csv(output)
        done_keys = set(zip(existing["slug"], existing["sha"]))
        print(f"Already fetched: {len(done_keys):,}. Remaining: {len(commits) - len(done_keys):,}")

    remaining = commits[~commits.apply(
        lambda r: (r["slug"], r["sha"]) in done_keys, axis=1
    )]

    new_rows = []
    for i, (_, row) in enumerate(remaining.iterrows()):
        author_login = fetch_author(row["slug"], row["sha"], github_token)
        modifier_type = classify_author(author_login)
        new_rows.append({
            "slug": row["slug"],
            "sha": row["sha"],
            "author_login": author_login,
            "modifier_type": modifier_type,
        })
        time.sleep(0.1)

        if (i + 1) % 100 == 0 or (i + 1) == len(remaining):
            df = pd.DataFrame(new_rows)
            if os.path.exists(output):
                df.to_csv(output, mode="a", header=False, index=False)
            else:
                df.to_csv(output, index=False)
            new_rows = []
            print(f"  [{i+1}/{len(remaining)}] fetched — last: "
                  f"{row['slug']} / {row['sha'][:8]} -> {modifier_type}")

    # Summary
    final = pd.read_csv(output)
    print(f"\nDone. Total: {len(final):,}")
    print(f"modifier_type distribution:\n{final['modifier_type'].value_counts()}")
    print(f"\nTop 20 agent author_logins:")
    agents = final[final["modifier_type"] == "agent"]
    print(agents["author_login"].value_counts().head(20))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent-data", default=P("data/death_commit_messages.csv"))
    parser.add_argument("--output", default=P("results/modifier_authors.csv"))
    parser.add_argument("--reclassify", action="store_true",
                        help="re-derive modifier_type from cached logins, no API calls")
    args = parser.parse_args()
    if args.reclassify:
        reclassify(args.output)
    else:
        run(args.intent_data, args.output)