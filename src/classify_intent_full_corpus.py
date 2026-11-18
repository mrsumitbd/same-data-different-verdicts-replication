"""
classify_intent_full_corpus.py

Classifies the primary intent of modification commits in
death_commit_messages.csv with an LLM, using the primary-intent prompt
(TAXONOMY_PROMPT, identical to the category
definitions in docs/annotation-guide.md).

Models (Together AI serverless, Oct 2026), each reported separately as its
own classifier, same prompt, temperature 0:
  deepseek  deepseek-ai/DeepSeek-V4.1-Flash
  minimax   MiniMaxAI/MiniMax-M3.1-Flash
  gpt_oss   openai/gpt-oss-120b
All three are reasoning models (hidden thinking before the answer), so
MAX_TOKENS leaves room for ~1,000+ thinking tokens.

Diff context:
  - Diffs come from results/diff_cache_full_corpus.csv; missing diffs are fetched from the
    GitHub "Get a commit" API in the same format: complete file list first,
    diff body truncated at 200,000 chars (first + last 100,000) if needed.

Restricting the run:
  --only <csv with slug, sha> classifies just those commits, e.g. the
  385-commit validation sample (results/intent_validation_sample_full.csv).

Resume safety:
  - Output is written incrementally every 20 classifications.
  - On re-run, already-classified (slug, sha) pairs are skipped.
  - Failed rows (api_error, unparseable) are retried on re-run.
  - API calls retry with exponential backoff (rate limits); --concurrency
    bounds parallel requests per model.

Usage (validation sample, one model):
    python src/classify_intent_full_corpus.py --model deepseek \
        --only results/intent_validation_sample_full.csv \
        --output results/intent_llm_deepseek_t0.csv

Usage (full corpus):
    python src/classify_intent_full_corpus.py --model minimax --max-cost 15 \
        --output results/intent_minimax_full_corpus_t0.csv
"""

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from dotenv import load_dotenv
from together import Together
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import P

load_dotenv()

MODELS = {
    "deepseek": "deepseek-ai/DeepSeek-V4.1-Flash",
    "minimax": "MiniMaxAI/MiniMax-M3.1-Flash",
    "gpt_oss": "openai/gpt-oss-120b",
}
# USD per 1M tokens (input, output), Together serverless, Oct 2026; for the --max-cost guard
PRICES = {
    "deepseek-ai/DeepSeek-V4.1-Flash": (0.30, 1.20),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "MiniMaxAI/MiniMax-M3.1-Flash": (0.15, 0.60),
}
BATCH_SIZE = 20
VALID_LABELS = {"corrective", "perfective", "adaptive", "preventive"}
MAX_TOKENS = 4000          # reasoning models think before answering
RETRY_MAX_TOKENS = 8000    # one retry for truncated/empty answers
MAX_RETRIES = 6           # exponential backoff covers rate-limit bursts
RETRY_BACKOFF_SECONDS = 5
FAILURE_LABELS = {"unparseable", "api_error"}

GITHUB_API_BASE = "https://api.github.com"
DIFF_TRUNCATE_LIMIT = 200_000
DIFF_HEAD_KEEP = 100_000
DIFF_TAIL_KEEP = 100_000

TAXONOMY_PROMPT = """You are classifying the intent of a software modification commit using Swanson's software maintenance taxonomy. Assign exactly ONE category based on the PRIMARY intent of the commit.

Categories:
- corrective: the main purpose is fixing a defect -- a bug, crash, regression, or incorrect behavior that existed before this commit
- perfective: the main purpose is improving existing, already-working code -- refactoring, performance optimization, cleanup, readability
- adaptive: the main purpose is adding new functionality or adapting to new requirements -- new features, API changes, dependency updates, migrations
- preventive: the main purpose is preventing future problems -- adding tests, documentation, CI configuration, type hints, validation

IMPORTANT -- classify by PRIMARY intent:
If a commit does multiple things, choose the category that best describes the DOMINANT purpose, weighted by the scope and magnitude of each type of change. Do not let a minor incidental element override the primary classification.

Examples of primary-intent reasoning:
- A commit that adds a major new feature across five files, and also fixes one small bug in passing -> adaptive (the feature is dominant)
- A commit that refactors a module for clarity, and happens to also fix a formatting issue -> perfective (the refactor is dominant)
- A commit whose entire purpose is fixing a crash, even if it also touches related code -> corrective (the fix is the reason this commit exists)
- A commit that adds comprehensive tests for existing functionality -> preventive

Commit message:
{commit_message}

Diff:
{diff_text}

Classify this commit's PRIMARY intent as exactly ONE of: corrective, perfective, adaptive, preventive.
Reply with ONLY the single category word, nothing else."""


def fetch_commit_diff(slug: str, sha: str, github_token: str) -> str:
    """
    Fetch a commit's full multi-file diff. Returns formatted diff text
    (file list always complete + truncated body if needed), or an empty
    string on failure.
    """
    url = f"{GITHUB_API_BASE}/repos/{slug}/commits/{sha}"
    headers = {"Authorization": f"token {github_token}"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                files = data.get("files", [])
                if not files:
                    return "[No files in this commit]"

                # File list: always complete
                file_list_lines = [f"Files changed ({len(files)} total):"]
                for f in files:
                    file_list_lines.append(
                        f"  - {f['filename']} ({f.get('status','?')}, "
                        f"+{f.get('additions',0)}/-{f.get('deletions',0)})"
                    )
                file_list = "\n".join(file_list_lines)

                # Diff body: concatenated patches
                patch_parts = []
                for f in files:
                    patch = f.get("patch", "")
                    if patch:
                        patch_parts.append(f"--- {f['filename']} ---\n{patch}")
                full_body = "\n\n".join(patch_parts)

                if not full_body:
                    return f"{file_list}\n\n[No patch content available]"

                original_len = len(full_body)
                if original_len <= DIFF_TRUNCATE_LIMIT:
                    return f"{file_list}\n\n--- DIFF ---\n\n{full_body}"

                head = full_body[:DIFF_HEAD_KEEP]
                tail = full_body[-DIFF_TAIL_KEEP:]
                truncated_body = (
                    f"{head}\n\n[... TRUNCATED: "
                    f"{original_len - DIFF_HEAD_KEEP - DIFF_TAIL_KEEP:,} chars omitted "
                    f"(original: {original_len:,} chars). File list above is complete. ...]"
                    f"\n\n{tail}"
                )
                return f"{file_list}\n\n--- DIFF (truncated) ---\n\n{truncated_body}"

            elif resp.status_code == 404:
                return "[404: commit not found]"
            elif resp.status_code == 403:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 2))
            else:
                time.sleep(RETRY_BACKOFF_SECONDS)
        except Exception:
            time.sleep(RETRY_BACKOFF_SECONDS)

    return "[fetch failed after retries]"


def parse_label(content: str) -> str:
    if not content:
        return "unparseable"
    cleaned = content.strip().lower().strip(".\"'")
    for label in VALID_LABELS:
        if label in cleaned:
            return label
    return "unparseable"


def classify_one(client: Together, model: str, temperature: float,
                 commit_message: str, diff_text: str) -> dict:
    """One classification call. A truncated (finish_reason=length) or empty
    answer is retried once with RETRY_MAX_TOKENS; token usage is recorded."""
    prompt = TAXONOMY_PROMPT.format(
        commit_message=commit_message,
        diff_text=diff_text,
    )

    last_error = None
    max_tokens = MAX_TOKENS
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            finish_reason = choice.finish_reason
            usage = resp.usage
            details = getattr(usage, "completion_tokens_details", None)
            result = {
                "label": parse_label(content),
                "raw_content": content,
                "finish_reason": finish_reason,
                "error": "",
                "max_tokens": max_tokens,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "reasoning_tokens": getattr(details, "reasoning_tokens", None),
            }
            if (finish_reason == "length" or not content.strip()) and max_tokens < RETRY_MAX_TOKENS:
                max_tokens = RETRY_MAX_TOKENS
                continue
            return result
        except Exception as e:
            last_error = str(e)
            attempt += 1
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))

    return {
        "label": "api_error",
        "raw_content": "",
        "finish_reason": "",
        "error": last_error,
        "max_tokens": max_tokens,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
    }


def load_diff_cache(cache_path: str) -> dict:
    """Load cached diffs as {(slug, sha): diff_text} dict."""
    if not os.path.exists(cache_path):
        return {}
    cache_df = pd.read_csv(cache_path)
    return {
        (row["slug"], row["sha"]): row["diff_text"]
        for _, row in cache_df.iterrows()
    }


def flush_diff_cache(new_entries: list, cache_path: str) -> None:
    new_df = pd.DataFrame(new_entries, columns=["slug", "sha", "diff_text"])
    if os.path.exists(cache_path):
        new_df.to_csv(cache_path, mode="a", header=False, index=False)
    else:
        new_df.to_csv(cache_path, index=False)


def run(intent_data: str, output: str, diff_cache_path: str, model: str,
        temperature: float, only: str, concurrency: int, max_cost: float) -> None:
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise RuntimeError("GITHUB_TOKEN not found in .env")

    together_key = os.getenv("TOGETHER_AI_API")
    if not together_key:
        raise RuntimeError("TOGETHER_AI_API not found in .env")

    client = Together(api_key=together_key)

    # Load corpus — unique (slug, sha) commits
    intent = pd.read_csv(intent_data)
    corpus = intent[["slug", "sha", "message"]].drop_duplicates(
        subset=["slug", "sha"]
    ).reset_index(drop=True)
    print(f"Total unique commits: {len(corpus):,}")
    if only:
        keep = pd.read_csv(only)[["slug", "sha"]].drop_duplicates()
        corpus = corpus.merge(keep, on=["slug", "sha"])
        print(f"Restricted to {len(corpus):,} commits in {only}")
    print(f"Model: {model}  temperature={temperature}  concurrency={concurrency}")

    # Load existing results — skip successful, retry failures
    done_keys = set()
    if os.path.exists(output):
        existing = pd.read_csv(output)
        n_total = len(existing)
        failed_mask = existing["label"].isin(FAILURE_LABELS)
        n_failed = failed_mask.sum()
        success = existing[~failed_mask]
        done_keys = set(zip(success["slug"], success["sha"]))
        if n_failed > 0:
            print(f"Found {n_total} existing rows: {len(success)} successful, "
                  f"{n_failed} failed (will retry).")
            success.to_csv(output, index=False)
        else:
            print(f"Found {len(success)} successfully classified rows.")

    # Load diff cache
    diff_cache = load_diff_cache(diff_cache_path)
    print(f"Diff cache loaded: {len(diff_cache):,} entries")

    remaining = corpus[~corpus.apply(
        lambda r: (r["slug"], r["sha"]) in done_keys, axis=1
    )]
    print(f"Remaining to classify: {len(remaining):,}\n")

    new_cache_entries = []
    fetch_count = 0
    jobs = []
    for _, row in remaining.iterrows():
        slug, sha, message = row["slug"], row["sha"], row["message"]
        cache_key = (slug, sha)
        if cache_key not in diff_cache:
            diff_cache[cache_key] = fetch_commit_diff(slug, sha, github_token)
            new_cache_entries.append([slug, sha, diff_cache[cache_key]])
            fetch_count += 1
            time.sleep(0.1)  # light pacing
        jobs.append((slug, sha, str(message), diff_cache[cache_key]))
    if fetch_count:
        print(f"  Fetched {fetch_count} new diffs")

    def classify_job(job):
        slug, sha, message, diff_text = job
        result = classify_one(client, model, temperature, message, diff_text)
        return {"slug": slug, "sha": sha, **result}

    # Cost guard: spend of this run so far, checked after every batch
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    spent = 0.0
    new_results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for start in range(0, len(jobs), BATCH_SIZE):
            batch = list(pool.map(classify_job, jobs[start:start + BATCH_SIZE]))
            new_results.extend(batch)
            _flush_results(new_results, output)
            new_results = []
            spent += sum((r["prompt_tokens"] or 0) * price_in + (r["completion_tokens"] or 0) * price_out
                         for r in batch) / 1e6
            print(f"  [{start + len(batch)}/{len(jobs)}] classified — ${spent:.2f} this run — "
                  f"last: {batch[-1]['slug']} -> {batch[-1]['label']}", flush=True)
            if max_cost and spent > max_cost:
                print(f"STOPPED: cost guard ${max_cost:.2f} exceeded; re-run to resume.")
                break

    # Final flush
    if new_cache_entries:
        flush_diff_cache(new_cache_entries, diff_cache_path)
    if new_results:
        _flush_results(new_results, output)

    # Summary
    final = pd.read_csv(output)
    print(f"\nDone. Total classified: {len(final):,}")
    print(f"Label distribution:\n{final['label'].value_counts()}")
    n_failed = final["label"].isin(FAILURE_LABELS).sum()
    if n_failed > 0:
        print(f"WARNING: {n_failed} rows still failed — re-run to retry.")


def _flush_results(rows: list, output: str) -> None:
    df = pd.DataFrame(rows)
    if os.path.exists(output):
        df.to_csv(output, mode="a", header=False, index=False)
    else:
        df.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent-data", default=P("data/death_commit_messages_v5.csv"))
    parser.add_argument("--output", required=True,
                        help="per-model output, e.g. results/intent_llm_minimax_t0.csv")
    parser.add_argument("--diff-cache", default=P("results/diff_cache_full_corpus.csv"))
    parser.add_argument("--model", choices=sorted(MODELS), default="deepseek")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--only", default=None,
                        help="CSV with slug, sha: classify only these commits")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-cost", type=float, default=None,
                        help="stop when this run's estimated spend (USD) exceeds this")
    args = parser.parse_args()
    run(args.intent_data, args.output, args.diff_cache, MODELS[args.model],
        args.temperature, args.only, args.concurrency, args.max_cost)