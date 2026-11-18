"""
sample_intent_validation.py

Draws the n=385 stratified sample (~96 per tier) for intent classifier
validation, from the SAME filtered population RQ3 actually uses:
hunks after outlier-PR + vendored-file exclusion, joined to commit
intent labels via death_sha == sha.

Key design decisions (see conversation log for full rationale):
  - Sampling unit is the UNIQUE COMMIT (slug, sha), not hunk-event.
    A commit's intent is a property of its message/diff, independent
    of how many hunks it touched.
  - A commit's tier is assigned as the MAJORITY tier among the hunks
    it modified (commits almost always touch hunks of one dominant
    tier within a coherent PR).
  - DIFFS ARE FETCHED LIVE VIA GITHUB API, not from any pre-mined
    file. pr_commit_details.parquet was investigated and rejected as
    a diff source: it covers commits belonging to the original
    BIRTH PRs in the cohort (used for Cox regression covariates), not
    the separate, later DEATH commits that modified that code. An
    empirical check found only 6/1,766 shas in common between the two
    populations -- confirming they are fundamentally different sets of
    commits, not a formatting/join bug. Diffs are therefore fetched
    on-demand via the GitHub 'Get a commit' API, and ONLY for the
    385 commits actually selected by sampling (not the full
    population), since this is far cheaper than mining diffs for
    every candidate commit upfront.
  - Diff bodies over 200,000 characters (DIFF_TRUNCATE_LIMIT) are
    truncated to the first 100,000 + last 100,000 characters with an
    explicit truncation note; the file list is never truncated.
  - Sampling happens BEFORE diff fetching (using only tier/message
    metadata, which is available for the full population), so we
    never fetch diffs we don't end up needing.
  - Random seed = 42 for reproducibility, consistent with other
    sampling in this pipeline (attribution_validation.csv).

Outputs:
  results/intent_validation_sample_full.csv
      Full sample with keyword label, tier, message, diff -- for
      LLM judges and internal audit. Includes a 'fetch_error' column
      for any commit where the GitHub API diff fetch failed (e.g.
      404 if the commit was later rebased/force-pushed away) --
      these should be reviewed and either re-sampled or excluded
      with disclosure, not silently dropped.
  results/intent_validation_sample_blind.csv
      Same sample WITHOUT keyword label or any model info -- for
      the first author's manual coding, to keep it blind to the existing
      classifier's output.

Usage:
    python src/sample_intent_validation.py \
        --hunk-data data/survival_events_hunk_v2.csv \
        --intent-data data/death_commit_messages.csv \
        --n-per-tier 96
"""

import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

OUTLIER_PR_KEYS = {('microsoft/typespec', 7115)}
GENERATED_PATH_PATTERNS = [
    r'/dist/', r'/third_party/', r'/vendor/', r'/generated/',
    r'_pb2\.py$', r'\.min\.js$', r'immutable_entry',
    r'_generated\.', r'zz_generated\.',
]

DIFF_TRUNCATE_LIMIT = 200_000  # confirmed: balances truncation rate (10.4% in
                                # n=385 sample) against token cost/latency
                                # across 1,155 LLM calls, given all 3 judge
                                # models' context windows (131K-262K tokens)
DIFF_HEAD_KEEP = 100_000
DIFF_TAIL_KEEP = 100_000

SEED = 42

# Equal allocation across 4 tiers for n=385: 97+96+96+96
TIER_ALLOCATION = {
    "human": 97,
    "pair_programmer": 96,
    "batch_generator": 96,
    "autonomous": 96,
}

GITHUB_API_BASE = "https://api.github.com"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3


def is_generated_file(unit_id: str) -> bool:
    filepath = str(unit_id).split('::')[0]
    return any(re.search(p, filepath) for p in GENERATED_PATH_PATTERNS)


def apply_original_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.copy()
    df['_key'] = list(zip(df['slug'], df['pr_number']))
    df = df[~df['_key'].isin(OUTLIER_PR_KEYS)]
    df = df[~df['unit_id'].apply(is_generated_file)]
    df = df.drop(columns=['_key'])
    print(f"Preprocessing (outlier PR + vendored/generated exclusion): "
          f"{before:,} -> {len(df):,} rows")
    return df


def build_commit_population(hunks: pd.DataFrame, intent: pd.DataFrame) -> pd.DataFrame:
    """
    Build a unique-commit-level population: one row per (slug, sha),
    with the majority tier among hunks that commit modified, and the
    existing keyword-based intent label. NOTE: no diff fetching here --
    diffs are fetched live via GitHub API only for the final sample,
    since death-commits are not covered by any pre-mined diff source
    (see conversation log: pr_commit_details.parquet covers birth-PR
    commits only, ~6/1766 overlap with death commits -- a different
    population entirely).
    """
    merged = hunks.merge(
        intent[["slug", "sha", "message", "intent"]],
        left_on=["slug", "death_sha"],
        right_on=["slug", "sha"],
        how="inner",
    )

    def majority_tier(series: pd.Series) -> str:
        return series.value_counts().index[0]

    commit_pop = (
        merged.groupby(["slug", "sha"])
        .agg(
            tier=("agent_tier", majority_tier),
            keyword_label=("intent", "first"),
            commit_message=("message", "first"),
            n_hunks_touched=("unit_id", "count"),
        )
        .reset_index()
    )
    commit_pop["tier"] = commit_pop["tier"].replace({"": "human"})
    return commit_pop


def fetch_commit_diff(slug: str, sha: str, github_token: str) -> dict:
    """
    Fetch a commit's full multi-file diff via the GitHub 'Get a commit'
    API endpoint.

    Format: a COMPLETE file list (filename, status, +/- line counts) is
    always included in full, regardless of truncation -- this is cheap
    (always available from the API response without needing the patch
    itself) and remains informative even when the diff body below it is
    heavily truncated. The 200,000-character truncation budget applies
    ONLY to the concatenated patch/diff body, not to the file list.

    Rationale for this split: investigation of the largest diffs in the
    n=385 sample (see conversation log) found genuinely large legitimate
    commits (platform rewrites, codebase-wide reformats, mass import
    updates) -- not contamination. For these, the file list alone is
    often highly informative for intent classification even when the
    diff content itself must be truncated.
    """
    url = f"{GITHUB_API_BASE}/repos/{slug}/commits/{sha}"
    headers = {"Authorization": f"token {github_token}"}

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                files = data.get("files", [])

                if not files:
                    return {"diff_text": "", "diff_truncated": False,
                             "diff_original_length": 0, "fetch_error": "no_files"}

                # File list: ALWAYS complete, never truncated.
                file_list_lines = [f"Files changed in this commit ({len(files)} total):"]
                for f in files:
                    status = f.get("status", "unknown")
                    additions = f.get("additions", 0)
                    deletions = f.get("deletions", 0)
                    file_list_lines.append(
                        f"  - {f['filename']} ({status}, +{additions}/-{deletions})"
                    )
                file_list_section = "\n".join(file_list_lines)

                # Diff body: concatenated patches, truncation applies HERE only.
                patch_parts = []
                for f in files:
                    patch = f.get("patch", "")
                    if patch:
                        patch_parts.append(f"--- {f['filename']} ---\n{patch}")
                full_body = "\n\n".join(patch_parts)

                if not full_body:
                    # File list still has value even with no patch content
                    # (e.g. binary files, or files too large for GitHub to
                    # return a patch for)
                    diff_text = f"{file_list_section}\n\n[No diff content available " \
                                f"for these files (binary or too large for patch).]"
                    return {"diff_text": diff_text, "diff_truncated": False,
                             "diff_original_length": 0, "fetch_error": ""}

                original_len = len(full_body)
                if original_len <= DIFF_TRUNCATE_LIMIT:
                    diff_text = f"{file_list_section}\n\n--- DIFF CONTENT ---\n\n{full_body}"
                    return {"diff_text": diff_text, "diff_truncated": False,
                             "diff_original_length": original_len, "fetch_error": ""}

                head = full_body[:DIFF_HEAD_KEEP]
                tail = full_body[-DIFF_TAIL_KEEP:]
                truncated_body = (
                    f"{head}\n\n[... TRUNCATED: "
                    f"{original_len - DIFF_HEAD_KEEP - DIFF_TAIL_KEEP:,} characters "
                    f"omitted (original total: {original_len:,} chars). "
                    f"See complete file list above, which is never truncated. ...]"
                    f"\n\n{tail}"
                )
                diff_text = f"{file_list_section}\n\n--- DIFF CONTENT (truncated) ---\n\n{truncated_body}"
                return {"diff_text": diff_text, "diff_truncated": True,
                         "diff_original_length": original_len, "fetch_error": ""}

            elif resp.status_code == 404:
                return {"diff_text": "", "diff_truncated": False,
                         "diff_original_length": 0, "fetch_error": "404_not_found"}
            elif resp.status_code == 403:
                last_error = "403_rate_limited"
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 2))
            else:
                last_error = f"http_{resp.status_code}"
                time.sleep(RETRY_BACKOFF_SECONDS)
        except Exception as e:
            last_error = str(e)
            time.sleep(RETRY_BACKOFF_SECONDS)

    return {"diff_text": "", "diff_truncated": False,
             "diff_original_length": 0, "fetch_error": last_error}



def run(hunk_data: str, intent_data: str, n_per_tier_override: int | None) -> None:
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise RuntimeError("GITHUB_TOKEN not found in .env -- required to fetch diffs.")

    hunks = pd.read_csv(hunk_data)
    intent = pd.read_csv(intent_data)

    hunks = apply_original_preprocessing(hunks)

    commit_pop = build_commit_population(hunks, intent)
    print(f"\nUnique-commit population: {len(commit_pop):,} commits")
    print(f"Tier distribution:\n{commit_pop['tier'].value_counts()}")

    # ── Stratified sampling FIRST (no diff needed yet) ────────────────
    allocation = TIER_ALLOCATION.copy()
    if n_per_tier_override is not None:
        allocation = {k: n_per_tier_override for k in allocation}
        print(f"\nOverriding allocation to {n_per_tier_override} per tier "
              f"(total {n_per_tier_override * 4})")

    rng = np.random.default_rng(SEED)
    samples = []
    for tier, n_needed in allocation.items():
        pool = commit_pop[commit_pop["tier"] == tier]
        if len(pool) < n_needed:
            print(f"  WARNING: {tier} pool has only {len(pool)} commits, "
                  f"fewer than requested {n_needed}. Taking all available.")
            n_needed = len(pool)
        sampled_idx = rng.choice(pool.index, size=n_needed, replace=False)
        samples.append(pool.loc[sampled_idx])
        print(f"  {tier:20s}: sampled {n_needed} / {len(pool)} available")

    sample = pd.concat(samples, ignore_index=True)
    sample["sample_id"] = [f"S{i+1:04d}" for i in range(len(sample))]
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle order

    print(f"\nSample size (pre-diff-fetch): {len(sample)}")

    # ── Fetch diffs live via GitHub API, ONLY for sampled commits ────
    print(f"\nFetching diffs from GitHub API for {len(sample)} sampled commits...")
    print("(This avoids the need to pre-mine diffs for the full population, "
          "since death-commits are not covered by pr_commit_details.parquet.)")

    diff_results = []
    for i, row in sample.iterrows():
        result = fetch_commit_diff(row["slug"], row["sha"], github_token)
        diff_results.append(result)
        if (i + 1) % 25 == 0 or (i + 1) == len(sample):
            print(f"  [{i+1}/{len(sample)}] fetched")
        time.sleep(0.1)  # light pacing to stay well within rate limits

    diff_df = pd.DataFrame(diff_results)
    sample = pd.concat([sample.reset_index(drop=True), diff_df], axis=1)

    n_errors = (sample["fetch_error"] != "").sum()
    n_no_files = (sample["fetch_error"] == "no_files").sum()
    n_404 = (sample["fetch_error"] == "404_not_found").sum()
    print(f"\nFetch results: {len(sample) - n_errors} succeeded, {n_errors} errors")
    if n_errors > 0:
        print(f"  no_files (commit has no file changes, e.g. empty merge): {n_no_files}")
        print(f"  404_not_found (commit may have been rebased/force-pushed away): {n_404}")
        other_errors = n_errors - n_no_files - n_404
        if other_errors > 0:
            print(f"  other errors: {other_errors}")
        print(f"\n  WARNING: {n_errors} commits have no usable diff. These should be "
              f"either re-sampled (replacement draws) or excluded with disclosure "
              f"in the methodology. Not auto-handled -- review the output file's "
              f"'fetch_error' column before proceeding.")

    n_truncated = sample["diff_truncated"].sum()
    n_with_diff = (sample["diff_text"] != "").sum()
    print(f"\nDiffs requiring truncation (>{DIFF_TRUNCATE_LIMIT:,} chars): "
          f"{n_truncated:,} / {n_with_diff:,} usable diffs "
          f"({n_truncated/max(n_with_diff,1)*100:.1f}%)")

    # ── Output: full version (internal + LLM judges) ─────────────────
    full_cols = ["sample_id", "slug", "sha", "tier", "keyword_label",
                 "commit_message", "diff_text", "diff_truncated",
                 "diff_original_length", "fetch_error", "n_hunks_touched"]
    sample[full_cols].to_csv(
        "results/intent_validation_sample_full.csv", index=False
    )
    print("\nWritten: results/intent_validation_sample_full.csv")

    # ── Output: blind version (manual coding) ─────────────────────────
    blind = sample[["sample_id", "slug", "sha", "commit_message", "diff_text"]].copy()
    blind["human_label"] = ""  # to be filled in manually
    blind.to_csv("results/intent_validation_sample_blind.csv", index=False)
    print("Written: results/intent_validation_sample_blind.csv (for manual coding)")

    print("\nNOTE: 'tier' and 'keyword_label' are deliberately EXCLUDED from the "
          "blind version to avoid biasing manual coding.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hunk-data", default="data/survival_events_hunk_v2.csv")
    parser.add_argument("--intent-data", default="data/death_commit_messages.csv")
    parser.add_argument("--n-per-tier", type=int, default=None,
                         help="Override equal allocation (default: 97/96/96/96 = 385)")
    args = parser.parse_args()
    run(args.hunk_data, args.intent_data, args.n_per_tier)