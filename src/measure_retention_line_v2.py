"""
measure_retention_line_v2.py

Measures code survival at LINE granularity using git blame.

Changes from v1:
- BUG FIX: find_line_death now uses pickaxe search (-S) to find the exact
  commit that removed the specific line, rather than the first commit that
  touched the file. This fixes incorrect duration timestamps in v1.
- Imports shared constants from config/constants.py
- Imports agent taxonomy from config/agent_taxonomy.py
- Adds death_magnitude field (trivial/moderate/substantial/deletion) using
  token edit distance at the survival event level
- Structured funnel logging for transparency
- v4 merge finder (Oct 2026): births are the commit that merged the PR into
  the default branch, taken from AIDev pr_timeline's 'merged' event when
  available, else the default-branch commit within 60 s of AIDev merged_at.
  PRs without merged_at are skipped. The old "#N" commit-message grep
  (which matched unrelated commits) is removed.

Definition of "death": A line introduced by the PR is attributed to a
different commit when running git blame at HEAD. Death timestamp is determined
by the first commit that specifically removed or replaced the line's content
(pickaxe search), not merely the first commit that touched the file (v1 bug).

Usage:
    python src/measure_retention_line_v2.py --output data/survival_events_line_v4.csv

Input:
    data/aidev_filtered_v1.csv
    data/all_pull_request.parquet
    data/human_pull_request.parquet
    data/pr_timeline.parquet

Output:
    data/survival_events_line_v4.csv

Schema:
    slug, pr_number, is_agent, agent_name, agent_tier, unit_id,
    event, duration_days, birth_sha, birth_date, death_sha, death_date,
    birth_content, death_content, token_edit_distance, death_magnitude,
    granularity
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import re
import argparse
import subprocess
import tempfile
import shutil
import random
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from datetime import timezone, timedelta

from dotenv import load_dotenv
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

from config.constants import (
    OBSERVATION_CUTOFF,
    SOURCE_EXTENSIONS,
    MAX_LINES_PER_FILE,
    MAX_FILES_PER_PR,
    THRESHOLD_TRIVIAL,
    THRESHOLD_MODERATE,
)
from config.agent_taxonomy import AGENT_TO_TIER, AGENT_KEYWORDS

# --- CONFIGURATION ---
INPUT_FILE = Path("data/aidev_filtered_v1.csv")
PR_DATA_DIR = Path("data")
OUTPUT_FILE = Path("data/survival_events_line_v4.csv")
MERGE_TIME_TOLERANCE_S = 60
MAX_WORKERS = 8

# Minimum content length to use pickaxe search.
# Lines shorter than this are too common to reliably distinguish via -S.
MIN_PICKAXE_LENGTH = 8


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def parse_git_date(date_str: str) -> Optional[datetime]:
    """
    Parse git date string to timezone-aware UTC datetime.
    
    Git format: '2025-02-25 23:03:10 +0800'
    Converts to UTC for exact comparison with GitHub's merged_at (UTC).
    """
    try:
        date_str = date_str.strip()
        # Split into datetime part and timezone offset
        # Format: 'YYYY-MM-DD HH:MM:SS +HHMM'
        parts = date_str.rsplit(' ', 1)
        if len(parts) == 2:
            dt_part, tz_part = parts
            # Parse offset: +0800 or -0500
            sign = 1 if tz_part[0] == '+' else -1
            tz_hours = int(tz_part[1:3])
            tz_mins = int(tz_part[3:5])
            offset_minutes = sign * (tz_hours * 60 + tz_mins)
            tz = timezone(timedelta(minutes=offset_minutes))
            # Parse datetime with offset
            dt = datetime.strptime(dt_part, "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=tz)
            # Convert to UTC naive for consistent comparison
            dt_utc = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt_utc
        else:
            # Fallback: parse without timezone
            return datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def identify_agent(row: pd.Series) -> str:
    """Identify which agent authored the PR."""
    # First check the explicit agent field if present
    if 'agent' in row.index:
        raw = str(row['agent']).strip()
        # Normalize underscore-separated names to space-separated
        normalized = raw.replace('_', ' ')
        if normalized in AGENT_TO_TIER:
            return normalized
        # Also try keyword matching on the raw agent field
        raw_lower = raw.lower()
        for keyword, agent_name in AGENT_KEYWORDS.items():
            if keyword in raw_lower:
                return agent_name

    # Fall back to checking user/author fields
    for field in ['user', 'author', 'user_login']:
        if field in row.index:
            val = str(row[field]).lower()
            for keyword, agent_name in AGENT_KEYWORDS.items():
                if keyword in val:
                    return agent_name

    return 'Unknown Agent'


def normalize_slug(url: str) -> str:
    """Normalize repository URL to slug format."""
    s = str(url).lower().strip().replace('.git', '').rstrip('/')
    if 'github.com/' in s:
        return s.split('github.com/')[-1]
    if 'api.github.com/repos/' in s:
        return s.split('api.github.com/repos/')[-1]
    parts = s.split('/')
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return s


def tokenize_line(content: str) -> List[str]:
    """
    Language-agnostic tokenization for token edit distance computation.
    Splits on whitespace and punctuation, preserving meaningful tokens.
    Suitable for Python, TypeScript, Go, Java, C/C++, C#, Rust.
    """
    if not content:
        return []
    return re.findall(r'[a-zA-Z_]\w*|[0-9]+|[^\s\w]', content)


def compute_token_edit_distance(content_a: str, content_b: str) -> float:
    """
    Compute token-level edit distance between two lines as a normalized
    Jaccard distance on token multisets.

    Returns a float in [0.0, 1.0]:
        0.0 = identical token content
        1.0 = completely different token content

    Using Jaccard on multisets (bags) rather than sets so that repeated
    tokens are handled correctly.
    """
    tokens_a = tokenize_line(content_a)
    tokens_b = tokenize_line(content_b)

    if not tokens_a and not tokens_b:
        return 0.0
    if not tokens_a or not tokens_b:
        return 1.0

    # Build token frequency dicts
    freq_a: Dict[str, int] = {}
    for t in tokens_a:
        freq_a[t] = freq_a.get(t, 0) + 1

    freq_b: Dict[str, int] = {}
    for t in tokens_b:
        freq_b[t] = freq_b.get(t, 0) + 1

    all_tokens = set(freq_a) | set(freq_b)
    intersection = sum(min(freq_a.get(t, 0), freq_b.get(t, 0)) for t in all_tokens)
    union = sum(max(freq_a.get(t, 0), freq_b.get(t, 0)) for t in all_tokens)

    return 1.0 - (intersection / union) if union > 0 else 0.0


def classify_death_magnitude(
    token_edit_distance: Optional[float],
    is_deletion: bool
) -> str:
    """
    Classify death magnitude based on token edit distance.

    Categories:
        deletion    - line was entirely removed from the file
        trivial     - < 25% of tokens changed (e.g., rename, minor edit)
        moderate    - 25-50% of tokens changed
        substantial - > 50% of tokens changed (near-rewrite)
    """
    if is_deletion:
        return 'deletion'
    if token_edit_distance is None:
        return 'unknown'
    if token_edit_distance < THRESHOLD_TRIVIAL:
        return 'trivial'
    if token_edit_distance < THRESHOLD_MODERATE:
        return 'moderate'
    return 'substantial'


# =============================================================================
# GIT OPERATIONS
# =============================================================================

def find_merge_commit(
    repo_dir: str,
    merged_at: Optional[str] = None,
    timeline_sha: Optional[str] = None,
) -> Tuple[Optional[str], Optional[datetime], str]:
    """
    Find the commit that merged a PR into the default branch.

    Returns (sha, committer date in naive UTC, reason). sha is None when the
    PR is skipped; reason says why (or which strategy matched).

    Strategy 1: the commit_id of the PR's 'merged' event in AIDev's
                pr_timeline, if it is reachable from the default branch.
    Strategy 2: the default-branch commit whose committer date is closest
                to AIDev merged_at, within MERGE_TIME_TOLERANCE_S.
    PRs without merged_at are never merged and are skipped. Commit-message
    grep for "#N" is not used: it matches unrelated commits.
    """
    if not merged_at or pd.isna(merged_at) or str(merged_at) in ('', 'nan', 'None'):
        return None, None, 'no merged_at timestamp'

    def on_default_branch(sha: str) -> bool:
        cmd = ["git", "merge-base", "--is-ancestor", sha, "HEAD"]
        return subprocess.run(cmd, cwd=repo_dir, capture_output=True).returncode == 0

    if timeline_sha and not pd.isna(timeline_sha):
        if on_default_branch(timeline_sha):
            cmd = ["git", "show", "-s", "--format=%ci", timeline_sha]
            result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace')
            date = parse_git_date(result.stdout) if result.returncode == 0 else None
            if date:
                return timeline_sha, date, 'timeline merge sha'

    try:
        merge_dt = datetime.strptime(str(merged_at)[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None, None, 'unparseable merged_at'

    # ±1 day only retrieves candidates; the match itself is on exact UTC time
    after  = (merge_dt - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    before = (merge_dt + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    cmd = [
        "git", "log", "HEAD",
        f"--after={after}",
        f"--before={before}",
        "--format=%H|%ci",
    ]
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace')
    best = None
    for line in result.stdout.strip().splitlines():
        parts = line.split('|')
        if len(parts) < 2:
            continue
        candidate_date = parse_git_date(parts[1])
        if candidate_date is None:
            continue
        gap = abs((candidate_date - merge_dt).total_seconds())
        if gap <= MERGE_TIME_TOLERANCE_S and (best is None or gap < best[0]):
            best = (gap, parts[0].strip(), candidate_date)
    if best:
        return best[1], best[2], 'merged_at match'

    reason = ('timeline merge sha not on default branch; no merged_at match'
              if timeline_sha and not pd.isna(timeline_sha)
              else f'no default-branch commit within {MERGE_TIME_TOLERANCE_S}s of merged_at')
    return None, None, reason


def get_added_lines(repo_dir: str, commit_sha: str) -> Dict[str, List[int]]:
    """
    Get lines ADDED by a commit.

    Uses git blame at the commit itself to find lines introduced by that
    commit, which is more reliable than diff parsing for squash merges.

    Returns:
        Dict mapping filepath -> list of line numbers (1-indexed) added
    """
    added_lines: Dict[str, List[int]] = {}

    # Get files changed in commit
    cmd = ["git", "show", "--name-only", "--format=", commit_sha]
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace')
    if result.returncode != 0:
        return added_lines

    files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
    source_files = [f for f in files if f.endswith(SOURCE_EXTENSIONS)]

    if len(source_files) > MAX_FILES_PER_PR:
        source_files = source_files[:MAX_FILES_PER_PR]

    for filepath in source_files:
        cmd = [
            "git", "blame",
            "--line-porcelain",
            commit_sha,
            "--", filepath
        ]
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace')
        if result.returncode != 0:
            continue

        lines_from_this_commit = []
        current_line_num = 0

        for line in result.stdout.splitlines():
            if re.match(r'^[0-9a-f]{40}', line):
                parts = line.split()
                line_sha = parts[0]
                if len(parts) >= 3:
                    current_line_num = int(parts[2])
                if line_sha.startswith(commit_sha[:8]):
                    lines_from_this_commit.append(current_line_num)

        if lines_from_this_commit:
            if len(lines_from_this_commit) > MAX_LINES_PER_FILE:
                lines_from_this_commit = sorted(
                    random.sample(lines_from_this_commit, MAX_LINES_PER_FILE)
                )
            added_lines[filepath] = lines_from_this_commit

    return added_lines


def get_line_content(
    repo_dir: str, filepath: str, line_number: int, commit_sha: str
) -> Optional[str]:
    """
    Retrieve the content of a specific line at a given commit.

    Args:
        repo_dir:    Path to local git repository
        filepath:    Relative path to file within repo
        line_number: 1-indexed line number
        commit_sha:  Commit SHA to inspect

    Returns:
        Line content string (stripped of trailing newline) or None if
        the file/line does not exist at that commit.
    """
    cmd = ["git", "show", f"{commit_sha}:{filepath}"]
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace')
    if result.returncode != 0:
        return None
    file_lines = result.stdout.splitlines()
    if 0 < line_number <= len(file_lines):
        return file_lines[line_number - 1]
    return None


def find_line_death_v2(
    repo_dir: str,
    filepath: str,
    birth_sha: str,
    birth_date: datetime,
    birth_content: Optional[str] = None,
) -> Tuple[int, float, Optional[str], Optional[datetime], Optional[str], Optional[float]]:
    """
    Find when a specific line was first modified after its birth commit.

    v2 FIX: Uses pickaxe search (git log -S) on the line's content to
    find the commit that specifically removed or replaced that content.
    This is accurate to the line, not just the file.

    Falls back to file-level search (v1 behavior) when:
    - birth_content is None or too short to distinguish via pickaxe
    - pickaxe returns no results (content may have been incrementally modified)

    Args:
        repo_dir:      Path to local git repository
        filepath:      Relative path to modified file
        birth_sha:     SHA of the commit that introduced this line
        birth_date:    Datetime of birth commit
        birth_content: Raw content of the line at birth (for pickaxe)

    Returns:
        Tuple of:
            event            (int)   1 = died, 0 = survived
            duration_days    (float) days from birth to death or cutoff
            death_sha        (str|None)
            death_date       (datetime|None)
            death_content    (str|None) line content at death commit
            token_edit_dist  (float|None) normalized edit distance [0,1]
    """
    # --- PICKAXE PATH ---
    # Use content-based search when we have a long enough, distinctive line.
    use_pickaxe = (
        birth_content is not None
        and len(birth_content.strip()) >= MIN_PICKAXE_LENGTH
    )

    if use_pickaxe:
        content_stripped = birth_content.strip()
        cmd = [
            "git", "log",
            "--reverse",
            "--format=%H|%ci",
            f"-S{content_stripped}",
            f"{birth_sha}..HEAD",
            "--", filepath
        ]
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace')

        if result.stdout.strip():
            for log_line in result.stdout.strip().splitlines():
                parts = log_line.split('|')
                if len(parts) < 2:
                    continue
                candidate_sha = parts[0].strip()
                candidate_date = parse_git_date(parts[1])
                if not candidate_date:
                    continue

                # Verify this commit REMOVED the content (not added it).
                # git log -S finds both additions and removals; we want removal.
                before_cmd = ["git", "show", f"{candidate_sha}^:{filepath}"]
                after_cmd = ["git", "show", f"{candidate_sha}:{filepath}"]

                before_result = subprocess.run(
                    before_cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace'
                )
                after_result = subprocess.run(
                    after_cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace'
                )

                content_before = (
                    content_stripped in before_result.stdout
                    if before_result.returncode == 0 else False
                )
                content_after = (
                    content_stripped in after_result.stdout
                    if after_result.returncode == 0 else False
                )

                if content_before and not content_after:
                    # This is the commit that removed our line.
                    duration = (
                        (candidate_date - birth_date).total_seconds() / 86400.0
                    )

                    # Try to get the death content: find the replacement line
                    # at the same approximate position in the file after the commit.
                    death_content = _find_replacement_content(
                        repo_dir, filepath, candidate_sha, content_stripped
                    )
                    token_edit_dist = (
                        compute_token_edit_distance(birth_content, death_content)
                        if death_content is not None
                        else None
                    )

                    return (
                        1, max(0.001, duration),
                        candidate_sha, candidate_date,
                        death_content, token_edit_dist
                    )

    # --- FALLBACK PATH (v1 behavior) ---
    # Either no birth_content, content too short, or pickaxe found no removal.
    # Use the first commit that touched the file (less precise, kept for
    # completeness and as a safety net).
    cmd = [
        "git", "log",
        "--reverse",
        "--format=%H|%ci",
        f"{birth_sha}..HEAD",
        "--", filepath
    ]
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace')

    if result.stdout.strip():
        parts = result.stdout.strip().splitlines()[0].split('|')
        if len(parts) >= 2:
            death_sha = parts[0].strip()
            death_date = parse_git_date(parts[1])
            if death_date:
                duration = (death_date - birth_date).total_seconds() / 86400.0
                return (
                    1, max(0.001, duration),
                    death_sha, death_date,
                    None, None  # No content info on fallback path
                )

    # No modifications found — line survived.
    duration = (OBSERVATION_CUTOFF - birth_date).total_seconds() / 86400.0
    return 0, max(0.001, duration), None, None, None, None


def _find_replacement_content(
    repo_dir: str, filepath: str, death_sha: str, original_content: str
) -> Optional[str]:
    """
    Attempt to find what replaced the original line after a death commit.

    Looks at the diff of the death commit and finds the first added line
    that is in the same hunk as the removed original content. This is
    a best-effort heuristic — it is used only for token edit distance
    computation, not for event detection.

    Returns:
        Replacement line content, or None if not found.
    """
    cmd = [
        "git", "show",
        "--unified=3",
        "--format=",
        death_sha,
        "--", filepath
    ]
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace')
    if result.returncode != 0:
        return None

    lines = result.stdout.splitlines()
    original_stripped = original_content.strip()

    # Walk through the diff, find the hunk containing our removed line,
    # then return the first added line in that hunk.
    in_relevant_hunk = False
    for i, line in enumerate(lines):
        if line.startswith('@@'):
            in_relevant_hunk = False  # Reset for each hunk
        elif line.startswith('-') and original_stripped in line[1:]:
            in_relevant_hunk = True
        elif in_relevant_hunk and line.startswith('+') and not line.startswith('+++'):
            return line[1:]  # Return first added line after the removed one

    return None


def check_line_survival(
    repo_dir: str,
    filepath: str,
    line_number: int,
    birth_sha: str,
    birth_date: datetime,
) -> Tuple[int, float, Optional[str], Optional[datetime], Optional[str], Optional[str], Optional[float], str]:
    """
    Check if a specific line survived to the observation cutoff.

    v2: Captures birth_content and death_content, computes token edit
    distance, and classifies death magnitude.

    Returns:
        Tuple of:
            event            (int)   1 = died, 0 = survived
            duration_days    (float)
            death_sha        (str|None)
            death_date       (datetime|None)
            birth_content    (str|None)
            death_content    (str|None)
            token_edit_dist  (float|None)
            death_magnitude  (str)  'deletion'|'trivial'|'moderate'|'substantial'|'survived'|'unknown'
    """
    # Capture birth content for pickaxe and token distance computation
    birth_content = get_line_content(repo_dir, filepath, line_number, birth_sha)

    # --- CASE 1: File was deleted ---
    cmd = ["git", "ls-files", "--", filepath]
    ls_result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace')

    if not ls_result.stdout.strip():
        # File deleted — find the deletion commit
        cmd = [
            "git", "log",
            "--reverse",
            "--format=%H|%ci",
            "--diff-filter=D",
            f"{birth_sha}..HEAD",
            "--", filepath
        ]
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace')

        if result.stdout.strip():
            parts = result.stdout.strip().splitlines()[0].split('|')
            if len(parts) >= 2:
                death_sha = parts[0].strip()
                death_date = parse_git_date(parts[1])
                if death_date:
                    duration = (death_date - birth_date).total_seconds() / 86400.0
                    return (
                        1, max(0.001, duration),
                        death_sha, death_date,
                        birth_content, None, None,
                        'deletion'
                    )

        # File deleted but couldn't locate deletion commit
        duration = (OBSERVATION_CUTOFF - birth_date).total_seconds() / 86400.0
        return 1, max(0.001, duration), None, None, birth_content, None, None, 'deletion'

    # --- CASE 2: File exists — check blame for this line ---
    cmd = [
        "git", "blame",
        "-L", f"{line_number},{line_number}",
        "--porcelain",
        "HEAD",
        "--", filepath
    ]
    blame_result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace')

    if blame_result.returncode != 0 or not blame_result.stdout.strip():
        # Line no longer exists at this position (file shrank)
        # Fall through to find_line_death_v2 which will handle this.
        pass
    else:
        first_line = blame_result.stdout.splitlines()[0]
        current_sha = first_line.split()[0]

        if (
            current_sha.startswith(birth_sha[:8])
            or birth_sha.startswith(current_sha[:8])
        ):
            # Line survived to HEAD — censored at observation cutoff
            duration = (OBSERVATION_CUTOFF - birth_date).total_seconds() / 86400.0
            return (
                0, max(0.001, duration),
                None, None,
                birth_content, None, None,
                'survived'
            )

    # --- CASE 3: Line was modified — find when and how ---
    event, duration, death_sha, death_date, death_content, token_edit_dist = (
        find_line_death_v2(repo_dir, filepath, birth_sha, birth_date, birth_content)
    )

    # is_deletion is False here — file exists and line was modified, not deleted.
    # death_content may be None if pickaxe fallback fired (content capture failed),
    # but that is not the same as the line being deleted.
    magnitude = classify_death_magnitude(token_edit_dist, is_deletion=False)

    return (
        event, duration,
        death_sha, death_date,
        birth_content, death_content, token_edit_dist,
        magnitude
    )


# =============================================================================
# REPOSITORY PROCESSING
# =============================================================================

def process_repo(slug: str, pr_df: pd.DataFrame) -> List[dict]:
    """Process all PRs for a single repository at line granularity."""
    results = []
    temp_dir = tempfile.mkdtemp()
    clone_url = (
        f"https://{GITHUB_TOKEN}@github.com/{slug}.git"
        if GITHUB_TOKEN
        else f"https://github.com/{slug}.git"
    )

    try:
        clone_result = subprocess.run(
            ["git", "clone", "--filter=blob:none", clone_url, temp_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True, errors='replace',
            timeout=300
        )
        if clone_result.returncode != 0:
            err = clone_result.stderr.strip().splitlines()
            print(f"  [CLONE FAILED] {slug}: {err[-1] if err else 'no stderr'}")
            return results

        for _, row in pr_df.iterrows():
            pr_number = row['number']
            is_agent = row['is_agent']

            if is_agent:
                agent_name = identify_agent(row)
                agent_tier = AGENT_TO_TIER.get(agent_name, 'unknown')
                if agent_tier == 'unknown':
                    print(f"  [WARN] Unrecognized agent name: '{agent_name}' "
                        f"(raw: {row.get('agent', 'N/A')})")
            else:
                agent_name = 'Human'
                agent_tier = 'human'

            merge_sha, merge_date, reason = find_merge_commit(
                temp_dir, row.get('merged_at'), row.get('timeline_merge_sha')
            )

            if not merge_sha or not merge_date:
                print(f"  [SKIP] {slug} PR#{pr_number} "
                    f"(is_agent={is_agent}): {reason}")
                continue
            print(f"  [MERGE] {slug} PR#{pr_number}: {merge_sha[:10]} ({reason})")

            added_lines = get_added_lines(temp_dir, merge_sha)

            for filepath, line_numbers in added_lines.items():
                for line_num in line_numbers:
                    (
                        event, duration,
                        death_sha, death_date,
                        birth_content, death_content,
                        token_edit_dist, magnitude
                    ) = check_line_survival(
                        temp_dir, filepath, line_num, merge_sha, merge_date
                    )

                    results.append({
                        'slug': slug,
                        'pr_number': pr_number,
                        'is_agent': is_agent,
                        'agent_name': agent_name,
                        'agent_tier': agent_tier,
                        'unit_id': f"{filepath}::{line_num}",
                        'event': event,
                        'duration_days': duration,
                        'birth_sha': merge_sha,
                        'birth_date': merge_date.isoformat(),
                        'death_sha': death_sha,
                        'death_date': death_date.isoformat() if death_date else None,
                        'birth_content': birth_content,
                        'death_content': death_content,
                        'token_edit_distance': token_edit_dist,
                        'death_magnitude': magnitude,
                        'granularity': 'line',
                    })

    except subprocess.TimeoutExpired as e:
        print(f"  [TIMEOUT] {slug}: {e.cmd[:2]} after {e.timeout}s "
              f"({len(results)} units kept)")
    except Exception as e:
        print(f"  [ERROR] {slug}: {type(e).__name__}: {e} "
              f"({len(results)} units kept)")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Measure line-level code survival'
    )
    parser.add_argument(
        '--repos-override',
        type=Path,
        default=None,
        help='CSV file with slug column — restrict processing to these repos only'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=OUTPUT_FILE,
        help=f'Output CSV path (default: {OUTPUT_FILE})'
    )
    args = parser.parse_args()
    output_file = args.output

    print("=" * 70)
    print("SURVIVAL MEASUREMENT - LINE LEVEL (v2)")
    print("=" * 70)
    print(f"Observation cutoff : {OBSERVATION_CUTOFF.date()}")
    print(f"Max lines per file : {MAX_LINES_PER_FILE}")
    print(f"Max files per PR   : {MAX_FILES_PER_PR}")
    print(f"Pickaxe min length : {MIN_PICKAXE_LENGTH} chars")
    print(f"Death thresholds   : trivial <{THRESHOLD_TRIVIAL}, "
          f"moderate <{THRESHOLD_MODERATE}, substantial >=0.5")
    print()

    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run filter_repos.py first.")
        return

    df_repos = pd.read_csv(INPUT_FILE)
    slugs = set(df_repos['slug'].unique())
    print(f"Filtered repositories : {len(slugs)}")

    print("Loading PR data...")
    df_agent = pd.read_parquet(PR_DATA_DIR / "all_pull_request.parquet")
    df_human = pd.read_parquet(PR_DATA_DIR / "human_pull_request.parquet")

    def find_url_col(df):
        return next(
            (c for c in ['repo_url', 'url', 'repository_url'] if c in df.columns),
            None
        )

    df_agent['slug'] = df_agent[find_url_col(df_agent)].apply(normalize_slug)
    df_agent['is_agent'] = True
    df_human['slug'] = df_human[find_url_col(df_human)].apply(normalize_slug)
    df_human['is_agent'] = False

    all_prs = pd.concat([df_agent, df_human], ignore_index=True)
    all_prs = all_prs[all_prs['slug'].isin(slugs)]

    # Merge commit SHAs from AIDev's PR timeline ('merged' events)
    timeline = pd.read_parquet(PR_DATA_DIR / "pr_timeline.parquet",
                               columns=['pr_id', 'event', 'commit_id'])
    merged_events = timeline[timeline['event'] == 'merged'].dropna(subset=['commit_id'])
    timeline_sha = merged_events.drop_duplicates('pr_id').set_index('pr_id')['commit_id']
    all_prs['timeline_merge_sha'] = all_prs['id'].map(timeline_sha)
    print(f"  With timeline merge SHA : {all_prs['timeline_merge_sha'].notna().sum():,}")

    print(f"PRs to process : {len(all_prs):,}")
    print(f"  Agent PRs    : {all_prs['is_agent'].sum():,}")
    print(f"  Human PRs    : {(~all_prs['is_agent']).sum():,}")

    if args.repos_override and args.repos_override.exists():
        override_slugs = set(
            pd.read_csv(args.repos_override)['slug'].unique()
        )
        all_prs = all_prs[all_prs['slug'].isin(override_slugs)]
        print(f"Repos override active: {len(override_slugs)} repos")
        print(f"  PRs after override : {len(all_prs):,}")

    repo_groups = list(all_prs.groupby('slug'))
    print(f"\nProcessing {len(repo_groups)} repositories with "
          f"{MAX_WORKERS} workers...")

    # Per-repo checkpoints: one CSV per finished repo, so the run can be
    # monitored and resumed (repos with a checkpoint file are skipped).
    partial_dir = Path(output_file).parent / f"{Path(output_file).stem}_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    def partial_path(slug: str) -> Path:
        return partial_dir / f"{slug.replace('/', '__')}.csv"

    todo = [(slug, group) for slug, group in repo_groups if not partial_path(slug).exists()]
    if len(todo) < len(repo_groups):
        print(f"Resuming: {len(repo_groups) - len(todo)} repos already in {partial_dir}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_repo, slug, group): slug
            for slug, group in todo
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Mining"):
            slug = futures[future]
            results = future.result()
            pd.DataFrame(results).to_csv(partial_path(slug), index=False)
            print(f"  [REPO DONE] {slug}: {len(results):,} units", flush=True)

    frames = [pd.read_csv(partial_path(slug)) for slug, _ in repo_groups
              if partial_path(slug).stat().st_size > 1]
    df_results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df_results = df_results.drop_duplicates(
        subset=['slug', 'pr_number', 'unit_id'], keep='first'
    )
    df_results.to_csv(output_file, index=False)

    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total survival events : {len(df_results):,}")

    if not df_results.empty:
        n_deaths = int(df_results['event'].sum())
        n_censored = len(df_results) - n_deaths
        print(f"  Deaths (modified)   : {n_deaths:,} "
              f"({100 * n_deaths / len(df_results):.1f}%)")
        print(f"  Censored (survived) : {n_censored:,} "
              f"({100 * n_censored / len(df_results):.1f}%)")

        print("\n  Death magnitude breakdown (deaths only):")
        dead = df_results[df_results['event'] == 1]
        for mag, count in dead['death_magnitude'].value_counts().items():
            pct = 100 * count / len(dead)
            print(f"    {mag:<15}: {count:,} ({pct:.1f}%)")

        print("\n  By agent tier:")
        for tier, grp in df_results.groupby('agent_tier'):
            dr = grp['event'].mean() * 100
            print(f"    {tier:<20}: {len(grp):,} lines, "
                  f"{dr:.1f}% death rate")

        print(f"\n  Median duration : {df_results['duration_days'].median():.1f} days")
        print(f"  Mean duration   : {df_results['duration_days'].mean():.1f} days")

    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()