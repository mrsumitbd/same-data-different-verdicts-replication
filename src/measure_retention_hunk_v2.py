"""
measure_retention_hunk_v2.py

Measures code survival at HUNK granularity.

Changes from v1:
- BUG FIX: parse_git_date now converts to UTC for correct timestamp comparison
- BUG FIX: find_merge_commit adds timestamp-based fallback for squash/rebase repos
- BUG FIX: identify_agent normalizes underscore agent names (e.g. OpenAI_Codex)
- BUG FIX: death timestamp now uses hunk-specific blame SHAs, not file-level
  git log (avoids attributing wrong death timestamp)
- Imports shared constants from config/constants.py
- Imports agent taxonomy from config/agent_taxonomy.py
- Adds agent_tier field (autonomous / pair_programmer / batch_generator / human)
- Stores continuous modification metrics:
    lines_modified    - count of hunk lines changed by a subsequent commit
    lines_survived    - count of hunk lines still attributed to birth commit
    modification_rate - lines_modified / hunk_size in [0.0, 1.0]
- REMOVED MAX_HUNKS_PER_FILE cap — all hunks retained
- REMOVED MAX_FILES_PER_PR cap — all files retained
- Adds pr_hunk_count and pr_file_count as Cox model covariates to control
  for PR scale as a confound (large PRs are qualitatively different from
  small PRs and must be controlled, not truncated)
- GitHub token used in clone URL to avoid rate limiting
- SKIP logging with reason (no merged_at vs commit not found)
- Deduplication before save
- v4 merge finder (Oct 2026): births are the commit that merged the PR into
  the default branch, taken from AIDev pr_timeline's 'merged' event when
  available, else the default-branch commit within 60 s of AIDev merged_at.
  PRs without merged_at are skipped. The old "#N" commit-message grep
  (which matched unrelated commits) is removed.

Definition of death:
    A hunk is born when its parent PR is merged. It dies when any subsequent
    commit modifies at least one line within its range. The modification_rate
    captures how much of the hunk changed, enabling sensitivity analysis at
    any threshold during analysis.

    Special case: file deletion -> event=1, lines_modified=hunk_size,
    lines_survived=0, modification_rate=1.0.

Usage:
    python src/measure_retention_hunk_v2.py --output data/survival_events_hunk_v3.csv

Input:
    data/aidev_filtered_v1.csv
    data/all_pull_request.parquet
    data/human_pull_request.parquet
    data/pr_timeline.parquet

Output:
    data/survival_events_hunk_v3.csv

Schema:
    slug, pr_number, is_agent, agent_name, agent_tier, unit_id,
    event, duration_days, birth_sha, birth_date, death_sha, death_date,
    lines_modified, lines_survived, modification_rate,
    pr_hunk_count, pr_file_count,
    granularity, hunk_start, hunk_end, hunk_size
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import re
import argparse
import subprocess
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

from config.constants import (
    OBSERVATION_CUTOFF,
    SOURCE_EXTENSIONS,
)
from config.agent_taxonomy import AGENT_TO_TIER, AGENT_KEYWORDS

# --- CONFIGURATION ---
INPUT_FILE  = Path("data/aidev_filtered_v1.csv")
PR_DATA_DIR = Path("data")
OUTPUT_FILE = Path("data/survival_events_hunk_v3.csv")
MERGE_TIME_TOLERANCE_S = 60
MAX_WORKERS = 8

# No hunk or file caps — all hunks retained.
# PR scale (pr_hunk_count, pr_file_count) is controlled as a Cox covariate
# rather than truncated, preserving all data while controlling confounding.


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Hunk:
    """Represents a diff hunk (contiguous block of added lines)."""
    filepath:   str
    start_line: int
    line_count: int
    hunk_index: int

    @property
    def end_line(self) -> int:
        return self.start_line + self.line_count - 1

    @property
    def unit_id(self) -> str:
        return f"{self.filepath}::hunk_{self.hunk_index}"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_git_date(date_str: str) -> Optional[datetime]:
    """
    Parse git date string to timezone-aware UTC datetime.

    Git format: '2025-02-25 23:03:10 +0800'
    Converts offset to UTC for exact comparison with GitHub's merged_at (UTC).
    """
    try:
        date_str = date_str.strip()
        parts = date_str.rsplit(' ', 1)
        if len(parts) == 2:
            dt_part, tz_part = parts
            sign        = 1 if tz_part[0] == '+' else -1
            tz_hours    = int(tz_part[1:3])
            tz_mins     = int(tz_part[3:5])
            offset_mins = sign * (tz_hours * 60 + tz_mins)
            tz          = timezone(timedelta(minutes=offset_mins))
            dt          = datetime.strptime(dt_part, "%Y-%m-%d %H:%M:%S")
            dt          = dt.replace(tzinfo=tz)
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            return datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def identify_agent(row: pd.Series) -> str:
    """Identify which agent authored the PR."""
    if 'agent' in row.index:
        raw        = str(row['agent']).strip()
        normalized = raw.replace('_', ' ')
        if normalized in AGENT_TO_TIER:
            return normalized
        raw_lower = raw.lower()
        for keyword, agent_name in AGENT_KEYWORDS.items():
            if keyword in raw_lower:
                return agent_name

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


def extract_hunks(repo_dir: str, commit_sha: str) -> List[Hunk]:
    """
    Extract ALL hunks from a commit's diff — no cap applied.

    PR scale (total hunk count, file count) is recorded as a covariate
    for the Cox model rather than being used to truncate data.
    """
    hunks: List[Hunk] = []

    cmd = [
        "git", "show",
        "--format=",
        "--unified=0",
        commit_sha
    ]
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace')
    if result.returncode != 0:
        return hunks

    current_file: Optional[str] = None
    file_hunk_counts: Dict[str, int] = {}

    for line in result.stdout.splitlines():
        if line.startswith('+++ b/'):
            current_file = line[6:]
            if not current_file.endswith(SOURCE_EXTENSIONS):
                current_file = None
            else:
                if current_file not in file_hunk_counts:
                    file_hunk_counts[current_file] = 0

        elif line.startswith('@@') and current_file:
            match = re.search(r'\+(\d+)(?:,(\d+))?', line)
            if match:
                start_line = int(match.group(1))
                line_count = int(match.group(2)) if match.group(2) else 1

                if line_count > 0:
                    hunk_idx = file_hunk_counts[current_file]
                    hunks.append(Hunk(
                        filepath=current_file,
                        start_line=start_line,
                        line_count=line_count,
                        hunk_index=hunk_idx,
                    ))
                    file_hunk_counts[current_file] += 1

    return hunks


def check_hunk_survival(
    repo_dir: str,
    hunk: Hunk,
    birth_sha: str,
    birth_date: datetime,
) -> Tuple[int, float, Optional[str], Optional[datetime], int, int]:
    """
    Check if a hunk survived to the observation cutoff.

    Returns:
        Tuple of:
            event           (int)   1 = died, 0 = survived
            duration_days   (float) days from birth to death or cutoff
            death_sha       (str|None)
            death_date      (datetime|None)
            lines_modified  (int)   number of hunk lines changed
            lines_survived  (int)   number of hunk lines unchanged
    """
    # --- CASE 1: File deleted ---
    cmd = ["git", "ls-files", "--", hunk.filepath]
    ls_result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace')

    if not ls_result.stdout.strip():
        cmd = [
            "git", "log",
            "--reverse",
            "--format=%H|%ci",
            "--diff-filter=D",
            f"{birth_sha}..HEAD",
            "--", hunk.filepath
        ]
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace')

        if result.stdout.strip():
            parts = result.stdout.strip().splitlines()[0].split('|')
            if len(parts) >= 2:
                death_sha  = parts[0].strip()
                death_date = parse_git_date(parts[1])
                if death_date:
                    duration = (death_date - birth_date).total_seconds() / 86400.0
                    return (
                        1, max(0.001, duration),
                        death_sha, death_date,
                        hunk.line_count, 0
                    )

        duration = (OBSERVATION_CUTOFF - birth_date).total_seconds() / 86400.0
        return 1, max(0.001, duration), None, None, hunk.line_count, 0

    # --- CASE 2: File exists — blame the hunk's line range at HEAD ---
    cmd = [
        "git", "blame",
        "-L", f"{hunk.start_line},{hunk.end_line}",
        "--porcelain",
        "HEAD",
        "--", hunk.filepath
    ]
    blame_result = subprocess.run(
        cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace'
    )

    if blame_result.returncode != 0 or not blame_result.stdout.strip():
        # Line range no longer exists — file shrank past hunk range
        cmd = [
            "git", "log",
            "--reverse",
            "--format=%H|%ci",
            f"{birth_sha}..HEAD",
            "--", hunk.filepath
        ]
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace')
        if result.stdout.strip():
            parts = result.stdout.strip().splitlines()[0].split('|')
            if len(parts) >= 2:
                death_sha  = parts[0].strip()
                death_date = parse_git_date(parts[1])
                if death_date:
                    duration = (death_date - birth_date).total_seconds() / 86400.0
                    return (
                        1, max(0.001, duration),
                        death_sha, death_date,
                        hunk.line_count, 0
                    )

        duration = (OBSERVATION_CUTOFF - birth_date).total_seconds() / 86400.0
        return 1, max(0.001, duration), None, None, hunk.line_count, 0

    # --- Count lines survived vs modified ---
    lines_survived = 0
    lines_modified = 0
    modified_shas: List[str] = []

    for blame_line in blame_result.stdout.splitlines():
        if re.match(r'^[0-9a-f]{40}', blame_line):
            parts       = blame_line.split()
            current_sha = parts[0]

            if (current_sha.startswith(birth_sha[:8]) or
                    birth_sha.startswith(current_sha[:8])):
                lines_survived += 1
            else:
                lines_modified += 1
                modified_shas.append(current_sha)

    if lines_modified == 0:
        # Hunk survived
        duration = (OBSERVATION_CUTOFF - birth_date).total_seconds() / 86400.0
        return 0, max(0.001, duration), None, None, 0, lines_survived

    # --- Find earliest death timestamp from modifying commits ---
    # Hunk-specific: uses actual SHAs from blame, not file-level log.
    death_sha  = None
    death_date = None
    best_delta = float('inf')

    seen_shas: set = set()
    for mod_sha in modified_shas:
        if mod_sha in seen_shas:
            continue
        seen_shas.add(mod_sha)

        cmd = ["git", "log", "--format=%ci", "-n", "1", mod_sha]
        result = subprocess.run(
            cmd, cwd=repo_dir, capture_output=True, text=True, errors='replace'
        )
        if result.stdout.strip():
            candidate_date = parse_git_date(result.stdout.strip())
            if candidate_date:
                delta = (candidate_date - birth_date).total_seconds()
                if delta > 0 and delta < best_delta:
                    best_delta = delta
                    death_sha  = mod_sha
                    death_date = candidate_date

    if death_date:
        duration = (death_date - birth_date).total_seconds() / 86400.0
        return (
            1, max(0.001, duration),
            death_sha, death_date,
            lines_modified, lines_survived
        )

    # Fallback — couldn't resolve exact death timestamp
    duration = (OBSERVATION_CUTOFF - birth_date).total_seconds() / 86400.0
    return 1, max(0.001, duration), None, None, lines_modified, lines_survived


# =============================================================================
# REPOSITORY PROCESSING
# =============================================================================

def process_repo(slug: str, pr_df: pd.DataFrame) -> List[dict]:
    """Process all PRs for a single repository at hunk granularity."""
    results: List[dict] = []
    temp_dir  = tempfile.mkdtemp()
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
            is_agent  = row['is_agent']

            if is_agent:
                agent_name = identify_agent(row)
                agent_tier = AGENT_TO_TIER.get(agent_name, 'unknown')
                if agent_tier == 'unknown':
                    print(f"  [WARN] Unrecognized agent: '{agent_name}' "
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

            hunks = extract_hunks(temp_dir, merge_sha)

            # PR-level covariates for Cox model
            pr_hunk_count = len(hunks)
            pr_file_count = len(set(h.filepath for h in hunks))

            for hunk in hunks:
                (
                    event, duration,
                    death_sha, death_date,
                    lines_modified, lines_survived
                ) = check_hunk_survival(
                    temp_dir, hunk, merge_sha, merge_date
                )

                modification_rate = (
                    lines_modified / hunk.line_count
                    if hunk.line_count > 0
                    else None
                )

                results.append({
                    'slug':              slug,
                    'pr_number':         pr_number,
                    'is_agent':          is_agent,
                    'agent_name':        agent_name,
                    'agent_tier':        agent_tier,
                    'unit_id':           hunk.unit_id,
                    'event':             event,
                    'duration_days':     duration,
                    'birth_sha':         merge_sha,
                    'birth_date':        merge_date.isoformat(),
                    'death_sha':         death_sha,
                    'death_date':        death_date.isoformat() if death_date else None,
                    'lines_modified':    lines_modified,
                    'lines_survived':    lines_survived,
                    'modification_rate': modification_rate,
                    'pr_hunk_count':     pr_hunk_count,
                    'pr_file_count':     pr_file_count,
                    'granularity':       'hunk',
                    'hunk_start':        hunk.start_line,
                    'hunk_end':          hunk.end_line,
                    'hunk_size':         hunk.line_count,
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
        description='Measure hunk-level code survival'
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
        help='Output CSV path (default: data/survival_events_hunk_v2.csv)'
    )
    args = parser.parse_args()
    output_file = args.output

    print("=" * 70)
    print("SURVIVAL MEASUREMENT - HUNK LEVEL (v2)")
    print("=" * 70)
    print(f"Observation cutoff : {OBSERVATION_CUTOFF.date()}")
    print(f"Hunk cap           : None (all hunks retained)")
    print(f"File cap           : None (all files retained)")
    print(f"PR scale covariates: pr_hunk_count, pr_file_count")
    print()

    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run filter_repos.py first.")
        return

    df_repos = pd.read_csv(INPUT_FILE)
    slugs    = set(df_repos['slug'].unique())
    print(f"Filtered repositories : {len(slugs)}")

    print("Loading PR data...")
    df_agent = pd.read_parquet(PR_DATA_DIR / "all_pull_request.parquet")
    df_human = pd.read_parquet(PR_DATA_DIR / "human_pull_request.parquet")

    def find_url_col(df):
        return next(
            (c for c in ['repo_url', 'url', 'repository_url'] if c in df.columns),
            None
        )

    df_agent['slug']     = df_agent[find_url_col(df_agent)].apply(normalize_slug)
    df_agent['is_agent'] = True
    df_human['slug']     = df_human[find_url_col(df_human)].apply(normalize_slug)
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

    # Deduplicate
    before = len(df_results)
    df_results = df_results.drop_duplicates(
        subset=['slug', 'pr_number', 'unit_id'], keep='first'
    )
    if len(df_results) < before:
        print(f"Removed {before - len(df_results):,} duplicate rows")

    df_results.to_csv(output_file, index=False)

    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total survival events : {len(df_results):,}")

    if not df_results.empty:
        n_deaths   = int(df_results['event'].sum())
        n_censored = len(df_results) - n_deaths
        print(f"  Deaths (modified)   : {n_deaths:,} "
              f"({100*n_deaths/len(df_results):.1f}%)")
        print(f"  Censored (survived) : {n_censored:,} "
              f"({100*n_censored/len(df_results):.1f}%)")

        dead = df_results[df_results['event'] == 1]
        print(f"\n  Modification rate distribution (deaths only):")
        print(f"    Mean   : {dead['modification_rate'].mean():.3f}")
        print(f"    Median : {dead['modification_rate'].median():.3f}")
        full_replace = (dead['modification_rate'] == 1.0).sum()
        print(f"    Full replacement (rate=1.0): {full_replace:,} "
              f"({100*full_replace/len(dead):.1f}%)")

        print(f"\n  PR scale covariates:")
        print(f"    Median pr_hunk_count : "
              f"{df_results['pr_hunk_count'].median():.0f}")
        print(f"    Max pr_hunk_count    : "
              f"{df_results['pr_hunk_count'].max():.0f}")
        print(f"    Median pr_file_count : "
              f"{df_results['pr_file_count'].median():.0f}")
        print(f"    Max pr_file_count    : "
              f"{df_results['pr_file_count'].max():.0f}")

        print("\n  By agent tier:")
        for tier, grp in df_results.groupby('agent_tier'):
            dr = grp['event'].mean() * 100
            n  = len(grp)
            print(f"    {tier:<20}: {n:>8,} hunks, {dr:.1f}% death rate")

        print(f"\n  Median duration  : "
              f"{df_results['duration_days'].median():.1f} days")
        print(f"  Mean duration    : "
              f"{df_results['duration_days'].mean():.1f} days")
        print(f"  Median hunk size : "
              f"{df_results['hunk_size'].median():.0f} lines")
        print(f"  Mean hunk size   : "
              f"{df_results['hunk_size'].mean():.1f} lines")

    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()