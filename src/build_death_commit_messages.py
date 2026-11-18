"""
build_death_commit_messages.py

Builds the modification-commit table used by the intent analyses:
(slug, sha, message, intent) for every commit that modifies a line unit,
with the keyword-based intent label.

The keyword classifier (classify_intent) and the GitHub message client are
restored verbatim from the deleted src/analyze_intent.py (git 2adf0cc^), so
labels for commits already in data/death_commit_messages.csv are reproduced
exactly; the script checks this on every run. Commit messages are cached in
data/intent_api_cache.json; existing messages are reused, only new commits are
fetched from the GitHub API.

Modification commits are the death_sha values of line units that remain
modified after the observation window (config/observation_window.py).

Outputs:
  data/death_commit_messages_v5.csv

Usage:
    python src/build_death_commit_messages.py \
        --line-data data/survival_events_line_v5_analysis.csv \
        --previous data/death_commit_messages.csv \
        --output data/death_commit_messages_v5.csv
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.observation_window import apply_observation_window
from config.paths import P

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
API_BASE = "https://api.github.com"
SLEEP_BETWEEN = 0.72   # seconds (5000/hour)
MAX_RETRIES = 3
CACHE_FILE = Path(P("data/intent_api_cache.json"))


# Conventional commit type → intent mapping
CONVENTIONAL_MAP = {
    # Corrective
    'fix':      'corrective',
    'bugfix':   'corrective',
    'hotfix':   'corrective',
    'patch':    'corrective',

    # Perfective
    'refactor': 'perfective',
    'perf':     'perfective',
    'style':    'perfective',
    'improve':  'perfective',
    'cleanup':  'perfective',
    'clean':    'perfective',

    # Adaptive
    'feat':     'adaptive',
    'feature':  'adaptive',
    'add':      'adaptive',
    'new':      'adaptive',
    'update':   'adaptive',
    'change':   'adaptive',

    # Preventive
    'test':     'preventive',
    'tests':    'preventive',
    'spec':     'preventive',
    'docs':     'preventive',
    'doc':      'preventive',
    'chore':    'preventive',
    'ci':       'preventive',
    'build':    'preventive',
    'lint':     'preventive',
    'types':    'preventive',
    'type':     'preventive',
}

# Free-text keyword patterns → intent
# Applied in order — first match wins
KEYWORD_PATTERNS = [
    # Corrective
    ('corrective', [
        r'\bfix(es|ed|ing)?\b',
        r'\bbug\b',
        r'\berror\b',
        r'\bcrash\b',
        r'\broken\b',
        r'\bbreaking\b',
        r'\bincorrect\b',
        r'\bwrong\b',
        r'\bresolve[sd]?\b',
        r'\bregression\b',
        r'\bdefect\b',
        r'\bproblem\b',
        r'\bissue\b',
        r'\bnpe\b',
        r'\bnull pointer\b',
        r'\bexception\b',
        r'\bfailure\b',
        r'\bpatch\b',
    ]),

    # Perfective
    ('perfective', [
        r'\brefactor\b',
        r'\boptimi[sz]e[sd]?\b',
        r'\boptimiz(ation|ing)\b',
        r'\bimprove[sd]?\b',
        r'\bimprovement\b',
        r'\bperformance\b',
        r'\bclean\s*up\b',
        r'\bcleanup\b',
        r'\bsimplif(y|ies|ied|ying)\b',
        r'\brestructure[sd]?\b',
        r'\bclarif(y|ies|ied)\b',
        r'\benhance[sd]?\b',
        r'\benhancement\b',
        r'\bformat(ting)?\b',
        r'\bstyle\b',
        r'\breach?abilit(y|ies)\b',
        r'\breadabilit(y|ies)\b',
        r'\bdeadcode\b',
        r'\bdead\s+code\b',
        r'\bcode\s+quality\b',
        r'\bdedup(licate|lication)?\b',
        r'\bextract\b',
        r'\binline\b',
        r'\brename\b',
        r'\bmove\b',
    ]),

    # Adaptive
    ('adaptive', [
        r'\badd(s|ed|ing)?\b',
        r'\bnew\b',
        r'\bfeature[sd]?\b',
        r'\bimplement(s|ed|ation)?\b',
        r'\bintroduce[sd]?\b',
        r'\bsupport(s|ed|ing)?\b',
        r'\bintegrat(e|es|ed|ion)\b',
        r'\bmigrat(e|es|ed|ion)\b',
        r'\bupgrad(e|es|ed|ing)\b',
        r'\bbump\b',
        r'\bdependenc(y|ies)\b',
        r'\bdeprecate[sd]?\b',
        r'\bremove[sd]?\b',
        r'\bdelete[sd]?\b',
        r'\bdrop\b',
        r'\bport(ing)?\b',
        r'\bupdate[sd]?\b',
        r'\bmodif(y|ied|ication)\b',
        r'\bchange[sd]?\b',
        r'\benable[sd]?\b',
        r'\bdisable[sd]?\b',
    ]),

    # Preventive
    ('preventive', [
        r'\btest(s|ed|ing)?\b',
        r'\bspec(s|ification)?\b',
        r'\bdoc(s|ument|umentation)?\b',
        r'\bcomment(s|ed)?\b',
        r'\bci\b',
        r'\bcd\b',
        r'\bworkflow[s]?\b',
        r'\blint(ing)?\b',
        r'\btype[ds]?\s+hint\b',
        r'\btype\s+check\b',
        r'\bvalidat(e|es|ion)\b',
        r'\bassert(s|ion)?\b',
        r'\bsanit(ize|ization)\b',
        r'\bsecuri(ty|tize)\b',
        r'\baudit\b',
        r'\bversion\b',
        r'\brelease\b',
        r'\bchore\b',
        r'\bbuild\b',
        r'\bpackage\b',
        r'\bscript[s]?\b',
        r'\bconfigurati?on\b',
    ]),
]


def classify_intent(message: str) -> str:
    """
    Classify a commit message by Swanson intent taxonomy.

    Strategy:
    1. Extract conventional commit prefix (e.g. "fix:", "feat(scope):")
       and check against CONVENTIONAL_MAP
    2. Extract Apache-style bracket prefix (e.g. "[fix]", "[improve]")
       and check against CONVENTIONAL_MAP
    3. Fall back to keyword pattern matching on lowercased full message
    4. Return 'unknown' if no signal found

    Returns one of: corrective, perfective, adaptive, preventive, unknown
    """
    if not message or not str(message).strip():
        return 'unknown'

    msg_lower = str(message).lower().strip()
    first_line = msg_lower.split('\n')[0].strip()

    # 1. Conventional commit prefix: "type:" or "type(scope):"
    conventional_match = re.match(
        r'^([a-z][a-z0-9_-]*)(?:\([^)]*\))?!?\s*:', first_line
    )
    if conventional_match:
        prefix = conventional_match.group(1).lower()
        if prefix in CONVENTIONAL_MAP:
            return CONVENTIONAL_MAP[prefix]

    # 2. Apache/bracket prefix: "[type]" or "[type][scope]"
    bracket_match = re.match(r'^\[([a-z][a-z0-9_/\s-]*)\]', first_line)
    if bracket_match:
        bracket_type = bracket_match.group(1).lower().strip()
        bracket_intent = None
        if bracket_type in CONVENTIONAL_MAP:
            bracket_intent = CONVENTIONAL_MAP[bracket_type]
        else:
            for word in bracket_type.split():
                if word in CONVENTIONAL_MAP:
                    bracket_intent = CONVENTIONAL_MAP[word]
                    break

        if bracket_intent:
            # Check if message body contains a strong corrective override
            # e.g. "[improve] Fixes ..." should be corrective not perfective
            corrective_override = re.search(
                r'\bfix(es|ed|ing)?\b|\bbug\b|\bcrash\b|\bregression\b',
                first_line, re.IGNORECASE
            )
            if corrective_override and bracket_intent != 'corrective':
                return 'corrective'
            return bracket_intent

    # 3. Keyword pattern matching on full message
    for intent, patterns in KEYWORD_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                return intent

    return 'unknown'


class GitHubClient:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        })
        self._count = 0
        self._hour_start = time.time()

    def _rate_limit(self):
        elapsed = time.time() - self._hour_start
        if elapsed >= 3600:
            self._count = 0
            self._hour_start = time.time()
        if self._count >= 4950:
            wait = 3600 - elapsed + 5
            print(f"\n  [Rate limit] sleeping {wait:.0f}s...")
            time.sleep(wait)
            self._count = 0
            self._hour_start = time.time()

    def get_commit_message(self, slug: str, sha: str) -> str:
        """Fetch commit message for a given SHA. Returns empty string on failure."""
        self._rate_limit()
        url = f"{API_BASE}/repos/{slug}/commits/{sha}"

        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, timeout=20)
                self._count += 1
                time.sleep(SLEEP_BETWEEN)

                if resp.status_code == 200:
                    data = resp.json()
                    return data.get('commit', {}).get('message', '').strip()
                elif resp.status_code == 404:
                    return ''
                elif resp.status_code == 403:
                    reset = int(resp.headers.get('X-RateLimit-Reset', 0))
                    wait = max(reset - time.time(), 60)
                    time.sleep(wait)
                    continue
                else:
                    time.sleep(2 ** attempt)
                    continue
            except Exception:
                time.sleep(2 ** attempt)
                continue

        return ''



def local_message(slug: str, sha: str):
    """Commit message from the local bare clone used by the v5 miner, if present."""
    repo = Path(P('data/temp_repos')) / (slug.replace('/', '__') + '.git')
    if not repo.is_dir():
        return None
    import subprocess
    r = subprocess.run(['git', '-C', str(repo), 'log', '-1', '--format=%B', sha],
                       capture_output=True, text=True, errors='replace')
    return r.stdout.strip() if r.returncode == 0 else None


def run(line_data: str, previous: str, output: str) -> None:
    line = pd.read_csv(line_data, usecols=[
        'slug', 'pr_number', 'is_agent', 'agent_tier', 'unit_id', 'death_sha',
        'event', 'duration_days', 'birth_date', 'death_date'])
    line = apply_observation_window(line, label='line').dropna(subset=['death_sha'])
    commits = line[['slug', 'death_sha']].drop_duplicates().rename(columns={'death_sha': 'sha'})

    prev = pd.read_csv(previous)
    reproduced = prev['message'].fillna('').map(classify_intent)
    mismatch = (reproduced != prev['intent']).sum()
    print(f"Classifier check on {len(prev):,} previous commits: {mismatch} label mismatches")
    if mismatch:
        sys.exit("Restored classifier does not reproduce the previous labels; stopping.")

    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    known = {(r.slug, r.sha): r.message for r in prev.itertuples()}
    client = GitHubClient(GITHUB_TOKEN)
    rows, fetched = [], 0
    for r in commits.itertuples():
        key = f"{r.slug}::{r.sha}"
        if (r.slug, r.sha) in known:
            message = '' if pd.isna(known[(r.slug, r.sha)]) else known[(r.slug, r.sha)]
        elif key in cache:
            message = cache[key]
        else:
            message = local_message(r.slug, r.sha)
            if message is None:
                message = client.get_commit_message(r.slug, r.sha)
                fetched += 1
            cache[key] = message
        rows.append({'slug': r.slug, 'sha': r.sha, 'message': message,
                     'intent': classify_intent(message)})
    CACHE_FILE.write_text(json.dumps(cache))
    out = pd.DataFrame(rows)
    out.to_csv(output, index=False)
    new = (~out.set_index(['slug', 'sha']).index.isin(list(known))).sum()
    print(f"{len(out):,} modification commits ({new} not in {previous}; {fetched} messages fetched)")
    print(out['intent'].value_counts().to_string())
    print(f"Written: {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--line-data', default=P('data/survival_events_line_v5_analysis.csv'))
    parser.add_argument('--previous',  default=P('data/death_commit_messages.csv'))
    parser.add_argument('--output',    default=P('data/death_commit_messages_v5.csv'))
    args = parser.parse_args()
    run(args.line_data, args.previous, args.output)
