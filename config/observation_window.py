"""
Observation-window enforcement shared by all analysis scripts.

Design (as stated in the paper): code units are born in PRs merged before
MERGE_CUTOFF and are tracked until OBSERVATION_CUTOFF, giving every unit at
least five months of follow-up.

apply_observation_window(df):
  1. drops units with birth_date >= MERGE_CUTOFF;
  2. administratively censors at OBSERVATION_CUTOFF: modifications observed
     after the cutoff become censored (event=0, death fields set to NaN);
  3. sets duration_days = OBSERVATION_CUTOFF - birth_date for every
     censored unit;
  4. logs the units dropped, censored and re-timed per tier.

Robustness check (VERIFIED_MERGE=1 in the environment): additionally keep
only units from PRs whose birth commit is verified against AIDev, i.e. the
PR has a merged_at and birth_date is within 1 day of it. In that mode
output_path() sends every result and figure to
results/robustness_verified_merge/ so the main results are untouched.
Example:
    VERIFIED_MERGE=1 python src/analyze_rq1_master.py

Timestamps follow the measurement scripts (measure_retention_line_v2.py,
measure_retention_hunk_v2.py): naive UTC datetimes, durations in days as
total_seconds / 86400.
"""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config.constants import OBSERVATION_CUTOFF, PR_INCLUSION_CUTOFF
from config.paths import P

MERGE_CUTOFF = PR_INCLUSION_CUTOFF   # exclusive: birth_date < 2025-08-01
DEATH_COLUMNS = ['death_sha', 'death_date', 'ast_death']
TIER_ORDER = ['human_supervised', 'fully_autonomous', 'human']

VERIFIED_MERGE = os.environ.get('VERIFIED_MERGE') == '1'
ROBUSTNESS_DIR = Path(P('results/robustness_verified_merge'))
MAX_MERGE_GAP_DAYS = 1.0
AGENT_PRS = 'data/all_pull_request.parquet'
HUMAN_PRS = 'data/human_pull_request.parquet'


def output_path(path) -> str:
    """Main-analysis path, or its robustness_verified_merge/ twin if VERIFIED_MERGE."""
    if not VERIFIED_MERGE:
        return str(path)
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    return str(ROBUSTNESS_DIR / Path(path).name)


@lru_cache(maxsize=1)
def _aidev_merged_at() -> pd.DataFrame:
    """(slug, pr_number, is_agent) -> AIDev merged_at, naive UTC."""
    frames = []
    for path, is_agent in [(AGENT_PRS, True), (HUMAN_PRS, False)]:
        prs = pd.read_parquet(path, columns=['number', 'repo_url', 'merged_at'])
        frames.append(pd.DataFrame({
            '_slug': prs['repo_url'].str.extract(r'repos/(.+)$')[0].str.lower(),
            'pr_number': prs['number'],
            'is_agent': is_agent,
            '_merged_at': pd.to_datetime(prs['merged_at'], utc=True, errors='coerce',
                                         format='ISO8601').dt.tz_localize(None),
        }))
    return pd.concat(frames).drop_duplicates(['_slug', 'pr_number', 'is_agent'])


def _two_tier(df: pd.DataFrame) -> pd.Series:
    """Vectorised two-tier label, for logging only."""
    tier = np.where(
        ~df['is_agent'].astype(bool), 'human',
        np.where(df['agent_tier'] == 'autonomous', 'fully_autonomous',
                 np.where(df['agent_tier'].isin(['pair_programmer', 'batch_generator']),
                          'human_supervised', 'unknown'))
    )
    return pd.Series(tier, index=df.index)


def _per_tier(tier: pd.Series, mask: pd.Series) -> str:
    counts = tier[mask].value_counts()
    return ', '.join(f'{t}={counts.get(t, 0):,}' for t in TIER_ORDER)


def apply_observation_window(df: pd.DataFrame, drop_late_births: bool = True,
                             label: str = 'units') -> pd.DataFrame:
    """Enforce the merge cutoff and administrative censoring (see module docstring).

    drop_late_births=False applies only the censoring steps (2-3), for
    analyses that vary the merge cutoff themselves.
    """
    df = df.copy()
    tier = _two_tier(df)
    print(f'Observation window ({label}): {len(df):,} units before')

    if VERIFIED_MERGE:
        merged = df[['slug', 'pr_number', 'is_agent']].assign(
            _slug=df['slug'].str.lower(), is_agent=df['is_agent'].astype(bool)
        ).merge(_aidev_merged_at(), on=['_slug', 'pr_number', 'is_agent'], how='left')
        merged.index = df.index
        gap = (pd.to_datetime(df['birth_date'], format='ISO8601')
               - merged['_merged_at']).dt.total_seconds().abs() / 86400.0
        never_merged = merged['_merged_at'].isna()
        misdated = ~never_merged & (gap > MAX_MERGE_GAP_DAYS)
        print(f'  [VERIFIED_MERGE] dropped, no AIDev merged_at: {never_merged.sum():,} '
              f'({_per_tier(tier, never_merged)})')
        print(f'  [VERIFIED_MERGE] dropped, birth > {MAX_MERGE_GAP_DAYS:g} day from merged_at: '
              f'{misdated.sum():,} ({_per_tier(tier, misdated)})')
        keep = ~never_merged & ~misdated
        df, tier = df[keep].copy(), tier[keep]

    birth = pd.to_datetime(df['birth_date'], format='ISO8601')

    if drop_late_births:
        late_birth = birth >= MERGE_CUTOFF
        print(f'  dropped, born >= {MERGE_CUTOFF.date()}: {late_birth.sum():,} '
              f'({_per_tier(tier, late_birth)})')
        df, birth, tier = df[~late_birth].copy(), birth[~late_birth], tier[~late_birth]

    death = pd.to_datetime(df['death_date'], format='ISO8601')
    censor_duration = (OBSERVATION_CUTOFF - birth).dt.total_seconds() / 86400.0

    late_death = (df['event'] == 1) & (death > OBSERVATION_CUTOFF)
    df.loc[late_death, 'event'] = 0
    df.loc[late_death, [c for c in DEATH_COLUMNS if c in df.columns]] = np.nan
    print(f'  censored, modified after {OBSERVATION_CUTOFF.date()}: {late_death.sum():,} '
          f'({_per_tier(tier, late_death)})')

    censored = df['event'] == 0
    retimed = censored & ~late_death & ~np.isclose(df['duration_days'], censor_duration)
    df.loc[censored, 'duration_days'] = censor_duration[censored]
    print(f'  already-censored units re-timed to the cutoff: {retimed.sum():,} '
          f'({_per_tier(tier, retimed)})')
    print(f'  {len(df):,} units after')
    return df
