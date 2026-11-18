"""
plot_rq1.py

Generates the two RQ1 figures with two-tier taxonomy and improved typography:
  1. rq1_km_def1_line.pdf  -- KM retention curves by tier, Definition 1 (syntactic)
  2. rq1_hunk_stratified_km.pdf -- KM curves by tier x block size (hunk level)

TWO-TIER TAXONOMY:
  human_supervised = pair_programmer + batch_generator
  fully_autonomous = autonomous (Devin only)
  human            = human baseline

Typography: base font 14pt, all labels/ticks/titles/legend bold,
colorblind-safe Wong 2011 palette, no top/right spines.

Usage:
    python src/plot_rq1.py \
        --line-data data/survival_events_line_v5_analysis.csv \
        --hunk-data data/survival_events_hunk_v5.csv

Robustness check (verified merges only; outputs to results/robustness_verified_merge/):
    VERIFIED_MERGE=1 python src/plot_rq1.py
"""

import argparse
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from pathlib import Path
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.observation_window import apply_observation_window, output_path
from config.paths import P

VISUALS_DIR = Path(output_path(P("visuals/_"))).parent
VISUALS_DIR.mkdir(exist_ok=True)

OUTLIER_PR_KEYS = {('microsoft/typespec', 7115)}
GENERATED_PATH_PATTERNS = [
    r'/dist/', r'/third_party/', r'/vendor/', r'/generated/',
    r'_pb2\.py$', r'\.min\.js$', r'immutable_entry',
    r'_generated\.', r'zz_generated\.',
]

HUMAN_SUPERVISED_TIERS = {'pair_programmer', 'batch_generator'}
FULLY_AUTONOMOUS_TIERS  = {'autonomous'}

# Colorblind-safe palette (Wong 2011)
TIER_COLORS = {
    'human_supervised': '#0072B2',   # blue
    'fully_autonomous': '#D55E00',   # vermilion
    'human':            '#009E73',   # green
}
TIER_LABELS = {
    'human_supervised': 'Human-supervised',
    'fully_autonomous': 'Fully-autonomous',
    'human':            'Human',
}

BASE_FONT = 14
plt.rcParams.update({
    'font.size':          BASE_FONT,
    'axes.titlesize':     BASE_FONT + 2,
    'axes.labelsize':     BASE_FONT,
    'xtick.labelsize':    BASE_FONT - 1,
    'ytick.labelsize':    BASE_FONT - 1,
    'legend.fontsize':    BASE_FONT - 1,
    'axes.titleweight':   'bold',
    'axes.labelweight':   'bold',
    'font.weight':        'bold',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})


def is_generated(unit_id: str) -> bool:
    filepath = str(unit_id).split('::')[0]
    return any(re.search(p, filepath) for p in GENERATED_PATH_PATTERNS)


def add_two_tier_column(df: pd.DataFrame) -> pd.DataFrame:
    def map_tier(row):
        if not row['is_agent']:
            return 'human'
        if row['agent_tier'] in HUMAN_SUPERVISED_TIERS:
            return 'human_supervised'
        if row['agent_tier'] in FULLY_AUTONOMOUS_TIERS:
            return 'fully_autonomous'
        return 'unknown'
    df = df.copy()
    df['two_tier'] = df.apply(map_tier, axis=1)
    return df


def preprocess_hunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['_key'] = list(zip(df['slug'], df['pr_number']))
    df = df[~df['_key'].isin(OUTLIER_PR_KEYS)]
    df = df[~df['unit_id'].apply(is_generated)]
    return df.drop(columns=['_key'])


def set_bold_ticks(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(BASE_FONT - 1)


def plot_km_def1(line: pd.DataFrame) -> None:
    """KM retention curves by two-tier + human baseline, Definition 1 (syntactic)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for tier in ['human_supervised', 'fully_autonomous', 'human']:
        sub = line[line['two_tier'] == tier]
        if len(sub) == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub['duration_days'], sub['event'],
                label=f"{TIER_LABELS[tier]} (n={len(sub):,})")
        kmf.plot_survival_function(
            ax=ax,
            color=TIER_COLORS[tier],
            ci_show=True,
            ci_alpha=0.15,
            linewidth=2.0,
        )

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.2,
               alpha=0.6, label='Median (0.5)')
    ax.set_xlim(0, 365)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Days since merge', fontweight='bold')
    ax.set_ylabel('Retention probability', fontweight='bold')
    ax.set_title('Kaplan-Meier Retention Curves by Tier\n(Definition 1: Syntactic)',
                 fontweight='bold', pad=12)
    ax.legend(loc='lower left', framealpha=0.9,
              prop={'weight': 'bold', 'size': BASE_FONT - 1})
    ax.grid(True, alpha=0.25, linestyle=':')
    set_bold_ticks(ax)
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        plt.savefig(VISUALS_DIR / f'rq1_km_def1_line.{ext}',
                    bbox_inches='tight', dpi=150)
    plt.close()
    print('  Saved: rq1_km_def1_line.pdf / .png')


def plot_hunk_stratified(hunk: pd.DataFrame) -> None:
    """KM retention curves by two-tier, split by hunk block size."""
    size_groups = {
        'Single-line (n=1)':  hunk['hunk_size'] == 1,
        'Small block (2-10)': (hunk['hunk_size'] >= 2) & (hunk['hunk_size'] <= 10),
        'Large block (11+)':  hunk['hunk_size'] >= 11,
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    plt.subplots_adjust(wspace=0.08)

    for col_idx, (size_label, mask) in enumerate(size_groups.items()):
        ax = axes[col_idx]
        hunk_sub = hunk[mask]

        for tier in ['human_supervised', 'fully_autonomous', 'human']:
            sub = hunk_sub[hunk_sub['two_tier'] == tier]
            if len(sub) < 10:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub['duration_days'], sub['event'],
                    label=TIER_LABELS[tier])
            kmf.plot_survival_function(
                ax=ax,
                color=TIER_COLORS[tier],
                ci_show=False,
                linewidth=2.0,
            )

        ax.axhline(y=0.5, color='gray', linestyle='--',
                   linewidth=1.0, alpha=0.5)
        ax.set_xlim(0, 365)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Days since merge', fontweight='bold')
        ax.set_title(size_label, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.25, linestyle=':')
        set_bold_ticks(ax)

        if col_idx == 0:
            ax.set_ylabel('Retention probability', fontweight='bold')
            ax.legend(loc='lower left', framealpha=0.9,
                      prop={'weight': 'bold', 'size': BASE_FONT - 2})
        else:
            ax.set_ylabel('')

    fig.suptitle('Hunk-Level Retention by Tier and Block Size',
                 fontweight='bold', fontsize=BASE_FONT + 2, y=1.02)

    for ext in ['pdf', 'png']:
        plt.savefig(VISUALS_DIR / f'rq1_hunk_stratified_km.{ext}',
                    bbox_inches='tight', dpi=150)
    plt.close()
    print('  Saved: rq1_hunk_stratified_km.pdf / .png')


def run(line_data: str, hunk_data: str) -> None:
    print('Loading data...')
    line = apply_observation_window(pd.read_csv(line_data), label='line')
    hunk = apply_observation_window(pd.read_csv(hunk_data), label='hunk')
    hunk = preprocess_hunk(hunk)

    line = add_two_tier_column(line)
    hunk = add_two_tier_column(hunk)

    if 'hunk_size' not in hunk.columns:
        print('  Warning: hunk_size column not found, deriving from unit_id...')
        def extract_size(uid: str) -> int:
            parts = str(uid).split('::')
            if len(parts) >= 3:
                try:
                    return int(parts[2]) - int(parts[1]) + 1
                except (ValueError, IndexError):
                    return 1
            return 1
        hunk['hunk_size'] = hunk['unit_id'].apply(extract_size)

    print('Plotting KM curves (Definition 1, line-level)...')
    plot_km_def1(line)

    print('Plotting hunk-level KM curves by block size...')
    plot_hunk_stratified(hunk)

    print('\nDone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--line-data', default=P('data/survival_events_line_v5_analysis.csv'))
    parser.add_argument('--hunk-data', default=P('data/survival_events_hunk_v5.csv'))
    args = parser.parse_args()
    run(args.line_data, args.hunk_data)