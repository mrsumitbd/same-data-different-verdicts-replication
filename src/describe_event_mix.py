"""
describe_event_mix.py

What the modification events are, before and after the re-mine.

v2 (data/survival_events_line_v4_ast.csv, window applied):
  death_magnitude mix by tier, and for whole-file "deletion" events the status
  of the file in the death commit (from the cached commit diffs): removed,
  renamed (path absent but a file with the same name renamed in that commit),
  file list truncated, or other.
v5 (data/survival_events_line_v5_analysis.csv):
  event_type mix by tier, whitespace-only share of content modifications, and
  the Definition 2 mix (leaf-only / structural / unclassified) among
  modifications.

Usage:
    python src/describe_event_mix.py
Output:
    results/event_mix.txt
"""

import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.observation_window import apply_observation_window
from config.paths import P, DATA_LABEL
from analyze_rq1_master import add_two_tier_column


def file_status(cache, slug, sha, path):
    t = cache.get((slug, sha))
    if t is None or pd.isna(t):
        return 'commit not in diff cache'
    head = str(t).split('--- DIFF ---')[0]
    m = re.search(r'- ' + re.escape(path) + r' \((\w+)', head)
    if m:
        return 'file ' + m.group(1)
    if any(os.path.basename(r) == os.path.basename(path) for r in re.findall(r'- (\S+) \(renamed', head)):
        return 'renamed (same file name, new path)'
    if '[... cut at' in head or len(re.findall(r'^\s+- ', head, re.M)) >= 300:
        return 'file list truncated'
    return 'other'


def pct(s):
    return (s * 100).round(1)


def main():
    out = ['=' * 70, f'Modification events: line-number tracking (v2) vs identity tracking ({DATA_LABEL})', '=' * 70]
    v2 = add_two_tier_column(apply_observation_window(pd.read_csv(P('data/survival_events_line_v4_ast.csv')), label='v2'))
    ev2 = v2[v2['event'] == 1]
    cat = ev2['death_magnitude'].map(lambda m: 'file deletion' if m == 'deletion' else
                                     ('no recorded change' if m == 'unknown' else 'content change'))
    out += ['', f'v2: {len(ev2):,} modification events', 'v2 event mix by tier (% of events):',
            pct(pd.crosstab(ev2['two_tier'], cat, normalize='index')).to_string(),
            f'overall: ' + ', '.join(f'{k} {v:.1f}%' for k, v in (cat.value_counts(normalize=True) * 100).items())]
    cache = pd.read_csv(P('results/diff_cache_full_corpus.csv')).drop_duplicates(['slug', 'sha']).set_index(['slug', 'sha'])['diff_text']
    de = ev2[ev2['death_magnitude'] == 'deletion'].assign(path=lambda x: x['unit_id'].str.rsplit('::', n=1).str[0])
    keys = de[['slug', 'death_sha', 'path']].drop_duplicates()
    keys['status'] = [file_status(cache, a, b, c) for a, b, c in keys.itertuples(index=False)]
    st = de.merge(keys, on=['slug', 'death_sha', 'path'])['status'].value_counts()
    out += ['', f'v2 "file deletion" events ({len(de):,}; {de["death_sha"].nunique():,} death commits) by file status in the death commit:',
            st.to_string()]

    v5 = add_two_tier_column(pd.read_csv(P('data/survival_events_line_v5_analysis.csv'), low_memory=False))
    ev5 = v5[v5['event'] == 1]
    out += ['', f'{DATA_LABEL}: {len(v5):,} units, {len(ev5):,} modification events', f'{DATA_LABEL} event_type mix by tier (% of units):',
            pct(pd.crosstab(v5['two_tier'], v5['event_type'], normalize='index')).to_string(),
            f'{DATA_LABEL} event_type mix by tier (% of modification events):',
            pct(pd.crosstab(ev5['two_tier'], ev5['event_type'], normalize='index')).to_string()]
    cm = ev5[ev5['event_type'] == 'content_modified']
    ws = cm['whitespace_only'].astype(str).eq('True')
    out += ['', f'whitespace-only share of content modifications: {ws.mean() * 100:.1f}% overall; by tier: '
            + ', '.join(f'{t} {v:.1f}%' for t, v in (ws.groupby(cm['two_tier']).mean() * 100).items())]
    a = ev5['ast_death'].map({1.0: 'structural', 0.0: 'leaf-only'}).fillna('unclassified')
    out += ['', 'Definition 2 among modifications (% by tier):', pct(pd.crosstab(ev5['two_tier'], a, normalize='index')).to_string(),
            'unclassified reasons (% of unclassified by tier):',
            pct(pd.crosstab(ev5.loc[a == 'unclassified', 'two_tier'], ev5.loc[a == 'unclassified', 'ast_exclusion_reason'], normalize='index')).to_string()]
    text = '\n'.join(out) + '\n'
    open(P('results/event_mix.txt'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
