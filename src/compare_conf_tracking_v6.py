"""
compare_conf_tracking_v6.py

Agreement between the journal's identity tracker (v6conf: measure_retention_v5
track phase on the conference born lines) and the conference's own tracker
(scripts/mine_survival.py: blame --reverse, death = first later commit that
touches the file), on the same lines.

Status per line:
  identity    modified (event 1) / censored (event 0) / unverified (event null)
  conference  died / censored (conf_kind)
Reported: verified vs unverified counts; status agreement (all lines, and
verified lines only); among lines both trackers call modified, the share with
the same modification commit; breakdown by vendored flag and by group.

Input:  data/survival_events_line_v6conf.csv (or the partial directory)
Output: results_v6/conf_tracking_agreement.txt
"""

import argparse
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ['slug', 'pr_number', 'is_agent', 'unit_id', 'vendored', 'blank', 'event', 'event_type', 'death_sha',
        'conf_kind', 'conf_death_sha', 'whitespace_only']


def load(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        files = [f for f in glob.glob(os.path.join(path, '*.csv'))
                 if not f.endswith(('.births.csv', '.pr_status.csv')) and 'stats' not in f and os.path.getsize(f) > 1]
        return pd.concat([pd.read_csv(f, usecols=COLS, low_memory=False) for f in files], ignore_index=True)
    return pd.read_csv(path, usecols=COLS, low_memory=False)


def block(d: pd.DataFrame, title: str) -> list:
    idm = d['event'].map({1: 'modified', 0: 'censored'}).fillna('unverified')
    cf = d['conf_kind'].map({'died': 'modified', 'censored': 'censored'})
    ver = idm != 'unverified'
    both_mod = (idm == 'modified') & (cf == 'modified')
    same_sha = both_mod & (d['death_sha'] == d['conf_death_sha'])
    out = [f'\n## {title}', f'lines {len(d):,}; repos {d["slug"].nunique()}; '
           f'PRs {d[["slug", "pr_number", "is_agent"]].drop_duplicates().shape[0]:,}']
    et = d['event_type'].value_counts()
    out.append('identity tracker event_type: ' + ', '.join(f'{k} {v:,} ({v / len(d) * 100:.2f}%)' for k, v in et.items()))
    out.append(f'verified {ver.sum():,} ({ver.mean() * 100:.2f}%), unverified {(~ver).sum():,} ({(~ver).mean() * 100:.2f}%)')
    out.append('\ncross-tab (rows identity, columns conference):')
    out.append(pd.crosstab(idm, cf, margins=True).to_string())
    out.append(f'\nsame status, all lines (unverified counts as disagreement): {(idm == cf).mean() * 100:.2f}%')
    out.append(f'same status, verified lines: {(idm[ver] == cf[ver]).mean() * 100:.2f}%')
    out.append(f'modified by both: {both_mod.sum():,}; same modification commit: {same_sha.sum():,} '
               f'({same_sha.sum() / max(both_mod.sum(), 1) * 100:.2f}%)')
    out.append(f'same status AND (censored, or modified in the same commit), verified lines: '
               f'{(((idm == cf) & ((idm == "censored") | same_sha))[ver]).mean() * 100:.2f}%')
    out.append(f'modification rate: identity {(idm[ver] == "modified").mean() * 100:.1f}% (verified), '
               f'conference {(cf == "modified").mean() * 100:.1f}%')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=os.path.join(ROOT, 'data', 'survival_events_line_v6conf.csv'))
    ap.add_argument('--output', default=os.path.join(ROOT, 'results_v6', 'conf_tracking_agreement.txt'))
    a = ap.parse_args()
    d = load(a.input)
    out = ['Identity tracker (journal) vs blame --reverse tracker (conference), same conference born lines',
           f'input: {a.input}']
    out += block(d, 'all conference lines')
    out += block(d[d['vendored'] == False], 'non-vendored lines')
    out += block(d[d['vendored'] == True], 'vendored lines')
    out += block(d[d['is_agent'] == True], 'agent lines')
    out += block(d[d['is_agent'] == False], 'human lines')
    text = '\n'.join(out)
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    open(a.output, 'w').write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
