"""
emulate_prior_studies.py

Emulates, on our v5 data, the operationalization of published post-merge
studies of agent-attributed code as closely as the data allow, reports the
direction we obtain, and shows which single change moves it. These are
approximations of each design (noted per block), not reproductions.

  Xia & Miller (2026): line units, all agents pooled, repository-stratified Cox
      (cluster-robust SE); corrective modifications as a cause-specific hazard,
      intent from a keyword-first classifier (keyword label, LLM majority for
      'unknown') vs the LLM majority vote.
  Sawada et al. (2026): file level, files touched by agent vs human PRs,
      modification commits per file per month over months 1-6 (distinct
      commits that modify a tracked line of the file; a lower bound), pooled
      agents; human share of commits modifying agent files.
  Popescu et al. (2026): survival at 3 days, 1 week, 3 weeks per agent (they use
      the first commit of each PR and count a commit as surviving when all its
      added lines are unchanged; we approximate at the line level because v5 does
      not identify the first commit),
      pooled across repositories (Kaplan-Meier); single change: within-repository
      Cox per agent.
  Kraishan (2026): per agent, churn c = 100 x follow-up commits touching the
      PR's most-changed file within 90 days of merge / PR changed lines
      (follow-up commits here = distinct commits modifying a tracked line of that
      file, a lower bound; changed lines = added lines); median and Cliff's delta
      vs human, pooled; single change: repository fixed effects on log(1 + c)
      with repository-clustered SE.
  Takerngsaksiri et al. (2026): PR level, Mantel-Haenszel odds ratio stratified
      over repositories with both cohorts, agents pooled vs human, 30-day window.
      Their outcome is a verified follow-up fix PR; our proxy is any modification
      of the PR's lines within 30 days, so the outcome differs.

Usage:
    python src/emulate_prior_studies.py
Output:
    results/emulation_prior_studies.txt
"""

import os
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from config.observation_window import apply_observation_window
from config.paths import P, DATA_LABEL
from analyze_rq1_master import add_two_tier_column, is_generated, OUTLIER_PR_KEYS

MODELS = ['deepseek', 'gpt_oss', 'minimax']


def load(path, label):
    df = add_two_tier_column(apply_observation_window(pd.read_csv(path, low_memory=False), label=label))
    df = df[~df['unit_id'].apply(is_generated)]
    return df[~pd.Series(list(zip(df['slug'], df['pr_number'])), index=df.index).isin(OUTLIER_PR_KEYS)]


def cox(d, stratified):
    d = d.copy()
    if stratified:
        d = d[d.groupby('s')['t'].transform('nunique') == 2]
        d['c'] = d['s']
        c = CoxPHFitter().fit(d[['d', 'e', 't', 's', 'c']], 'd', 'e', strata=['s'], cluster_col='c')
    else:
        c = CoxPHFitter().fit(d[['d', 'e', 't']], 'd', 'e')
    r = c.summary.loc['t']
    return f"HR={r['exp(coef)']:.3f} [{r['exp(coef) lower 95%']:.3f}, {r['exp(coef) upper 95%']:.3f}] p={r['p']:.2g}"


def frame(df, treat_mask, event):
    return pd.DataFrame({'d': df['duration_days'].values, 'e': event.astype(int).values,
                         't': treat_mask.astype(int).values, 's': df['slug'].values})


def intent_labels():
    msg = pd.read_csv(P('data/death_commit_messages_v5.csv'))[['slug', 'sha', 'intent']].drop_duplicates(['slug', 'sha'])
    for m in MODELS:
        r = pd.read_csv(P(f'results/intent_{m}_full_corpus_t0.csv'))[['slug', 'sha', 'label']].drop_duplicates(['slug', 'sha'])
        msg = msg.merge(r.rename(columns={'label': m}), on=['slug', 'sha'], how='left')
    votes = msg[MODELS]
    maj = votes.mode(axis=1)[0].where(votes.nunique(axis=1) < 3, 'undecided')
    msg['llm_majority'] = maj
    msg['keyword_first'] = np.where(msg['intent'].isin(['corrective', 'perfective', 'adaptive', 'preventive']),
                                    msg['intent'], msg['llm_majority'])
    return msg.rename(columns={'sha': 'death_sha'})[['slug', 'death_sha', 'keyword_first', 'llm_majority']]


def main():
    v5 = load(P('data/survival_events_line_v5_analysis.csv'), 'v5')
    v2 = load(P('data/survival_events_line_v4_ast.csv'), 'v2')
    out = ['=' * 70, f'Emulating published post-merge studies on the {DATA_LABEL} data', '=' * 70]

    # --- Xia & Miller
    out += ['', '--- Xia & Miller (2026): pooled agents vs human, line units, repository-stratified Cox']
    for name, df in [(f'{DATA_LABEL} (identity tracking, all lines)', v5), ('v2 tracking, capped (single change: tracking + caps)', v2)]:
        f = frame(df, df['is_agent'].astype(bool), df['event'] == 1)
        out.append(f'  {name}: unadjusted {cox(f, False)} | stratified {cox(f, True)}')
    lab = intent_labels()
    x = v5.merge(lab, on=['slug', 'death_sha'], how='left')
    for col in ['keyword_first', 'llm_majority']:
        ev = (x['event'] == 1) & (x[col] == 'corrective')
        f = frame(x, x['is_agent'].astype(bool), ev)
        out.append(f'  corrective modifications ({col}): unadjusted {cox(f, False)} | stratified {cox(f, True)}')
    out.append('  Xia & Miller report: overall HR 1.11 (NS, stratified); corrective HR 1.49 (Table V).')

    # --- Sawada et al.
    out += ['', '--- Sawada et al. (2026): file level, modification commits per file per month (months 1-6)']
    v5['path'] = v5['unit_id'].str.rsplit('::', n=1).str[0]
    ev = v5[(v5['event'] == 1) & (v5['duration_days'] <= 183)]
    files = v5.groupby(['slug', 'pr_number', 'path']).agg(agent=('is_agent', 'first'), tier=('two_tier', 'first')).reset_index()
    commits = ev.groupby(['slug', 'pr_number', 'path'])['death_sha'].nunique().rename('commits')
    files = files.merge(commits, on=['slug', 'pr_number', 'path'], how='left').fillna({'commits': 0})
    files['per_month'] = files['commits'] / 6.0
    for t, g in files.groupby('tier'):
        out.append(f'  {t:17s} files {len(g):6,d}  mean commits/file/month {g["per_month"].mean():.3f}  '
                   f'share of files ever modified {100 * (g["commits"] > 0).mean():.1f}%')
    mods = pd.read_csv(P('results/modifier_authors.csv')).rename(columns={'sha': 'death_sha'})
    am = ev[ev['is_agent'].astype(bool)].drop_duplicates(['slug', 'death_sha']).merge(mods, on=['slug', 'death_sha'], how='left')
    out.append(f'  human share of commits modifying agent-touched lines (months 1-6): {100 * (am["modifier_type"] == "human").mean():.1f}%'
               f' (Sawada et al. report about 83%)')

    # --- Popescu et al.
    out += ['', '--- Popescu et al. (2026): line survival at 3 d / 1 w / 3 w, per agent, pooled across repositories']
    groups = [('Human', ~v5['is_agent'].astype(bool))] + [(a, v5['agent_name'] == a) for a in
              v5.loc[v5['is_agent'].astype(bool), 'agent_name'].value_counts().index]
    human = ~v5['is_agent'].astype(bool)
    for name, m in groups:
        g = v5[m]
        if len(g) < 200:
            continue
        km = KaplanMeierFitter().fit(g['duration_days'], g['event'])
        s = km.survival_function_at_times([3, 7, 21]).values * 100
        line = f'  {name:16s} n={len(g):7,d}  S(3d)={s[0]:.1f}%  S(7d)={s[1]:.1f}%  S(21d)={s[2]:.1f}%'
        if name != 'Human':
            sub = v5[m | human]
            f = frame(sub, (sub['agent_name'] == name), sub['event'] == 1)
            line += f'  | within-repo {cox(f, True)}'
        out.append(line)

    # --- Kraishan
    import statsmodels.formula.api as smf
    out += ['', '--- Kraishan (2026): churn per changed line on the most-changed file, 90 days after merge']
    sizes = v5.groupby(['slug', 'pr_number']).size().rename('n_lines')
    top = v5.groupby(['slug', 'pr_number', 'path']).size().rename('n').reset_index()
    top = top.sort_values('n').drop_duplicates(['slug', 'pr_number'], keep='last')
    mf = v5.merge(top[['slug', 'pr_number', 'path']], on=['slug', 'pr_number', 'path'])
    F = mf[(mf['event'] == 1) & (mf['duration_days'] <= 90)].groupby(['slug', 'pr_number'])['death_sha'].nunique().rename('F')
    prs = v5.groupby(['slug', 'pr_number']).agg(agent=('agent_name', 'first')).join(sizes).join(F).fillna({'F': 0}).reset_index()
    prs['c'] = 100 * prs['F'] / prs['n_lines']
    hum = prs.loc[prs['agent'] == 'Human', 'c'].values

    def cliff(x, y):
        x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
        return ((x > y).sum() - (x < y).sum()) / (x.size * y.size)
    for a_ in prs['agent'].value_counts().index:
        g = prs[prs['agent'] == a_]
        line = f'  {a_:16s} PRs {len(g):4d}  median c {g["c"].median():.3f}  mean c {g["c"].mean():.3f}'
        if a_ != 'Human':
            sub = prs[prs['agent'].isin([a_, 'Human'])].copy()
            sub = sub[sub.groupby('slug')['agent'].transform('nunique') == 2]
            sub['t'] = (sub['agent'] == a_).astype(int); sub['y'] = np.log1p(sub['c'])
            line += f'  | pooled Cliff delta vs human {cliff(g["c"], hum):+.2f}'
            if sub['slug'].nunique() >= 3:
                m = smf.ols('y ~ t + C(slug)', sub).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(sub['slug'])[0]})
                line += f' | within-repo log(1+c) diff {m.params["t"]:+.3f} [{m.conf_int().loc["t", 0]:+.3f}, {m.conf_int().loc["t", 1]:+.3f}] p={m.pvalues["t"]:.2g} ({sub["slug"].nunique()} repos)'
        out.append(line)
    out.append('  Kraishan reports (normalized churn): Claude Code lowest (median 0.5 vs 5.3 human), Copilot delta -.16, Cursor -.08, Codex and Devin at human level.')

    # --- Takerngsaksiri et al.
    from statsmodels.stats.contingency_tables import StratifiedTable
    out += ['', '--- Takerngsaksiri et al. (2026): PR-level, 30 days, Mantel-Haenszel OR stratified by repository (proxy outcome: any modification)']
    pr = v5.groupby(['slug', 'pr_number']).apply(
        lambda g: pd.Series({'tier': g['two_tier'].iloc[0],
                             'first': g.loc[g['event'] == 1, 'duration_days'].min() if (g['event'] == 1).any() else np.nan,
                             'follow': g['duration_days'].max()})).reset_index()
    pr['e'] = pr['first'].notna().astype(int)
    pr['d'] = pr['first'].fillna(pr['follow'])
    pr['y30'] = ((pr['e'] == 1) & (pr['d'] <= 30)).astype(int)
    for t_, g in pr.groupby('tier'):
        out.append(f'  {t_:17s} PRs {len(g):4d}  share with a modification within 30 days {100 * g["y30"].mean():.1f}%')

    def mh(sub, treat):
        sub = sub[sub.groupby('slug')['tier'].transform(lambda s: s.isin(treat).any() and (s == 'human').any())]
        tabs = []
        for _, g in sub.groupby('slug'):
            x = g['tier'].isin(treat)
            tb = np.array([[((x) & (g['y30'] == 1)).sum(), ((x) & (g['y30'] == 0)).sum()],
                           [((~x) & (g['y30'] == 1)).sum(), ((~x) & (g['y30'] == 0)).sum()]], dtype=float)
            tabs.append(tb)
        st = StratifiedTable(tabs)
        lo, hi = st.oddsratio_pooled_confint()
        pooled = np.sum(tabs, axis=0)
        crude = (pooled[0, 0] * pooled[1, 1]) / (pooled[0, 1] * pooled[1, 0])
        return f'crude OR, same repositories {crude:.2f} | MH OR {st.oddsratio_pooled:.2f} [{lo:.2f}, {hi:.2f}] p={st.test_null_odds().pvalue:.2g} ({len(tabs)} repos)'
    out.append(f'  agents pooled vs human: {mh(pr, ["human_supervised", "fully_autonomous"])}')
    for t_ in ['human_supervised', 'fully_autonomous']:
        out.append(f'  {t_:17s} vs human: {mh(pr[pr["tier"].isin([t_, "human"])], [t_])}')
    out.append('  Takerngsaksiri et al. report: verified-fix MH OR 1.62 [1.10, 2.39] (218 repositories), agents pooled.')

    text = '\n'.join(out) + '\n'
    open(P('results/emulation_prior_studies.txt'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
