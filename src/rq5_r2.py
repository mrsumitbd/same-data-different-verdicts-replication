"""
rq5_r2.py 

Basis (same as results_v6/rq4_results.txt and src/rq4_self_modification.py):
  - lines: data/survival_events_line_v6confnv_analysis.csv, observation window applied,
    microsoft/typespec#7115 and generated paths dropped (analyze_rq4.get_birth_tier_map);
  - commit unit: each of the 5,300 tier-mapped modification commits once, under the majority
    birth group of the lines it modifies (1,453 human-supervised / 617 Devin / 3,230 human);
  - modifier: commit author login (results_v6/modifier_authors.csv), agent by config/agent_logins.py;
  - self-modification: the modifier is a coding agent and at least one line the commit modifies
    was written by that agent (rq4_self_modification.py rule).
Lag (window): days from the merge of the original PR to the modifying commit; for a commit,
the shortest lag among the lines of its assigned group that it modifies (for a line, its own
duration). Windows: 7 days, 30 days, full window.

E6  unit x window: commit, modified line, follow-up PR. A follow-up PR is a merged PR that
    contains a modifying commit (GET /repos/{o}/{r}/commits/{sha}/pulls, cached by
    src/fetch_commit_pulls_r2.py), other than the PR that wrote the lines; if the commit is in
    several merged PRs, the earliest created one. Its modifier is the PR author. A follow-up PR is
    counted once per group whose lines (within the window) it modifies (and, as a check, once per
    (original PR, follow-up PR) pair).
E7  identification rule (commit unit, full window): (a) committer login (API); (b) author OR any
    Co-authored-by trailer naming a coding agent (Copilot, Cursor, Claude, Codex, Devin).
E8  the commits that modify lines of more than one group: (a) counted under every group they
    modify, (b) dropped.
E9  self-modification lags (Copilot, Devin); intent of agent-made vs human-made modifications per
    tier under every classification approach and the human labels; Copilot self-modification
    annotation sheet (20 of 143, seed 42).

Usage: DATA_TAG=v6confnv RESULTS_DIR=results_v6 python src/rq5_r2.py [e6] [e7] [e8] [e9]
Outputs: results_v6/round2/e6_modifier_unit_window.txt/.csv, e7_identification_rule.txt/.csv,
         e8_mixed_commits.txt/.csv, e9_self_modification.txt/.csv, copilot_selfmod_sample.xlsx
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)
from config.agent_logins import is_agent_login
from config.observation_window import apply_observation_window
from analyze_rq4 import get_birth_tier_map, is_generated, OUTLIER_PR_KEYS, add_two_tier_column

OUT = 'results_v6/round2'
GROUPS = ['human_supervised', 'fully_autonomous', 'human']
GLAB = {'human_supervised': 'Human-supervised', 'fully_autonomous': 'Devin', 'human': 'Human'}
WINDOWS = [('7 days', 7), ('30 days', 30), ('Full', np.inf)]
# login (lower case) -> AIDev agent that commits through it
LOGIN_AGENT = {'copilot': 'GitHub Copilot', 'copilot-swe-agent[bot]': 'GitHub Copilot', 'copilot-swe-agent': 'GitHub Copilot',
               'devin-ai-integration[bot]': 'Devin', 'devin-ai-integration': 'Devin',
               'chatgpt-codex-connector[bot]': 'OpenAI Codex', 'chatgpt-codex-connector': 'OpenAI Codex', 'codex': 'OpenAI Codex',
               'cursoragent': 'Cursor', 'cursor[bot]': 'Cursor', 'cursor': 'Cursor',
               'claude': 'Claude Code', 'claude[bot]': 'Claude Code'}
# Co-authored-by trailers that name a coding agent: matched on the trailer's name and e-mail
TRAILER = re.compile(r'^\s*co-authored-by:\s*(?P<name>[^<\n]*?)\s*<(?P<email>[^>\n]*)>', re.I | re.M)
TRAILER_AGENT = [  # (agent, regex on "name <email>", lower case)
    # Copilot coding agent / Copilot accounts; not "Copilot Autofix" (github-advanced-security[bot], code-scanning fixes)
    ('GitHub Copilot', re.compile(r'^(?!.*(autofix|advanced-security)).*copilot')),
    ('Cursor', re.compile(r'cursor(agent| agent)|@cursor\.(com|sh)|^cursor\b')),
    ('Claude Code', re.compile(r'^claude\b|anthropic\.com|claude\[bot\]')),
    ('OpenAI Codex', re.compile(r'codex')),
    ('Devin', re.compile(r'devin-ai-integration|devin\.ai|^devin ai\b')),
]


def agent_of_login(login):
    return LOGIN_AGENT.get(str(login).strip().lower()) if isinstance(login, str) else None


def trailer_agents(msg):
    found = []
    for m in TRAILER.finditer(msg if isinstance(msg, str) else ''):
        s = f"{m.group('name').strip().lower()} <{m.group('email').strip().lower()}>"
        for ag, rx in TRAILER_AGENT:
            if rx.search(s):
                found.append((ag, s))
    return found


# ── data ────────────────────────────────────────────────────────────────

def load_lines():
    cols = ['slug', 'pr_number', 'is_agent', 'agent_tier', 'agent_name', 'unit_id', 'death_sha', 'event',
            'duration_days', 'birth_date', 'death_date']
    lines = pd.read_csv('data/survival_events_line_v6confnv_analysis.csv', usecols=cols, low_memory=False)
    lines = apply_observation_window(lines, label='line').dropna(subset=['death_sha'])
    tier_map = get_birth_tier_map(lines)
    lines = lines[~pd.Series(list(zip(lines['slug'], lines['pr_number'])), index=lines.index).isin(OUTLIER_PR_KEYS)]
    lines = lines[~lines['unit_id'].apply(is_generated)]
    lines = add_two_tier_column(lines).rename(columns={'death_sha': 'sha'})
    assert (lines['event'] == 1).all(), 'every line with a death_sha should be modified'
    return lines, tier_map


def load_commits(lines, tier_map):
    mods = pd.read_csv('data/death_commit_messages_v6confnv.csv')[['slug', 'sha', 'message']].drop_duplicates(['slug', 'sha'])
    auth = pd.read_csv('results_v6/modifier_authors.csv')[['slug', 'sha', 'author_login', 'modifier_type']].drop_duplicates(['slug', 'sha'])
    c = mods.merge(auth, on=['slug', 'sha'], how='left').merge(tier_map, on=['slug', 'sha'], how='inner')
    g = lines.groupby(['slug', 'sha'])
    per = g.agg(agents=('agent_name', lambda s: set(s.dropna()) - {'Human'}), groups=('two_tier', lambda s: set(s)),
                n_lines=('unit_id', 'size'), lag_all=('duration_days', 'min')).reset_index()
    lag_g = lines.groupby(['slug', 'sha', 'two_tier'])['duration_days'].min().unstack()
    lag_g.columns = [f'lag_{x}' for x in lag_g.columns]
    c = c.merge(per, on=['slug', 'sha'], how='left').merge(lag_g.reset_index(), on=['slug', 'sha'], how='left')
    c['lag'] = [getattr(r, f'lag_{r.birth_tier}') for r in c.itertuples()]
    c['author_agent'] = c['author_login'].map(agent_of_login)
    c['is_agent'] = c['modifier_type'] == 'agent'
    c['self'] = [bool(a) and a in ags for a, ags in zip(c['author_agent'], c['agents'])]
    return c


def shares(df, agent_col='is_agent', self_col='self'):
    n = len(df)
    a = int(df[agent_col].sum())
    s = int(df[self_col].sum())
    return dict(n=n, agent=a, agent_pct=100 * a / n if n else np.nan, self=s, self_pct=100 * s / n if n else np.nan)


def fmt(r, with_self=True):
    if not r['n']:
        return 'n=0'
    t = f"{r['agent_pct']:5.1f}% ({r['agent']}/{r['n']})"
    if with_self:
        t += f" self {r['self_pct']:4.1f}% ({r['self']})"
    return t


# ── E6 ──────────────────────────────────────────────────────────────────

def followup_map(c, lines):
    pulls = json.load(open(os.path.join(OUT, 'commit_pulls_cache.json')))
    orig = lines.groupby(['slug', 'sha'])['pr_number'].agg(lambda s: set(int(x) for x in s)).to_dict()
    rows = []
    for (s, h) in zip(c['slug'], c['sha']):
        k = f'{s}@{h}'
        if k not in pulls:
            rows.append(dict(slug=s, sha=h, fetched=False, fpr=None, fpr_user=None, n_merged_prs=0))
            continue
        prs = [p for p in (pulls[k] or []) if p.get('merged_at') and p['number'] not in orig.get((s, h), set())]
        prs.sort(key=lambda p: p.get('created_at') or '')
        rows.append(dict(slug=s, sha=h, fetched=True, fpr=prs[0]['number'] if prs else None,
                         fpr_user=prs[0]['user'] if prs else None, n_merged_prs=len(prs)))
    return pd.DataFrame(rows)


def e6(lines, c):
    out = ['E6. Share of modifications made by agents, and (self) by the agent that wrote the code, by unit and window',
           'Groups: commit unit = majority birth group (Table modifier_type); line and follow-up PR units = group of each line.',
           'Window = days from the merge of the original PR to the modifying commit.', '']
    rows = []
    fm = followup_map(c, lines)
    cf = c.merge(fm, on=['slug', 'sha'], how='left')
    lf = lines.drop(columns=['is_agent']).merge(
        cf[['slug', 'sha', 'author_login', 'author_agent', 'is_agent', 'fetched', 'fpr', 'fpr_user']], on=['slug', 'sha'], how='inner')
    lf['self'] = lf['author_agent'].notna() & (lf['author_agent'] == lf['agent_name'])
    n_fetched = int(fm['fetched'].sum())
    out.append(f'commit-to-PR map: {n_fetched:,} of {len(fm):,} commits fetched; '
               f'{int(fm["fpr"].notna().sum()):,} in a merged follow-up PR, {int((fm["fetched"] & fm["fpr"].isna()).sum()):,} in none '
               f'(direct pushes or unmerged); {int((fm["n_merged_prs"] > 1).sum())} in more than one merged PR (earliest taken)')
    out.append('')
    for unit in ['Commit', 'Line', 'Follow-up PR', 'Follow-up PR (pairs)']:
        for wl, w in WINDOWS:
            line = f'{unit:21s} {wl:8s}'
            for g in GROUPS:
                if unit == 'Commit':
                    d = c[(c['birth_tier'] == g) & (c['lag'] <= w)]
                    r = shares(d)
                else:
                    ll = lf[(lf['two_tier'] == g) & (lf['duration_days'] <= w)]
                    if unit == 'Line':
                        r = shares(ll)
                    else:
                        ll = ll[ll['fpr'].notna()]
                        keys = ['slug', 'fpr'] if unit == 'Follow-up PR' else ['slug', 'pr_number', 'fpr']
                        p = ll.groupby(keys).agg(user=('fpr_user', 'first'),
                                                 agents=('agent_name', lambda s: set(s) - {'Human'})).reset_index()
                        p['is_agent'] = p['user'].apply(is_agent_login)
                        p['self'] = [agent_of_login(u) in a for u, a in zip(p['user'], p['agents'])]
                        r = shares(p)
                r.update(unit=unit, window=wl, group=g)
                rows.append(r)
                line += f' | {GLAB[g]:16s} {fmt(r)}'
            out.append(line)
        out.append('')
    # agents who make the follow-up PRs
    ll = lf[lf['fpr'].notna()]
    p = ll.groupby(['two_tier', 'slug', 'fpr'])['fpr_user'].first().reset_index()
    p = p[p['fpr_user'].apply(is_agent_login)]
    out.append('Agent authors of follow-up PRs (full window), per group: ' +
               '; '.join(f'{GLAB[g]}: ' + ', '.join(f'{u} {n}' for u, n in p[p['two_tier'] == g]['fpr_user'].value_counts().items())
                         for g in GROUPS))
    out.append('Note: the commit rows reproduce Table modifier_type in the full window (1,453 / 617 / 3,230 commits).')
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, 'e6_modifier_unit_window.csv'), index=False)
    write('e6_modifier_unit_window.txt', out)
    return res


# ── E7 ──────────────────────────────────────────────────────────────────

def e7(c):
    meta = json.load(open(os.path.join(OUT, 'commit_meta_cache.json')))
    c = c.copy()
    c['meta'] = [meta.get(f'{s}@{h}') for s, h in zip(c['slug'], c['sha'])]
    c['committer_login'] = [m.get('committer_login') if m else None for m in c['meta']]
    c['api_author'] = [m.get('author_login') if m else None for m in c['meta']]
    c['api_message'] = [m.get('message') if m else None for m in c['meta']]
    out = ['E7. Identification rule (commit unit, full window, majority birth group)', '']
    fetched = c['meta'].notna()
    out.append(f'API metadata for {int(fetched.sum()):,} of {len(c):,} commits; author login agrees with modifier_authors.csv '
               f'for {int((c.loc[fetched, "api_author"].fillna("") == c.loc[fetched, "author_login"].fillna("")).sum()):,}; '
               f'API message equals the cached message for {int((c.loc[fetched, "api_message"].fillna("").str.strip() == c.loc[fetched, "message"].fillna("").str.strip()).sum()):,}')
    # (a) committer
    c['cm_agent'] = c['committer_login'].apply(is_agent_login)
    c['cm_agent_name'] = c['committer_login'].map(agent_of_login)
    c['cm_self'] = [bool(a) and a in ags for a, ags in zip(c['cm_agent_name'], c['agents'])]
    vc = c.loc[fetched, 'committer_login'].fillna('(none)').value_counts()
    out.append('Most frequent committer logins: ' + ', '.join(f'{k} {v}' for k, v in vc.head(8).items()))
    # (b) author or co-author trailer
    msg = c['message'].where(c['message'].notna(), c['api_message'])
    tr = msg.apply(trailer_agents)
    c['tr_agents'] = tr.apply(lambda x: sorted(set(a for a, _ in x)))
    c['co_agent'] = c['is_agent'] | c['tr_agents'].apply(bool)
    c['co_self'] = c['self'] | np.array([any(a in ags for a in tas) for tas, ags in zip(c['tr_agents'], c['agents'])])
    allt = pd.Series([s for x in tr for _, s in x])
    out.append(f'Commits with a Co-authored-by trailer naming a coding agent: {int(c["tr_agents"].apply(bool).sum())} '
               f'(any Co-authored-by trailer: {int(msg.fillna("").str.contains("co-authored-by", case=False).sum())})')
    out.append('Matched trailer identities (name <email>, count): ' + '; '.join(f'{k} {v}' for k, v in allt.value_counts().items()))
    out.append('')
    rows = []
    for rule, ac, sc in [('author (default)', 'is_agent', 'self'), ('committer', 'cm_agent', 'cm_self'),
                         ('author or co-author trailer', 'co_agent', 'co_self')]:
        line = f'{rule:28s}'
        base = c if rule != 'committer' else c[fetched]
        for g in GROUPS + ['both tiers']:
            d = base[base['birth_tier'].isin(['human_supervised', 'fully_autonomous'])] if g == 'both tiers' else base[base['birth_tier'] == g]
            r = shares(d, ac, sc)
            r.update(rule=rule, group=g, human_pct=100 - r['agent_pct'])
            rows.append(r)
            line += f' | {GLAB.get(g, g):16s} agent {fmt(r)}'
        out.append(line)
    out.append('')
    both = c[c['birth_tier'] != 'human']
    out.append('Agents named in trailers on agent-code commits: ' +
               ', '.join(f'{k} {v}' for k, v in pd.Series([a for x in both['tr_agents'] for a in x]).value_counts().items()))
    out.append('Not counted as a coding agent: "Copilot Autofix powered by AI" (github-advanced-security[bot]) trailers, '
               'which come from code-scanning autofix rather than the Copilot coding agent: '
               f'{int(msg.fillna("").str.contains("advanced-security", case=False).sum())} commits.')
    out.append('API author login missing for some commits (deleted or unlinked accounts); where the API returns a login it never '
               'changes the agent/human class of modifier_authors.csv.')
    out.append('Note: the committer of a PR merged through the GitHub web interface is web-flow, which the rule classifies as '
               'human (not an agent login).')
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, 'e7_identification_rule.csv'), index=False)
    write('e7_identification_rule.txt', out)
    c[['slug', 'sha', 'birth_tier', 'author_login', 'committer_login', 'tr_agents']].to_csv(
        os.path.join(OUT, 'work', 'e7_commit_rules.csv'), index=False)
    return res


# ── E8 ──────────────────────────────────────────────────────────────────

def e8(c):
    out = ['E8. Commits that modify lines of more than one group (author rule, commit unit, full window)', '']
    mixed = c['groups'].apply(len) > 1
    out.append(f'mixed commits: {int(mixed.sum())} of {len(c):,} tier-mapped commits; '
               + ', '.join(f'{"+".join(sorted(GLAB[x] for x in k))} {v}' for k, v in c.loc[mixed, 'groups'].apply(frozenset).value_counts().items()))
    out.append('')
    rows = []
    for g in GROUPS:
        base = shares(c[c['birth_tier'] == g])
        every = shares(c[c['groups'].apply(lambda s: g in s)])
        drop = shares(c[(c['birth_tier'] == g) & ~mixed])
        for name, r in [('majority (default)', base), ('every group', every), ('dropped', drop)]:
            rows.append(dict(group=g, variant=name, **r, human_pct=100 - r['agent_pct'],
                             d_agent_pp=r['agent_pct'] - base['agent_pct'], d_self_pp=r['self_pct'] - base['self_pct']))
        out.append(f'{GLAB[g]:16s} majority {fmt(base)} | every group {fmt(every)} | dropped {fmt(drop)}')
    res = pd.DataFrame(rows)
    out.append('')
    out.append(f'max |change| in the agent (= human) share: {res["d_agent_pp"].abs().max():.2f} pp; '
               f'in the self-modification share: {res["d_self_pp"].abs().max():.2f} pp')
    for g in ['both tiers']:
        d0 = c[c['birth_tier'] != 'human']
        d1 = c[c['groups'].apply(lambda s: bool(s & {'human_supervised', 'fully_autonomous'}))]
        d2 = d0[~mixed[d0.index]]
        out.append(f'human share of modifications to agent code (both tiers): majority {100 - shares(d0)["agent_pct"]:.1f}% '
                   f'({len(d0) - shares(d0)["agent"]}/{len(d0)}), every group {100 - shares(d1)["agent_pct"]:.1f}% '
                   f'({len(d1) - shares(d1)["agent"]}/{len(d1)}), dropped {100 - shares(d2)["agent_pct"]:.1f}% ({len(d2) - shares(d2)["agent"]}/{len(d2)})')
    res.to_csv(os.path.join(OUT, 'e8_mixed_commits.csv'), index=False)
    write('e8_mixed_commits.txt', out)
    return res


# ── E9 ──────────────────────────────────────────────────────────────────

def intent_corpus():
    """Per-commit intent labels of every approach, built exactly as analyze_intent_aggregation.run_full
    with the 618 human labels (results_v6/labels618), plus the human label where one exists."""
    import analyze_intent_aggregation as A
    L = 'results_v6/labels618'
    A.LLM_RUNS.update({m: os.path.join(L, f'intent_llm_{m}_t0.csv') for m in A.NEW_LLMS})
    A.FULL_RUNS.update({m: f'results_v6/intent_{m}_full_corpus_t0.csv' for m in A.NEW_LLMS})
    smp = A.load(os.path.join(L, 'gold.csv'), os.path.join(L, 'sample.csv'), os.path.join(L, 'keyword.csv'))
    k = {m: max(A.kappa(smp['gold'], smp[m]), 0.0) for m in A.NEW_LLMS}
    order = sorted(A.NEW_LLMS, key=lambda m: -k[m])
    kw = pd.read_csv('data/death_commit_messages_v6confnv.csv').drop_duplicates(['slug', 'sha'])
    corpus = kw[['slug', 'sha', 'intent']].rename(columns={'intent': 'keyword'})
    for m, path in A.FULL_RUNS.items():
        run = pd.read_csv(path).drop_duplicates(['slug', 'sha'], keep='last')
        corpus = corpus.merge(run[['slug', 'sha', 'label']].rename(columns={'label': m}), on=['slug', 'sha'], how='left')
    for col in ['keyword'] + A.NEW_LLMS:
        corpus[col] = corpus[col].fillna('missing').astype(str).str.strip().str.lower()
        corpus.loc[~corpus[col].isin(A.INTENT_LABELS + ['unknown']), col] = 'missing'
    corpus['mv3_undecided'] = A.vote(corpus[A.NEW_LLMS], {m: 1.0 for m in A.NEW_LLMS}, [])
    corpus['mv3_tiebest'] = A.vote(corpus[A.NEW_LLMS], {m: 1.0 for m in A.NEW_LLMS}, order)
    corpus['wv3'] = A.vote(corpus[A.NEW_LLMS], k, order)
    corpus['ds3'] = A.dawid_skene(corpus, A.NEW_LLMS)
    gold = smp[['slug', 'sha', 'gold']].drop_duplicates(['slug', 'sha'])
    corpus = corpus.merge(gold, on=['slug', 'sha'], how='left')
    return corpus, A.INTENT_LABELS


APPROACHES = [('keyword', 'Keyword'), ('deepseek', 'DeepSeek'), ('gpt_oss', 'gpt-oss'), ('minimax', 'MiniMax'),
              ('mv3_undecided', 'Majority vote'), ('mv3_tiebest', 'Majority vote, ties to best'), ('wv3', 'Weighted vote'),
              ('ds3', 'Dawid-Skene'), ('gold', 'Human labels (618, unweighted)')]


def intent_test(df, col, by, labels, a='agent', b='human'):
    sub = df[df[col].isin(labels + ['unknown'])]
    t = pd.crosstab(sub[by], sub[col]).reindex(index=[b, a], fill_value=0)
    t = t.loc[:, t.sum() > 0]
    sh = {x: sub.loc[sub[by] == x, col].value_counts(normalize=True).reindex(labels, fill_value=0) * 100 for x in [b, a]}
    n = {x: int((sub[by] == x).sum()) for x in [b, a]}
    if min(n.values()) == 0 or t.shape[1] < 2:
        return sh, n, np.nan, np.nan
    chi2, p, _, _ = chi2_contingency(t)
    v = np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))
    return sh, n, p, v


def selfmod_sample(c, lines):
    """20 of the 143 Copilot self-modifications (seed 42, sorted by (slug, sha) as for the Devin sample)."""
    g = c[(c['birth_tier'] == 'human_supervised') & (c['author_login'] == 'Copilot') & c['self']]
    g = g.sort_values(['slug', 'sha']).reset_index(drop=True)
    s = g.sample(20, random_state=42).sort_values(['slug', 'sha'])
    pr = pd.read_parquet('data/pull_request.parquet', columns=['number', 'title', 'body', 'repo_url', 'html_url'])
    pr['slug'] = pr['html_url'].str.extract(r'github\.com/([^/]+/[^/]+)/pull/')[0].str.lower()
    cop = lines[lines['agent_name'] == 'GitHub Copilot']
    rows = []
    for r in s.itertuples():
        ll = cop[(cop['slug'] == r.slug) & (cop['sha'] == r.sha)]
        top = ll['pr_number'].value_counts()
        n = int(top.index[0])
        meta = pr[(pr['slug'] == r.slug.lower()) & (pr['number'] == n)]
        title = meta['title'].iloc[0] if len(meta) else ''
        body = meta['body'].iloc[0] if len(meta) else ''
        rows.append({'repo': r.slug, 'sha': r.sha, 'commit message': r.message,
                     'modifying-commit URL': f'https://github.com/{r.slug}/commit/{r.sha}',
                     'original PR URL': f'https://github.com/{r.slug}/pull/{n}' + (
                         f' (also {", ".join("#" + str(int(x)) for x in top.index[1:])})' if len(top) > 1 else ''),
                     'original PR title and description': f'{title}\n\n{body or ""}'.strip()[:32000],
                     'lag (days)': round(float(ll['duration_days'].min()), 2),
                     'label': '', 'notes': ''})
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, 'copilot_selfmod_sample.xlsx')
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        df.to_excel(xw, index=False, sheet_name='sample')
        pd.DataFrame({'label': ['same task', 'next item of a series', 'later maintenance'],
                      'meaning': ['fixes or extends the feature / fix that the original PR added',
                                  'same requester, same kind of change, a different item (e.g., one of a series of similar PRs)',
                                  'a different task on the same code (later maintenance)']}).to_excel(xw, index=False, sheet_name='labels')
        ws = xw.sheets['sample']
        from openpyxl.styles import Alignment
        from openpyxl.worksheet.datavalidation import DataValidation
        widths = {'A': 28, 'B': 14, 'C': 60, 'D': 40, 'E': 40, 'F': 80, 'G': 10, 'H': 22, 'I': 40}
        for col, wd in widths.items():
            ws.column_dimensions[col].width = wd
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical='top')
        dv = DataValidation(type='list', formula1='"same task,next item of a series,later maintenance"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f'H2:H{len(df) + 1}')
    return df


def e9(c, lines):
    out = ['E9. Self-modification', '']
    rows = []
    for tier, login, agent in [('human_supervised', 'Copilot', 'GitHub Copilot'), ('fully_autonomous', 'devin-ai-integration[bot]', 'Devin')]:
        g = c[(c['birth_tier'] == tier) & (c['author_login'] == login)]
        sm = g[g['self']]
        own = lines[lines['agent_name'] == agent].groupby(['slug', 'sha'])['duration_days'].min().rename('lag_own')
        sm = sm.merge(own, left_on=['slug', 'sha'], right_index=True, how='left')
        r = dict(tier=tier, agent=agent, commits=len(g), self=len(sm), self_share_of_tier=len(sm) / (c['birth_tier'] == tier).sum(),
                 median_lag_own=sm['lag_own'].median(), median_lag_all=sm['lag_all'].median(), median_lag_all_151=g['lag_all'].median(),
                 within1=int((sm['lag_own'] <= 1).sum()), within7=int((sm['lag_own'] <= 7).sum()), within30=int((sm['lag_own'] <= 30).sum()),
                 q1=sm['lag_own'].quantile(.25), q3=sm['lag_own'].quantile(.75))
        rows.append(r)
        out.append(f'{agent} on {GLAB[tier]} code: {len(g)} commits by {login}, {len(sm)} self-modifications '
                   f'({len(sm)}/{(c["birth_tier"] == tier).sum()} = {100 * r["self_share_of_tier"]:.1f}% of the group\'s commits); '
                   f'median lag since the merge of the agent\'s own PR {r["median_lag_own"]:.2f} days (IQR {r["q1"]:.1f} to {r["q3"]:.1f}); '
                   f'within 1 / 7 / 30 days: {r["within1"]} / {r["within7"]} / {r["within30"]}; '
                   f'(shortest lag over all modified lines: median {r["median_lag_all"]:.2f}; all {len(g)} commits by {login}: {r["median_lag_all_151"]:.2f})')
    out += ['', 'Intent of agent-made vs human-made modifications, by tier of the modified code (commit unit, majority birth group).',
            'Shares corrective / perfective / adaptive / preventive (%); chi-square agent vs human modifier (Fisher exact on corrective vs not in brackets).', '']
    corpus, labels = intent_corpus()
    d = c[['slug', 'sha', 'birth_tier', 'modifier_type']].merge(corpus, on=['slug', 'sha'], how='left')
    trows = []
    for tier in ['human_supervised', 'fully_autonomous', 'both tiers', 'all groups']:
        x = d[d['birth_tier'] != 'human'] if tier == 'both tiers' else d if tier == 'all groups' else d[d['birth_tier'] == tier]
        out.append(f'  {GLAB.get(tier, tier)}:')
        for col, name in APPROACHES:
            sh, n, p, v = intent_test(x, col, 'modifier_type', labels)
            sub = x[x[col].isin(labels + ['unknown'])]
            ft = pd.crosstab(sub['modifier_type'] == 'agent', sub[col] == 'corrective').reindex(index=[False, True], columns=[False, True], fill_value=0)
            fp = fisher_exact(ft.values)[1] if ft.values.sum() and ft.sum(1).min() > 0 else np.nan
            out.append(f'    {name:30s} human-made (n={n["human"]:4d}) ' + '/'.join(f'{sh["human"][l]:.1f}' for l in labels) +
                       f'   agent-made (n={n["agent"]:3d}) ' + '/'.join(f'{sh["agent"][l]:.1f}' for l in labels) +
                       f'   p={p:.3g} V={v:.3f} [corrective Fisher p={fp:.3g}]')
            tr = dict(tier=tier, approach=col, n_human=n['human'], n_agent=n['agent'], p=p, V=v, p_fisher_corrective=fp)
            for l in labels:
                tr[f'human_{l}'] = sh['human'][l]
                tr[f'agent_{l}'] = sh['agent'][l]
            trows.append(tr)
        out.append('')
    out.append('The "all groups" rows reproduce results_v6/intent_aggregation_full_corpus_618.txt Section B '
               '(human labels: unweighted counterpart of rq4_intent_by_modifier_618.txt).')
    sample = selfmod_sample(c, lines)
    out += ['', f'Annotation sheet: results_v6/round2/copilot_selfmod_sample.xlsx ({len(sample)} of the 143 Copilot self-modifications, '
                'sorted by (slug, sha), DataFrame.sample(20, random_state=42), as the Devin sample); '
                'columns repo, sha, commit message, modifying-commit URL, original PR URL, original PR title and description, '
                'lag, label (blank; drop-down: same task / next item of a series / later maintenance), notes.']
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'e9_self_modification.csv'), index=False)
    pd.DataFrame(trows).to_csv(os.path.join(OUT, 'e9_intent_by_modifier_tier.csv'), index=False)
    write('e9_self_modification.txt', out)


def write(name, out):
    txt = '\n'.join(out) + '\n'
    open(os.path.join(OUT, name), 'w').write(txt)
    print(txt, flush=True)


def main():
    todo = [a for a in sys.argv[1:] if a.startswith('e')] or ['e6', 'e7', 'e8', 'e9']
    lines, tier_map = load_lines()
    c = load_commits(lines, tier_map)
    print('commits by group:', c['birth_tier'].value_counts().to_dict(), '; agent:',
          c.groupby('birth_tier')['is_agent'].sum().to_dict(), '; self:', c.groupby('birth_tier')['self'].sum().to_dict(), flush=True)
    if 'e6' in todo:
        e6(lines, c)
    if 'e7' in todo:
        e7(c)
    if 'e8' in todo:
        e8(c)
    if 'e9' in todo:
        e9(c, lines)


if __name__ == '__main__':
    main()
