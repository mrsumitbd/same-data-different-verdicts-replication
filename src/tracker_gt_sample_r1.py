"""Draw the tracker ground-truth sample.

Draws about 300 lines from the 276,360 verified units that all three trackers
cover (identity tracking, the tracker of the earlier study, line-number
tracking), stratified by the pattern of the three trackers' claims
(modified / retained within the observation window). Each tracker x claim
stratum then holds at least 50 lines. Writes, under results_v6/round1/tracker_gt/:

  tracker_groundtruth_sheet.xlsx      labeling sheet (no tracker claims, no tier)
  evidence/Lxxx.html                  per-line history evidence
  tracker_groundtruth_key.csv tracker claims, strata, weights (do not open
                                      before labeling)
  sampling_frame.csv                  cell sizes, eligibility rates, sample sizes

Run from the repo root:  python3 src/tracker_gt_sample_r1.py
"""
import html
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.constants import OBSERVATION_CUTOFF  # noqa: E402
from config.observation_window import apply_observation_window  # noqa: E402

SEED = 42
TOTAL = 300
MAX_COMMITS = 12          # first-parent commits touching the file in the window
MAX_PATCH_LINES = 300     # per commit, for the file
MIN_CHARS = 6             # non-whitespace characters in the line
MAX_PER_PR = 2
MAX_PER_REPO = 12
REPOS = ROOT / 'data' / 'temp_repos'
OUT = ROOT / 'results_v6' / 'round1' / 'tracker_gt'
EVID = OUT / 'evidence'
CUT = pd.Timestamp(OBSERVATION_CUTOFF)


def git(repo, *args, timeout=60):
    r = subprocess.run(['git', '-C', str(repo), *args], capture_output=True,
                       text=True, errors='replace', timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:300])
    return r.stdout


def repo_dir(slug):
    return REPOS / (slug.replace('/', '__') + '.git')


def load():
    k = ['slug', 'pr_number', 'unit_id']
    ident = pd.read_csv(ROOT / 'data' / 'survival_events_line_pos_v6confnv_idtrack.csv')
    pos = pd.read_csv(ROOT / 'data' / 'survival_events_line_pos_v6confnv.csv',
                      usecols=k + ['event', 'death_sha', 'death_date'])
    pos = pos.rename(columns={'event': 'pos_event', 'death_sha': 'pos_death_sha',
                              'death_date': 'pos_death_date'})
    ident = ident[ident.event_type != 'unverified']
    win = apply_observation_window(ident, label='identity (tracker GT)')
    df = win.merge(pos, on=k, how='inner')
    dd = lambda c: pd.to_datetime(df[c], errors='coerce', utc=True, format='mixed').dt.tz_localize(None)
    df['I'] = df['event'].astype(int)
    df['E'] = ((df['conf_event'] == 1) & (dd('conf_death_date') <= CUT)).astype(int)
    df['L'] = ((df['pos_event'] == 1) & (dd('pos_death_date') <= CUT)).astype(int)
    df['cell'] = df['I'].astype(str) + df['E'].astype(str) + df['L'].astype(str)
    return df


_LOG = {}


def _fp_log(repo, birth, end):
    key = (str(repo), birth, end)
    if key not in _LOG:
        out = git(repo, 'log', '--first-parent', '--reverse', '-M', '--name-status',
                  '--format=%x01%H%x02%cI%x02%s', f'{birth}..{end}', timeout=300)
        log = []
        for chunk in out.split('\x01')[1:]:
            head, _, body = chunk.partition('\n')
            sha, date, subj = head.split('\x02', 2)
            ents = [ln.split('\t') for ln in body.splitlines() if ln.strip()]
            log.append((sha, date, subj, ents))
        _LOG[key] = log
    return _LOG[key]


def history(repo, birth, end, path):
    """First-parent commits after birth up to end that touch the file, following renames forward."""
    commits, cur = [], path
    for sha, date, subj, ents in _fp_log(repo, birth, end):
        hit = None
        for e in ents:
            st = e[0]
            if st.startswith('R') and len(e) == 3 and e[1] == cur:
                hit = (e[1], e[2]); break
            if st[0] in 'MTD' and len(e) == 2 and e[1] == cur:
                hit = (e[1], None if st[0] == 'D' else e[1]); break
        if not hit:
            continue
        old, new = hit
        paths = [old] + ([new] if new and new != old else [])
        patch = git(repo, 'diff', '-M', '--no-color', f'{sha}^1', sha, '--', *paths)
        commits.append(dict(sha=sha, date=date, subject=subj, patch=patch.strip('\n'), path_after=new))
        if new is None:
            break
        cur = new
    return commits


def eligible(row, commits):
    if len(commits) > MAX_COMMITS:
        return False
    return all(c['patch'].count('\n') + 1 <= MAX_PATCH_LINES for c in commits)


def same_text_elsewhere(repo, sha, path, text):
    try:
        out = git(repo, 'diff', '-U0', '--no-color', '-M', f'{sha}^1', sha, timeout=30)
    except Exception:
        return []
    hits, cur = set(), None
    t = text.strip()
    for ln in out.splitlines():
        if ln.startswith('+++ '):
            cur = ln[6:] if ln.startswith('+++ b/') else None
        elif ln.startswith('+') and not ln.startswith('+++') and cur and cur != path and ln[1:].strip() == t:
            hits.add(cur)
    return sorted(hits)


def first_parent_of(repo, fp_list, sha):
    """Map a commit to the first first-parent commit that contains it."""
    if not isinstance(sha, str) or not sha:
        return ''
    if sha in fp_list:
        return sha
    lo, hi = 0, len(fp_list) - 1
    ok = lambda i: subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', sha, fp_list[i]]).returncode == 0
    if hi < 0 or not ok(hi):
        return 'outside_window'
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return fp_list[lo]


CSS = """body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}
pre{background:#f6f8fa;padding:8px;overflow-x:auto;font-size:12px;line-height:1.35}
.t{background:#fff3b0;font-weight:bold}.a{color:#116329}.d{color:#82071e}.h{color:#6639ba}
.m{background:#fff3b0}h2{border-top:1px solid #ccc;padding-top:12px;font-size:16px}
.k{color:#555;font-size:13px}"""


def render_patch(patch, text):
    t = text.strip()
    rows = []
    for ln in patch.splitlines():
        cls = 'a' if ln.startswith('+') else 'd' if ln.startswith('-') else 'h' if ln.startswith('@@') else ''
        mark = ' m' if ln[1:].strip() == t and t else ''
        e = html.escape(ln)
        rows.append(f'<span class="{cls}{mark}">{e}</span>' if (cls or mark) else e)
    return '\n'.join(rows)


def evidence(lid, row, repo, commits, birth_lines):
    slug, path, n, text = row.slug, row.path, int(row.line), row.birth_content
    gh = f'https://github.com/{slug}'
    lo, hi = max(1, n - 8), min(len(birth_lines), n + 8)
    ctx = '\n'.join((f'<span class="t">{i:5d} | {html.escape(birth_lines[i-1])}</span>' if i == n
                     else f'{i:5d} | {html.escape(birth_lines[i-1])}') for i in range(lo, hi + 1))
    parts = [f'<!doctype html><html><head><meta charset="utf-8"><title>{lid}</title><style>{CSS}</style></head><body>',
             f'<h1>{lid}</h1><p><b>Repository:</b> <a href="{gh}">{slug}</a><br><b>File at merge:</b> {html.escape(path)}, line {n}<br>',
             f'<b>Merge commit (birth):</b> <a href="{gh}/commit/{row.birth_sha}">{row.birth_sha[:10]}</a> ({row.birth_date})<br>',
             f'<b>Window ends:</b> {CUT.date()} (end snapshot <a href="{gh}/tree/{row.end_sha}">{row.end_sha[:10]}</a>)</p>',
             f'<p><b>The line:</b></p><pre><span class="t">{html.escape(text)}</span></pre>',
             f'<h2>File at the merge commit, lines {lo} to {hi}</h2><pre>{ctx}</pre>',
             f'<p class="k">Below are all {len(commits)} first-parent commits of the default branch, in time order, '
             'that touch this file (following renames) after the merge commit and up to the end of the window. '
             'Each shows the diff of this file against the commit\'s first parent. Lines whose text equals the '
             'line above are shaded. Identical text is not proof that it is the same line.</p>']
    cur = path
    for i, c in enumerate(commits, 1):
        nxt = c['path_after']
        try:
            cnt = git(repo, 'show', f'{c["sha"]}:{nxt}').splitlines().count(text) if nxt else 0
        except Exception:
            cnt = 0
        moved = same_text_elsewhere(repo, c['sha'], nxt or cur, text)
        parts.append(f'<h2>Commit #{i}: <a href="{gh}/commit/{c["sha"]}">{c["sha"][:10]}</a> ({c["date"]})</h2>'
                     f'<p class="k">{html.escape(c["subject"])}<br>Path after this commit: {html.escape(str(nxt)) if nxt else "file deleted"}. '
                     f'Lines with exactly this text in the file after the commit: {cnt}.'
                     + (f' Identical text added in other files in this commit: {html.escape(", ".join(moved[:5]))}.' if moved else '')
                     + f'</p><pre>{render_patch(c["patch"], text)}</pre>')
        cur = nxt or cur
    if not commits:
        parts.append('<p><b>No commit touches this file in the window.</b></p>')
    parts.append('</body></html>')
    (EVID / f'{lid}.html').write_text(''.join(parts), encoding='utf-8')


BUDGET_S = float(os.environ.get('TGT_BUDGET_S', '1e9'))  # stop and checkpoint after this many seconds
CKPT = OUT / '_checkpoint.pkl'


def main():
    import pickle
    import time
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    EVID.mkdir(exist_ok=True)
    df = load()
    content = df['birth_content'].fillna('')
    df = df[(df['blank'] == 0) & (content.str.replace(r'\s', '', regex=True).str.len() >= MIN_CHARS)]
    cells = sorted(df['cell'].unique())
    pop = df.groupby('cell').size()
    print('population after content filter:', pop.to_dict(), flush=True)
    quota = {c: TOTAL // len(cells) for c in cells}
    for c in cells[: TOTAL - sum(quota.values())]:
        quota[c] += 1
    rng = np.random.default_rng(SEED)
    order = {c: rng.permutation(df.index[df['cell'] == c].to_numpy()) for c in cells}
    pending = sorted(cells, key=lambda c: pop[c])  # smallest first; leftover quota moves to larger cells
    if CKPT.exists():
        st = pickle.loads(CKPT.read_bytes())
    else:
        st = dict(ci=0, pos=0, tried=0, acc=0, want=None, leftover=0, per_pr={}, per_repo={},
                  picked=[], stats={}, rows=None, done_rows=0)
    save = lambda: CKPT.write_bytes(pickle.dumps(st))

    # phase 1: draw
    while st['ci'] < len(pending):
        c = pending[st['ci']]
        if st['want'] is None:
            left = st['leftover']
            st['want'] = quota[c] + (left // (len(pending) - st['ci']) if left else 0)
            st['leftover'] -= st['want'] - quota[c]
        ids = order[c]
        while st['acc'] < st['want'] and st['pos'] < len(ids):
            if time.time() - t0 > BUDGET_S:
                save(); print('checkpoint (draw)', c, st['acc'], flush=True); return
            idx = ids[st['pos']]; st['pos'] += 1
            r = df.loc[idx]
            if st['per_pr'].get((r.slug, r.pr_number), 0) >= MAX_PER_PR or st['per_repo'].get(r.slug, 0) >= MAX_PER_REPO:
                continue
            repo = repo_dir(r.slug)
            st['tried'] += 1
            try:
                blines = git(repo, 'show', f'{r.birth_sha}:{r.path}').splitlines()
                if blines[int(r.line) - 1] != r.birth_content:
                    continue
                commits = history(repo, r.birth_sha, r.end_sha, r.path)
            except Exception as e:
                print('skip', r.slug, r.path, str(e)[:80], flush=True)
                continue
            if not eligible(r, commits):
                continue
            st['acc'] += 1
            st['per_pr'][(r.slug, r.pr_number)] = st['per_pr'].get((r.slug, r.pr_number), 0) + 1
            st['per_repo'][r.slug] = st['per_repo'].get(r.slug, 0) + 1
            st['picked'].append((idx, commits, blines))
        st['stats'][c] = dict(cell=c, population=int(pop[c]), tried=st['tried'], eligible=st['acc'],
                              eligible_rate=st['acc'] / st['tried'] if st['tried'] else np.nan, sampled=st['acc'])
        st['leftover'] += st['want'] - st['acc']
        print(f"cell {c}: want {st['want']}, sampled {st['acc']} of {st['tried']} tried", flush=True)
        st.update(ci=st['ci'] + 1, pos=0, tried=0, acc=0, want=None)
        save()

    # phase 2: evidence, sheet and key
    if st['rows'] is None:
        random.Random(SEED).shuffle(st['picked'])  # labeling order does not reveal the stratum
        st['rows'] = []
        save()
    fp_cache = {}
    for i in range(st['done_rows'], len(st['picked'])):
        if time.time() - t0 > BUDGET_S:
            save(); print('checkpoint (evidence)', i, flush=True); return
        idx, commits, blines = st['picked'][i]
        lid = f'L{i + 1:03d}'
        r = df.loc[idx]
        repo = repo_dir(r.slug)
        evidence(lid, r, repo, commits, blines)
        fpk = (r.slug, r.birth_sha, r.end_sha)
        if fpk not in fp_cache:
            fp_cache[fpk] = git(repo, 'rev-list', '--first-parent', '--reverse', f'{r.birth_sha}..{r.end_sha}').split()
        fpl = fp_cache[fpk]
        num = {c['sha']: j for j, c in enumerate(commits, 1)}

        def claim(ev, sha):
            if not ev:
                return 'retained', '', ''
            fp = first_parent_of(repo, fpl, sha)
            return 'modified', fp, num.get(fp, 'not_listed')
        ci, cs, cn = claim(r.I, r.death_sha)
        ei, es, en = claim(r.E, r.conf_death_sha)
        li, ls, ln = claim(r.L, r.pos_death_sha)
        sheet = dict(line_id=lid, repository=r.slug, file_at_merge=r.path, line_number=int(r.line),
                     line_text=r.birth_content, n_commits_listed=len(commits),
                     evidence=f'evidence/{lid}.html',
                     merge_commit=f'https://github.com/{r.slug}/commit/{r.birth_sha}',
                     status='', first_change_commit_no='', kind='', moved_in_commit_no='',
                     moved_to_file='', confident='', notes='')
        key = dict(line_id=lid, slug=r.slug, pr_number=r.pr_number, unit_id=r.unit_id, cell=r.cell,
                   agent_tier=r.agent_tier, identity=ci, identity_sha=cs, identity_commit_no=cn,
                   identity_event_type=r.event_type, identity_whitespace_only=r.whitespace_only,
                   earlier=ei, earlier_sha_raw=r.conf_death_sha if r.E else '', earlier_sha=es,
                   earlier_commit_no=en, line_number_tracker=li, line_number_sha_raw=r.pos_death_sha if r.L else '',
                   line_number_sha=ls, line_number_commit_no=ln,
                   commit_shas=' '.join(c['sha'] for c in commits))
        st['rows'].append((sheet, key))
        st['done_rows'] = i + 1
        if i % 20 == 0:
            save()
    save()
    pd.DataFrame([k for _, k in st['rows']]).to_csv(OUT / 'tracker_groundtruth_key.csv', index=False)
    fr = pd.DataFrame(st['stats'].values()).sort_values('cell')
    fr['eligible_population_est'] = (fr['population'] * fr['eligible_rate']).round(0)
    fr.to_csv(OUT / 'sampling_frame.csv', index=False)
    write_sheet(pd.DataFrame([s for s, _ in st['rows']]))
    print('wrote', len(st['rows']), 'lines; DONE', flush=True)


def write_sheet(s):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = 'labels'
    cols = list(s.columns)
    ws.append(cols)
    for _, r in s.iterrows():
        ws.append([r[c] for c in cols])
    fill = PatternFill('solid', fgColor='FFF3B0')
    for j, c in enumerate(cols, 1):
        cell = ws.cell(row=1, column=j)
        cell.font = Font(bold=True)
        if c in ('status', 'first_change_commit_no', 'kind', 'moved_in_commit_no', 'moved_to_file', 'confident', 'notes'):
            cell.fill = fill
    ei, mi = cols.index('evidence') + 1, cols.index('merge_commit') + 1
    for i in range(2, len(s) + 2):
        for j, label in ((ei, None), (mi, 'merge commit')):
            cell = ws.cell(row=i, column=j)
            cell.hyperlink = cell.value
            cell.style = 'Hyperlink'
            if label:
                cell.value = label
    lists = {'status': 'retained,modified,moved to other file,unclear', 'confident': 'yes,no',
             'kind': 'content changed,whitespace only,line deleted,file deleted,moved and changed'}
    n = len(s) + 1
    for c, vals in lists.items():
        dv = DataValidation(type='list', formula1=f'"{vals}"', allow_blank=True)
        L = ws.cell(row=1, column=cols.index(c) + 1).column_letter
        dv.add(f'{L}2:{L}{n}')
        ws.add_data_validation(dv)
    widths = dict(line_id=7, repository=26, file_at_merge=40, line_number=8, line_text=60, n_commits_listed=9,
                  evidence=22, merge_commit=14, status=11, first_change_commit_no=10, kind=18,
                  moved_in_commit_no=10, moved_to_file=26, confident=9, notes=30)
    for j, c in enumerate(cols, 1):
        ws.column_dimensions[ws.cell(row=1, column=j).column_letter].width = widths.get(c, 12)
    ws.freeze_panes = 'B2'
    for row in ws.iter_rows(min_row=2, min_col=cols.index('line_text') + 1, max_col=cols.index('line_text') + 1):
        for cell in row:
            cell.alignment = Alignment(wrap_text=False)
    wb.save(OUT / 'tracker_groundtruth_sheet.xlsx')


if __name__ == '__main__':
    main()
