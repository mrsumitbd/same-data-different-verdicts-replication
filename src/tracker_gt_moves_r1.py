"""Follow-up evidence for lines moved to another file.

The labeled tracker ground-truth sheet marks some lines as "moved to other
file" and names the destination file and the move commit. Under the main
convention (a move is not a modification), whether such a line was modified
in the window depends on what happens to it in the destination file after
the move. This script follows each destination file from the move commit to
the end of the window and writes:

  results_v6/round1/tracker_gt/evidence_moves/Lxxx_moved.html
  results_v6/round1/tracker_gt/tracker_groundtruth_moves_sheet.xlsx

  results_v6/round1/tracker_gt/tracker_groundtruth_moves_commits.csv

The sheet has the same status/kind columns as the main sheet; commit numbers
refer to the follow-up page, whose commit shas are in the CSV. Lines with no
commit after the move are prefilled as retained. An existing sheet is never
overwritten. Tracker claims are not shown.

Run from the repo root:  python3 src/tracker_gt_moves_r1.py
"""
import html
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import tracker_gt_sample_r1 as T  # noqa: E402

OUT = T.OUT
EV = OUT / 'evidence_moves'
SHEET = OUT / 'tracker_groundtruth_sheet_labelled.xlsx'
MAX_SHOWN = 1500   # patch lines shown per commit; longer patches are cut with a link


def end_shas(ids):
    cols = ['slug', 'pr_number', 'unit_id', 'end_sha']
    d = pd.read_csv(ROOT / 'data' / 'survival_events_line_pos_v6confnv_idtrack.csv',
                    usecols=cols, dtype=str)
    return d.merge(ids, on=['slug', 'pr_number', 'unit_id'], how='inner').drop_duplicates('line_id')


def page(lid, slug, src_path, text, move_sha, move_no, dest, occ, dest_lines, commits, end_sha):
    gh = f'https://github.com/{slug}'
    ctx = []
    for n in occ[:3]:
        lo, hi = max(1, n - 8), min(len(dest_lines), n + 8)
        ctx.append(f'<h2>Destination file at the move commit, lines {lo} to {hi}</h2><pre>' + '\n'.join(
            (f'<span class="t">{i:5d} | {html.escape(dest_lines[i-1])}</span>' if i == n
             else f'{i:5d} | {html.escape(dest_lines[i-1])}') for i in range(lo, hi + 1)) + '</pre>')
    p = [f'<!doctype html><html><head><meta charset="utf-8"><title>{lid} (moved)</title><style>{T.CSS}</style></head><body>',
         f'<h1>{lid}: after the move</h1><p><b>Repository:</b> <a href="{gh}">{slug}</a><br>',
         f'<b>Original file:</b> {html.escape(src_path)} (see {lid}.html)<br>',
         f'<b>Move commit:</b> <a href="{gh}/commit/{move_sha}">{move_sha[:10]}</a> (commit #{move_no} on {lid}.html)<br>',
         f'<b>Destination file (your label):</b> {html.escape(dest)}<br>',
         f'<b>Window ends:</b> {T.CUT.date()} (end snapshot <a href="{gh}/tree/{end_sha}">{end_sha[:10]}</a>)</p>',
         f'<p><b>The line:</b></p><pre><span class="t">{html.escape(text)}</span></pre>',
         f'<p class="k">Lines with exactly this text in the destination file at the move commit: {len(occ)}'
         + (f' (lines {", ".join(map(str, occ[:10]))}).' if occ else '. The text may have been reformatted in the move.') + '</p>',
         *ctx,
         f'<p class="k">Below are all {len(commits)} first-parent commits of the default branch, in time order, '
         'that touch the destination file (following renames) after the move commit and up to the end of the window. '
         'Lines whose text equals the line above are shaded. Identical text is not proof that it is the same line.</p>']
    for i, c in enumerate(commits, 1):
        nxt = c['path_after']
        try:
            cnt = T.git(T.repo_dir(slug), 'show', f'{c["sha"]}:{nxt}').splitlines().count(text) if nxt else 0
        except Exception:
            cnt = 0
        moved = T.same_text_elsewhere(T.repo_dir(slug), c['sha'], nxt or dest, text)
        lines = c['patch'].splitlines()
        cut = len(lines) > MAX_SHOWN
        shown = '\n'.join(lines[:MAX_SHOWN])
        p.append(f'<h2>Commit #{i}: <a href="{gh}/commit/{c["sha"]}">{c["sha"][:10]}</a> ({c["date"]})</h2>'
                 f'<p class="k">{html.escape(c["subject"])}<br>Path after this commit: {html.escape(str(nxt)) if nxt else "file deleted"}. '
                 f'Lines with exactly this text in the file after the commit: {cnt}.'
                 + (f' Identical text added in other files in this commit: {html.escape(", ".join(moved[:5]))}.' if moved else '')
                 + (f' <b>Patch cut at {MAX_SHOWN} of {len(lines)} lines; see the commit on GitHub.</b>' if cut else '')
                 + f'</p><pre>{T.render_patch(shown, text)}</pre>')
    if not commits:
        p.append('<p><b>No commit touches the destination file after the move in the window.</b></p>')
    p.append('</body></html>')
    (EV / f'{lid}_moved.html').write_text(''.join(p), encoding='utf-8')


def main():
    EV.mkdir(exist_ok=True)
    s = pd.read_excel(SHEET, dtype=str)
    k = pd.read_csv(OUT / 'tracker_groundtruth_key.csv', dtype=str)
    m = s[s.status.str.strip() == 'moved to other file'].merge(
        k[['line_id', 'slug', 'pr_number', 'unit_id', 'commit_shas']], on='line_id')
    m = m.merge(end_shas(m[['line_id', 'slug', 'pr_number', 'unit_id']])[['line_id', 'end_sha']], on='line_id')
    rows = []
    for r in m.itertuples():
        repo = T.repo_dir(r.slug)
        shas = r.commit_shas.split()
        no = int(float(r.moved_in_commit_no))
        move_sha = shas[no - 1]
        dest = str(r.moved_to_file).strip()
        try:
            dest_lines = T.git(repo, 'show', f'{move_sha}:{dest}').splitlines()
        except Exception:
            dest_lines = []
        text = r.line_text
        occ = [i for i, ln in enumerate(dest_lines, 1) if ln == text] or \
              [i for i, ln in enumerate(dest_lines, 1) if ln.strip() == text.strip()]
        commits = T.history(repo, move_sha, r.end_sha, dest) if dest_lines else []
        page(r.line_id, r.slug, r.file_at_merge, text, move_sha, no, dest, occ, dest_lines, commits, r.end_sha)
        rows.append(dict(line_id=r.line_id, repository=r.slug, destination_file=dest,
                         destination_found='yes' if dest_lines else 'NO (check path)',
                         occurrences_at_move=len(occ), n_commits_listed=len(commits),
                         evidence=f'evidence_moves/{r.line_id}_moved.html',
                         status='', first_change_commit_no='', kind='', confident='', notes='',
                         _shas=' '.join(c['sha'] for c in commits)))
        print(r.line_id, len(dest_lines) > 0, len(occ), len(commits), flush=True)
    out = pd.DataFrame(rows)
    out[['line_id', '_shas']].rename(columns={'_shas': 'followup_commit_shas'}).to_csv(
        OUT / 'tracker_groundtruth_moves_commits.csv', index=False)
    out = out.drop(columns='_shas')
    z = out.n_commits_listed == 0
    out.loc[z, 'status'], out.loc[z, 'confident'] = 'retained', 'yes'
    out.loc[z, 'notes'] = 'Prefilled: no commit touches the destination file after the move in the window.'
    sheet = OUT / 'tracker_groundtruth_moves_sheet.xlsx'
    if sheet.exists():
        print('sheet exists, not overwritten:', sheet)
    else:
        out.to_excel(sheet, index=False)
    print('wrote', len(out), 'rows; destination missing:', (out.destination_found != 'yes').sum())


if __name__ == '__main__':
    main()
