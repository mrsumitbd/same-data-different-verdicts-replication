# Tracker ground truth: labeling guideline

This sheet checks the three line trackers against what actually happened to each line. The labels give, for each tracker, how often its "modified" and "retained" claims are right.

## 1. What you have

All files are in `results_v6/round1/tracker_gt/`.

- `tracker_groundtruth_sheet.xlsx`: the sheet you fill in, with one row per line (300 lines).
- `evidence/Lxxx.html`: one page per line. Open it from the `evidence` link in the sheet. The evidence folder has to stay next to the sheet for the links to work.
- `tracker_groundtruth_key.csv`: what each tracker claims for each line. **Do not open it before you finish labeling.**

The sheet gives no tracker output, no agent tier, and no PR author. The rows are in random order, so the order doesn't reveal which group a line was drawn from.

## 2. The question for each line

> Between the merge commit and the end of the window (31 December 2025), is this line modified on the default branch? If it is, in which commit is it modified first?

"On the default branch" means the commits listed on the evidence page. These are the first-parent commits of the default branch that touch the line's file after the merge, up to the end snapshot, with renames followed. For a merge commit, the diff is against its first parent. That diff shows what the merge brought into the default branch, so treat a merge commit like any other commit. Ignore everything after the last listed commit.

## 3. What counts as a modification

We use the paper's Definition 1 (syntactic). A line is **modified** at the first listed commit that does any of these:

| Kind (column `kind`) | Meaning |
|---|---|
| `content changed` | Any change to the line's text: an edited expression, a renamed identifier, a changed literal, or a comment edit. |
| `whitespace only` | The new line differs only in whitespace. This includes re-indentation (for example, the block is wrapped in a new `if`), trailing spaces, tabs versus spaces, and line endings. **This counts as modified.** |
| `line deleted` | The line is removed and nothing takes its place. |
| `file deleted` | The file is deleted and is **not** renamed. A rename shows as `rename from` / `rename to` and is not a deletion. |
| `moved and changed` | The line is moved (within the file or to another file) and its text changes in the same commit. |

The following are **not** modifications. The line keeps its identity, and you keep following it:

- **Position changes.** Lines are inserted or deleted above or below it, so its line number changes.
- **File renames or moves.** The evidence page follows the new path for you ("Path after this commit").
- **Unchanged moves within the file.** A commit removes a block that contains the line and adds the same block elsewhere in the same file, for example when functions are reordered. Write the commit number in `moved_in_commit_no` and keep following the line at its new place.
- **Unchanged moves to another file.** The page says "Identical text added in other files in this commit: ...". If the context confirms that the block moved, set `status = moved to other file` and fill `moved_in_commit_no` and `moved_to_file`. Stop there: the page cannot show the other file's history.
- **Diff artefacts.** Sometimes git shows the line as removed (`-`) and an identical line added (`+`) in the same hunk, while the code around it stays the same. Git simply aligned the hunk differently, and the line is not modified. The count "Lines with exactly this text in the file after the commit" helps here: if it doesn't drop and the context is unchanged, the line survived.

**A rewrite is not a move.** Sometimes a commit deletes a whole block and writes a new version that happens to contain the same text. If the surrounding statements change (it's a new version of the function, not the same function relocated), label it `content changed` if the line now sits in changed code with a different role. Label it `line deleted` if the block is gone. Use `confident = no` when it's a judgment call, and say why in `notes`.

## 4. Identical lines

Many lines are not unique. Examples are `return None`, `import os`, and closing tags. Use the context shown at the merge commit (8 lines above and below) to decide which occurrence is yours, and follow that one. The shaded lines in each diff only mark identical text. **Identical text is not proof that it is the same line.**

## 5. How to label one line

1. Open the evidence page. Read the line and its context at the merge commit.
2. Go through the commits in order (#1, #2, ...). For each, check whether the diff touches *your* occurrence of the line.
3. At the first commit that modifies it (Section 3), set `status = modified`, write the commit number in `first_change_commit_no`, and pick the `kind`. You can stop there.
4. If no listed commit modifies it, set `status = retained`. This includes lines whose file is never touched again ("No commit touches this file in the window").
5. Set `confident = yes` or `no`, and use `notes` for anything unusual.
6. Use `unclear` only if you cannot decide even with the GitHub links, for example a huge generated diff. Unclear lines are left out of the scores and counted.

Use only the evidence page and, when needed, the GitHub commit links on it, for example to see where a moved block went. Please don't run `git blame` or any of our trackers, because the labels have to be independent of them.

## 6. Sheet columns

| Column | Fill in? | Content |
|---|---|---|
| `line_id`, `repository`, `file_at_merge`, `line_number`, `line_text`, `n_commits_listed` | no | The line and how many commits its page lists |
| `evidence`, `merge_commit` | no | Links |
| `status` | **yes** | `retained` / `modified` / `moved to other file` / `unclear` |
| `first_change_commit_no` | if modified | The number of the commit (#k on the page) |
| `kind` | if modified | See the table in Section 3 |
| `moved_in_commit_no` | if moved | The number of the commit that moved the line unchanged |
| `moved_to_file` | if moved to another file | The destination path |
| `confident` | **yes** | `yes` / `no` |
| `notes` | optional | Free text |

## 7. Short examples

- Commit #1 inserts 20 lines at the top of the file. Commit #2 changes `timeout=30` to `timeout=60` on your line. → `modified`, commit 2, `content changed`.
- Commit #1 renames `utils/a.py` to `core/a.py`. No later commit touches your line. → `retained`.
- Commit #1 wraps your line's block in `try:`, so the line is only indented further. → `modified`, commit 1, `whitespace only`.
- Commit #1 moves the function that holds your line, unchanged, below another function. Commit #3 deletes it. → `modified`, commit 3, `line deleted`, with `moved_in_commit_no = 1`.
- Commit #2 cuts the function from `api.py` and pastes it unchanged into `api_v2.py`. → `moved to other file`, `moved_in_commit_no = 2`, `moved_to_file = api_v2.py`.

## 8. How the sample was drawn

The lines come from the 276,360 verified units that all three trackers cover (the units of RQ1, without vendored code), of which 202,384 are non-blank lines with at least 6 non-whitespace characters. Each unit gets a pattern of three claims: identity tracking, the tracker of the earlier study, and line-number tracking, each saying modified or retained in the window. We drew about the same number of lines from each pattern (seed 42). That gives every tracker at least 50 lines it calls modified and at least 50 it calls retained (table below). To keep labeling quick, a line is eligible only under these conditions:

- It is not blank and has at least 6 non-whitespace characters.
- At most 12 listed commits touch its file in the window, and each of those diffs has at most 300 lines for the file.
- The sample takes at most 2 lines per PR and 12 per repository.

| Pattern (identity, earlier, line-number) | Units | Eligible (est.) | Sampled |
|---|---|---|---|
| retained, retained, retained | 76,667 | 70,988 | 50 |
| retained, retained, modified | 61,538 | 38,461 | 50 |
| retained, modified, retained | 844 | 391 | 50 |
| retained, modified, modified | 5,671 | 1,956 | 50 |
| modified, retained, retained | 1 | 0 | 0 |
| modified, retained, modified | 15 | 2 | 2 |
| modified, modified, retained | 6,550 | 3,527 | 49 |
| modified, modified, modified | 51,098 | 21,219 | 49 |

Lines per tracker and claim in the sample:

| Tracker | Claims modified | Claims retained |
|---|---|---|
| Identity tracking | 100 | 200 |
| Earlier study's tracker | 198 | 102 |
| Line-number tracking | 151 | 149 |

The 300 lines come from 91 repositories. Their pages list 0 to 12 commits (median 2.5).

The scores weight each line by its pattern's estimated eligible population, so they describe the eligible population, not the raw sample. 95% intervals come from a stratified bootstrap. We will state these eligibility restrictions in the paper, because they favor short, simple histories, where every tracker should do best.

## 9. After you label

Save the sheet under the same name and tell me. `src/tracker_gt_score_r1.py` then reports, for each tracker:

- The precision of its "modified" and "retained" claims, the recall of modified lines, and its accuracy.
- How often it names the right first commit.

It reports these under two conventions: a move is not a modification (the paper's definition), and a move is a modification.

Time estimate: most pages list 0 to 3 commits, so about 1 to 2 minutes per line, roughly 6 to 10 hours in all. You can label in several sittings.

## Addendum (7 Nov 2026): lines moved to another file

For the 43 lines you labeled "moved to other file", `tracker_groundtruth_moves_sheet.xlsx` follows the destination file you named from the move commit to the end of the window. Each row links to `evidence_moves/Lxxx_moved.html`, which shows the destination file around the line at the move commit and every later first-parent commit that touches it.

Fill the columns exactly as in the main sheet, judging the line in its new place:

- `status`: `retained`, `modified`, `moved to other file` (moved again; give the new file in `notes`), or `unclear`.
- `first_change_commit_no`: the commit number on the follow-up page (not on the original page).
- `kind`, `confident`, `notes`: as in the main sheet.

The 8 rows with no commit after the move are prefilled as `retained`; please glance at them and change any you disagree with. A move into the destination file is not itself a modification here; only what happens to the line after the move counts.
