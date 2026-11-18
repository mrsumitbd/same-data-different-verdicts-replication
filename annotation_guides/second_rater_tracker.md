# What happened to this line: annotation guide

## Purpose

The study follows lines of code after they are merged and records whether, and in which commit, each line is first modified. You label 100 lines independently from their file history. Your labels are compared with those of another rater to measure inter-rater agreement (Cohen's kappa).

## What you have

- `tracker_sheet.xlsx`: the sheet you fill in, sheet `sample`, one row per line. Sheet `labels` lists the allowed values of `status`.
- `evidence/TRK-xxx.html`: one page per line. Open it from the `evidence` column. Keep the `evidence` folder next to the sheet so the links work.
- This guide.

The rows are in shuffled order. The sheet contains no information about who wrote the line, and no output of any tool.

## The question for each line

> Between the merge commit and the end of the window (31 December 2025), is this line modified on the default branch? If it is, in which commit is it modified first?

"On the default branch" means the commits listed on the evidence page: the first-parent commits of the default branch that touch the line's file after the merge, up to the end of the window, with renames followed. For a merge commit, the diff is against its first parent, so treat a merge commit like any other commit. Ignore everything after the last listed commit.

## What counts as a modification

A line is **modified** at the first listed commit that does any of these:

| `kind` | Meaning |
|---|---|
| `content changed` | Any change to the line's text: an edited expression, a renamed identifier, a changed literal, or a comment edit. |
| `whitespace only` | The new line differs only in whitespace: re-indentation (for example, the block is wrapped in a new `if`), trailing spaces, tabs versus spaces, or line endings. **This counts as modified.** |
| `line deleted` | The line is removed and nothing takes its place. |
| `file deleted` | The file is deleted and **not** renamed. A rename shows as `rename from` / `rename to` and is not a deletion. |
| `moved and changed` | The line is moved (within the file or to another file) and its text changes in the same commit. |

The following are **not** modifications. The line keeps its identity, and you keep following it:

- **Position changes.** Lines are inserted or deleted above or below it, so its line number changes.
- **File renames or moves.** The evidence page follows the new path for you ("Path after this commit").
- **Unchanged moves within the file.** A commit removes a block that contains the line and adds the same block elsewhere in the same file, for example when functions are reordered. Keep following the line at its new place, and mention the move in `notes`.
- **Diff artifacts.** Sometimes git shows the line as removed (`-`) and an identical line added (`+`) in the same hunk while the code around it stays the same. Git aligned the hunk differently, and the line is not modified. The count "Lines with exactly this text in the file after the commit" helps here: if it does not drop and the context is unchanged, the line survived.

**Unchanged move to another file.** If the page says "Identical text added in other files in this commit: ..." and the context confirms that the block moved, set `status = moved to other file`, put the commit number in `first_change_commit_no`, and name the destination in `notes`. Stop there.

**A rewrite is not a move.** If a commit deletes a whole block and writes a new version that happens to contain the same text, and the surrounding statements change, label it `content changed` when the line now sits in changed code with a different role, or `line deleted` when the block is gone. Explain judgment calls in `notes`.

## Identical lines

Many lines are not unique, for example `return None`, `import os`, or closing tags. Use the context shown at the merge commit (8 lines above and below) to decide which occurrence is yours, and follow that one. The shaded lines in each diff only mark identical text. **Identical text is not proof that it is the same line.**

## How to label one line

1. Open the evidence page. Read the line and its context at the merge commit.
2. Go through the commits in order (#1, #2, ...). For each, check whether the diff touches *your* occurrence of the line.
3. At the first commit that modifies it, set `status = modified`, write the commit number in `first_change_commit_no`, and pick the `kind`. You can stop there.
4. If no listed commit modifies it, set `status = retained`.
5. Use `unclear` only if you cannot decide even with the GitHub links on the page, and say why in `notes`.

Use only the evidence page and, when needed, the GitHub links on it. Please do not run `git blame` or any line-tracking tool.

## Columns

| Column | Fill in? | Content |
|---|---|---|
| `rater2_id` | no | Row id (TRK-001 ...). Do not edit; it is the join key. |
| `repository`, `file_at_merge`, `line_number`, `line_text`, `n_commits_listed` | no | The line, and how many commits its page lists. |
| `evidence` | no | Link to the evidence page. |
| `status` | **yes** | `modified`, `retained`, `moved to other file`, or `unclear`. |
| `first_change_commit_no` | if modified or moved | The number of the commit (#k on the page). |
| `kind` | if modified | See the table above. |
| `notes` | optional | Free text. |

## Short examples

- Commit #1 inserts 20 lines at the top of the file. Commit #2 changes `timeout=30` to `timeout=60` on your line: `modified`, 2, `content changed`.
- Commit #1 renames `utils/a.py` to `core/a.py`. No later commit touches your line: `retained`.
- Commit #1 wraps your line's block in `try:`, so the line is only indented further: `modified`, 1, `whitespace only`.
- Commit #1 moves the function that holds your line, unchanged, below another function. Commit #3 deletes it: `modified`, 3, `line deleted`.

## When you finish

Save the sheet under the same name and return it. Every page lists between 1 and a few commits, so most lines take one to two minutes.
