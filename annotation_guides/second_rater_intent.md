# Commit intent: annotation guide

## Purpose

The study classifies the **intent** of commits that modify previously merged code, using an extended Swanson maintenance taxonomy. You label 100 commits independently by their **primary intent**: the single dominant reason the commit exists. Your labels are compared with those of another rater to measure inter-rater agreement (Cohen's kappa).

## What you have

- `intent_sheet.xlsx`: the sheet you fill in, sheet `sample`, one row per commit. Sheet `labels` lists the allowed labels.
- This guide.

The rows are in shuffled order. The sheet contains no information about who wrote the code, and no label from any classifier or rater.

## Columns

| Column | Fill in? | Meaning |
|---|---|---|
| `rater2_id` | no | Row id (INT-001 ...). Do not edit; it is the join key. |
| `commit_url` | no | The commit on GitHub, with the full diff. **Authoritative.** |
| `commit_message` | no | The full commit message. |
| `files_changed` | no | Every file the commit touched, with its status (added, modified, deleted, renamed) and `+added/-deleted` line counts. |
| `diff_length` | no | Size of the diff in characters. |
| `label` | **yes** | `corrective`, `perfective`, `adaptive`, or `preventive` (drop-down). |
| `notes` | optional | Anything unclear, ambiguous, or worth discussing. |

Assign **exactly one** category per commit. There is no "mixed" or "unsure" option. If a case is hard, pick your best label and say why in `notes`.

## The taxonomy

- **Corrective**: the main purpose is fixing a defect: a bug, crash, regression, or incorrect behavior that existed before this commit.
- **Perfective**: the main purpose is improving existing, already-working code: refactoring, performance optimization, cleanup, readability.
- **Adaptive**: the main purpose is adding new functionality or adapting to new requirements: new features, API changes, dependency updates, migrations.
- **Preventive**: the main purpose is preventing future problems: adding tests, documentation, CI configuration, type hints, validation.

## Classify by primary intent

Many commits do more than one thing. Choose the category that best describes the **dominant** purpose, weighted by the scope and magnitude of each kind of change (how much of the diff it accounts for, and how central it is to the commit). Do not let a minor incidental element decide the label.

- A commit that adds a major new feature across five files and also fixes one small bug in passing: **adaptive**.
- A commit that refactors a module for clarity and also fixes a formatting issue: **perfective**.
- A commit whose entire purpose is fixing a crash, even if it also touches related code: **corrective**.
- A commit that adds tests for existing functionality: **preventive**.

**Tie-break.** When no change clearly dominates by size, ask why this commit exists, as the message and the diff together show. A one-line fix that the whole commit is about is corrective even if it is small. A large diff that is mostly a mechanical consequence (regenerated files, renames, a version bump) does not outweigh the change that motivated it. Record hard tie-breaks in `notes`.

Fix vocabulary in the message ("fix", "bug", "issue") is evidence, not a rule. `feat: add export; fix typo` is adaptive if the export is the change. `chore: bump deps` is corrective if the diff shows that the bump fixes a reported crash.

## Reading the commit

- Read the message first, then `files_changed`, then the diff on GitHub. The file list often settles the scope (only test files suggest preventive; only CI or build files suggest preventive; lockfiles plus a manifest suggest adaptive), but check the diff before you decide.
- Use `commit_url` for the diff. The sheet does not carry diff bodies, because long diffs would be cut by the spreadsheet's cell limit.
- If a `files_changed` cell ends with `[... cut at 32,000 ...]`, read the file list on GitHub.

## Calibration examples

These examples were written for this guide and are not in the sample.

1. *"Fix crash when user uploads a file larger than 2GB"*, `upload_handler.py` (+8/-2). The commit exists to fix a crash: **corrective**.
2. *"Refactor authentication module to reduce duplicate code"*, `auth/login.py` (+40/-85), `auth/session.py` (+12/-30). Working code made cleaner, with no defect and no new behavior: **perfective**.
3. *"Add support for exporting reports to PDF"*, `reports/export.py` (added, +210/-0). New functionality: **adaptive**.
4. *"Add unit tests for the payment retry logic"*, `tests/test_payment_retry.py` (added, +95/-0). Only tests that guard existing behavior: **preventive**.
5. *"Update dependency versions; also fixes null pointer exception in parser caused by old library version"*, `requirements.txt` (+3/-3), `parser.py` (+2/-1). If the bump was made to fix the parser crash, the fix is the reason: **corrective**. If the bump is routine and the parser change incidental: **adaptive**. Read the diff and any linked issue, and note your reasoning.
6. *"Improve query performance and add caching layer for new analytics dashboard"*, with a new dashboard and cache making up most of the changed lines. The optimization serves the new feature: **adaptive**.

## Procedure

1. Label every row on its own, without discussing rows with the other rater and without using any automatic classifier.
2. Pick a label from the drop-down and add a note for hard cases.
3. Save the sheet under the same name and return it.

Most rows take one to three minutes.
