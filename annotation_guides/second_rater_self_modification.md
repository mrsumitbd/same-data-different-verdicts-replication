# Agent self-modifications: annotation guide

## Purpose

Coding agents sometimes modify code that the same agent wrote in an earlier pull request (PR). The study asks whether such a commit continues the work of the original PR or is later maintenance that happens to touch the same lines. You label 40 commits independently. Your labels are compared with those of another rater to measure inter-rater agreement (Cohen's kappa).

## What you have

- `selfmod_sheet.xlsx`: the sheet you fill in, sheet `sample`, one row per commit. Sheet `labels` lists the three labels.
- This guide.

The rows are in shuffled order and come from two agents, GitHub Copilot and Devin.

## The question for each commit

> Compared with the PR that wrote the lines, is this commit the same task, the next item of a series, or a different piece of later maintenance?

You compare two pieces of work: the modifying commit and the original PR. You do not need to work out which exact lines were modified.

## The labels

Go through these in order, and stop at the first one that fits.

**1. `same task`.** The commit fixes, completes, extends, or undoes the specific feature or fix that the original PR delivered.

- A bug in what the PR added is fixed.
- The PR's feature gets a follow-up it clearly needed.
- The PR is reverted, or partly reverted.

**2. `next item of a series`.** The commit does the same kind of change as the original PR, but on a different item. It does not change what the original PR delivered.

- Signs: "part 2", "part 5"; the same issue or tracking issue; the same change applied to another file, function, package, or command.
- Check who requested the work if you can see it. Many Devin PR descriptions have a "Requested by" line. The same requester supports this label, but you can use the label without it.

**3. `later maintenance`.** Neither of the above: the commit is a different piece of work that happens to touch these lines.

- A repository-wide refactor, migration, rename, or lint pass.
- A new feature whose code passes through the same lines.
- A fix to something other than what the original PR delivered.

## Examples

These examples were written for this guide and are not in the sample.

| Situation | Label |
|---|---|
| The original PR adds a CSV export. The commit fixes a wrong column order in that export. | same task |
| The original PR adds null-safety annotations to one package. The commit, "part 3" of the same issue, adds them to another package. | next item of a series |
| The original PR adds an error message. The commit moves all user-facing strings, including that message, into a localization file. | later maintenance |

## Rules for the hard cases

- **Ignore the time since merge.** Decide from the content of the work only. GitHub shows dates when you open a commit or PR; please do not compare them. A fix of the PR's feature 150 days later is still `same task`.
- **Same task or series?** Ask whether the commit changes what the PR delivered. If it does, use `same task`. If it applies the same kind of change to something else, use `next item of a series`.
- **A commit that does several things.** Label its main purpose, as stated in the message and as the bulk of the diff shows. If only a small side change relates to the original PR, it is `later maintenance`; say so in `notes`.
- **Several original PRs.** Some rows have `(also #...)` in the URL column. The first PR is the one with the most modified lines, and its text is what the sheet shows. Check the others too, and use the label for the closest relation. If you label against a PR other than the first, name it in `notes`.
- **Not sure.** Pick the best label, and write `low confidence` and the reason in `notes`.

## How to label one commit

1. Read the commit message and the original PR's title and description.
2. If those decide it, set the label and move on.
3. If not, open the commit and the PR on GitHub and read the diff, the linked issue, and the discussion.
4. Set `label` from the drop-down, and add `notes` if needed.

Label every row on its own, without discussing rows with the other rater.

## Columns

| Column | Fill in? | Content |
|---|---|---|
| `rater2_id` | no | Row id (SM-001 ...). Do not edit; it is the join key. |
| `repo`, `sha` | no | Repository and modifying commit. |
| `commit message` | no | The commit's full message. |
| `modifying-commit URL` | no | Link to the commit on GitHub. |
| `original PR URL` | no | Link to the PR that wrote the lines, plus any other PRs. |
| `original PR title and description` | no | The PR's title and description. |
| `label` | **yes** | `same task`, `next item of a series`, or `later maintenance`. |
| `notes` | if needed | Anything unusual, or `low confidence`. |

## When you finish

Save the sheet under the same name and return it.
