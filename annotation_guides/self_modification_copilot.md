# Copilot self-modifications: labeling guideline

This sheet classifies GitHub Copilot's self-modifications: whether a later change by the agent to its own code continues the task of the original PR or is later maintenance (RQ5).

## 1. What you have

The file is `results_v6/round2/copilot_selfmod_sample.xlsx`.

- Sheet `sample`: the one you fill in, with one row per commit (20 rows).
- Sheet `labels`: the three labels, for reference.

**The sample.** There are 143 commits in the corpus that Copilot authored and that change at least one line written by a Copilot PR. The 20 were drawn from these at random (seed 42). Twelve are in `dotnet/aspire`. That is what the draw gave, so please don't adjust for it.

**Before you start, please hide column G (`lag (days)`).** See Section 5.

## 2. The question for each commit

> Compared with the PR that wrote the lines, is this commit the same task, the next item of a series, or a different piece of later maintenance?

You are comparing two pieces of work: the modifying commit, and the original PR. You don't need to work out which exact lines were modified.

## 3. The labels

Go through these in order, and stop at the first one that fits.

**1. `same task`.** The commit fixes, completes, extends, or undoes the specific feature or fix that the original PR delivered.

- A bug in what the PR added is fixed.
- The PR's feature gets a follow-up it clearly needed.
- The PR is reverted, or partly reverted.

**2. `next item of a series`.** The commit does the same kind of change as the original PR, but on a different item. It does not change what the original PR delivered.

- Signs: "part 2", "part 5"; the same issue or tracking issue; the same change applied to another file, function, package, or command.
- Check the requester if you can see it on the PR page (who assigned Copilot or opened the issue). The same requester supports this label, but you can use the label without it. Copilot PRs often don't show the requester in the description, so write `requester not visible` in `notes` when that is the case.

**3. `later maintenance`.** Neither of the above: the commit is a different piece of work that happens to touch these lines.

- A repository-wide refactor, migration, rename, or lint pass.
- A new feature whose code passes through the same lines.
- A fix to something other than what the original PR delivered.

## 4. Examples from the Devin sample

These come from the Devin sample rather than from your sheet, so that they don't prime any of your rows.

| Commit | Original PR | Label | Why |
|---|---|---|---|
| lingo.dev: Packagist endpoint fix (#701) | #698 | same task | It fixes what the PR added. |
| crewai: telemetry-disable fix (#2946), 62 days later | #2537, which added the variable | same task | It fixes the feature. The long lag doesn't change the label. |
| nearcore: deprecate the next protocol features | an earlier deprecation by the same requester | next item of a series | Same kind of change, different features. |
| gumroad: Tailwind migration touching review-response code | the PR that added that code | later maintenance | A migration that merely passes through these lines. |

## 5. Rules for the hard cases

- **Ignore the time since merge.** Decide from the content of the work only. The lag is analyzed separately, and if it informs the labels then the two results are no longer independent. This is why column G should stay hidden. A fix of the PR's feature 150 days later is still `same task`. Mention it in `notes` if the long lag makes you uneasy about the label.
- **Same task or series?** Ask whether the commit changes what the PR delivered. If it does, use `same task`. If it applies the same kind of change to something else, use `next item of a series`.
- **A commit that does several things.** Label its main purpose, as stated in the message and as the bulk of the diff shows. If only a small side change relates to the original PR, it is `later maintenance`, and you can say so in `notes`.
- **Several original PRs.** Three rows have `(also #...)` in the URL column. The first PR is the one with the most modified lines, and its text is what the sheet shows. Check the others too, and use the label for the closest relation. If you label against a PR other than the first, name it in `notes`.
- **Not sure.** Pick the best label, and write `low confidence` and the reason in `notes`. There is no "unclear" label, because 20 is already a small sample.

## 6. How to label one commit

1. Read the commit message (column C) and the original PR's title and description (column F).
2. If those decide it, set the label and move on.
3. If not, open the commit (column D) and the PR (column E) on GitHub and read the diff, the linked issue, and the discussion.
4. Set `label` from the drop-down, and add `notes` if needed.

Please label without looking at the other rows' labels again, and don't use intent labels from the classifiers or from RQ4.

## 7. Sheet columns

| Column | Fill in? | Content |
|---|---|---|
| A `repo` | no | Repository. |
| B `sha` | no | The modifying commit. |
| C `commit message` | no | Its full message. |
| D `modifying-commit URL` | no | Link to the commit on GitHub. |
| E `original PR URL` | no | Link to the PR that wrote the lines, plus any other PRs. |
| F `original PR title and description` | no | The PR's title and description, as in the AIDev data. |
| G `lag (days)` | no | Hide this column (Section 5). |
| H `label` | **yes** | `same task`, `next item of a series`, or `later maintenance`. |
| I `notes` | if needed | Anything unusual, `low confidence`, or `requester not visible`. |

## 8. When you finish

Save the file as `results_v6/round2/copilot_selfmod_sample_labelled.xlsx`. The counts per label are reported in RQ5.
