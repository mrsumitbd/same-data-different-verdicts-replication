# Devin self-modifications: labeling guideline

This sheet classifies Devin's self-modifications: whether a later change by the agent to its own code continues the task of the original PR or is later maintenance (RQ5). It uses the same labels as the Copilot sheet.

## 1. What you have

The file is `results_v6/round2/devin_selfmod_sample.xlsx`.

- Sheet `sample`: the one you fill in, with one row per commit (20 rows, in shuffled order).
- Sheet `labels`: the three labels, for reference.

**The sample.** There are 85 commits in the corpus that Devin authored and that change at least one line written by a Devin PR. The 20 are the same commits as in round 1, drawn from these 85 with seed 42. Please don't adjust for how many come from any one repository.

**Please label before you look at anything else about these commits.** Until you finish, don't open earlier labels of these commits or the Devin sentences of RQ5 in the paper. They contain earlier judgments or the lags.

**Partial blinding.** You have already seen a few of the round-1 judgments (examples in the earlier Copilot guide, and the totals). Judge each row on its content anyway. If you recognize a row and think it affected your label, write `saw round-1 label` in `notes`.

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
- Check the requester if you can see it (who asked Devin for the work). Many Devin PR descriptions have a "Requested by" line. The same requester supports this label, but you can use the label without it. Write `requester not visible` in `notes` when you can't see it.

**3. `later maintenance`.** Neither of the above: the commit is a different piece of work that happens to touch these lines.

- A repository-wide refactor, migration, rename, or lint pass.
- A new feature whose code passes through the same lines.
- A fix to something other than what the original PR delivered.

## 4. Examples from your Copilot labels

These are your own labels from `copilot_selfmod_sample_labelled.xlsx`, so that the examples don't prime any row of this sheet.

| Commit | Your label | Your reason |
|---|---|---|
| dotnet/aspire: fix to a console rendering glitch (7e95af73) | same task | It adjusts the guidance-message helper that the original PR added. |
| dotnet/android: "Fix NRT annotations in MSBuild Tasks, part 5" (#10313) | next item of a series | The original PR was an earlier part of the same series, on different files. |
| dotnet/runtime: GeneratedRegex fixer, multiline verbatim strings (e317e1e1) | next item of a series | The same kind of fix on a different pattern; it does not change what the original PR delivered. |
| owncast/owncast: BrowserNotifyModal moved to the Localization component (#4444) | later maintenance | Part of an i18n pass, unrelated to the original PR's error message styling. |

## 5. Rules for the hard cases

- **Ignore the time since merge.** Decide from the content of the work only. The lag is analyzed separately, and if it informs the labels then the two results are no longer independent. This is why the sheet has no lag column. GitHub shows dates when you open a commit or PR, so please don't compare them. A fix of the PR's feature 150 days later is still `same task`.
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
| G `label` | **yes** | `same task`, `next item of a series`, or `later maintenance`. |
| H `notes` | if needed | Anything unusual, `low confidence`, or `requester not visible`. |

## 8. When you finish

Save the file as `results_v6/round2/devin_selfmod_sample_labelled.xlsx`. The counts per label are reported in RQ5.
