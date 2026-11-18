# Human-baseline pull requests: annotation guide

## Purpose

The study compares code from AI coding agents with code from pull requests (PRs) that are not attributed to an agent, which we call the human baseline. Some human-baseline PRs may still come from an automation bot, or carry signs that an AI tool wrote the code or the description. You label 100 human-baseline PRs independently. Another rater labels the same sheet, and the two sets of labels are compared to measure inter-rater agreement (Cohen's kappa).

## What you have

- `human_baseline_sheet.xlsx`: the sheet you fill in, sheet `sample`, one row per PR. Sheet `labels` lists the allowed labels.
- This guide.

The rows are in shuffled order. The sheet carries no earlier verdict or automatic signal.

## The question for each PR

> Who or what produced this PR: a person, an automation bot, or, as the PR shows, an AI tool?

## The labels

Go through these in order, and stop at the first one that fits.

**`bot`.** The PR is opened by an automation account rather than a person: dependency updaters (Dependabot, Renovate), release or CI bots, translation or sync bots. Their account names usually end in `[bot]` or `-bot`. A coding agent's account (for example `devin-ai-integration[bot]`, `copilot-swe-agent`) is not `bot`; label it `ai-attributed`.

**`ai-attributed`.** The PR shows that an AI tool produced the code or the PR description. Examples:

- The description says that an AI tool generated the change or the description ("Generated with Claude Code", "This description was created by Ellipsis", "Co-authored with Copilot").
- The commits carry an AI co-author trailer or are authored by an AI tool's account.
- The PR was opened by, or on behalf of, an AI agent.

**`human`.** A person opened the PR, and any AI mention is incidental and does not show that an AI tool wrote the code or the description. Examples:

- The PR adds support for an AI provider, model, or SDK (for example "Support OpenAI reasoning models").
- The repository belongs to an AI company or integrates with an AI service.
- An AI review bot only commented on or reviewed the PR. A review is not authorship.
- The word matches by accident (for example "cursor" meaning the text cursor).
- The PR configures an AI tool (for example a GitHub Action) without that tool writing the change.

**`cannot assess`.** The PR gives too little to decide, for example an empty description, and the commits and conversation on GitHub show nothing either way. Before you choose this label for an empty description, open the PR and check the commits.

## How to label one PR

1. Read the author (`user`), the title, and the description (`body`).
2. If they decide it, set the label.
3. If not, open the PR on GitHub (`github_url`) and check the commits (authors and co-author trailers) and the conversation.
4. Set `label` from the drop-down, and add `notes` for anything borderline.

Label every row on its own, without discussing rows with the other rater and without looking up any earlier verdict on these PRs.

## Columns

| Column | Fill in? | Content |
|---|---|---|
| `rater2_id` | no | Row id (HB-001 ...). Do not edit; it is the join key. |
| `github_url` | no | Link to the PR. |
| `title`, `user`, `body` | no | The PR's title, author, and description. |
| `label` | **yes** | `human`, `bot`, `ai-attributed`, or `cannot assess`. |
| `notes` | optional | Free text. |

## When you finish

Save the sheet under the same name and return it.
