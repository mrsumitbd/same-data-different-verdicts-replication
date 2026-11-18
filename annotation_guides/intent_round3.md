# Intent label top-up (v6): instructions

**File to label:** `results_v6/intent_topup_v6_sheet.xlsx` (the `.csv` file next to it holds the same rows). There are 256 commits, V001–V256.

**How to label:** follow `docs/annotation-guide.md`. It is the same task, columns and categories as the first two rounds (the 385-commit sample and the 148-commit top-up):

- Give each commit one primary intent in `human_label`: `corrective`, `perfective`, `adaptive` or `preventive`. The .xlsx has a dropdown.
- Use `notes` for anything that is unclear or ambiguous.
- The commit URL on GitHub is authoritative.

**Blind sheet:** the sheet does not show the author tier, the keyword label or any LLM label. Rows are in random order inside each block, so the order gives nothing away either. Please label from the message and diff only.

**Rows without a diff:** 22 rows show a file list with no diff text ("[No patch content available]"). These are large commits (25–300 files) for which GitHub's API returns no patch. Open the commit URL for them. The 148-commit top-up had 4 such rows.

**Stopping early:** rows V001–V100 are themselves a proportional stratified sample. If you stop after V100, the labels are still a valid sample. Please label in order from V001.

## What was sampled

- **Corpus:** data/death_commit_messages_v6conf.csv, 5,373 modification commits. These are the conference cohort's born lines tracked by the journal's identity tracker.
- **Already labelled:** 369 of the 533 earlier human labels are in this corpus (224 from the 385-commit sample and 145 from the 148-commit top-up). All of them are in the 156 journal repositories.
- **Pool:** the 2,202 modification commits in the 55 new repositories that have modification commits. None of them has a human label.
- **Tier of a commit:** the majority tier among the hunks it modifies, the same rule the 385-commit sample used (`sample_intent_validation.py`).
  - Human: human PRs.
  - Human-supervised: Copilot, Cursor, Claude Code and Codex.
  - Devin.
- **Allocation:** proportional to the pool (largest remainder), seeded random draw. The seed is 20261103, in `src/make_intent_topup_sheet_v6.py`.

| Tier | Pool | Sample (256) | First 100 (V001–V100) |
|---|---:|---:|---:|
| Human | 1,313 | 153 | 60 |
| Human-supervised | 730 | 85 | 33 |
| Devin | 159 | 18 | 7 |
| **Total** | **2,202** | **256** | **100** |

The 369 existing labels plus these 256 make 625. That is a 59/41 split between old-repo and new-repo commits, the same split as the corpus (3,171 / 2,202).

## Files

- `results_v6/intent_topup_v6_sheet.xlsx` and `.csv`: the blind sheet to fill in.
- `results_v6/intent_topup_v6_sample.csv`: the key (sample_id, slug, sha, tier, block, keyword label). It is not blind, so please don't open it while labelling.

## Scoring

When the sheet is done, run:

```
python src/intent_kappa_v6.py --topup-v6
```

This computes kappa for keyword, each LLM and the combined approaches on all labels (533 + 256), on the 625 labels in the v6conf corpus, and on the top-up alone. While labelling is in progress, add `--partial` to score the rows labelled so far.
