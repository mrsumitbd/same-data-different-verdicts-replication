# Leaf-only or structural modification: annotation guide

## Purpose

The study follows each line of code after it is merged and records the first commit that modifies it. One of its definitions of a modification counts only changes to the structure of the code. For each modified line, it finds the smallest *enclosing unit* around the line (a function, method, class, or similar named construct, or, at module level, an import or a simple assignment) and asks whether the commit changed that unit's structure or only its leaf tokens.

You label 100 such modifications independently. Your labels are compared with those of another rater to measure inter-rater agreement (Cohen's kappa).

## What you have

- `ast_sheet.xlsx`: the sheet you fill in, sheet `sample`, one row per modification. Sheet `labels` lists the allowed labels.
- This guide.

The rows are in shuffled order. The sheet contains no information about who wrote the code or about any other label.

## Columns

| Column | Fill in? | Meaning |
|---|---|---|
| `rater2_id` | no | Row id (AST-001 ...). Do not edit; it is the join key. |
| `repo`, `file`, `language` | no | Where the line lives. |
| `modifying_commit_url` | no | The commit that first modified the line. This is the change you judge. |
| `line_before` | no | The line just before the modifying commit. |
| `line_after` | no | A guess at the line that replaced it. `[line removed or empty]` means the line or the file was deleted. The guess can be wrong when several lines change together, so trust `diff_context` over this column. |
| `enclosing_node` | no | The type of the enclosing unit (for example `function_definition`, `method_declaration`, `impl_item`, `import_statement`). It tells you which unit to judge, not the answer. |
| `hunk_header` | no | Git's guess at the enclosing function or class. |
| `diff_context` | no | The file's diff in the modifying commit (5 lines of context per hunk), with the modified line marked `>>`. Long diffs are cut to about 15 lines on either side of the line and start with `[long diff ...]`. Deleted files start with `[file deleted in the modifying commit]`. |
| `label` | **yes** | `leaf-only`, `structural`, or `unclear` (drop-down). |
| `notes` | optional | Write a note whenever the case is borderline. |

If the context is not enough, open `modifying_commit_url` on GitHub.

## The question

> **Did this commit change the structure of the unit that encloses the line, or only its leaf tokens?**

Judge the **enclosing unit**, not only the marked line. If the commit adds an `if` five lines below the marked line in the same function, the unit's structure changed, so the label is `structural`, even if the marked line itself only had a rename.

### leaf-only

The unit has the same shape before and after: the same statements, in the same order, with the same kinds of expressions. Only the values at the leaves differ. Typical cases:

- Renaming a variable, parameter, attribute, or type used inside the unit (for example `e` to `_e`).
- Changing a literal: a number, string, boolean, or the text of a message (for example `timeout=30` to `timeout=60`).
- Changing quote style or formatting, or a whitespace-only change (for example a formatter run or line wrapping).
- Swapping an operator of the same kind (for example `<` to `<=`, `and` to `or`).
- Editing the text of an existing comment or docstring.

### structural

The shape of the unit changed. Typical cases:

- A statement is added, removed, or moved inside the unit (a call, an assignment, a `return`, a log line).
- Control flow changes (an `if`/`else`, a loop, `try`/`except`, an early return is added or removed).
- An expression changes kind (for example a literal becomes a function call, `x` becomes `x.strip()`, or an argument is added to or removed from a call).
- The signature changes (a parameter, decorator, or type annotation is added or removed).
- The unit is removed, or the whole file is deleted.
- The marked line is deleted, and with it a statement or expression of the unit.
- The unit itself is renamed (for example `def foo` becomes `def bar`).

### unclear

Use `unclear` only when you cannot decide from the diff and the GitHub view, for example when the diff is too large to locate the unit or the language is one you cannot read. Say why in `notes`.

## Edge cases

Label these by the concept above, and write a short note:

1. **Module-level imports and constants.** The unit is that one statement. Changing an imported name or a constant's value is `leaf-only`. Adding or removing an imported name from a list is `structural`. Note `module-level`.
2. **Comments.** Editing a comment's text is `leaf-only`. Adding or removing a whole comment inside the unit: label it as you see it, and note `comment`.
3. **Several changes in one unit.** If any change in the unit is structural, the label is `structural`.
4. **Moves.** A statement moved within the unit is `structural`, because the order changed. A whole unit moved elsewhere in the file without edits is `leaf-only`; note `moved`.
5. **A deleted line in an otherwise identical unit.** Deleting a blank line or a comment line is `leaf-only`. Deleting a code line is `structural`.

## Procedure

1. For each row, read `line_before`, `line_after`, and the `>>` line in `diff_context`. Then scan the rest of `diff_context` for other changes inside the same unit. Open `modifying_commit_url` when you need more.
2. Pick a label from the drop-down. Add a note for borderline cases.
3. Label every row on its own, without discussing rows with the other rater.
4. Save the sheet under the same name and return it.

Most rows take well under a minute.
