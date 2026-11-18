"""
make_annotation_sheet.py

Builds the Excel-safe annotation sheet for the 385-commit intent
validation sample from results/intent_validation_sample_blind.csv.

The blind CSV embeds full diffs (up to ~250,000 characters), which Excel
and Numbers silently truncate at 32,767 characters per cell. This sheet
keeps only what fits in a cell and points to GitHub for the full diff:

    sample_id       join key for compute_intent_kappa.py
    commit_url      full diff on GitHub (authoritative)
    commit_message  full commit message
    files_changed   file list with status and +/- line counts
    diff_length     characters of diff body in the blind CSV
    human_label     to fill: corrective / perfective / adaptive / preventive
    notes           optional free text

No tier, keyword or LLM information is included. Every cell is capped at
MAX_CELL_CHARS. The .xlsx has a dropdown (data validation) on human_label
and is written with the standard library only (no openpyxl). The .csv twin
prefixes cells starting with = + - @ with a space so spreadsheet apps do
not read them as formulas.

Outputs:
  results/intent_annotation_sheet.xlsx
  results/intent_annotation_sheet.csv

Usage:
    python src/make_annotation_sheet.py \
        --blind results/intent_validation_sample_blind.csv \
        --output results/intent_annotation_sheet.xlsx
"""

import argparse
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

LABELS = ['corrective', 'perfective', 'adaptive', 'preventive']
COLUMNS = ['sample_id', 'commit_url', 'commit_message', 'files_changed',
           'diff_length', 'human_label', 'notes']
COL_WIDTHS = [10, 45, 60, 70, 12, 16, 40]
MAX_CELL_CHARS = 32000          # Excel's hard limit is 32,767
DIFF_MARKER = '--- DIFF CONTENT ---'
_ILLEGAL_XML = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')


def split_diff(diff_text: str) -> tuple:
    """Return (file list, diff body) from a blind-sample diff_text."""
    text = '' if pd.isna(diff_text) else str(diff_text)
    head, _, body = text.partition(DIFF_MARKER)
    return head.strip(), body.strip()


def cap(text: str) -> str:
    """Cap a cell at MAX_CELL_CHARS with an explicit marker."""
    text = '' if pd.isna(text) else str(text)
    if len(text) <= MAX_CELL_CHARS:
        return text
    note = f'\n[... cut at {MAX_CELL_CHARS:,} of {len(text):,} characters; see commit_url ...]'
    return text[:MAX_CELL_CHARS - len(note)] + note


def build_rows(blind: pd.DataFrame) -> pd.DataFrame:
    files, lengths = [], []
    for diff_text in blind['diff_text']:
        file_list, body = split_diff(diff_text)
        files.append(cap(file_list))
        lengths.append(len(body))
    return pd.DataFrame({
        'sample_id': blind['sample_id'],
        'commit_url': blind['commit_url'],
        'commit_message': blind['commit_message'].map(cap),
        'files_changed': files,
        'diff_length': lengths,
        'human_label': '',
        'notes': '',
    })[COLUMNS]


def _cell(ref: str, value) -> str:
    if isinstance(value, (int, float)) and not pd.isna(value):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(_ILLEGAL_XML.sub('', '' if pd.isna(value) else str(value)))
    return f'<c r="{ref}" t="inlineStr" s="1"><is><t xml:space="preserve">{text}</t></is></c>'


def write_xlsx(rows: pd.DataFrame, path: Path) -> None:
    """Minimal .xlsx: one sheet, wrapped text, frozen header, label dropdown."""
    letters = [chr(ord('A') + i) for i in range(len(COLUMNS))]
    label_col = letters[COLUMNS.index('human_label')]
    sheet_rows = ['<row r="1">' + ''.join(
        _cell(f'{c}1', name) for c, name in zip(letters, COLUMNS)) + '</row>']
    for i, record in enumerate(rows.itertuples(index=False), start=2):
        sheet_rows.append(f'<row r="{i}">' + ''.join(
            _cell(f'{c}{i}', v) for c, v in zip(letters, record)) + '</row>')
    last = len(rows) + 1
    cols = ''.join(f'<col min="{i}" max="{i}" width="{w}" customWidth="1"/>'
                   for i, w in enumerate(COL_WIDTHS, start=1))
    sheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<cols>{cols}</cols><sheetData>{"".join(sheet_rows)}</sheetData>'
        f'<dataValidations count="1"><dataValidation type="list" allowBlank="1" '
        f'showErrorMessage="1" errorTitle="Invalid label" '
        f'error="Choose corrective, perfective, adaptive or preventive." '
        f'sqref="{label_col}2:{label_col}{last}"><formula1>"{",".join(LABELS)}"</formula1>'
        '</dataValidation></dataValidations></worksheet>'
    )
    files = {
        '[Content_Types].xml':
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '</Types>',
        '_rels/.rels':
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>',
        'xl/workbook.xml':
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="annotation" sheetId="1" r:id="rId1"/></sheets></workbook>',
        'xl/_rels/workbook.xml.rels':
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '</Relationships>',
        'xl/styles.xml':
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill>'
            '<fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1">'
            '<alignment vertical="top" wrapText="1"/></xf></cellXfs></styleSheet>',
        'xl/worksheets/sheet1.xml': sheet,
    }
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)


def write_csv(rows: pd.DataFrame, path: Path) -> None:
    def guard(v):
        return f' {v}' if isinstance(v, str) and v[:1] in ('=', '+', '-', '@') else v
    rows.apply(lambda col: col.map(guard)).to_csv(path, index=False, encoding='utf-8-sig')


def run(blind_path: str, output: str) -> None:
    blind = pd.read_csv(blind_path)
    rows = build_rows(blind)
    longest = rows[['commit_message', 'files_changed']].apply(lambda c: c.str.len()).max().max()
    assert longest <= MAX_CELL_CHARS, longest
    xlsx = Path(output)
    write_xlsx(rows, xlsx)
    write_csv(rows, xlsx.with_suffix('.csv'))
    print(f'{len(rows)} commits; longest cell {longest:,} chars '
          f'(limit {MAX_CELL_CHARS:,}); diff body > 32,767 chars in '
          f'{(rows["diff_length"] > 32767).sum()} commits (read those on GitHub)')
    print(f'Written: {xlsx} and {xlsx.with_suffix(".csv")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blind',  default='results/intent_validation_sample_blind.csv')
    parser.add_argument('--output', default='results/intent_annotation_sheet.xlsx')
    args = parser.parse_args()
    run(args.blind, args.output)
