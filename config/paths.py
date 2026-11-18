"""
Output/input path switch for re-running the analyses on another dataset
without touching the journal (v5) files.

    DATA_TAG=v6conf RESULTS_DIR=results_v6 python src/analyze_rq1_master.py

P(path) rewrites a repository-relative path:
  data/..._v5...            -> data/..._<DATA_TAG>...   (DATA_TAG set)
  data/<position-based v2>  -> its <DATA_TAG> twin (POSITION_DATA)
  results/<name>            -> <RESULTS_DIR>/<name>     (RESULTS_DIR set), except the
                               cohort-independent inputs in SHARED_RESULTS (human
                               labels, the 385-commit validation sample and its LLM
                               labels), which are always read from results/
  visuals/<name>, paper-overfeal-files/visuals/<name> -> <RESULTS_DIR>/visuals/<name>
With neither variable set, P is the identity (the v5 paths). DATA_TAG without
RESULTS_DIR is refused, so another dataset can never write into results/ or the paper.
"""

import os
import re

DATA_TAG = os.environ.get('DATA_TAG')
DATA_LABEL = DATA_TAG or 'v5'   # name of the dataset in printed headers
RESULTS_DIR = os.environ.get('RESULTS_DIR')
if DATA_TAG and not RESULTS_DIR:
    raise SystemExit('DATA_TAG is set but RESULTS_DIR is not: refusing to write into results/')

SHARED_RESULTS = {
    'intent_annotation_sheet.xlsx', 'intent_validation_sample_full.csv', 'intent_validation_sample_blind.csv',
    'intent_llm_deepseek_t0.csv', 'intent_llm_gpt_oss_t0.csv', 'intent_llm_minimax_t0.csv',
    'intent_annotation_topup.xlsx', 'intent_annotation_topup_filled.csv', 'intent_topup_sample.csv',
}
# position-based (line-number) tracking: journal file -> file for another dataset
POSITION_DATA = {
    'data/survival_events_line_v4_ast.csv': 'data/survival_events_line_pos_{tag}_ast.csv',
    'data/survival_events_line_v4.csv': 'data/survival_events_line_pos_{tag}.csv',
    'data/survival_events_line_v4units_v5track.csv': 'data/survival_events_line_pos_{tag}_idtrack.csv',
    'data/survival_events_line_v4units_v5track_ast.csv': 'data/survival_events_line_pos_{tag}_idtrack_ast.csv',
    'data/survival_events_hunk_v3.csv': 'data/survival_events_hunk_pos_{tag}.csv',
}
PAPER_VISUALS = 'paper-overfeal-files/visuals/'


def P(path) -> str:
    s = str(path)
    if DATA_TAG and s.startswith('data/'):
        s = POSITION_DATA[s].format(tag=DATA_TAG) if s in POSITION_DATA else re.sub(r'_v5(?=[_.])', f'_{DATA_TAG}', s)
    if RESULTS_DIR:
        if s.startswith('results/') and s[len('results/'):] not in SHARED_RESULTS:
            s = os.path.join(RESULTS_DIR, s[len('results/'):])
        elif s.startswith(PAPER_VISUALS):
            s = os.path.join(RESULTS_DIR, 'visuals', s[len(PAPER_VISUALS):])
        elif s.startswith('visuals/'):
            s = os.path.join(RESULTS_DIR, 'visuals', s[len('visuals/'):])
        if s.startswith(RESULTS_DIR):
            os.makedirs(os.path.dirname(s) or '.', exist_ok=True)
    return s
