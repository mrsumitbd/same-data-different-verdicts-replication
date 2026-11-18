"""
Shared constants for the survival analysis pipeline.
Single source of truth — import from here, never hardcode elsewhere.
"""

from datetime import datetime
from pathlib import Path

# --- OBSERVATION WINDOW ---
# Inclusive: modifications committed up to 2025-12-31 23:59:59 UTC are events;
# the miner's end snapshot is the last first-parent commit at or before this instant.
OBSERVATION_CUTOFF = datetime(2025, 12, 31, 23, 59, 59)
PR_INCLUSION_CUTOFF = datetime(2025, 8, 1)

# --- PATHS ---
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
VISUALS_DIR = Path("visuals")
MODELS_DIR = Path("models")

# --- SOURCE CODE EXTENSIONS ---
SOURCE_EXTENSIONS = (
    '.py', '.js', '.ts', '.jsx', '.tsx',
    '.java', '.c', '.cpp', '.cc', '.h', '.hpp',
    '.cs', '.go', '.rb', '.php', '.rs', '.swift', '.kt',
    '.scala', '.m', '.mm'
)

# --- SURVIVAL OPERATIONALIZATION ---
# Token edit distance thresholds for death magnitude classification
THRESHOLD_TRIVIAL = 0.25     # < 25% tokens changed = trivial
THRESHOLD_MODERATE = 0.50    # 25-50% tokens changed = moderate
# > 50% = substantial, line removed entirely = deletion

# Death definitions for sensitivity analysis
DEATH_DEFINITIONS = {
    'any_change': 0.0,       # current baseline: any SHA change
    'moderate':   0.25,      # token edit distance > 25%
    'substantial': 0.50,     # token edit distance > 50%
}

# --- SAMPLING LIMITS (from v1, kept for consistency) ---
MAX_LINES_PER_FILE = 100
MAX_FILES_PER_PR = 20

# --- STATISTICAL ---
ALPHA = 0.05
MIN_STARS = 500