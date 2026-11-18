"""
measure_ast_survival.py

Post-processing script that adds AST-level death classification to
survival_events_line_v4.csv, producing survival_events_line_v4_ast.csv.

Definition 2 (AST structural death):
    A line "dies" at the AST level when the structural signature of its
    enclosing named AST node changes between birth_sha and death_sha.
    Leaf-only changes (renamed variables, changed literals, formatting)
    are NOT considered structural deaths — the line survives at AST level.

Scope of AST analysis:
    - Lines inside named functions, methods, or classes: full structural
      comparison of the enclosing node subtree.
    - Module-level import statements and simple assignments: matched by
      node identity; compared structurally.
    - Other module-level constructs: excluded from Definition 2, analyzed
      under Definition 1 only (ast_exclusion_reason = 'module_level_complex').

Exclusions:
    - Survived lines (event == 0): ast_death = null (not applicable)
    - Unsupported languages (Swift, Dart, etc.): ast_death = null
    - Excluded-by-design languages (CSS, HTML, etc.): ast_death = null
    - Files that fail to parse: ast_death = null
    - Lines with no enclosing named node: ast_death = null

New columns added to output:
    ast_death             int|null  1 = structural change, 0 = leaf-only change
    ast_enclosing_node    str|null  type of enclosing AST node
    ast_exclusion_reason  str|null  reason if ast_death is null

Usage:
    python src/measure_ast_survival.py

Input:
    data/survival_events_line_v4.csv

Output:
    data/survival_events_line_v4_ast.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='tree_sitter')
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import subprocess
import argparse
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict, Any
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# --- CONFIGURATION ---
INPUT_FILE  = Path("data/survival_events_line_v4.csv")
OUTPUT_FILE = Path("data/survival_events_line_v4_ast.csv")
MAX_WORKERS = 8


# =============================================================================
# LANGUAGE MAPPINGS
# =============================================================================

EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    '.py':   'python',
    '.js':   'javascript',
    '.ts':   'typescript',
    '.jsx':  'javascript',
    '.tsx':  'typescript',
    '.java': 'java',
    '.c':    'c',
    '.cpp':  'cpp',
    '.cc':   'cpp',
    '.h':    'cpp',
    '.hpp':  'cpp',
    '.cs':   'c_sharp',
    '.go':   'go',
    '.rb':   'ruby',
    '.php':  'php',
    '.rs':   'rust',
    '.kt':   'kotlin',
    '.scala':'scala',
    '.m':    'objc',
    '.mm':   'objc',
    '.lua':  'lua',
    '.jl':   'julia',
    '.ex':   'elixir',
    '.exs':  'elixir',
}

# Excluded by design: no meaningful function/class structure
EXCLUDED_BY_DESIGN = {
    '.css', '.scss', '.less',
    '.html', '.htm',
    '.md', '.mdx',
    '.json', '.yaml', '.yml',
    '.toml', '.ini', '.cfg',
    '.tf', '.hcl',
    '.mustache', '.hbs',
    '.typespec',
    '.spec',
    '.svg', '.xml',
}

# Unsupported: no tree-sitter grammar in our build
UNSUPPORTED = {
    '.swift', '.dart',
    '.ps1', '.psm1',
    '.vue', '.svelte',
    '.cmake',
    '.vim',
    '.nim',
}


# =============================================================================
# ENCLOSING NODE TYPES PER LANGUAGE
# Lines inside these node types receive full structural comparison.
# =============================================================================

ENCLOSING_NODE_TYPES: Dict[str, set] = {
    'python': {
        'function_definition',
        'async_function_definition',
        'class_definition',
        'decorated_definition',
    },
    'javascript': {
        'function_declaration',
        'function_expression',
        'arrow_function',
        'method_definition',
        'class_declaration',
        'class_expression',
    },
    'typescript': {
        'function_declaration',
        'function_expression',
        'arrow_function',
        'method_definition',
        'method_signature',
        'class_declaration',
        'class_expression',
        'abstract_method_signature',
        'function_type',
    },
    'java': {
        'method_declaration',
        'constructor_declaration',
        'class_declaration',
        'interface_declaration',
        'enum_declaration',
        'annotation_type_declaration',
    },
    'c': {
        'function_definition',
    },
    'cpp': {
        'function_definition',
        'class_specifier',
        'struct_specifier',
        'template_declaration',
    },
    'c_sharp': {
        'method_declaration',
        'constructor_declaration',
        'class_declaration',
        'interface_declaration',
        'struct_declaration',
        'enum_declaration',
        'property_declaration',
        'operator_declaration',
    },
    'go': {
        'function_declaration',
        'method_declaration',
        'type_declaration',
        'func_literal',
    },
    'rust': {
        'function_item',
        'impl_item',
        'trait_item',
        'closure_expression',
    },
    'ruby': {
        'method',
        'singleton_method',
        'class',
        'module',
        'do_block',
    },
    'php': {
        'function_definition',
        'method_declaration',
        'class_declaration',
        'interface_declaration',
        'trait_declaration',
    },
    'kotlin': {
        'function_declaration',
        'class_declaration',
        'object_declaration',
        'anonymous_function',
        'lambda_literal',
    },
    'scala': {
        'function_definition',
        'class_definition',
        'object_definition',
        'trait_definition',
        'val_definition',
    },
    'objc': {
        'function_definition',
        'method_definition',
        'class_interface',
        'class_implementation',
    },
    'lua': {
        'function_definition',
        'local_function',
        'function_statement',
    },
    'julia': {
        'function_definition',
        'short_function_definition',
        'macro_definition',
        'struct_definition',
    },
    'elixir': {
        'call',      # def, defp, defmodule are represented as 'call' nodes
        'do_block',
    },
}

# Module-level node types eligible for identity-based matching
# (imports and simple assignments only)
MODULE_LEVEL_ELIGIBLE: Dict[str, set] = {
    'python': {
        'import_statement',
        'import_from_statement',
        'expression_statement',   # covers simple assignments like X = 3
    },
    'javascript': {'import_statement', 'lexical_declaration', 'variable_declaration'},
    'typescript': {'import_statement', 'lexical_declaration', 'variable_declaration'},
    'java':       {'import_declaration'},
    'c':          {'declaration'},
    'cpp':        {'declaration', 'using_declaration'},
    'c_sharp':    {'using_directive', 'global_statement'},
    'go':         {'import_declaration', 'var_declaration', 'const_declaration'},
    'rust':       {'use_declaration', 'static_item', 'const_item'},
    'ruby':       {'assignment', 'call'},
    'php':        {'namespace_use_declaration', 'expression_statement'},
    'kotlin':     {'import_header', 'property_declaration'},
    'scala':      {'import_declaration', 'val_definition'},
    'objc':       {'preproc_include', 'declaration'},
    'lua':        {'assignment_statement', 'local_statement'},
    'julia':      {'import_statement', 'using_statement', 'assignment'},
    'elixir':     {'call'},
}


# =============================================================================
# AST UTILITIES
# =============================================================================

def get_parser(language: str):
    """Return a tree-sitter parser for the given language."""
    from tree_sitter_languages import get_parser as _get_parser
    return _get_parser(language)


def get_structural_signature(node) -> List[Tuple[int, str]]:
    """
    Serialize an AST subtree to a list of (depth, node_type) tuples.

    Ignores:
    - Anonymous nodes (punctuation, operators, keywords stored as literals)
    - Leaf node values (identifiers, literals, strings)

    Only captures tree shape and named node types — the structural skeleton.
    Two subtrees are structurally identical iff their signatures are equal.
    """
    result = []

    def walk(n, depth: int):
        if not n.is_named:
            return
        result.append((depth, n.type))
        for child in n.children:
            walk(child, depth + 1)

    walk(node, 0)
    return result


def find_enclosing_node(tree, line_number: int, language: str):
    """
    Find the smallest enclosing named AST node for a given line number.

    Search strategy:
    1. Walk the tree looking for named nodes whose range contains line_number.
    2. Among all candidates, select the smallest (deepest) one whose type
       is in ENCLOSING_NODE_TYPES for this language.
    3. If none found, check MODULE_LEVEL_ELIGIBLE for module-level handling.
    4. If still none, return (None, 'no_enclosing_node').

    Returns:
        Tuple of (node, category) where category is one of:
            'enclosing'      - named function/class/method node
            'module_eligible' - module-level import or simple assignment
            None             - no suitable node found
    """
    enclosing_types = ENCLOSING_NODE_TYPES.get(language, set())
    module_types    = MODULE_LEVEL_ELIGIBLE.get(language, set())

    # line_number is 1-indexed; tree-sitter uses 0-indexed rows
    target_row = line_number - 1

    best_enclosing = None
    best_enclosing_size = float('inf')
    best_module = None
    best_module_size = float('inf')

    def walk(node):
        nonlocal best_enclosing, best_enclosing_size
        nonlocal best_module, best_module_size

        start_row = node.start_point[0]
        end_row   = node.end_point[0]

        if not (start_row <= target_row <= end_row):
            return  # Node doesn't contain our line

        size = end_row - start_row

        if node.is_named:
            if node.type in enclosing_types:
                if size < best_enclosing_size:
                    best_enclosing      = node
                    best_enclosing_size = size
            elif node.type in module_types and start_row == target_row:
                # Module-level: node must START on our line
                if size < best_module_size:
                    best_module      = node
                    best_module_size = size

        for child in node.children:
            walk(child)

    walk(tree.root_node)

    if best_enclosing is not None:
        return best_enclosing, 'enclosing'
    if best_module is not None:
        return best_module, 'module_eligible'
    return None, None


def get_node_identity(node, language: str) -> Optional[str]:
    """
    Extract a stable identity string for a node used to find its
    counterpart in the death version of the file.

    For functions/methods: the function name.
    For classes: the class name.
    For imports/assignments: the full node text (normalized whitespace).

    Returns None if identity cannot be determined.
    """
    node_type = node.type

    # Node types where the first named child is the name identifier
    named_first_child_types = {
        'function_definition', 'async_function_definition',
        'function_declaration', 'method_definition', 'method_declaration',
        'constructor_declaration', 'class_declaration', 'class_definition',
        'interface_declaration', 'enum_declaration',
        'function_item', 'impl_item', 'trait_item',
        'function_declaration',  # go
        'method_declaration',    # go
        'type_declaration',      # go
    }

    if node_type in named_first_child_types:
        for child in node.children:
            if child.is_named and child.type in (
                'identifier', 'name', 'type_identifier',
                'field_identifier', 'property_identifier'
            ):
                return child.text.decode('utf-8', errors='replace')

    # For module-level nodes: use normalized full text as identity
    text = node.text.decode('utf-8', errors='replace')
    # Normalize whitespace for stable comparison
    return ' '.join(text.split())


def find_node_in_tree(tree, identity: str, node_type: str,
                      category: str, language: str):
    """
    Find a node in a parse tree by its identity string and type.

    For 'enclosing' category: match by node type + identity (name).
    For 'module_eligible' category: match by normalized full text.

    Returns the matching node or None if not found.
    """
    enclosing_types = ENCLOSING_NODE_TYPES.get(language, set())
    module_types    = MODULE_LEVEL_ELIGIBLE.get(language, set())

    candidates = []

    def walk(node):
        if not node.is_named:
            for child in node.children:
                walk(child)
            return

        if category == 'enclosing' and node.type in enclosing_types:
            node_identity = get_node_identity(node, language)
            if node_identity == identity:
                candidates.append(node)

        elif category == 'module_eligible' and node.type in module_types:
            node_text = ' '.join(
                node.text.decode('utf-8', errors='replace').split()
            )
            if node_text == identity:
                candidates.append(node)

        for child in node.children:
            walk(child)

    walk(tree.root_node)

    # Return the first (shallowest) match
    return candidates[0] if candidates else None


def classify_ast_death(
    birth_content: str,
    death_content: str,
    line_number: int,
    language: str,
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Classify whether a line's death is an AST structural change or
    a leaf-only change.

    Args:
        birth_content: Full file content at birth commit
        death_content: Full file content at death commit
        line_number:   1-indexed line number of the dying line
        language:      tree-sitter language identifier

    Returns:
        Tuple of (ast_death, ast_enclosing_node, ast_exclusion_reason)
        ast_death:
            1  = structural change detected
            0  = leaf-only change (line survives at AST level)
            None = could not classify (see ast_exclusion_reason)
    """
    try:
        parser = get_parser(language)
    except Exception:
        return None, None, 'parser_unavailable'

    # Parse birth version
    try:
        birth_tree = parser.parse(birth_content.encode('utf-8'))
    except Exception:
        return None, None, 'parse_failed_birth'

    # Parse death version
    try:
        death_tree = parser.parse(death_content.encode('utf-8'))
    except Exception:
        return None, None, 'parse_failed_death'

    # Check for fundamental parse failures (ERROR at root)
    if birth_tree.root_node.type == 'ERROR':
        return None, None, 'parse_failed_birth'
    if death_tree.root_node.type == 'ERROR':
        return None, None, 'parse_failed_death'

    # Find enclosing node in birth tree
    birth_node, category = find_enclosing_node(birth_tree, line_number, language)

    if birth_node is None or category is None:
        return None, None, 'no_enclosing_node'

    # For complex module-level constructs, exclude from AST analysis
    if category == 'module_eligible':
        # Only allow imports and simple assignments
        simple_module_types = {
            'import_statement', 'import_from_statement',
            'import_declaration', 'using_directive', 'using_declaration',
            'use_declaration', 'namespace_use_declaration',
            'import_header', 'lexical_declaration',
        }
        if birth_node.type not in simple_module_types:
            # Check if it's a simple assignment (single name on LHS)
            is_simple_assignment = _is_simple_assignment(birth_node, language)
            if not is_simple_assignment:
                return None, birth_node.type, 'module_level_complex'

    # Get identity for cross-version matching
    birth_identity = get_node_identity(birth_node, language)
    if birth_identity is None:
        return None, birth_node.type, 'identity_extraction_failed'

    # Find the same node in death tree
    death_node = find_node_in_tree(
        death_tree, birth_identity, birth_node.type, category, language
    )

    if death_node is None:
        # Node no longer exists in death version — structural change
        return 1, birth_node.type, None

    # Compare structural signatures
    birth_sig = get_structural_signature(birth_node)
    death_sig  = get_structural_signature(death_node)

    if birth_sig == death_sig:
        # Identical structure — only leaf values changed
        return 0, birth_node.type, None
    else:
        # Structure changed
        return 1, birth_node.type, None


def _is_simple_assignment(node, language: str) -> bool:
    """
    Heuristic: check if a module-level node is a simple name = value
    assignment (e.g. MAX_RETRIES = 3, VERSION = '1.0').

    Returns True only for assignments with a single identifier on the LHS.
    """
    node_text = node.text.decode('utf-8', errors='replace').strip()

    # Simple check: single identifier before = sign, no complex expressions
    import re
    # Match: IDENTIFIER = anything (but not def, class, or function-like)
    pattern = r'^[A-Za-z_][A-Za-z0-9_]*\s*='
    if re.match(pattern, node_text):
        # Exclude if it looks like a function or class definition
        if any(kw in node_text[:20] for kw in ('def ', 'class ', 'async ')):
            return False
        return True
    return False


# =============================================================================
# FILE CONTENT FETCHING
# =============================================================================

def fetch_file_at_commit(
    repo_dir: str, filepath: str, commit_sha: str
) -> Optional[str]:
    """Fetch the full content of a file at a specific commit."""
    cmd = ['git', 'show', f'{commit_sha}:{filepath}']
    result = subprocess.run(
        cmd, cwd=repo_dir, capture_output=True, text=True,
        errors='replace'
    )
    if result.returncode != 0:
        return None
    return result.stdout


def get_language_for_file(filepath: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine the tree-sitter language for a file path.

    Returns:
        Tuple of (language, exclusion_reason)
        If language is None, exclusion_reason explains why.
    """
    ext = Path(filepath).suffix.lower()

    if ext in EXCLUDED_BY_DESIGN:
        return None, 'excluded_by_design'
    if ext in UNSUPPORTED:
        return None, 'unsupported_language'

    language = EXTENSION_TO_LANGUAGE.get(ext)
    if language is None:
        return None, 'unknown_extension'

    return language, None


# =============================================================================
# REPOSITORY PROCESSING
# =============================================================================

def process_repo(slug: str, deaths_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Process all death events for a single repository.

    For each death event:
    1. Short-circuit deletions as unambiguous structural deaths (ast_death=1)
    2. Fetch file content at birth_sha and death_sha
    3. Classify AST death
    4. Return result dict with new columns

    Args:
        slug:      Repository slug (owner/repo)
        deaths_df: DataFrame of death events for this repo

    Returns:
        List of result dicts with keys:
            slug, pr_number, unit_id, ast_death,
            ast_enclosing_node, ast_exclusion_reason
    """
    results = []
    temp_dir = tempfile.mkdtemp()
    clone_url = (
        f"https://{GITHUB_TOKEN}@github.com/{slug}.git"
        if GITHUB_TOKEN
        else f"https://github.com/{slug}.git"
    )
    # AST_LOCAL_CLONES=1: clone from the miners' local bare clone (same objects) instead of GitHub
    local = Path(__file__).resolve().parent.parent / 'data' / 'temp_repos' / (slug.replace('/', '__') + '.git')
    if os.environ.get('AST_LOCAL_CLONES') == '1' and local.is_dir():
        clone_url = str(local)

    def _skip(row, reason: str):
        """Helper to append a skipped result with all key fields."""
        results.append({
            'slug':                 slug,
            'pr_number':            int(row['pr_number']),
            'unit_id':              row['unit_id'],
            'ast_death':            None,
            'ast_enclosing_node':   None,
            'ast_exclusion_reason': reason,
        })

    try:
        clone_result = subprocess.run(
            ['git', 'clone', '--filter=blob:none', clone_url, temp_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300
        )
        if clone_result.returncode != 0:
            for _, row in deaths_df.iterrows():
                _skip(row, 'clone_failed')
            return results

        # Cache file contents to avoid redundant git show calls
        # Key: (commit_sha, filepath) -> content
        file_cache: Dict[Tuple[str, str], Optional[str]] = {}

        for _, row in deaths_df.iterrows():
            unit_id   = row['unit_id']
            birth_sha = row['birth_sha']
            death_sha = row['death_sha']

            # Parse filepath and line number from unit_id
            # Format: "filepath::line_number"
            if '::' not in str(unit_id):
                _skip(row, 'invalid_unit_id')
                continue

            parts = str(unit_id).rsplit('::', 1)
            filepath = parts[0]
            try:
                line_number = int(parts[1])
            except (ValueError, IndexError):
                _skip(row, 'invalid_unit_id')
                continue

            # Skip if death_sha is missing
            if pd.isna(death_sha) or not death_sha:
                _skip(row, 'missing_death_sha')
                continue

            # --- SHORT-CIRCUIT: file deletions are unambiguous structural deaths ---
            # When a file is deleted, the line is gone entirely — this is always
            # a structural death by definition. No AST comparison needed.
            # (Attempting to fetch a deleted file produces fetch_failed_death.)
            death_magnitude = str(row.get('death_magnitude', ''))
            if death_magnitude == 'deletion':
                results.append({
                    'slug':                 slug,
                    'pr_number':            int(row['pr_number']),
                    'unit_id':              unit_id,
                    'ast_death':            1,
                    'ast_enclosing_node':   None,
                    'ast_exclusion_reason': None,
                })
                continue

            # Determine language
            language, exclusion_reason = get_language_for_file(filepath)
            if language is None:
                _skip(row, exclusion_reason)
                continue

            # Fetch file contents (with caching)
            birth_key = (birth_sha, filepath)
            death_key = (death_sha, filepath)

            if birth_key not in file_cache:
                file_cache[birth_key] = fetch_file_at_commit(
                    temp_dir, filepath, birth_sha
                )
            if death_key not in file_cache:
                file_cache[death_key] = fetch_file_at_commit(
                    temp_dir, filepath, death_sha
                )

            birth_content = file_cache[birth_key]
            death_content = file_cache[death_key]

            if birth_content is None:
                _skip(row, 'fetch_failed_birth')
                continue

            if death_content is None:
                _skip(row, 'fetch_failed_death')
                continue

            # Classify AST death
            ast_death, ast_node, ast_reason = classify_ast_death(
                birth_content, death_content, line_number, language
            )

            results.append({
                'slug':                 slug,
                'pr_number':            int(row['pr_number']),
                'unit_id':              unit_id,
                'ast_death':            ast_death,
                'ast_enclosing_node':   ast_node,
                'ast_exclusion_reason': ast_reason,
            })

    except subprocess.TimeoutExpired:
        for _, row in deaths_df.iterrows():
            _skip(row, 'clone_timeout')
    except Exception as e:
        for _, row in deaths_df.iterrows():
            _skip(row, f'error: {type(e).__name__}')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AST-level survival classification'
    )
    parser.add_argument(
        '--repos-override',
        type=Path,
        default=None,
        help='CSV file with slug column — restrict processing to these repos only'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path("data/survival_events_line_v4.csv"),
        help='Input survival CSV'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("data/survival_events_line_v4_ast.csv"),
        help='Output CSV path'
    )
    args = parser.parse_args()
    input_file  = args.input
    output_file = args.output

    print("=" * 70)
    print("AST SURVIVAL CLASSIFICATION (v1)")
    print("=" * 70)
    print(f"Input  : {input_file}")
    print(f"Output : {output_file}")
    print(f"Workers: {MAX_WORKERS}")
    print()

    if not input_file.exists():
        print(f"ERROR: {input_file} not found. Run measure_retention_line_v2.py first.")
        return

    df = pd.read_csv(input_file, low_memory=False)
    # Deduplicate source data — duplicates arise from parallel worker overlap
    before = len(df)
    df = df.drop_duplicates(subset=['slug', 'pr_number', 'unit_id'], keep='first')
    if len(df) < before:
        print(f"Removed {before - len(df)} duplicate rows from source data")
    print(f"Total rows loaded : {len(df):,}")

    if args.repos_override and args.repos_override.exists():
        override_slugs = set(
            pd.read_csv(args.repos_override)['slug'].unique()
        )
        df = df[df['slug'].isin(override_slugs)]
        print(f"Repos override active: {len(override_slugs)} repos")

    # Separate deaths from survivors
    deaths    = df[df['event'] == 1].copy()
    survivors = df[df['event'] == 0].copy()

    print(f"Death events      : {len(deaths):,}")
    print(f"Survived (censored): {len(survivors):,}")
    print()

    # Add placeholder columns to survivors — not applicable
    survivors = survivors.copy()
    survivors['ast_death']            = None
    survivors['ast_enclosing_node']   = None
    survivors['ast_exclusion_reason'] = 'survived'

    # Process deaths by repository
    repo_groups = list(deaths.groupby('slug'))
    print(f"Repositories to process: {len(repo_groups)}")
    print()

    all_ast_results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_repo, slug, group): slug
            for slug, group in repo_groups
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="AST classification"
        ):
            result = future.result()
            if result:
                all_ast_results.extend(result)

    # Build AST results DataFrame
    ast_df = pd.DataFrame(all_ast_results)
    print(f"\nAST results computed: {len(ast_df):,}")

    # Merge AST results back into deaths DataFrame
    deaths = deaths.merge(
        ast_df[['slug', 'pr_number', 'unit_id', 'ast_death',
                'ast_enclosing_node', 'ast_exclusion_reason']],
        on=['slug', 'pr_number', 'unit_id'],
        how='left'
    )

    # Fill any unmatched deaths (shouldn't happen, but safety net)
    deaths['ast_enclosing_node']   = deaths['ast_enclosing_node'].where(
        deaths['ast_enclosing_node'].notna(), other=pd.NA
    )
    deaths['ast_exclusion_reason'] = deaths['ast_exclusion_reason'].fillna(
        'not_processed'
    )

    # Align columns before concat to avoid NA-column dtype issues
    for col in ['ast_death', 'ast_enclosing_node', 'ast_exclusion_reason']:
        if col not in survivors.columns:
            survivors[col] = pd.NA
        if col not in deaths.columns:
            deaths[col] = pd.NA

    final_df = pd.concat([deaths, survivors], ignore_index=True)
    final_df = final_df.sort_values(['slug', 'pr_number', 'unit_id']).reset_index(drop=True)

    # Save output
    final_df.to_csv(output_file, index=False)

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total rows in output : {len(final_df):,}")
    print()

    # AST classification coverage
    death_rows = final_df[final_df['event'] == 1]
    classified = death_rows[death_rows['ast_death'].notna()]
    print(f"Deaths classified    : {len(classified):,} "
          f"({100*len(classified)/len(death_rows):.1f}%)")
    print()

    # AST death breakdown
    print("AST death classification (classified deaths only):")
    ast_dead  = int((classified['ast_death'] == 1).sum())
    ast_alive = int((classified['ast_death'] == 0).sum())
    print(f"  Structural change (ast_death=1) : {ast_dead:,} "
          f"({100*ast_dead/len(classified):.1f}%)")
    print(f"  Leaf-only change  (ast_death=0) : {ast_alive:,} "
          f"({100*ast_alive/len(classified):.1f}%)")
    print()

    # Exclusion reasons
    print("Exclusion reasons (unclassified deaths):")
    unclassified = death_rows[death_rows['ast_death'].isna()]
    reason_counts = unclassified['ast_exclusion_reason'].value_counts()
    for reason, count in reason_counts.items():
        pct = 100 * count / len(death_rows)
        print(f"  {reason:<35}: {count:,} ({pct:.1f}%)")
    print()

    # Enclosing node type distribution
    print("Enclosing node types (classified deaths):")
    node_counts = classified['ast_enclosing_node'].value_counts().head(15)
    for node_type, count in node_counts.items():
        print(f"  {node_type:<40}: {count:,}")
    print()

    # AST death rate by agent tier
    print("AST death rate by agent tier (classified deaths only):")
    for tier, grp in classified.groupby('agent_tier'):
        ast_dr = grp['ast_death'].mean() * 100
        print(f"  {tier:<20}: {len(grp):>8,} lines, "
              f"{ast_dr:.1f}% AST death rate")

    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()