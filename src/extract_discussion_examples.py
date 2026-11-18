"""
extract_discussion_examples.py

Pulls concrete, real examples from the dataset to illustrate the three
measurement traps described in the Discussion section:

  Trap 1 (Aggregation): Show the within-tier modification rate spread vs
          the pooled result -- illustrates why aggregation obscures heterogeneity.

  Trap 2a (Operationalization -- Definition): Find commits that are
          syntactically modified (Def 1 event=1) but NOT structurally
          modified (ast_death=0) -- i.e., leaf-only changes. These are
          real examples of what Def 2 filters out.

  Trap 2b (Operationalization -- Intent): Find commits where keyword
          classifier says 'corrective' but all three LLMs (DeepSeek,
          gpt-oss, MiniMax) agree on 'adaptive' or 'perfective'. These
          concretely show the priority-rule failure.

  Trap 3 (Confounding): Show repos where a tier's modification rate
          looks very different in pooled vs within-repo comparison.
          Pull the most extreme examples.

Usage:
    python src/extract_discussion_examples.py
"""

import re
import pandas as pd
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import P

OUTLIER_PR_KEYS = {('microsoft/typespec', 7115)}
GENERATED_PATH_PATTERNS = [
    r'/dist/', r'/third_party/', r'/vendor/', r'/generated/',
    r'_pb2\.py$', r'\.min\.js$', r'immutable_entry',
    r'_generated\.', r'zz_generated\.',
]


def is_generated(unit_id: str) -> bool:
    filepath = str(unit_id).split('::')[0]
    return any(re.search(p, filepath) for p in GENERATED_PATH_PATTERNS)


def main():
    print("Loading data...")
    line = pd.read_csv(P('data/survival_events_line_v5_analysis.csv'))
    intent_kw = pd.read_csv(P('data/death_commit_messages_v5.csv'))
    llm = intent_kw[['slug', 'sha']].copy()
    for m in ['deepseek', 'gpt_oss', 'minimax']:
        run = pd.read_csv(P(f'results/intent_{m}_full_corpus_t0.csv'))[['slug', 'sha', 'label']]
        llm = llm.merge(run.rename(columns={'label': m}), on=['slug', 'sha'], how='left')
    # unanimous LLM label, else NaN
    unanimous = (llm['deepseek'] == llm['gpt_oss']) & (llm['gpt_oss'] == llm['minimax'])
    intent_llm = llm.assign(label=llm['deepseek'].where(unanimous))[['slug', 'sha', 'label']]

    # ── TRAP 2a: Leaf-only changes (Def 1 event but NOT Def 2 event) ─
    print("\n" + "="*70)
    print("TRAP 2a: LEAF-ONLY CHANGES (Def 1 modified, Def 2 retained)")
    print("="*70)
    print("These are real lines that were syntactically modified but the")
    print("enclosing AST node did NOT change structurally.\n")

    leaf_only = line[
        (line['event'] == 1) &
        (line['ast_death'] == 0.0) &
        (line['birth_content'].notna()) &
        (line['death_content'].notna())
    ].copy()

    print(f"Total leaf-only modification events: {len(leaf_only):,}")
    print(f"Breakdown by tier:")
    print(leaf_only.groupby('agent_tier').size().to_string())
    print()

    # Sample 5 interesting examples (short lines, visible diff)
    leaf_only['birth_len'] = leaf_only['birth_content'].str.len()
    leaf_only['death_len'] = leaf_only['death_content'].str.len()
    candidates = leaf_only[
        (leaf_only['birth_len'] < 100) &
        (leaf_only['death_len'] < 100) &
        (leaf_only['birth_len'] > 5) &
        (leaf_only['birth_content'] != leaf_only['death_content'])
    ].copy()

    print(f"Short leaf-only examples (birth + death lines shown):")
    sample = candidates.groupby('agent_tier').head(2)
    for _, row in sample.iterrows():
        print(f"\n  Repo: {row['slug']}")
        print(f"  Tier: {row['agent_tier']}")
        print(f"  AST node: {row['ast_enclosing_node']}")
        print(f"  Birth: {repr(str(row['birth_content']).strip())}")
        print(f"  Death: {repr(str(row['death_content']).strip())}")

    # ── TRAP 2b: Keyword=corrective but all LLMs=adaptive/perfective ────
    print("\n" + "="*70)
    print("TRAP 2b: KEYWORD says CORRECTIVE but all three LLMs disagree")
    print("="*70)
    print("These commits concretely show the priority-rule failure mode.\n")

    kw_corr = intent_kw[intent_kw['intent'] == 'corrective'][['slug','sha','message']].copy()
    llm_non_corr = intent_llm[
        intent_llm['label'].isin(['adaptive','perfective'])
    ][['slug','sha','label']].copy()
    llm_non_corr.columns = ['slug','sha','llm_label']

    disagreements = kw_corr.merge(llm_non_corr, on=['slug','sha'], how='inner')
    print(f"Keyword=corrective, LLMs=adaptive/perfective: {len(disagreements)} commits")
    print()

    # Show 8 examples with full commit messages (truncated to 300 chars)
    for _, row in disagreements.sample(min(8, len(disagreements)),
                                        random_state=42).iterrows():
        msg = str(row['message']).split('\n')[0][:200]
        print(f"  [{row['slug']}]")
        print(f"  Keyword: corrective | LLMs: {row['llm_label']}")
        print(f"  Message: {msg}")
        print()

    # ── TRAP 2b reverse: Keyword=adaptive/perfective but all LLMs=corrective
    print("="*70)
    print("TRAP 2b (reverse): KEYWORD says ADAPTIVE/PERFECTIVE but all three LLMs say CORRECTIVE")
    print("="*70)
    kw_non_corr = intent_kw[
        intent_kw['intent'].isin(['adaptive','perfective'])
    ][['slug','sha','message','intent']].copy()
    kw_non_corr.columns = ['slug','sha','message','kw_label']

    llm_corr = intent_llm[
        intent_llm['label'] == 'corrective'
    ][['slug','sha']].copy()

    reverse_disagree = kw_non_corr.merge(llm_corr, on=['slug','sha'], how='inner')
    print(f"Keyword=adaptive/perfective, LLMs=corrective: {len(reverse_disagree)} commits")
    print()
    for _, row in reverse_disagree.sample(min(5, len(reverse_disagree)),
                                           random_state=42).iterrows():
        msg = str(row['message']).split('\n')[0][:200]
        print(f"  [{row['slug']}]")
        print(f"  Keyword: {row['kw_label']} | LLMs: corrective")
        print(f"  Message: {msg}")
        print()

    # ── TRAP 3: Confounding -- repos with extreme pooled vs within-repo diff
    print("="*70)
    print("TRAP 3: CONFOUNDING -- Repos where tier modification rate looks")
    print("        very different from the overall pooled rate for that tier")
    print("="*70)
    print("These repos show WHY repo confounding matters.\n")

    for tier in ['pair_programmer', 'batch_generator', 'autonomous']:
        tier_lines = line[line['agent_tier'] == tier]
        human_lines = line[line['is_agent'] == False]
        pooled_rate = tier_lines['event'].mean()

        # Per-repo: tier mod rate vs human mod rate in same repo
        per_repo = []
        for repo in tier_lines['slug'].unique():
            t = tier_lines[tier_lines['slug'] == repo]
            h = human_lines[human_lines['slug'] == repo]
            if len(t) < 20 or len(h) < 20:
                continue
            per_repo.append({
                'slug': repo,
                'n_tier': len(t),
                'n_human': len(h),
                'tier_rate': t['event'].mean(),
                'human_rate': h['event'].mean(),
                'gap': t['event'].mean() - h['event'].mean(),
            })

        if not per_repo:
            continue
        repo_df = pd.DataFrame(per_repo).sort_values('gap')

        print(f"\n{tier} (pooled modification rate: {pooled_rate:.1%}):")
        print(f"  Repos where tier is MUCH LESS modified than human in same repo:")
        print(repo_df.head(3)[['slug','n_tier','n_human',
                                'tier_rate','human_rate','gap']].to_string(index=False))
        print(f"  Repos where tier is MUCH MORE modified than human in same repo:")
        print(repo_df.tail(3)[['slug','n_tier','n_human',
                                'tier_rate','human_rate','gap']].to_string(index=False))


if __name__ == '__main__':
    main()