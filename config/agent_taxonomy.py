"""
Agent taxonomy for survival analysis.

Tier definitions based on human-AI collaboration mode:
- autonomous: minimal human oversight, end-to-end task execution
- pair_programmer: human retains authorship identity, inline suggestion model  
- batch_generator: programmatic, variable human involvement
"""

AGENT_TIERS = {
    'autonomous': ['Devin'],
    'pair_programmer': ['GitHub Copilot', 'Cursor', 'Claude Code'],
    'batch_generator': ['OpenAI Codex'],
}

# Reverse lookup: agent name -> tier
AGENT_TO_TIER = {
    agent: tier
    for tier, agents in AGENT_TIERS.items()
    for agent in agents
}

# Display order for tables and figures
TIER_ORDER = ['pair_programmer', 'batch_generator', 'autonomous']

# Human baseline label
HUMAN_LABEL = 'Human'

AGENT_KEYWORDS = {
    'copilot': 'GitHub Copilot',
    'devin': 'Devin',
    'cursor': 'Cursor',
    'codex': 'OpenAI Codex',
    'claude': 'Claude Code',
}