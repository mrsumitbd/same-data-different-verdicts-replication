"""
Agent / bot login classification shared by the modifier analysis
(src/fetch_modifier_authors.py) and the bot exclusion
(src/analyze_rq1_bot_excluded.py).

A login counts as an agent or bot when it
  - is a GitHub App account (ends with "[bot]"),
  - ends with "-bot" (bot accounts without the App suffix), or
  - is one of the known coding-agent or automation accounts below.

Substring matching on vendor names is deliberately not used: it labels
human staff accounts such as "rm-openai" or "jakobbotsch" as agents.
"""

KNOWN_AGENT_LOGINS = {
    # coding agents
    'copilot', 'copilot-swe-agent', 'cursoragent', 'devin-ai-integration',
    'codex', 'chatgpt-codex-connector', 'claude', 'google-labs-jules',
    'openhands-agent', 'sweep-ai',
    # automation accounts without a bot suffix (audited in the cohort data)
    'mlflow-automation', 'cometactions', 'modularbot', 'ckanbot',
    'syliusbot', 'repobirdbot', 'robot-ci-heartex',
}


def is_agent_login(login) -> bool:
    if not isinstance(login, str) or not login.strip():
        return False
    name = login.strip().lower()
    return (name.endswith('[bot]') or name.endswith('-bot')
            or name in KNOWN_AGENT_LOGINS)
