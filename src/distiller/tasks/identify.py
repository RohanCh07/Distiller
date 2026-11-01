"""Header first; otherwise a configured system-prompt match; otherwise "unassigned".

Unassigned calls are still logged (with a short hash of their system prompt), so
repetitive untagged traffic can be found and turned into a task later.
"""

import hashlib

from distiller.gateway.config import GatewayConfig

HEADER = "x-distiller-task"


def system_prompt(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")


def identify(headers, messages: list[dict], config: GatewayConfig) -> str:
    tagged = headers.get(HEADER)
    if tagged:
        return tagged.strip()
    prompt = system_prompt(messages)
    for name, task in config.tasks.items():
        if task.match_system_prompt and task.match_system_prompt in prompt:
            return name
    if not prompt:
        return "unassigned"
    return "unassigned:" + hashlib.blake2b(prompt.encode(), digest_size=4).hexdigest()
