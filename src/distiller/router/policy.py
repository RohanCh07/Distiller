"""Routing decisions and drift protection."""

import random

from distiller.gateway.config import GatewayConfig, Mode
from distiller.storage.db import Store


def should_audit(config: GatewayConfig) -> bool:
    """Send a small random share of student-served requests to the teacher too."""
    return random.random() < config.audit_rate


def check_drift(store: Store, config: GatewayConfig, task: str) -> bool:
    """Demote serve -> shadow when audited agreement falls below the floor. Returns True if demoted."""
    samples, agreement = store.recent_agreement(task, config.demote.window)
    if samples >= config.demote.min_samples and agreement < config.demote.min_agreement:
        store.set_mode(
            task,
            Mode.shadow.value,
            reason=f"auto-demoted: audit agreement {agreement:.1%} over {samples} requests",
        )
        return True
    return False
