"""Permission broker: default-deny gate for anything with side effects.

AG has broad *capability* but narrow *default authority*. Any external or
irreversible action (Chrome/browser control, network calls, file deletion, sending
messages, spawning many sub-agents) must pass through here and is denied unless the
user has explicitly granted it for this session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set


class PermissionError_(RuntimeError):
    """Raised when a gated capability is used without a grant."""


# Capabilities that are ALWAYS gated regardless of config.
GATED = {
    "browser",        # Chrome / web automation
    "network",        # arbitrary outbound HTTP (the model API itself is exempt)
    "filesystem_write_outside_repo",
    "delete",         # any destructive deletion
    "send_message",   # email / chat / any outbound comms
    "spawn_agent",    # creating sub-agents
    "shell",          # running arbitrary shell commands
}


@dataclass
class PermissionBroker:
    allow_external_tools: bool = False
    grants: Set[str] = field(default_factory=set)
    _audit: list = field(default_factory=list)

    def grant(self, capability: str) -> None:
        self.grants.add(capability)
        self._audit.append(("grant", capability))

    def revoke(self, capability: str) -> None:
        self.grants.discard(capability)
        self._audit.append(("revoke", capability))

    def check(self, capability: str) -> bool:
        allowed = capability in self.grants and (
            self.allow_external_tools or capability not in GATED
        )
        self._audit.append(("check", capability, allowed))
        return allowed

    def require(self, capability: str) -> None:
        if not self.check(capability):
            raise PermissionError_(
                f"Capability '{capability}' is not granted. "
                f"Enable allow_external_tools and grant it explicitly to proceed."
            )

    @property
    def audit_log(self):
        return list(self._audit)
