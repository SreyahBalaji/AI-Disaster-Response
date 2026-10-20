
"""
RESPONDA — Phase 29
Local approval access control.

Protects simulated approval and reset endpoints.

The approval passphrase is provided through an
environment variable, never hardcoded in source.

Prototype-only security:
- No per-user identity
- No role-based access control
- No production authentication system
"""

import os
import secrets

from fastapi import Header, HTTPException


ENV_NAME = "RESPONDA_APPROVAL_KEY"


def require_approval_key(
    x_approval_key: str | None = Header(
        default=None,
        alias="X-Approval-Key",
    ),
):
    configured_key = os.getenv(ENV_NAME, "")

    # Fail closed if authorization has not
    # been configured correctly.
    if len(configured_key) < 16:
        raise HTTPException(
            status_code=503,
            detail=(
                "Approval authorization is not configured "
                "on this server."
            ),
        )

    if not x_approval_key:
        raise HTTPException(
            status_code=401,
            detail="Approval authorization is required",
        )

    if not secrets.compare_digest(
        x_approval_key,
        configured_key,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid approval authorization",
        )

    return None
