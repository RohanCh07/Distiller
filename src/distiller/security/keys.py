"""Gateway API keys.

Keys look like `dk-<random>` and are shown once, when created. Only a SHA-256 hash is
stored. Apps can send their key either as `X-Distiller-Key: dk-...` or as the normal
`Authorization: Bearer dk-...` (so an OpenAI SDK only needs OPENAI_API_KEY=dk-...). In
the second case the gateway uses its own configured teacher key upstream.
"""

import hashlib
import hmac
import secrets

PREFIX = "dk-"
ROLES = ("app", "admin")


def generate_key() -> str:
    return PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def same(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def key_from_headers(headers, query_params=None) -> tuple[str | None, bool]:
    """(key, came_from_authorization). The dashboard may also pass ?key=."""
    key = headers.get("x-distiller-key")
    if key:
        return key.strip(), False
    auth = headers.get("authorization") or ""
    if auth.lower().startswith("bearer ") and auth[7:].strip().startswith(PREFIX):
        return auth[7:].strip(), True
    if query_params is not None and query_params.get("key"):
        return query_params["key"], False
    return None, False
