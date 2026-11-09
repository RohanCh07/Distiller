"""Encryption at rest for logged request content.

Text columns that hold customer content (input, request, teacher/student output) are
encrypted with Fernet (AES-128-CBC + HMAC-SHA256) before they reach the database. Keys
come from an environment variable as a comma-separated list: the first key encrypts,
all keys decrypt, so keys can be rotated without losing old logs (`distiller keys
rotate-encryption` re-encrypts old rows with the first key).

Values written before encryption was turned on have no prefix and are read as-is.
"""

PREFIX = "enc:v1:"


class FieldCipher:
    def __init__(self, keys: list[str]):
        try:
            from cryptography.fernet import Fernet, MultiFernet
        except ImportError as e:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "Encryption at rest needs the 'cryptography' package: pip install 'distiller[security]'"
            ) from e
        keys = [k.strip() for k in keys if k.strip()]
        if not keys:
            raise ValueError("no encryption key given")
        self._fernet = MultiFernet([Fernet(k.encode()) for k in keys])

    @classmethod
    def from_env_value(cls, value: str) -> "FieldCipher":
        return cls(value.split(","))

    @staticmethod
    def generate_key() -> str:
        from cryptography.fernet import Fernet

        return Fernet.generate_key().decode()

    def encrypt(self, value: str | None) -> str | None:
        if value is None or value.startswith(PREFIX):
            return value
        return PREFIX + self._fernet.encrypt(value.encode()).decode()

    def decrypt(self, value: str | None) -> str | None:
        if value is None or not value.startswith(PREFIX):
            return value
        return self._fernet.decrypt(value[len(PREFIX) :].encode()).decode()

    def rotate(self, value: str | None) -> str | None:
        """Re-encrypt with the current (first) key."""
        if value is None or not value.startswith(PREFIX):
            return self.encrypt(value)
        return PREFIX + self._fernet.rotate(value[len(PREFIX) :].encode()).decode()
