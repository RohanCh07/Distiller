import json

import pytest

pytest.importorskip("cryptography")

from distiller.security.crypto import FieldCipher  # noqa: E402
from distiller.security.pii import Scrubber  # noqa: E402
from distiller.storage.db import Store  # noqa: E402


def test_encrypt_roundtrip_and_passthrough():
    c = FieldCipher([FieldCipher.generate_key()])
    token = c.encrypt("secret text")
    assert token.startswith("enc:v1:") and "secret" not in token
    assert c.decrypt(token) == "secret text"
    assert c.encrypt(token) == token  # never double-encrypts
    assert c.decrypt("legacy plaintext") == "legacy plaintext"
    assert c.encrypt(None) is None


def test_rotation_with_multiple_keys():
    old, new = FieldCipher.generate_key(), FieldCipher.generate_key()
    token = FieldCipher([old]).encrypt("x")
    both = FieldCipher([new, old])
    assert both.decrypt(token) == "x"
    rotated = both.rotate(token)
    assert FieldCipher([new]).decrypt(rotated) == "x"
    with pytest.raises(ValueError):
        FieldCipher([" "])


def test_store_encrypts_and_scrubs_content(tmp_path):
    key = FieldCipher.generate_key()
    store = Store(tmp_path / "d.db", cipher=FieldCipher([key]), scrubber=Scrubber())
    msgs = [{"role": "user", "content": "my email is a@b.com"}]
    cid = store.log_call(
        task="t",
        mode="observe",
        route="teacher",
        input="my email is a@b.com",
        request=msgs,
        teacher_output="reply to a@b.com",
    )
    raw = store.query("SELECT input, request, teacher_output FROM calls")[0]
    assert all(v.startswith("enc:v1:") for v in raw.values())
    assert "a@b.com" not in json.dumps(raw)
    row = store.get_call(cid)
    assert row["input"] == "my email is <EMAIL>"
    assert json.loads(row["request"])[0]["content"] == "my email is <EMAIL>"
    assert list(store.export_training_rows("t"))[0]["messages"][-1]["content"] == "reply to <EMAIL>"
    store.update_call(cid, student_output="b@c.org")
    assert store.get_call(cid)["student_output"] == "<EMAIL>"
    # Rotate to a new key; old key can be dropped afterwards.
    new = FieldCipher.generate_key()
    store.cipher = FieldCipher([new, key])
    assert store.rotate_encryption() == 1
    store.cipher = FieldCipher([new])
    assert store.get_call(cid)["input"] == "my email is <EMAIL>"


def test_store_without_key_cannot_read_encrypted(tmp_path):
    Store(tmp_path / "d.db", cipher=FieldCipher([FieldCipher.generate_key()])).log_call(
        task="t", mode="observe", route="teacher", input="hello"
    )
    assert Store(tmp_path / "d.db").get_call(1)["input"].startswith("enc:v1:")
