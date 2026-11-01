import pytest

from distiller.security.keys import PREFIX, generate_key, hash_key, key_from_headers
from distiller.security.pii import Scrubber, iban_ok, luhn_ok, verhoeff_ok
from distiller.storage.db import Store

# ---------- PII ----------


def test_checksums():
    assert luhn_ok("4111 1111 1111 1111") and not luhn_ok("4111 1111 1111 1112")
    assert verhoeff_ok("234123412346") and not verhoeff_ok("234123412345")
    assert iban_ok("GB82 WEST 1234 5698 7654 32") and not iban_ok("GB82 WEST 1234 5698 7654 33")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("mail me at ravi.k+bank@example.co.in now", "mail me at <EMAIL> now"),
        ("card 4111-1111-1111-1111 declined", "card <CARD> declined"),
        ("call +91 98765 43210 or 09876543210", "call <PHONE> or <PHONE>"),
        ("ring +44 20 7946 0958 please", "ring <PHONE> please"),
        ("aadhaar 2341 2341 2346 linked", "aadhaar <AADHAAR> linked"),
        ("PAN ABCPE1234F on file", "PAN <PAN> on file"),
        ("IBAN GB82WEST12345698765432 ok", "IBAN <IBAN> ok"),
        ("from 192.168.1.20 login", "from <IP> login"),
    ],
)
def test_scrubs_each_kind(text, expected):
    assert Scrubber().scrub(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Rs.450.00 debited to VPA swiggy@ybl (UPI Ref 628811234567)",  # ref fails Verhoeff/Luhn
        "Order 1234567890123 shipped",  # 13 digits, fails Luhn
        "Rs 1,299 paid on 12-10-26",
        "version 1.2.3 released",
    ],
)
def test_keeps_non_pii(text):
    assert Scrubber().scrub(text) == text


def test_kinds_are_configurable():
    s = Scrubber(["email"])
    assert s.scrub("a@b.com 9876543210") == "<EMAIL> 9876543210"
    with pytest.raises(ValueError):
        Scrubber(["dna"])


def test_scrub_messages_and_parts():
    msgs = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": [{"type": "text", "text": "x@y.com"}, {"type": "image"}]},
        {"role": "user", "content": "phone 9876543210"},
    ]
    out = Scrubber().scrub_messages(msgs)
    assert out[1]["content"][0]["text"] == "<EMAIL>" and out[1]["content"][1] == {"type": "image"}
    assert out[2]["content"] == "phone <PHONE>"
    assert msgs[2]["content"] == "phone 9876543210"  # input not mutated
    assert Scrubber().scrub(None) is None


# ---------- API keys ----------


def test_generate_and_hash():
    k = generate_key()
    assert k.startswith(PREFIX) and len(k) > 40 and generate_key() != k
    assert hash_key(k) == hash_key(k) and len(hash_key(k)) == 64


def test_key_from_headers():
    assert key_from_headers({"x-distiller-key": " dk-a "}) == ("dk-a", False)
    assert key_from_headers({"authorization": "Bearer dk-b"}) == ("dk-b", True)
    assert key_from_headers({"authorization": "Bearer sk-openai"}) == (None, False)
    assert key_from_headers({}, {"key": "dk-c"}) == ("dk-c", False)
    assert key_from_headers({}, None) == (None, False)


def test_store_keys(tmp_path):
    store = Store(tmp_path / "d.db")
    assert not store.has_admin_key()
    store.add_api_key("app1", hash_key("dk-1"), "app")
    store.add_api_key("boss", hash_key("dk-2"), "admin")
    assert store.find_api_key(hash_key("dk-1"))["name"] == "app1"
    assert store.has_admin_key()
    assert store.revoke_api_key("boss") and not store.revoke_api_key("boss")
    assert not store.has_admin_key() and store.find_api_key(hash_key("dk-2")) is None
    assert [k["name"] for k in store.list_api_keys()] == ["app1", "boss"]
    assert "key_hash" not in store.list_api_keys()[0]
