"""Remove personal data from text before it is stored.

Each kind is a regex plus, where one exists, a checksum (Luhn for cards, Verhoeff for
Aadhaar, mod-97 for IBAN), so long reference numbers are not mistaken for card numbers.
Matches are replaced by a placeholder such as <EMAIL>, which keeps the sentence shape
for training while dropping the value.

UPI handles (name@bank) are not scrubbed: they usually name the merchant, which is
exactly what tasks like transaction extraction need.
"""

import re
from collections.abc import Callable, Iterable

ALL_KINDS = ("email", "card", "aadhaar", "phone", "pan", "iban", "ip")


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in _digits(number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_ok(number: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(_digits(number))):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def iban_ok(value: str) -> bool:
    s = value.replace(" ", "").upper()
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    return int("".join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1


# (kind, pattern, optional validator). Order matters: longer digit runs first.
_RULES: list[tuple[str, re.Pattern, Callable[[str], bool] | None]] = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), None),
    ("card", re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])"), luhn_ok),
    ("aadhaar", re.compile(r"(?<!\w)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\w)"), verhoeff_ok),
    (
        "phone",
        re.compile(
            r"(?<![\w+])(?:\+?91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\w)"  # Indian mobile
            r"|(?<![\w+])\+\d{1,3}[ -]?\(?\d{1,4}\)?(?:[ -]?\d){6,10}(?!\w)"  # international
        ),
        None,
    ),
    ("pan", re.compile(r"(?<!\w)[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z](?!\w)"), None),
    ("iban", re.compile(r"(?<!\w)[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}(?!\w)"), iban_ok),
    (
        "ip",
        re.compile(
            r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"
        ),
        None,
    ),
]


class Scrubber:
    def __init__(self, kinds: Iterable[str] = ALL_KINDS):
        kinds = set(kinds)
        unknown = kinds - set(ALL_KINDS)
        if unknown:
            raise ValueError(f"unknown PII kinds {sorted(unknown)}; choose from {ALL_KINDS}")
        self.rules = [r for r in _RULES if r[0] in kinds]

    def scrub(self, text: str | None) -> str | None:
        if not text:
            return text
        for kind, pattern, check in self.rules:
            placeholder = f"<{kind.upper()}>"

            def repl(m: re.Match, check=check, placeholder=placeholder) -> str:
                return placeholder if check is None or check(m.group(0)) else m.group(0)

            text = pattern.sub(repl, text)
        return text

    def scrub_messages(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            m = dict(m)
            if isinstance(m.get("content"), str):
                m["content"] = self.scrub(m["content"])
            elif isinstance(m.get("content"), list):  # content parts
                m["content"] = [
                    {**p, "text": self.scrub(p["text"])}
                    if isinstance(p, dict) and "text" in p
                    else p
                    for p in m["content"]
                ]
            out.append(m)
        return out
