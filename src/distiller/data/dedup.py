"""Exact-duplicate removal, label-conflict resolution and near-duplicate grouping.

Exact duplicates are dropped. Inputs that appear with different outputs are label
noise: we keep the majority output and count the conflict. Near-duplicates are kept
(they are natural variants of real traffic) but grouped, so the splitter can keep a
whole group in one split and the test set never contains near-copies of training rows.
"""

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from distiller.data.io import Example, serialize_output

_WS = re.compile(r"\s+")
_PRIME = (1 << 31) - 1


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return _WS.sub(" ", text).strip()


@dataclass
class DedupResult:
    examples: list[Example]
    exact_duplicates: int
    conflicts: int  # distinct inputs that had more than one output
    conflict_samples: list[str]


def remove_exact_duplicates(examples: list[Example]) -> DedupResult:
    by_input: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_input[normalize(ex.input)].append(ex)

    kept, exact, conflicts, samples = [], 0, 0, []
    for group in by_input.values():
        outputs = Counter(serialize_output(ex.output) for ex in group)
        if len(outputs) > 1:
            conflicts += 1
            if len(samples) < 5:
                samples.append(f"{group[0].input[:80]!r} -> {dict(outputs)}")
        majority = outputs.most_common(1)[0][0]
        winner = next(ex for ex in group if serialize_output(ex.output) == majority)
        kept.append(winner)
        exact += len(group) - 1
    return DedupResult(kept, exact, conflicts, samples)


def _shingles(text: str, k: int = 5) -> set[str]:
    text = normalize(text)
    if len(text) <= k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _stable_hash(s: str) -> int:
    # Python's hash() is salted per process; use a fixed FNV-1a so runs are reproducible.
    h = 0xCBF29CE484222325
    for b in s.encode():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h % _PRIME


def near_duplicate_groups(
    examples: list[Example], threshold: float, num_perm: int = 64, bands: int = 16, seed: int = 0
) -> list[int]:
    """MinHash + LSH over character shingles. Returns a group id per example."""
    n = len(examples)
    rows = num_perm // bands
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, _PRIME, size=num_perm, dtype=np.uint64)

    shingle_sets = [_shingles(ex.input) for ex in examples]
    signatures = np.empty((n, num_perm), dtype=np.uint64)
    for i, sh in enumerate(shingle_sets):
        h = np.fromiter((_stable_hash(s) for s in sh), dtype=np.uint64, count=len(sh))
        signatures[i] = ((a[:, None] * h[None, :] + b[:, None]) % _PRIME).min(axis=1)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    checked: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        block = signatures[:, band * rows : (band + 1) * rows]
        for i in range(n):
            buckets[block[i].tobytes()].append(i)
        for members in buckets.values():
            for j in members[1:]:
                i = members[0]
                pair = (i, j)
                if pair in checked or find(i) == find(j):
                    continue
                checked.add(pair)
                s1, s2 = shingle_sets[i], shingle_sets[j]
                if len(s1 & s2) / len(s1 | s2) >= threshold:
                    parent[find(j)] = find(i)

    roots: dict[int, int] = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]
