"""Group-aware, label-stratified train/val/test split."""

import random
from collections import Counter, defaultdict

from distiller.data.io import Example, serialize_output


def split_examples(
    examples: list[Example],
    groups: list[int],
    val_fraction: float,
    test_fraction: float,
    seed: int,
    stratify: bool,
) -> tuple[list[Example], list[Example], list[Example]]:
    """Every near-duplicate group lands in exactly one split.

    With stratify=True, groups are bucketed by their majority label and each label is
    split separately, so rare labels still show up in val/test in proportion.
    """
    rng = random.Random(seed)
    by_group: dict[int, list[Example]] = defaultdict(list)
    for ex, g in zip(examples, groups):
        by_group[g].append(ex)

    strata: dict[str, list[list[Example]]] = defaultdict(list)
    for members in by_group.values():
        key = (
            Counter(serialize_output(ex.output) for ex in members).most_common(1)[0][0]
            if stratify
            else "_all"
        )
        strata[key].append(members)

    train, val, test = [], [], []
    for key in sorted(strata):
        group_list = strata[key]
        rng.shuffle(group_list)
        total = sum(len(g) for g in group_list)
        want_test, want_val = round(total * test_fraction), round(total * val_fraction)
        n_test = n_val = 0
        for i, members in enumerate(group_list):
            # Always keep at least the first group for training so every label is learnable.
            if i > 0 and n_test < want_test:
                test.extend(members)
                n_test += len(members)
            elif i > 0 and n_val < want_val:
                val.extend(members)
                n_val += len(members)
            else:
                train.extend(members)
    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test
