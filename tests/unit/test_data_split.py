from collections import Counter

from distiller.data.io import Example
from distiller.data.split import split_examples


def _data(n_labels=4, per_label=50, group_size=1):
    examples, groups = [], []
    g = 0
    for li in range(n_labels):
        for i in range(per_label):
            examples.append(Example(f"{li}-{i}", f"text {li} {i}", f"label{li}"))
            groups.append(g)
            if (i + 1) % group_size == 0:
                g += 1
        g += 1
    return examples, groups


def test_sizes_and_disjoint():
    exs, groups = _data()
    train, val, test = split_examples(exs, groups, 0.1, 0.2, seed=1, stratify=True)
    assert len(train) + len(val) + len(test) == len(exs)
    ids = [set(e.id for e in part) for part in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert len(test) == 40 and len(val) == 20


def test_stratified_every_label_in_every_split():
    exs, groups = _data(n_labels=5, per_label=30)
    train, val, test = split_examples(exs, groups, 0.1, 0.1, seed=0, stratify=True)
    for part in (train, val, test):
        assert set(Counter(e.output for e in part)) == {f"label{i}" for i in range(5)}


def test_groups_never_cross_splits():
    exs, groups = _data(per_label=60, group_size=4)
    group_of = {e.id: g for e, g in zip(exs, groups)}
    train, val, test = split_examples(exs, groups, 0.15, 0.15, seed=3, stratify=True)
    seen = [set(group_of[e.id] for e in part) for part in (train, val, test)]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])


def test_single_group_label_stays_in_train():
    exs, groups = _data(n_labels=2, per_label=20)
    exs.append(Example("rare", "rare text", "rare_label"))
    groups.append(10_000)
    train, val, test = split_examples(exs, groups, 0.3, 0.3, seed=0, stratify=True)
    assert "rare" in {e.id for e in train}


def test_seeded_and_deterministic():
    exs, groups = _data()
    a = split_examples(exs, groups, 0.1, 0.1, seed=7, stratify=True)
    b = split_examples(exs, groups, 0.1, 0.1, seed=7, stratify=True)
    c = split_examples(exs, groups, 0.1, 0.1, seed=8, stratify=True)
    assert [[e.id for e in p] for p in a] == [[e.id for e in p] for p in b]
    assert [e.id for e in a[2]] != [e.id for e in c[2]]


def test_unstratified_handles_dict_outputs():
    exs = [Example(str(i), f"t{i}", {"k": i % 3}) for i in range(100)]
    train, val, test = split_examples(exs, list(range(100)), 0.1, 0.1, seed=0, stratify=False)
    assert (len(train), len(val), len(test)) == (80, 10, 10)
