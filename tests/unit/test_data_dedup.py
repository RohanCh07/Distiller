from distiller.data.dedup import near_duplicate_groups, normalize, remove_exact_duplicates
from distiller.data.io import Example


def ex(i, text, out="a"):
    return Example(str(i), text, out)


def test_normalize():
    assert normalize("  Hello\tWORLD \n") == "hello world"
    assert normalize("ｆｕｌｌ") == "full"  # NFKC folds full-width characters


def test_exact_duplicates_after_normalisation():
    res = remove_exact_duplicates([ex(1, "Hi there"), ex(2, "hi   THERE"), ex(3, "other")])
    assert res.exact_duplicates == 1
    assert res.conflicts == 0
    assert [e.id for e in res.examples] == ["1", "3"]


def test_conflicts_keep_majority():
    rows = [ex(1, "x", "a"), ex(2, "x", "b"), ex(3, "X", "b"), ex(4, "y", "c")]
    res = remove_exact_duplicates(rows)
    assert res.conflicts == 1
    assert res.exact_duplicates == 2
    kept = {e.input.lower(): e.output for e in res.examples}
    assert kept == {"x": "b", "y": "c"}
    assert len(res.conflict_samples) == 1 and "'b': 2" in res.conflict_samples[0]


def test_conflicts_on_dict_outputs():
    res = remove_exact_duplicates(
        [ex(1, "q", {"k": 1}), ex(2, "q", {"k": 1}), ex(3, "q", {"k": 2})]
    )
    assert res.conflicts == 1 and res.examples[0].output == {"k": 1}


def test_near_duplicates_grouped():
    rows = [
        ex(0, "my card has not arrived yet, it has been two weeks"),
        ex(1, "my card has not arrived yet, it has been two weeks!"),
        ex(2, "what exchange rate do you use for euros"),
        ex(3, "completely unrelated sentence about pizza"),
    ]
    g = near_duplicate_groups(rows, threshold=0.8)
    assert g[0] == g[1]
    assert len({g[0], g[2], g[3]}) == 3


def test_near_duplicates_threshold_and_short_texts():
    rows = [ex(0, "abc"), ex(1, "abc"), ex(2, "abd")]
    g = near_duplicate_groups(rows, threshold=0.85)
    assert g[0] == g[1] != g[2]


def test_near_duplicate_groups_are_deterministic():
    rows = [ex(i, f"sentence number {i % 7} with some padding text") for i in range(40)]
    assert near_duplicate_groups(rows, 0.85) == near_duplicate_groups(rows, 0.85)
    assert len(set(near_duplicate_groups(rows, 0.85))) == 7


def test_group_ids_are_dense():
    rows = [ex(i, t) for i, t in enumerate(["aaaa bbbb", "cccc dddd", "aaaa bbbb", "eeee ffff"])]
    g = near_duplicate_groups(rows, 0.9)
    assert sorted(set(g)) == list(range(len(set(g))))
