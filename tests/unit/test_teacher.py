"""Teacher client against a mocked OpenAI-compatible API (no network)."""

import json

import httpx
import pytest

from distiller import teacher as teacher_mod
from distiller.data.io import Example
from distiller.spec import TaskSpec
from distiller.teacher import Teacher, TeacherConfig, to_row

SCHEMA = {
    "type": "object",
    "required": ["amount"],
    "additionalProperties": False,
    "properties": {"amount": {"type": ["number", "null"]}},
}


def _spec(type_="classification", **kw):
    return TaskSpec.model_validate(
        {"name": "t", "type": type_, "instruction": "Do the task.", "data": {"path": "x"}, **kw}
    )


def make_teacher(handler, concurrency=2) -> tuple[Teacher, list[dict]]:
    """Teacher whose HTTP calls go to handler(body) -> str | httpx.Response."""
    calls: list[dict] = []

    def transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append({"body": body, "headers": dict(request.headers), "path": request.url.path})
        out = handler(body)
        if isinstance(out, httpx.Response):
            return out
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": out}}]}
        )

    t = Teacher(
        TeacherConfig("https://teacher.test/v1", "sk-test", "big-model"), concurrency=concurrency
    )
    t.client = httpx.Client(
        base_url="https://teacher.test/v1",
        headers={"Authorization": "Bearer sk-test"},
        transport=httpx.MockTransport(transport),
    )
    return t, calls


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(teacher_mod.time, "sleep", lambda s: None)


def test_from_env(monkeypatch):
    monkeypatch.delenv("DISTILLER_TEACHER_MODEL", raising=False)
    with pytest.raises(ValueError):
        TeacherConfig.from_env()
    monkeypatch.setenv("DISTILLER_TEACHER_MODEL", "m")
    monkeypatch.delenv("DISTILLER_TEACHER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fallback")
    monkeypatch.delenv("DISTILLER_TEACHER_BASE_URL", raising=False)
    cfg = TeacherConfig.from_env()
    assert (cfg.model, cfg.api_key, cfg.base_url) == ("m", "fallback", "https://api.openai.com/v1")
    monkeypatch.setenv("DISTILLER_TEACHER_API_KEY", "primary")
    assert TeacherConfig.from_env("override").api_key == "primary"
    assert TeacherConfig.from_env("override").model == "override"


def test_chat_sends_model_and_auth():
    t, calls = make_teacher(lambda body: "hello")
    assert t.chat([{"role": "user", "content": "hi"}], temperature=0.3) == "hello"
    assert calls[0]["path"] == "/v1/chat/completions"
    assert calls[0]["body"]["model"] == "big-model" and calls[0]["body"]["temperature"] == 0.3
    assert calls[0]["headers"]["authorization"] == "Bearer sk-test"


def test_chat_retries_on_rate_limit_then_succeeds():
    responses = iter([httpx.Response(429), httpx.Response(503), "ok"])
    t, calls = make_teacher(lambda body: next(responses))
    assert t.chat([{"role": "user", "content": "x"}]) == "ok"
    assert len(calls) == 3


def test_chat_gives_up_and_raises():
    t, calls = make_teacher(lambda body: httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        t.chat([{"role": "user", "content": "x"}], retries=2)
    assert len(calls) == 3


def test_chat_does_not_retry_client_errors():
    t, calls = make_teacher(lambda body: httpx.Response(401))
    with pytest.raises(httpx.HTTPStatusError):
        t.chat([{"role": "user", "content": "x"}])
    assert len(calls) == 1


def test_chat_retries_transport_errors():
    state = {"n": 0}

    def handler(body):
        state["n"] += 1
        if state["n"] < 3:
            raise httpx.ConnectError("down")
        return "back"

    t, _ = make_teacher(handler)
    assert t.chat([{"role": "user", "content": "x"}]) == "back"


def test_label_classification_validates_answers():
    answers = {"card": ' "card_arrival" ', "rate": "exchange_rate", "junk": "something else"}

    def handler(body):
        system = body["messages"][0]["content"]
        assert "Allowed labels: card_arrival, exchange_rate" in system
        assert "exactly one label" in system
        return answers[body["messages"][1]["content"]]

    t, _ = make_teacher(handler)
    spec = _spec(labels=["card_arrival", "exchange_rate"])
    assert t.label(spec, ["card", "rate", "junk"]) == ["card_arrival", "exchange_rate", None]


def test_label_extraction_parses_json_in_text():
    replies = {
        "a": 'Sure! {"amount": 12.5}',
        "b": '```json\n{"amount": null}\n```',
        "c": '{"amount": "lots"}',  # schema violation
        "d": "no json here",
        "e": '{"amount": 1, ',  # broken JSON
    }

    def handler(body):
        assert "JSON Schema for the answer" in body["messages"][0]["content"]
        return replies[body["messages"][1]["content"]]

    t, _ = make_teacher(handler)
    out = t.label(_spec("extraction", output_schema=SCHEMA), list(replies))
    assert out == [{"amount": 12.5}, {"amount": None}, None, None, None]


def test_label_http_failure_becomes_none():
    t, _ = make_teacher(lambda body: httpx.Response(400))
    assert t.label(_spec(), ["x"]) == [None]


def test_augment_classification_tops_up_rare_labels_with_consistency_filter():
    def handler(body):
        msgs = body["messages"]
        if len(msgs) == 1:  # input-writing prompt
            prompt = msgs[0]["content"]
            target = prompt.split('whose correct answer is "')[1].split('"')[0]
            n = int(prompt.split("Write ")[1].split(" ")[0])
            assert body["temperature"] == 0.9
            return json.dumps([f"{target} question {i}" for i in range(n)])
        text = msgs[1]["content"]
        # Teacher disagrees on every third generated input: those must be dropped.
        idx = int(text.rsplit(" ", 1)[1])
        return text.split(" ")[0] if idx % 3 else "other"

    t, _ = make_teacher(handler)
    spec = _spec(labels=["a", "b", "other"])
    seeds = [Example(str(i), f"seed a {i}", "a") for i in range(5)] + [
        Example("o", "seed other", "other")
    ]
    seeds += [Example(f"x{i}", f"seed other {i}", "other") for i in range(9)]
    new = t.augment(spec, seeds, per_label=10, batch=4)
    by_label: dict[str, int] = {}
    for ex in new:
        assert ex.meta == {"synthetic": True}
        by_label[ex.output] = by_label.get(ex.output, 0) + 1
    # a needs 5 more: batches of 4 + 1 -> idx 0,1,2,3,0 -> teacher agrees on idx 1,2
    # b needs 10: batches 4 + 4 + 2 -> agrees on 1,2 / 1,2 / 1
    assert by_label == {"a": 2, "b": 5}
    assert "other" not in by_label  # already had 10


def test_augment_stops_when_teacher_returns_nothing_usable():
    t, calls = make_teacher(lambda body: "I cannot do that")
    new = t.augment(_spec(labels=["a"]), [Example("1", "s", "a")], per_label=5)
    assert new == [] and len(calls) == 1


def test_augment_extraction_total():
    def handler(body):
        msgs = body["messages"]
        if len(msgs) == 1:
            n = int(msgs[0]["content"].split("Write ")[1].split(" ")[0])
            return "Here you go: " + json.dumps([f"paid {i}" for i in range(n)])
        amount = int(msgs[1]["content"].split()[1])
        return json.dumps({"amount": amount}) if amount % 2 == 0 else "oops"

    t, _ = make_teacher(handler)
    new = t.augment(
        _spec("extraction", output_schema=SCHEMA),
        [Example("1", "paid 5", {"amount": 5})],
        total=7,
        batch=3,
    )
    # batches of 3 + 3 + 1 -> amounts 0,1,2,0,1,2,0 -> odd ones fail to parse
    assert [ex.output["amount"] for ex in new] == [0, 2, 0, 2, 0]


def test_to_row():
    assert json.loads(to_row(Example("", "x", {"a": 1}, {"synthetic": True}))) == {
        "input": "x",
        "output": {"a": 1},
        "synthetic": True,
    }
    assert "synthetic" not in json.loads(to_row(Example("", "x", "y")))
