import pytest

from distiller.spec import TaskSpec
from distiller.training.base import default_candidates, get_backend, warmup_steps
from distiller.training.presets import hyperparams, sweep_space


def spec(type_="classification", **student):
    return TaskSpec.model_validate(
        {"name": "t", "type": type_, "data": {"path": "x"}, "student": student}
    )


@pytest.mark.parametrize("n,enc,llm", [(500, 10, 5), (5_000, 5, 3), (50_000, 3, 2)])
def test_epochs_scale_with_data_size(n, enc, llm):
    assert hyperparams("encoder", spec(), n)["epochs"] == enc
    assert hyperparams("llm-lora", spec(), n)["epochs"] == llm


def test_models_and_lengths_come_from_spec():
    s = spec(encoder_model="my/enc", llm_model="my/llm", max_length=77)
    assert hyperparams("encoder", s, 10)["model"] == "my/enc"
    llm = hyperparams("llm-lora", s, 10)
    assert llm["model"] == "my/llm" and llm["max_length"] == 77


def test_max_new_tokens_by_task_type():
    assert hyperparams("llm-lora", spec(), 10)["max_new_tokens"] == 16
    assert hyperparams("llm-lora", spec("extraction"), 10)["max_new_tokens"] == 256


def test_overrides_win():
    s = spec(overrides={"encoder": {"epochs": 1, "new_key": True}})
    hp = hyperparams("encoder", s, 100_000)
    assert hp["epochs"] == 1 and hp["new_key"] is True
    assert hyperparams("llm-lora", s, 100_000)["epochs"] == 2  # other backend untouched


def test_unknown_backend():
    with pytest.raises(ValueError):
        hyperparams("svm", spec(), 10)
    with pytest.raises(ValueError):
        sweep_space("svm")
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("svm")


def test_sweep_spaces():
    assert [p["learning_rate"] for p in sweep_space("encoder")] == [3e-5, 5e-5, 8e-5]
    llm = sweep_space("llm-lora")
    assert len(llm) == 4 and all(p["lora_alpha"] == 2 * p["lora_r"] for p in llm)


def test_default_candidates():
    from distiller.spec import TaskType

    assert default_candidates(TaskType.classification) == ["encoder", "llm-lora"]
    assert default_candidates(TaskType.extraction) == ["llm-lora"]
    assert default_candidates(TaskType.generation) == ["llm-lora"]


def test_warmup_steps():
    # 1000 examples / (32*1) = 32 steps per epoch, 3 epochs, 10% warmup -> 9
    assert warmup_steps(1000, 32, 1, {"warmup_ratio": 0.1, "epochs": 3}) == 9
    assert warmup_steps(10, 8, 2, {"warmup_ratio": 0.05, "epochs": 1}) == 0
