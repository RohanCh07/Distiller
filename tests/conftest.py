"""Shared fixtures: task specs on disk and tiny random models built offline.

The tiny models are created locally (random weights, tokenizer trained on the fly), so
end-to-end tests need no network and run in seconds on CPU. Their scores mean nothing;
they only prove that every step of the pipeline runs and produces the right files.
"""

import json
import random
from pathlib import Path

import pytest
import yaml

from distiller.spec import TaskSpec

INTENTS = {
    "card_arrival": [
        "my card has not arrived yet",
        "when will my new card arrive",
        "still waiting for the card you sent",
        "card delivery is taking too long",
    ],
    "top_up": [
        "how do i top up my account",
        "top up failed with apple pay",
        "can i add money by bank transfer",
        "my top up is still pending",
    ],
    "exchange_rate": [
        "what exchange rate do you use",
        "the currency conversion rate looks wrong",
        "how much is one euro in pounds today",
        "why did i get a bad exchange rate",
    ],
}
FILLERS = ["", " please", " thanks", " asap", " help", " today", " again", " now"]


def classification_rows(n_per_label: int = 60, seed: int = 0) -> list[dict]:
    """Keyword-separable intents with small variations, so a tiny model can learn them."""
    rng = random.Random(seed)
    rows = []
    for label, templates in INTENTS.items():
        for i in range(n_per_label):
            text = rng.choice(templates) + rng.choice(FILLERS) + f" #{i}"
            rows.append({"input": text, "output": label})
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def write_task(directory: Path, rows: list[dict], **spec) -> Path:
    """Write data.jsonl + task.yaml into directory and return the task.yaml path."""
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "data.jsonl", rows)
    base = {
        "name": "demo",
        "type": "classification",
        "instruction": "Classify the message.",
        "data": {"path": "data.jsonl"},
        "output_dir": "runs",
    }
    for key, value in spec.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    path = directory / "task.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False))
    return path


@pytest.fixture
def make_task(tmp_path):
    """make_task(rows, **spec_fields) -> loaded TaskSpec in a fresh folder."""
    counter = iter(range(1000))

    def _make(rows: list[dict], **spec) -> TaskSpec:
        return TaskSpec.load(write_task(tmp_path / f"task{next(counter)}", rows, **spec))

    return _make


# ---------- tiny offline models ----------


def _corpus_texts() -> list[str]:
    texts = [r["input"] for r in classification_rows(20)]
    texts += list(INTENTS)
    seeds = Path(__file__).parent.parent / "examples" / "transactions" / "seed.jsonl"
    for line in seeds.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        texts += [row["input"], json.dumps(row["output"], ensure_ascii=False)]
    return texts


@pytest.fixture(scope="session")
def tiny_encoder(tmp_path_factory) -> str:
    """A 2-layer BERT with a WordPiece vocab trained on the test corpus."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    out = tmp_path_factory.mktemp("tiny-encoder")
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(
        _corpus_texts(), trainers.WordPieceTrainer(vocab_size=600, special_tokens=special)
    )
    cls_id, sep_id = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=128,
    )
    fast.save_pretrained(out)
    config = BertConfig(
        vocab_size=tok.get_vocab_size(),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    BertModel(config).save_pretrained(out)
    return str(out)


@pytest.fixture(scope="session")
def tiny_llm(tmp_path_factory) -> str:
    """A 2-layer Llama-style causal LM with a byte-level BPE vocab (no chat template)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out = tmp_path_factory.mktemp("tiny-llm")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        _corpus_texts(),
        trainers.BpeTrainer(
            vocab_size=800,
            special_tokens=["<eos>", "<pad>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<eos>", pad_token="<pad>", model_max_length=256
    )
    fast.save_pretrained(out)
    config = LlamaConfig(
        vocab_size=tok.get_vocab_size(),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
        bos_token_id=None,
    )
    LlamaForCausalLM(config).save_pretrained(out)
    return str(out)
