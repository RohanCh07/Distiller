"""Convert the RoBERTa-base encoder inside spaCy's en_core_web_trf to Hugging Face format.

Used for the Banking77 run in an environment where the Hugging Face Hub was blocked but
GitHub release downloads were not. Usage:

  curl -LO https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl
  unzip en_core_web_trf-3.8.0-py3-none-any.whl 'en_core_web_trf/*' -d spacy_model
  pip install srsly
  python scripts/convert_spacy_roberta.py spacy_model/en_core_web_trf/en_core_web_trf-3.8.0 models/roberta-base-spacy

Note: these weights were fine-tuned by spaCy for tagging/parsing/NER, so they are a real
pretrained RoBERTa-base encoder, but not the original checkpoint.
"""

import io
import re
import sys
from pathlib import Path

import srsly
import torch
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

src, out = Path(sys.argv[1]), Path(sys.argv[2])
d = srsly.msgpack_loads((src / "transformer/model").read_bytes())
sd = torch.load(
    io.BytesIO(srsly.msgpack_loads(d["shims"][6][0])["state"]),
    map_location="cpu",
    weights_only=True,
)
bpe = srsly.msgpack_loads(d["attrs"][3]["byte_bpe_processor"])

cfg = RobertaConfig(
    vocab_size=50265,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=514,
    type_vocab_size=1,
    layer_norm_eps=1e-5,
    pad_token_id=1,
    bos_token_id=0,
    eos_token_id=2,
)
new = {}
for k, v in sd.items():
    k = k.replace("curated_encoder.", "")
    if k.startswith("embeddings.inner."):
        k = "embeddings." + k[len("embeddings.inner.") :].replace("layer_norm", "LayerNorm")
        new[k] = v
        continue
    m = re.match(r"layers\.(\d+)\.(.*)", k)
    i, rest = m.group(1), m.group(2)
    p = f"encoder.layer.{i}."
    if rest.startswith("mha.input."):
        kind = rest.split(".")[-1]
        q, kk, vv = v.chunk(3, dim=0)
        for name, t in (("query", q), ("key", kk), ("value", vv)):
            new[f"{p}attention.self.{name}.{kind}"] = t.clone()
    else:
        mapping = {
            "mha.output": "attention.output.dense",
            "attn_output_layernorm": "attention.output.LayerNorm",
            "ffn.intermediate": "intermediate.dense",
            "ffn.output": "output.dense",
            "ffn_output_layernorm": "output.LayerNorm",
        }
        base, kind = rest.rsplit(".", 1)
        new[f"{p}{mapping[base]}.{kind}"] = v
model = RobertaModel(cfg, add_pooling_layer=False)
missing, unexpected = model.load_state_dict(new, strict=False)
assert not unexpected and all("position_ids" in m for m in missing), (missing, unexpected)
out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out)
# Tensors only (weights_only=True): never unpickle arbitrary objects from a download.
tok = RobertaTokenizerFast(
    vocab=bpe["vocab"], merges=[tuple(m) for m in bpe["merges"]], model_max_length=512
)
tok.save_pretrained(out)
ids = tok("Hello world")["input_ids"]
assert ids == [0, 31414, 232, 2], ids  # same ids as the original roberta-base tokenizer
print(f"Wrote {out}")
