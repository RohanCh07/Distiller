"""Small causal LLM student fine-tuned with LoRA.

Handles every task type: the answer is generated as text (a label, a JSON object or a
short string). Loss is computed only on answer tokens, so the model learns to answer,
not to repeat prompts. Confidence is the geometric mean of the generated tokens'
probabilities, calibrated later in eval.

For extraction tasks with an output_schema, decoding can be constrained to the schema
(needs the optional `lm-format-enforcer` package); without it, outputs that fail the
schema are treated as "no answer" and fall back to the teacher.
"""

import json
import math
import time
from pathlib import Path
from typing import Any

import jsonschema

from distiller.data.io import Example, serialize_output
from distiller.spec import TaskType
from distiller.training.base import Prediction, Student, warmup_steps

IGNORE = -100


class LoraLLMStudent(Student):
    name = "llm-lora"
    supports = (TaskType.classification, TaskType.extraction, TaskType.generation)

    # ---------- prompt formatting ----------

    def _prompt(self, input_text: str) -> str:
        instruction = self.spec.instruction
        if self.tokenizer.chat_template:
            messages = [{"role": "system", "content": instruction}] if instruction else []
            messages.append({"role": "user", "content": input_text})
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        prefix = f"{instruction}\n\n" if instruction else ""
        return f"{prefix}Input: {input_text}\nOutput: "

    def _encode_prompt(self, text: str) -> list[int]:
        # Chat templates already contain their special tokens as text.
        return self.tokenizer(text, add_special_tokens=not self.tokenizer.chat_template)["input_ids"]

    def _encode_train(self, ex: Example) -> dict[str, list[int]]:
        answer = self.tokenizer(serialize_output(ex.output), add_special_tokens=False)["input_ids"]
        answer = answer + [self.tokenizer.eos_token_id]
        prompt = self._encode_prompt(self._prompt(ex.input))
        room = self.hparams["max_length"] - len(answer)
        if room <= 0:
            prompt, answer = prompt[-8:], answer[: self.hparams["max_length"] - 8]
        elif len(prompt) > room:
            prompt = prompt[-room:]  # keep the end of the prompt, next to the answer
        return {"input_ids": prompt + answer, "labels": [IGNORE] * len(prompt) + answer}

    # ---------- loading ----------

    def _model_kwargs(self) -> dict[str, Any]:
        import torch

        kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            kwargs["dtype"] = self.dtype
            if self.hparams.get("load_in_4bit"):
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = {"": 0}
        else:
            self.dtype = torch.float32
        return kwargs

    def _load_tokenizer(self, source: str | Path) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_weights(self, model_dir: Path) -> None:
        from transformers import AutoModelForCausalLM

        self._load_tokenizer(model_dir)
        if (model_dir / "adapter_config.json").exists():
            from peft import PeftModel

            base = AutoModelForCausalLM.from_pretrained(self.hparams["model"], **self._model_kwargs())
            self.model = PeftModel.from_pretrained(base, model_dir)
        else:  # merged export
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, **self._model_kwargs())
        self.model.eval()

    # ---------- training ----------

    def fit(self, train: list[Example], val: list[Example], out_dir: Path) -> dict[str, Any]:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

        hp = self.hparams
        self._load_tokenizer(hp["model"])
        base = AutoModelForCausalLM.from_pretrained(hp["model"], **self._model_kwargs())
        if hp.get("load_in_4bit") and torch.cuda.is_available():
            base = prepare_model_for_kbit_training(base)
        lora = LoraConfig(
            task_type="CAUSAL_LM",
            r=hp["lora_r"],
            lora_alpha=hp["lora_alpha"],
            lora_dropout=hp["lora_dropout"],
            target_modules="all-linear",
        )
        self.model = get_peft_model(base, lora)

        train_rows = [self._encode_train(ex) for ex in train]
        val_rows = [self._encode_train(ex) for ex in val]
        longest_answer = max(sum(1 for t in r["labels"] if t != IGNORE) for r in train_rows)
        self.hparams["max_new_tokens"] = min(longest_answer + 8, 512)

        pad_id = self.tokenizer.pad_token_id

        def collate(rows: list[dict]) -> dict[str, torch.Tensor]:
            width = max(len(r["input_ids"]) for r in rows)
            ids, labels, mask = [], [], []
            for r in rows:
                pad = width - len(r["input_ids"])
                ids.append(r["input_ids"] + [pad_id] * pad)
                labels.append(r["labels"] + [IGNORE] * pad)
                mask.append([1] * len(r["input_ids"]) + [0] * pad)
            return {
                "input_ids": torch.tensor(ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask),
            }

        cuda = torch.cuda.is_available()
        args = TrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            num_train_epochs=hp["epochs"],
            learning_rate=hp["learning_rate"],
            per_device_train_batch_size=hp["batch_size"],
            per_device_eval_batch_size=hp["batch_size"],
            gradient_accumulation_steps=hp["grad_accum"],
            warmup_steps=warmup_steps(len(train_rows), hp["batch_size"], hp["grad_accum"], hp),
            lr_scheduler_type="cosine",
            eval_strategy="epoch" if val_rows else "no",
            save_strategy="epoch" if val_rows else "no",
            load_best_model_at_end=bool(val_rows),
            metric_for_best_model="eval_loss",
            save_total_limit=1,
            bf16=cuda and self.dtype == torch.bfloat16,
            fp16=cuda and self.dtype == torch.float16,
            logging_steps=25,
            report_to="none",
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_rows,
            eval_dataset=val_rows or None,
            data_collator=collate,
        )
        start = time.time()
        result = trainer.train()
        self.model.eval()
        trainable, total = self.model.get_nb_trainable_parameters()
        return {
            "train_seconds": round(time.time() - start, 1),
            "train_loss": round(result.training_loss, 4),
            "parameters": total,
            "trainable_parameters": trainable,
        }

    # ---------- inference ----------

    def _parse(self, text: str) -> Any:
        text = text.strip()
        if self.spec.type == TaskType.extraction:
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                return None
            if not isinstance(value, dict):
                return None
            if self.spec.output_schema:
                try:
                    jsonschema.validate(value, self.spec.output_schema)
                except jsonschema.ValidationError:
                    return None
            return value
        return text or None

    def _constraint(self):
        """prefix_allowed_tokens_fn forcing schema-valid JSON, or None if unavailable."""
        if not (
            self.spec.type == TaskType.extraction
            and self.spec.output_schema
            and self.hparams.get("constrained_decoding", True)
        ):
            return None
        try:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import (
                build_transformers_prefix_allowed_tokens_fn,
            )
        except ImportError as e:
            print(f"constrained decoding unavailable ({e}); using schema validation only")
            return None
        try:
            parser = JsonSchemaParser(self.spec.output_schema)
            return build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)
        except Exception as e:  # unusual tokenizer or schema: fall back to validate-after
            print(f"constrained decoding unavailable ({e}); using schema validation only")
            return None

    def predict(self, inputs: list[str]) -> list[Prediction]:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not hasattr(self.model, "hf_device_map"):
            self.model.to(device)
        self.tokenizer.padding_side = "left"
        eos = self.tokenizer.eos_token_id
        constraint = self._constraint()
        # The enforcer tracks state per sequence; batch size 1 keeps that simple and safe.
        batch_size = 1 if constraint else self.hparams["batch_size"] * 2
        preds: list[Prediction] = []
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                prompts = [self._prompt(x) for x in inputs[i : i + batch_size]]
                enc = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.hparams["max_length"],
                    add_special_tokens=not self.tokenizer.chat_template,
                ).to(device)
                def generate(**extra):
                    return self.model.generate(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        max_new_tokens=self.hparams["max_new_tokens"],
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **extra,
                    )

                if constraint:
                    try:
                        out = generate(prefix_allowed_tokens_fn=constraint)
                    except ValueError:  # constraint unsatisfiable for this tokenizer/input
                        out = generate()
                else:
                    out = generate()
                scores = self.model.compute_transition_scores(
                    out.sequences, out.scores, normalize_logits=True
                ).float().cpu()
                generated = out.sequences[:, enc["input_ids"].shape[1] :].cpu()
                for tokens, logprobs in zip(generated.tolist(), scores.tolist()):
                    n = tokens.index(eos) + 1 if eos in tokens else len(tokens)
                    finite = [lp for lp in logprobs[:n] if math.isfinite(lp)]
                    mean_lp = sum(finite) / max(len(finite), 1)
                    text = self.tokenizer.decode(tokens[:n], skip_special_tokens=True)
                    preds.append(Prediction(self._parse(text), math.exp(mean_lp)))
        self.tokenizer.padding_side = "right"
        return preds

    # ---------- saving ----------

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)  # LoRA adapter only (tens of MB)
        self.tokenizer.save_pretrained(out_dir)
        self._write_meta(out_dir)

    def export_merged(self, out_dir: Path) -> None:
        """Fold the adapter into the base weights: one standalone model for vLLM/llama.cpp."""
        out_dir.mkdir(parents=True, exist_ok=True)
        merged = self.model.merge_and_unload() if hasattr(self.model, "merge_and_unload") else self.model
        merged.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        self._write_meta(out_dir)
