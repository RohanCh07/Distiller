"""Encoder classifier student (BERT-style model + classification head).

For pure classification this is usually the best student: tens of millions of
parameters, runs on CPU in milliseconds, and its softmax is easy to calibrate.
"""

import time
from pathlib import Path
from typing import Any

from distiller.data.io import Example
from distiller.spec import TaskType
from distiller.training.base import Prediction, Student, warmup_steps


class EncoderStudent(Student):
    name = "encoder"
    supports = (TaskType.classification,)

    def fit(self, train: list[Example], val: list[Example], out_dir: Path) -> dict[str, Any]:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

        hp = self.hparams
        label2id = {lbl: i for i, lbl in enumerate(self.labels)}
        self.tokenizer = AutoTokenizer.from_pretrained(hp["model"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hp["model"],
            num_labels=len(self.labels),
            id2label=dict(enumerate(self.labels)),
            label2id=label2id,
            ignore_mismatched_sizes=True,  # base checkpoints may ship a different head
        )

        def encode(examples: list[Example]) -> list[dict]:
            enc = self.tokenizer(
                [ex.input for ex in examples], truncation=True, max_length=hp["max_length"]
            )
            return [
                {"input_ids": ids, "attention_mask": mask, "labels": label2id[ex.output]}
                for ids, mask, ex in zip(enc["input_ids"], enc["attention_mask"], examples)
            ]

        args = TrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            num_train_epochs=hp["epochs"],
            learning_rate=hp["learning_rate"],
            per_device_train_batch_size=hp["batch_size"],
            per_device_eval_batch_size=hp["batch_size"] * 2,
            weight_decay=hp["weight_decay"],
            warmup_steps=warmup_steps(len(train), hp["batch_size"], 1, hp),
            eval_strategy="epoch" if val else "no",
            save_strategy="epoch" if val else "no",
            load_best_model_at_end=bool(val),
            metric_for_best_model="eval_loss",
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            report_to="none",
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=encode(train),
            eval_dataset=encode(val) if val else None,
            data_collator=DataCollatorWithPadding(self.tokenizer),
        )
        start = time.time()
        result = trainer.train()
        self.model.eval()
        return {
            "train_seconds": round(time.time() - start, 1),
            "train_loss": round(result.training_loss, 4),
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }

    def predict(self, inputs: list[str]) -> list[Prediction]:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        batch_size = self.hparams["batch_size"] * 2
        preds: list[Prediction] = []
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                enc = self.tokenizer(
                    inputs[i : i + batch_size],
                    truncation=True,
                    max_length=self.hparams["max_length"],
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                logits = self.model(**enc).logits.float().cpu()
                probs = logits.softmax(-1)
                for row_logits, row_probs in zip(logits, probs):
                    idx = int(row_probs.argmax())
                    preds.append(
                        Prediction(self.labels[idx], float(row_probs[idx]), row_logits.tolist())
                    )
        return preds

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        self._write_meta(out_dir)

    def _load_weights(self, model_dir: Path) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
