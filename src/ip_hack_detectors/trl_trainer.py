"""LoRA training with TRL's SFTTrainer and peft (PREREGISTRATION.md section 4).

torch, transformers, trl and peft live in the optional extra `train` and are imported inside
the functions that need them. Examples are tokenized here (`build_example`) and handed to
SFTTrainer pre-tokenized with our own collator, so the loss mask is exactly ours.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

from .records import read_jsonl
from .training import TrainJob
from .transformers_backend import check_stop_tokens

END_OF_TEXT = "<|endoftext|>"
# Pad token for batches: any token other than the two end tokens works, because padding is
# masked by position (attention mask 0, label -100), never by token id. <|im_start|> is used
# because Qwen2's tokenizer has no dedicated pad token; that it also opens every chat turn does
# not matter, since padded positions never enter attention or loss.
PAD_TOKEN = "<|im_start|>"


@dataclass(frozen=True)
class Example:
    input_ids: list[int]
    labels: list[int]  # -100 = not supervised
    truncated: bool


def build_example(tokenizer, messages: list[dict[str, str]], max_seq_length: int) -> Example:
    """Tokenize one chat example with a response-only loss.

    Text: the chat template of the full conversation plus "<|endoftext|>". Everything up to and
    including the assistant header (the template of the conversation without the assistant
    turn, with add_generation_prompt) is masked. The supervised part is the response,
    <|im_end|>, the newline and <|endoftext|>. Longer examples are truncated at max_seq_length.
    """
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(messages, tokenize=False) + END_OF_TEXT
    if not full.startswith(prompt):
        raise ValueError("chat template: prompt is not a prefix of the full conversation")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("tokenization of the prompt differs from its place in the full text")
    labels = [-100] * len(prompt_ids) + ids[len(prompt_ids) :]
    truncated = len(ids) > max_seq_length
    return Example(ids[:max_seq_length], labels[:max_seq_length], truncated)


class _Collator:
    """Right-pads input_ids, labels and attention mask."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        n = max(len(f["input_ids"]) for f in features)
        ids, labels, mask = [], [], []
        for f in features:
            pad = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            mask.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


class TrlTrainer:
    def train(self, job: TrainJob) -> dict:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import SFTConfig, SFTTrainer

        hp = job.training
        set_seed(job.seed)
        cuda = torch.cuda.is_available()
        tokenizer = AutoTokenizer.from_pretrained(job.model.name, revision=job.model.revision)
        check_stop_tokens(tokenizer)
        pad_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        if pad_id is None or pad_id == tokenizer.convert_tokens_to_ids(END_OF_TEXT):
            raise ValueError(f"tokenizer has no usable pad token {PAD_TOKEN}")
        tokenizer.pad_token = PAD_TOKEN

        rows = read_jsonl(job.dataset)
        examples = [build_example(tokenizer, r["messages"], hp.max_seq_length) for r in rows]
        n_truncated = sum(e.truncated for e in examples)
        dataset = Dataset.from_list(
            [{"input_ids": e.input_ids, "labels": e.labels} for e in examples]
        )

        model = AutoModelForCausalLM.from_pretrained(
            job.model.name,
            revision=job.model.revision,
            dtype=torch.bfloat16 if cuda else torch.float32,
        )
        lora = LoraConfig(
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            target_modules=hp.target_modules,
            bias="none",
            use_rslora=hp.use_rslora,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)

        args = SFTConfig(
            output_dir=str(job.out_dir / "_trainer"),
            seed=job.seed,
            data_seed=job.seed,
            learning_rate=job.learning_rate,
            weight_decay=hp.weight_decay,
            lr_scheduler_type=hp.lr_scheduler,
            warmup_steps=hp.warmup_steps,
            optim=hp.optimizer,
            per_device_train_batch_size=hp.batch_size,
            gradient_accumulation_steps=hp.gradient_accumulation,
            num_train_epochs=hp.epochs,
            max_steps=hp.max_steps if hp.max_steps is not None else -1,
            max_length=hp.max_seq_length,
            packing=False,
            bf16=cuda,
            gradient_checkpointing=hp.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
            save_strategy="no",
            logging_steps=1,
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=_Collator(pad_id),
        )
        result = trainer.train()

        merged = trainer.model.merge_and_unload()
        job.out_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(job.out_dir)
        tokenizer.save_pretrained(job.out_dir)
        report = {
            "kind": "trl",
            "examples": len(examples),
            "truncated": n_truncated,
            "steps": trainer.state.global_step,
            "train_loss": result.training_loss,
            "losses": [
                {"step": e["step"], "loss": e["loss"], "learning_rate": e.get("learning_rate")}
                for e in trainer.state.log_history
                if "loss" in e
            ],
        }
        # Release GPU memory before the next training run or the vLLM worker starts.
        del merged, trainer, model
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        return report
