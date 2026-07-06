from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Qwen2 attention + MLP projections. Standard QLoRA target set for this family.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="QLoRA SFT to distill the Sonnet judge into the local 7B "
        "(faithfulness + quality). bf16 base, bnb 4-bit NF4, peft LoRA."
    )
    ap.add_argument("--data", type=Path, default=Path("eval/datasets/judge_sft"))
    ap.add_argument("--out", type=Path, default=Path("eval/models/judge-qlora"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seq-len", type=int, default=2560)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="2 steps on a tiny subset to validate wiring.")
    args = ap.parse_args()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )

    ds = load_dataset(
        "json",
        data_files={
            "train": str(args.data / "train.jsonl"),
            "val": str(args.data / "val.jsonl"),
        },
    )
    ds = ds.select_columns(["messages"])
    if args.smoke:
        ds["train"] = ds["train"].select(range(16))
        ds["val"] = ds["val"].select(range(8))

    cfg = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.seq_len,
        packing=False,
        optim="paged_adamw_8bit",
        logging_steps=1 if args.smoke else 5,
        eval_strategy="no" if args.smoke else "epoch",
        save_strategy="no" if args.smoke else "epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    result = trainer.train()
    print(f"train_loss={result.training_loss:.4f}")
    if not args.smoke:
        trainer.save_model(str(args.out))
        print(f"adapter saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
