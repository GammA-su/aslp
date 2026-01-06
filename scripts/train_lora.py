#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from huggingface_hub.errors import GatedRepoError
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aslp.projector import ASLPProjector, random_orthonormal
from aslp.trainer_patch import ProjectingCallback, ProjectingTrainer
from aslp.utils import (
    coerce_float,
    coerce_optional_float,
    get_hf_token,
    get_middle_third_indices,
    get_model_device,
    load_yaml,
    select_module_names,
    set_seed,
    setup_logging,
)


def _resolve_value(cli_value: Any, cfg: Dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if key in cfg:
        return cfg[key]
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA with optional ASLP projection")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", dest="model_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_jsonl", type=str, default=None)
    parser.add_argument("--eval_jsonl", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["baseline", "aslp", "random"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--adam_beta1", type=float, default=None)
    parser.add_argument("--adam_beta2", type=float, default=None)
    parser.add_argument("--adam_epsilon", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--u_path", type=str, default=None)
    parser.add_argument("--r_pca", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--proj_log_steps", type=int, default=None)
    return parser.parse_args()


def _load_u_subspaces(path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "u_by_module" in payload:
        return payload["u_by_module"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Invalid u_subspaces file format")


def _build_random_u(model: torch.nn.Module, module_names: list[str], r_pca: int, seed: int) -> Dict[str, torch.Tensor]:
    name_to_module = dict(model.named_modules())
    u_by_module: Dict[str, torch.Tensor] = {}
    for idx, name in enumerate(module_names):
        module = name_to_module.get(name)
        if module is None:
            matches = [full_name for full_name in name_to_module.keys() if full_name.endswith(name)]
            if len(matches) == 1:
                module = name_to_module[matches[0]]
            elif len(matches) > 1:
                raise ValueError(f"Ambiguous module name for random projection: {name}")
        if module is None:
            raise ValueError(f"Module not found for random projection: {name}")
        if hasattr(module, "lora_B") and "default" in module.lora_B:
            d_out = module.lora_B["default"].weight.shape[0]
        elif hasattr(module, "out_features"):
            d_out = int(module.out_features)
        elif hasattr(module, "weight"):
            d_out = module.weight.shape[0]
        else:
            raise ValueError(f"Unable to infer output dimension for {name}")
        u_by_module[name] = random_orthonormal(d_out, r_pca, seed + idx)
    return u_by_module


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config) if args.config else {}

    model_name = _resolve_value(args.model_name, cfg, "model_name", "meta-llama/Llama-3.1-8B-Instruct")
    train_jsonl = _resolve_value(args.train_jsonl, cfg, "train_jsonl", "data/domain_train.jsonl")
    eval_jsonl = _resolve_value(args.eval_jsonl, cfg, "eval_jsonl", "data/domain_eval.jsonl")
    mode = _resolve_value(args.mode, cfg, "mode", "baseline")
    seed = _resolve_value(args.seed, cfg, "seed", 42)
    max_seq_len = _resolve_value(args.max_seq_len, cfg, "max_seq_len", 256)
    per_device_train_batch_size = _resolve_value(
        args.per_device_train_batch_size, cfg, "per_device_train_batch_size", 1
    )
    per_device_eval_batch_size = _resolve_value(
        args.per_device_eval_batch_size, cfg, "per_device_eval_batch_size", 1
    )
    gradient_accumulation_steps = _resolve_value(
        args.gradient_accumulation_steps, cfg, "gradient_accumulation_steps", 8
    )
    learning_rate = coerce_float("learning_rate", _resolve_value(args.learning_rate, cfg, "learning_rate", 2e-4))
    weight_decay = coerce_optional_float("weight_decay", _resolve_value(args.weight_decay, cfg, "weight_decay", None))
    warmup_ratio = coerce_optional_float("warmup_ratio", _resolve_value(args.warmup_ratio, cfg, "warmup_ratio", None))
    adam_beta1 = coerce_optional_float("adam_beta1", _resolve_value(args.adam_beta1, cfg, "adam_beta1", None))
    adam_beta2 = coerce_optional_float("adam_beta2", _resolve_value(args.adam_beta2, cfg, "adam_beta2", None))
    adam_epsilon = coerce_optional_float("adam_epsilon", _resolve_value(args.adam_epsilon, cfg, "adam_epsilon", None))
    max_grad_norm = coerce_optional_float("max_grad_norm", _resolve_value(args.max_grad_norm, cfg, "max_grad_norm", None))
    max_steps = _resolve_value(args.max_steps, cfg, "max_steps", 100)
    logging_steps = _resolve_value(args.logging_steps, cfg, "logging_steps", 10)
    eval_steps = _resolve_value(args.eval_steps, cfg, "eval_steps", 50)
    save_steps = _resolve_value(args.save_steps, cfg, "save_steps", 50)
    lora_r = _resolve_value(args.lora_r, cfg, "lora_r", 8)
    lora_alpha = _resolve_value(args.lora_alpha, cfg, "lora_alpha", 16)
    lora_dropout = _resolve_value(args.lora_dropout, cfg, "lora_dropout", 0.05)
    u_path = _resolve_value(args.u_path, cfg, "u_path", "artifacts/u_subspaces.pt")
    r_pca = _resolve_value(args.r_pca, cfg, "r_pca", 16)
    output_dir = _resolve_value(args.output_dir, cfg, "output_dir", "")
    proj_log_steps = _resolve_value(args.proj_log_steps, cfg, "proj_log_steps", 20)

    if not output_dir:
        output_dir = f"artifacts/runs/{mode}_seed{seed}"

    logger = setup_logging()
    set_seed(seed)

    hf_token = get_hf_token()
    token_kwargs = {"token": hf_token} if hf_token else {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **token_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            **token_kwargs,
        )
    except GatedRepoError:
        print("Model is gated; set --model to one you can access or request access", file=sys.stderr)
        raise SystemExit(1)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["o_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    raw_datasets = load_dataset("json", data_files={"train": train_jsonl, "validation": eval_jsonl})

    def tokenize_batch(batch: Dict[str, list[str]]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_len)

    tokenized = raw_datasets.map(
        tokenize_batch,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    projector: Optional[ASLPProjector] = None
    if mode == "aslp":
        if not Path(u_path).exists():
            raise FileNotFoundError(f"Missing subspace file: {u_path}")
        u_by_module = _load_u_subspaces(u_path)
        projector = ASLPProjector(model, u_by_module)
        logger.info("Loaded %d subspaces for ASLP", len(u_by_module))
    elif mode == "random":
        module_names: Optional[list[str]] = None
        if u_path and Path(u_path).exists():
            u_by_module_src = _load_u_subspaces(u_path)
            module_names = list(u_by_module_src.keys())
            if not module_names:
                raise ValueError("u_subspaces file contains no modules")
        else:
            num_layers = int(getattr(model.config, "num_hidden_layers", 0))
            layer_indices = get_middle_third_indices(num_layers)
            module_names = select_module_names(model, ["self_attn.o_proj", "mlp.down_proj"], layer_indices)
        u_by_module = _build_random_u(model, module_names, r_pca, seed)
        projector = ASLPProjector(model, u_by_module)
        logger.info("Built %d random subspaces", len(u_by_module))

    training_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "report_to": [],
        "remove_unused_columns": False,
        "fp16": torch.cuda.is_available(),
        "seed": seed,
        "data_seed": seed,
    }
    if weight_decay is not None:
        training_kwargs["weight_decay"] = weight_decay
    if warmup_ratio is not None:
        training_kwargs["warmup_ratio"] = warmup_ratio
    if adam_beta1 is not None:
        training_kwargs["adam_beta1"] = adam_beta1
    if adam_beta2 is not None:
        training_kwargs["adam_beta2"] = adam_beta2
    if adam_epsilon is not None:
        training_kwargs["adam_epsilon"] = adam_epsilon
    if max_grad_norm is not None:
        training_kwargs["max_grad_norm"] = max_grad_norm

    training_args = TrainingArguments(
        **training_kwargs,
    )

    callbacks = []
    if projector is not None:
        callbacks.append(ProjectingCallback(projector, proj_log_steps))

    trainer = ProjectingTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved LoRA adapter to %s", output_dir)


if __name__ == "__main__":
    main()
