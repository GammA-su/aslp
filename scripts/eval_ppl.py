#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aslp.utils import get_hf_token, get_model_device, load_yaml, set_seed, setup_logging


def _resolve_value(cli_value: Any, cfg: Dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if key in cfg:
        return cfg[key]
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute perplexity on general or domain datasets")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", dest="model_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--eval_type", type=str, choices=["general", "domain"], default=None)
    parser.add_argument("--eval_jsonl", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_sequences", type=int, default=None)
    parser.add_argument("--general_dataset_name", type=str, default=None)
    parser.add_argument("--general_dataset_config", type=str, default=None)
    parser.add_argument("--general_fallback_config", type=str, default=None)
    parser.add_argument("--general_split", type=str, default=None)
    parser.add_argument("--text_field", type=str, default=None)
    parser.add_argument("--out_json", type=str, default=None)
    return parser.parse_args()


def _compute_ppl(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dataset_iter,
    max_seq_len: int,
    num_sequences: int,
    batch_size: int,
) -> float:
    device = get_model_device(model)
    model.eval()
    losses = []
    batch_texts: list[str] = []
    with torch.no_grad():
        for sample in dataset_iter:
            text = sample.get("text", "")
            if not text or not isinstance(text, str) or not text.strip():
                continue
            batch_texts.append(text)
            if len(batch_texts) < batch_size:
                continue
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_seq_len,
                padding=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            if input_ids.shape[1] < 2:
                batch_texts = []
                continue
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
            token_losses = token_losses.view(shift_labels.shape)
            token_losses = token_losses * shift_mask
            denom = shift_mask.sum(dim=1)
            valid_mask = denom > 0
            if valid_mask.any():
                seq_losses = token_losses.sum(dim=1)[valid_mask] / denom[valid_mask]
                losses.extend(seq_losses.detach().float().tolist())
            batch_texts = []
            if len(losses) >= num_sequences:
                break
        if batch_texts and len(losses) < num_sequences:
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_seq_len,
                padding=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            if input_ids.shape[1] >= 2:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                shift_mask = attention_mask[:, 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                token_losses = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
                token_losses = token_losses.view(shift_labels.shape)
                token_losses = token_losses * shift_mask
                denom = shift_mask.sum(dim=1)
                valid_mask = denom > 0
                if valid_mask.any():
                    seq_losses = token_losses.sum(dim=1)[valid_mask] / denom[valid_mask]
                    losses.extend(seq_losses.detach().float().tolist())
    if not losses:
        return float("nan")
    mean_loss = sum(losses) / len(losses)
    return float(math.exp(mean_loss))


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config) if args.config else {}

    model_name = _resolve_value(args.model_name, cfg, "model_name", "meta-llama/Llama-3.1-8B-Instruct")
    mode = _resolve_value(args.mode, cfg, "mode", "baseline")
    seed = _resolve_value(args.seed, cfg, "seed", 42)
    lora_path = _resolve_value(args.lora_path, cfg, "lora_path", "")
    eval_type = _resolve_value(args.eval_type, cfg, "eval_type", "general")
    eval_jsonl = _resolve_value(args.eval_jsonl, cfg, "domain_eval_jsonl", "data/domain_eval.jsonl")
    max_seq_len = _resolve_value(args.max_seq_len, cfg, "max_seq_len", 256)
    batch_size = _resolve_value(args.batch_size, cfg, "batch_size", 1)
    num_sequences = _resolve_value(args.num_sequences, cfg, "num_sequences", 128)
    general_dataset_name = _resolve_value(args.general_dataset_name, cfg, "general_dataset_name", "wikitext")
    general_dataset_config = _resolve_value(
        args.general_dataset_config, cfg, "general_dataset_config", "wikitext-103-raw-v1"
    )
    general_fallback_config = _resolve_value(
        args.general_fallback_config, cfg, "general_fallback_config", "wikitext-2-raw-v1"
    )
    general_split = _resolve_value(args.general_split, cfg, "general_split", "test")
    text_field = _resolve_value(args.text_field, cfg, "text_field", "text")
    out_json = _resolve_value(args.out_json, cfg, "out_json", "")

    logger = setup_logging()
    set_seed(seed)

    hf_token = get_hf_token()
    token_kwargs = {"token": hf_token} if hf_token else {}
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
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)

    if eval_type == "general":
        try:
            dataset = load_dataset(general_dataset_name, general_dataset_config, split=general_split, streaming=True)
        except Exception as exc:
            logger.warning("Falling back to %s due to error: %s", general_fallback_config, exc)
            dataset = load_dataset(general_dataset_name, general_fallback_config, split=general_split, streaming=True)
        dataset_iter = (sample for sample in dataset if sample.get(text_field))
    else:
        dataset = load_dataset("json", data_files=eval_jsonl, split="train")
        dataset_iter = (sample for sample in dataset if sample.get("text"))

    ppl = _compute_ppl(model, tokenizer, dataset_iter, max_seq_len, num_sequences, batch_size)
    result = {
        "mode": mode,
        "seed": seed,
        "eval_type": eval_type,
        "ppl": ppl,
        "num_sequences": num_sequences,
        "max_seq_len": max_seq_len,
    }

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
