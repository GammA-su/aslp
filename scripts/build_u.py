#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from huggingface_hub.errors import GatedRepoError
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aslp.fd_sketch import FrequentDirections
from aslp.hooks import extract_tensor, register_activation_hooks
from aslp.utils import (
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


def _sample_tokens(x: torch.Tensor, max_tokens: Optional[int], generator: torch.Generator) -> torch.Tensor:
    if max_tokens is None or x.shape[0] <= max_tokens:
        return x
    idx = torch.randperm(x.shape[0], generator=generator)[:max_tokens]
    return x[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build protected subspaces via Frequent Directions PCA")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", dest="model_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--fallback_dataset_config", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--text_field", type=str, default=None)
    parser.add_argument("--num_sequences", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--tokens_per_batch", type=int, default=None)
    parser.add_argument("--r_pca", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--layer_range", type=str, default=None)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config) if args.config else {}

    model_name = _resolve_value(args.model_name, cfg, "model_name", "meta-llama/Llama-3.1-8B-Instruct")
    dataset_name = _resolve_value(args.dataset_name, cfg, "dataset_name", "wikitext")
    dataset_config = _resolve_value(args.dataset_config, cfg, "dataset_config", "wikitext-103-raw-v1")
    fallback_dataset_config = _resolve_value(
        args.fallback_dataset_config, cfg, "fallback_dataset_config", "wikitext-2-raw-v1"
    )
    split = _resolve_value(args.split, cfg, "split", "train")
    text_field = _resolve_value(args.text_field, cfg, "text_field", "text")
    num_sequences = _resolve_value(args.num_sequences, cfg, "num_sequences", 256)
    batch_size = _resolve_value(args.batch_size, cfg, "batch_size", 1)
    max_seq_len = _resolve_value(args.max_seq_len, cfg, "max_seq_len", 256)
    tokens_per_batch = _resolve_value(args.tokens_per_batch, cfg, "tokens_per_batch", 512)
    r_pca = _resolve_value(args.r_pca, cfg, "r_pca", 16)
    seed = _resolve_value(args.seed, cfg, "seed", 42)
    layer_range = _resolve_value(args.layer_range, cfg, "layer_range", "middle_third")
    out_path = _resolve_value(args.out_path, cfg, "out_path", "artifacts/u_subspaces.pt")
    log_every = _resolve_value(args.log_every, cfg, "log_every", 100)
    if log_every is None:
        log_every = 0
    log_every = int(log_every)
    if log_every < 0:
        log_every = 0

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
    model.eval()

    num_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if num_layers <= 0:
        raise ValueError("Model config does not define num_hidden_layers")

    if layer_range == "middle_third":
        layer_indices = get_middle_third_indices(num_layers)
    elif layer_range == "all":
        layer_indices = set(range(num_layers))
    else:
        raise ValueError(f"Unsupported layer_range: {layer_range}")

    module_suffixes = ["self_attn.o_proj", "mlp.down_proj"]
    module_names = select_module_names(model, module_suffixes, layer_indices)
    if not module_names:
        raise ValueError("No target modules matched for subspace building")

    logger.info("Building subspaces for %d modules", len(module_names))

    try:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    except Exception as exc:
        logger.warning("Falling back to dataset config %s due to error: %s", fallback_dataset_config, exc)
        dataset = load_dataset(dataset_name, fallback_dataset_config, split=split, streaming=True)

    sketches: Dict[str, FrequentDirections] = {}
    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(seed)

    def callback(name: str, output: Any) -> None:
        tensor = extract_tensor(output)
        if tensor is None:
            return
        if tensor.ndim == 2:
            flat = tensor
        elif tensor.ndim == 3:
            flat = tensor.reshape(-1, tensor.shape[-1])
        else:
            return
        flat_cpu = flat.detach().to(dtype=torch.float32, device="cpu")
        flat_cpu = _sample_tokens(flat_cpu, tokens_per_batch, cpu_gen)
        if name not in sketches:
            sketches[name] = FrequentDirections(d=flat_cpu.shape[1], k=r_pca)
        sketches[name].update(flat_cpu)

    handles = register_activation_hooks(model, module_names, callback)

    device = get_model_device(model)
    seen = 0
    batch_texts: list[str] = []
    start_time = time.time()
    next_log = log_every if log_every > 0 else 0

    def maybe_log_progress(seen_count: int, next_threshold: int) -> int:
        if log_every <= 0 or seen_count < next_threshold:
            return next_threshold
        elapsed = time.time() - start_time
        rate = seen_count / elapsed if elapsed > 0 else 0.0
        pct = (seen_count / num_sequences) * 100 if num_sequences else 0.0
        logger.info("Processed %d/%d sequences (%.1f%%, %.2f seq/s)", seen_count, num_sequences, pct, rate)
        while next_threshold <= seen_count:
            next_threshold += log_every
        return next_threshold

    with torch.no_grad():
        for sample in dataset:
            text = sample.get(text_field, "")
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
            lengths = enc["attention_mask"].sum(dim=1)
            valid_mask = lengths >= 2
            if valid_mask.any():
                input_ids = enc["input_ids"][valid_mask].to(device)
                attention_mask = enc["attention_mask"][valid_mask].to(device)
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                seen += int(valid_mask.sum().item())
                next_log = maybe_log_progress(seen, next_log)
            batch_texts = []
            if seen >= num_sequences:
                break
        if batch_texts and seen < num_sequences:
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_seq_len,
                padding=True,
                return_tensors="pt",
            )
            lengths = enc["attention_mask"].sum(dim=1)
            valid_mask = lengths >= 2
            if valid_mask.any():
                input_ids = enc["input_ids"][valid_mask].to(device)
                attention_mask = enc["attention_mask"][valid_mask].to(device)
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                seen += int(valid_mask.sum().item())
                next_log = maybe_log_progress(seen, next_log)

    for handle in handles:
        handle.remove()

    if seen == 0:
        raise RuntimeError("No usable sequences found to build subspaces")

    u_by_module: Dict[str, torch.Tensor] = {}
    for name, sketch in sketches.items():
        B = sketch.get_sketch()
        if B.shape[0] < r_pca:
            raise RuntimeError(f"Sketch for {name} has rank {B.shape[0]} < r_pca={r_pca}")
        _, _, Vh = torch.linalg.svd(B, full_matrices=False)
        U = Vh[:r_pca].T
        Q, _ = torch.linalg.qr(U)
        u_by_module[name] = Q[:, :r_pca].contiguous()

    metadata = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "num_sequences": num_sequences,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "tokens_per_batch": tokens_per_batch,
        "r_pca": r_pca,
        "seed": seed,
        "layer_range": layer_range,
        "log_every": log_every,
        "module_suffixes": module_suffixes,
        "module_names": sorted(list(u_by_module.keys())),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"u_by_module": u_by_module, "metadata": metadata}, out_path)
    logger.info("Saved subspaces to %s", out_path)


if __name__ == "__main__":
    main()
