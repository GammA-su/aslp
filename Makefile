PY := uv run python
BUILD_U_CFG := configs/build_u.yaml
TRAIN_CFG := configs/train.yaml
EVAL_CFG := configs/eval.yaml
MODE ?= baseline
SEED ?= 42
MODEL ?= meta-llama/Llama-3.1-8B-Instruct
RUN_DIR ?= artifacts/runs/$(MODE)_seed$(SEED)
LORA_PATH ?= $(RUN_DIR)
RUN_DIRS ?=

.PHONY: build_u toy_data train eval_general eval_domain summarize

build_u:
	$(PY) scripts/build_u.py --config $(BUILD_U_CFG) --model $(MODEL)

toy_data:
	$(PY) scripts/make_toy_domain_data.py --out_dir data

train:
	$(PY) scripts/train_lora.py --config $(TRAIN_CFG) --mode $(MODE) --seed $(SEED) --output_dir $(RUN_DIR) --model $(MODEL)

eval_general:
	$(PY) scripts/eval_ppl.py --config $(EVAL_CFG) --eval_type general --mode $(MODE) --seed $(SEED) --lora_path $(LORA_PATH) --out_json $(RUN_DIR)/general.json --model $(MODEL)

eval_domain:
	$(PY) scripts/eval_ppl.py --config $(EVAL_CFG) --eval_type domain --mode $(MODE) --seed $(SEED) --eval_jsonl data/domain_eval.jsonl --lora_path $(LORA_PATH) --out_json $(RUN_DIR)/domain.json --model $(MODEL)

summarize:
	$(PY) scripts/summarize_results.py $(foreach dir,$(RUN_DIRS),--run_dir $(dir))
