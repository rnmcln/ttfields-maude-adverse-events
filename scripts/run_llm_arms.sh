#!/usr/bin/env bash
# Re-runs both LLM arms on a machine with Ollama serving qwen2.5:7b and llama3.1:8b (Q4_K_M).
# Outputs go to data/derived/llm_rerun/ so that the archived study outputs in data/annotations/llm/ are not overwritten.
set -euo pipefail
cd "$(dirname "$0")"
OUT=../data/derived/llm_rerun; mkdir -p "$OUT"
python3 llm_classify.py qwen2.5:7b "$OUT/qwen2.5_7b.jsonl"
ollama stop qwen2.5:7b || true
python3 llm_classify.py llama3.1:8b "$OUT/llama3.1_8b.jsonl"
