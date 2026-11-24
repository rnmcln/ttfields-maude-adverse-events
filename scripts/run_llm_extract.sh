#!/bin/bash
# Cross-check extraction with qwen2.5:7b (Ollama), then both classification arms and the extraction on the
# eligible reports that the original query did not return. Outputs are written to data/derived/llm_rerun.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../data/derived/llm_rerun
python3 llm_extract.py qwen2.5:7b ../data/derived/llm_rerun/qwen2.5_7b_extract.jsonl ../data/annotations/llm_extract_order.txt
export REPORTS_FILE="$(cd .. && pwd)/data/derived/missed_reports_for_llm.jsonl"
python3 llm_extract.py qwen2.5:7b ../data/derived/llm_rerun/missed_qwen2.5_7b_extract.jsonl
python3 llm_classify.py qwen2.5:7b ../data/derived/llm_rerun/missed_qwen2.5_7b.jsonl
python3 llm_classify.py llama3.1:8b ../data/derived/llm_rerun/missed_llama3.1_8b.jsonl
