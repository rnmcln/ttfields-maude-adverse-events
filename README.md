# Adverse event reports for Tumor Treating Fields devices in the FDA MAUDE database

Code, classification codebook and annotations for an analysis of all adverse event reports for Tumor Treating Fields (TTFields; Optune, Optune Gio, Optune Lua, NovoTTF-100A) in the US Food and Drug Administration Manufacturer and User Facility Device Experience (MAUDE) database, received between 21 February 2012 and 30 September 2025, together with a comparison of human, rule-based and large language model classification of the report narratives.

Department of Neurosurgery, Jena University Hospital, Friedrich Schiller University Jena, Germany.
Authors: Sophie Nickel, Mareike Kienow, Christian Senft, Falko Schwarz, Aaron Lawson McLean.

## Summary

All 548 TTFields reports retrieved from openFDA on 17 October 2025 were classified on a five-dimension codebook (primary event category, severity, causality, device component, secondary event) by a trained rater; a second rater independently classified a random sample of 80 reports. A rule-based classifier and two locally run open-weight language models (Qwen 2.5 7B and Llama 3.1 8B, via Ollama) classified the same reports using the same codebook.

- Infection (34.1%) and non-infectious skin reactions (27.4%) accounted for 61.5% of reports, followed by falls or entanglement (13.3%) and neurological events (11.5%). No report was classified as a primary hardware malfunction.
- Of wound and infection reports, 53.4% involved a prior surgical site and 24.9% described surgical wound revision.
- A death was described in 23 reports, of which 5 carried the MAUDE event type "death".
- Agreement on the primary category: two human raters κ = 0.57; human reference against the rule-based classifier κ = 0.59, Qwen 2.5 7B κ = 0.63, Llama 3.1 8B κ = 0.57.

MAUDE contains no exposure denominator; all proportions describe the composition of reports, not event risk.

## Repository structure

```
codebook/            classification codebook (v1.0), LLM system prompt, pilot prompt (v1, not used in the analysis)
data/annotations/    reports.csv               study report numbers with SHA-256 checksums of each narrative and its text segments
                     rater1_full.csv           first human rater, all reports (reference standard)
                     rater2_sample.csv         second human rater, random sample of 80 reports
                     double_coding_sample_key.csv
                     rule_based.csv            rule-based classifier output
                     llm/                      LLM classifications (JSON lines: codes, settings, timings and SHA-256 checksums of
                                               prompt, report text and raw response; free-text responses are omitted because
                                               they can quote report narratives)
data/raw/            openFDA download (created by 00_fetch_openfda.py; not distributed)
data/derived/        files containing report narratives (created by the pipeline; not distributed)
results/             results JSON files, tables (CSV) and figures (PDF, PNG)
scripts/             analysis pipeline
```

## Reproducing the analysis

Python 3.10 or later; install dependencies with `pip install -r requirements.txt`, then run `./run_all.sh`. An openFDA API key is optional (`export OPENFDA_API_KEY=...`).

1. `00_fetch_openfda.py` downloads all reports matching the original query from openFDA.
2. `00b_select_reports.py` applies the inclusion filter (manufacturer Novocure, brand name Optune or NovoTTF, or product code NZK) and rebuilds the 548 study narratives in the order of the original snapshot using the segment checksums in `data/annotations/reports.csv`.
3. `rule_based_classifier.py` re-runs the rule-based arm and compares it with the archived output.
4. `01_build_dataset.py` to `05_figures_concordance.py` and `10_interrater.py` produce the results, tables and figures; `09_double_coding_sample.py` re-draws the double-coding sample and checks it against the archived key.

The language model arm can be re-run with `scripts/run_llm_arms.sh` on a machine running Ollama with `qwen2.5:7b` and `llama3.1:8b` (Q4_K_M); settings are temperature 0, seed 20251017, context 8,192 tokens, maximum 512 output tokens and JSON-constrained output. Re-runs are written to `data/derived/llm_rerun/`.

### Differences between a new download and the study snapshot

MAUDE reports receive follow-up submissions after their initial publication. In a download made on 23 November 2026, 546 of the 548 study narratives could be rebuilt exactly from their text segments; two reports (3010457505-2025-00548 and 3010457505-2025-00584) contained segments edited after the snapshot, and their current text is used. Twelve further reports received by 17 October 2025 meet the inclusion criteria but were published in openFDA after the retrieval; they are not part of the study set. With that download, all classification and agreement results reproduced exactly; the count of reports with follow-up submissions, the product problem field and one text-length quartile differed because these structured fields had been updated.

## Data sources and licences

- MAUDE data are retrieved from the openFDA device adverse event API; openFDA content is public domain (CC0 1.0). Report narratives are not redistributed here.
- Code: MIT licence (`LICENSE`).
- Codebook, annotations and results: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Citation

See `CITATION.cff`.
