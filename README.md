# Wound, surgical-site and fall-related complications reported with Tumor Treating Fields in the FDA MAUDE database

Code, classification definitions, extraction rules and annotations for a narrative analysis of adverse event reports for Tumor Treating Fields (TTFields; Optune, Optune Gio, Optune Lua, NovoTTF-100A) in the US Food and Drug Administration Manufacturer and User Facility Device Experience (MAUDE) database, received between 21 February 2012 and 30 September 2025.

Department of Neurosurgery, Jena University Hospital, Friedrich Schiller University Jena, Germany.
Authors: Sophie Nickel, Mareike Kienow, Christian Senft, Falko Schwarz, Aaron Lawson McLean.

## Summary

The study cohort comprises 548 reports retrieved from openFDA on 17 October 2025. Each report was classified by a trained rater (event category, severity, causality, device component, secondary event); a second rater recoded a random sample of 80. Clinical attributes (operative-site involvement, dehiscence, exposure of bone or implants, infection, operative procedures, fall mechanisms and injuries, death attribution) were extracted from the narratives with sentence-level rules and cross-checked with a locally run language model (Qwen 2.5 7B). A rule-based classifier and two language models (Qwen 2.5 7B, Llama 3.1 8B) classified the event category for a secondary comparison.

- Retrieval audit: independent openFDA search arms and the FDA bulk MAUDE device files identified 590 eligible Novocure reports; the original query had retrieved 548. The 42 missed reports (2012 to 2015, filed under spelling variants of NovoTTF-100A) are described with automated methods only.
- Of 326 wound and infection reports concerning devices for intracranial tumours, 53.7% involved a prior operative site, 16.0% described exposed bone or implants and 40.2% an operative procedure for the complication (rule-based estimates; two-method ranges in the results files).
- Of 73 fall reports, 32.9% described a cable or device-carrying mechanism of the fall.
- Death was mentioned in 23 narratives; 5 were filed as death reports and 19 attributed the death to another cause.
- Report content changed from early 2024 (Optune Gio brand, higher volume, lower share of reports graded severe), so proportions are reported by period.
- The language-model extraction agreed closely with the rules for the main wound attributes but was unreliable for fall mechanisms and causal statements.

MAUDE contains no exposure denominator; all proportions describe reports, not event risk.

## Repository structure

```
codebook/            classification definitions (v1.0), classification prompt, pilot prompt (v1, not used),
                     cross-check extraction prompt (llm_extract_prompt.txt)
data/annotations/    reports.csv                     study report numbers with SHA-256 checksums of each narrative and its segments
                     rater1_full.csv                 first human rater, all reports (reference)
                     rater2_sample.csv               second human rater, random sample of 80 reports
                     double_coding_sample_key.csv
                     rule_based.csv                  rule-based classifier output
                     llm/                            language-model classifications (codes, settings, timings, checksums;
                                                     free-text responses omitted)
                     attributes_rule_based_v2.csv    clinical attributes, sentence-level rules (study and missed reports)
                     llm_extract/, llm_missed/       language-model extraction and classification of the missed reports
                                                     (field values and checksums; raw responses omitted)
                     attributes_combined.csv         rule-based and language-model attributes merged
                     missed_reports_rule_based.csv   rule-based classification of the missed reports
data/raw/            openFDA download (not distributed)
data/derived/        files containing narratives or FDA record fields (created by the pipeline; not distributed)
results/             results JSON files, tables (CSV), figures (PDF, PNG), audit metadata
scripts/             analysis pipeline
```

## Reproducing the analysis

Python 3.10 or later; install dependencies with `pip install -r requirements.txt`, then run `./run_all.sh`. An openFDA API key is optional (`export OPENFDA_API_KEY=...`). The published results were produced with Python 3.13, pandas 3.0, SciPy 1.18, statsmodels 0.15 and scikit-learn 1.9.

1. `00_fetch_openfda.py` and `00b_select_reports.py` download the reports and rebuild the 548 study narratives from the checksums in `data/annotations/reports.csv`.
2. `rule_based_classifier.py`, `01_build_dataset.py` to `05_figures_concordance.py` and `10_interrater.py` reproduce the classification and agreement results.
3. `11_cohort_audit.py` runs the openFDA search arms; `11b_bulk_maude_scan.sh` scans the FDA bulk device files (about 700 MB are downloaded and discarded); `12_missed_reports.py` retrieves the reports that the original query did not return.
4. `13_descriptors_v2.py` extracts the clinical attributes with sentence-level rules; `14_extraction_compare.py` merges them with the language-model extraction; `15_clinical_v2.py`, `16_concordance_v2.py` and `17_figures_v2.py` produce the revised results, tables and figures.

The language-model steps can be re-run on a machine running Ollama with `qwen2.5:7b` and `llama3.1:8b` (Q4_K_M): `scripts/run_llm_arms.sh` (classification) and `scripts/run_llm_extract.sh` (cross-check extraction and missed reports); settings are temperature 0, seed 20251017, context 8,192 tokens, maximum 512 output tokens and JSON-constrained output. Re-runs are written to `data/derived/llm_rerun/`.

### Differences between a new download and the study snapshot

MAUDE reports receive follow-up submissions after their initial publication. In a download made on 23 November 2026, 546 of the 548 study narratives could be rebuilt exactly from their text segments; two reports (3010457505-2025-00548 and 3010457505-2025-00584) contained segments edited after the snapshot, and their current text is used. Twelve further reports received between 1 and 16 October 2025 meet the inclusion criteria but were published in openFDA after the retrieval; they lie outside the study window. Structured fields such as follow-up counts, reporter country and source type are taken from the current record.

## Data sources and licences

- MAUDE data are retrieved from openFDA and the FDA bulk MAUDE files; openFDA content is public domain (CC0 1.0). Report narratives and FDA record fields are not redistributed here.
- Code: MIT licence (`LICENSE`).
- Definitions, annotations and results: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Citation

See `CITATION.cff`.
