#!/usr/bin/env bash
# Reproduces the analysis from the openFDA download and the archived classifications.
set -euo pipefail
cd "$(dirname "$0")/scripts"
python3 00_fetch_openfda.py          # downloads data/raw/maude_ttfields_masterfile.csv (public domain, not redistributed)
python3 00b_select_reports.py        # inclusion filter; rebuilds the 548 study narratives
python3 rule_based_classifier.py     # re-runs the rule-based arm and compares with the archived output
python3 01_build_dataset.py
python3 02_clinical.py
python3 03_figures_clinical.py
python3 04_concordance.py
python3 10_interrater.py
python3 05_figures_concordance.py
python3 09_double_coding_sample.py   # re-draws the double-coding sample and checks it against the archived key
