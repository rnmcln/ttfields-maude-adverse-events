#!/usr/bin/env python3
"""Apply the inclusion filter to the openFDA download and reconstruct the study narratives.

1. Inclusion: manufacturer contains NOVOCURE, brand name contains OPTUNE / NOVO-TTF / NOVOTTF, or FDA
   product code NZK. Reports received after 17 October 2025 are outside the study period.
2. The study set is the 548 reports in data/annotations/reports.csv. For each report, the narrative is
   rebuilt from the downloaded text segments in the order of the study snapshot, using the per-segment
   SHA-256 checksums. Reports whose segments were edited after the snapshot cannot be rebuilt exactly;
   their current text is used and they are listed in data/derived/reconstruction_report.csv.
Output: data/derived/maude_ttfields_clean.csv, data/derived/reports_for_llm.jsonl
"""
import hashlib, json
import pandas as pd
from paths import RAW, DERIVED, ANN

H = lambda t: hashlib.sha256(t.encode()).hexdigest()
raw = pd.read_csv(RAW / "maude_ttfields_masterfile.csv", dtype=str)
brand = raw.device_0_brand_name.fillna("").str.upper(); mfr = raw.device_0_manufacturer_d_name.fillna("").str.upper()
pcode = raw.device_0_device_report_product_code.fillna("").str.upper()
incl = raw[(brand.str.contains("OPTUNE|NOVO-TTF|NOVOTTF") | mfr.str.contains("NOVOCURE") | (pcode == "NZK"))
           & (raw.date_received.astype(int) <= 20251017)].copy()
study = pd.read_csv(ANN / "reports.csv", dtype={"report_number": str})
print(f"Downloaded records: {len(raw)}; meeting inclusion criteria up to 17 Oct 2025: {len(incl)}; study reports: {len(study)}")
print(f"Study reports missing from download: {len(set(study.report_number) - set(incl.report_number))}; "
      f"eligible reports not in study set: {len(set(incl.report_number) - set(study.report_number))}")

raw = raw.set_index("report_number")
rows, recon = [], []
for r in study.itertuples():
    rec = raw.loc[r.report_number].to_dict()
    segs = str(rec["mdr_text_concat"]).split(" | ")
    by_hash = {H(s): s for s in segs}
    want = r.segment_sha256.split("|")
    if all(h in by_hash for h in want):
        text = " | ".join(by_hash[h] for h in want); status = "exact" if text == rec["mdr_text_concat"] else "reordered or follow-up segments removed"
    else:
        text = rec["mdr_text_concat"]; status = "changed after snapshot (current text used)"
    assert status == "changed after snapshot (current text used)" or H(text) == r.text_sha256
    recon.append({"row_id": r.row_id, "report_number": r.report_number, "status": status})
    rows.append({"row_id": r.row_id, "report_number": r.report_number, **{k: v for k, v in rec.items() if k != "mdr_text_concat"}, "mdr_text_concat": text})
clean = pd.DataFrame(rows)
model_map = {"TFH-9000": "TFH-9000", "TFH9000": "TFH-9000", "TFH-9100": "TFH-9100", "TFH 9100": "TFH-9100", "TFH9100": "TFH-9100",
             "TFH9100 LBL": "TFH-9100", "TFL9050": "TFL-9050", "TFT9200": "TFT-9200"}
clean["device_model_std"] = clean.device_0_model_number.fillna("").str.strip().map(model_map).fillna("UNKNOWN")
clean["date_received"] = clean.date_received.astype(int)
clean["year_received"] = clean.date_received // 10000
clean.to_csv(DERIVED / "maude_ttfields_clean.csv", index=False)
pd.DataFrame(recon).to_csv(DERIVED / "reconstruction_report.csv", index=False)
with open(DERIVED / "reports_for_llm.jsonl", "w") as fh:
    for r in clean.itertuples():
        fh.write(json.dumps({"row_id": int(r.row_id), "report_number": r.report_number, "sha256": H(str(r.mdr_text_concat)), "text": str(r.mdr_text_concat)}) + "\n")
print(pd.Series([x["status"] for x in recon]).value_counts().to_string())
