#!/usr/bin/env python3
"""Retrieve the Novocure reports within the study window that the original query did not return
(identified by 11_cohort_audit.py), in the same format as the original extraction, and apply the
rule-based classifier. The narratives are kept locally (analysis/data) and are not redistributed.
Outputs: analysis/data/missed_reports.csv, analysis/data/missed_reports_for_llm.jsonl,
analysis/outputs/cohort_audit/missed_reports_regex.csv
"""
import json, hashlib, pathlib, sys, urllib.request, urllib.parse, os
from paths import ROOT, DERIVED, ANN, RESULTS, TABLES, FIGURES, CODEBOOK
import pandas as pd

import rule_based_classifier as ps

aud = pd.read_csv(DERIVED / "cohort_audit" / "openfda_audit_records.csv", dtype=str)
nov = aud.manufacturer_d_name.fillna("").str.upper().str.contains("NOVO") | aud.product_code.isin(["NZK", "SDA", "QGZ"])
miss = aud[(aud.in_study == "False") & nov & (aud.date_received <= "20250930")]
rns = sorted(miss.report_number)
s = " OR ".join(f'report_number:"{r}"' for r in rns)
q = {"search": f"({s})", "limit": 100}
if os.environ.get("OPENFDA_API_KEY"): q["api_key"] = os.environ["OPENFDA_API_KEY"]
res = json.load(urllib.request.urlopen("https://api.fda.gov/device/event.json?" + urllib.parse.urlencode(q), timeout=60))["results"]
assert len(res) == len(rns), (len(res), len(rns))
rows = []
for r in res:
    d0 = (r.get("device") or [{}])[0]
    rows.append({"report_number": r["report_number"], "event_type": r.get("event_type"), "date_received": r.get("date_received"),
                 "date_of_event": r.get("date_of_event"), "manufacturer_d_name": d0.get("manufacturer_d_name"),
                 "report_source_code": r.get("report_source_code"), "initial_report_to_fda": r.get("initial_report_to_fda"),
                 "type_of_report": "|".join(r.get("type_of_report") or []),
                 "product_problems": "|".join(dict.fromkeys(r.get("product_problems") or [])),
                 "device_0_brand_name": d0.get("brand_name"), "device_0_generic_name": d0.get("generic_name"),
                 "device_0_manufacturer_d_name": d0.get("manufacturer_d_name"),
                 "device_0_device_report_product_code": d0.get("device_report_product_code"),
                 "mdr_text_concat": " | ".join(t.get("text", "") for t in (r.get("mdr_text") or []))})
m = pd.DataFrame(rows).sort_values(["date_received", "report_number"]).reset_index(drop=True)
m.insert(0, "row_id", range(1001, 1001 + len(m)))
m.to_csv(DERIVED / "missed_reports.csv", index=False)
with open(DERIVED / "missed_reports_for_llm.jsonl", "w") as fh:
    for _, r in m.iterrows():
        fh.write(json.dumps({"row_id": int(r.row_id), "report_number": r.report_number, "text": r.mdr_text_concat,
                             "sha256": hashlib.sha256(str(r.mdr_text_concat).encode()).hexdigest()}) + "\n")
rx = m.apply(ps.classify_regex, axis=1, result_type="expand")
out = pd.concat([m[["row_id", "report_number", "date_received", "event_type", "device_0_brand_name"]], rx], axis=1)
out.to_csv(ANN / "missed_reports_rule_based.csv", index=False)
print(len(m), m.event_type.value_counts().to_dict())
print(out.filter(like="primary").iloc[:, 0].value_counts().to_dict())
