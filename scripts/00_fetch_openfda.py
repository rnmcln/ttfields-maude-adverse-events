#!/usr/bin/env python3
"""Retrieve TTFields-related device adverse event reports from the openFDA device event API.

Reproduces the retrieval of 17 October 2025 (same query, same field layout). openFDA data are
public domain (CC0); they are not redistributed in this repository and must be downloaded with
this script. Reports can receive follow-up submissions after the original retrieval, so a new
download can differ from the study snapshot; 01_build_dataset.py compares each narrative with the
SHA-256 checksum of the study snapshot and reports any differences.

Usage: python scripts/00_fetch_openfda.py            (optional: set OPENFDA_API_KEY)
Output: data/raw/maude_ttfields_masterfile.csv
"""
import json, os, time, urllib.parse, urllib.request
import pandas as pd
from paths import RAW

ENDPOINT = "https://api.fda.gov/device/event.json"
QUERY = ('(mdr_text.text:"tumor treating fields" OR mdr_text.text:"ttfields" OR mdr_text.text:"ttf" '
         'OR device.brand_name:OPTUNE OR device.brand_name:"OPTUNE S" OR device.generic_name:"tumor treating fields" '
         'OR manufacturer_d_name:NOVOCURE OR manufacturer_d_name:"NOVOCURE GMBH" OR manufacturer_d_name:"NOVOCURE INC")')
DEVICE_FIELDS = ["brand_name", "generic_name", "manufacturer_d_name", "model_number", "catalog_number",
                 "lot_number", "device_report_product_code", "device_operator"]

def get(params):
    key = os.environ.get("OPENFDA_API_KEY")
    if key: params = {**params, "api_key": key}
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"openFDA request failed: {url}")

def flatten(rec):
    dev = (rec.get("device") or [{}])[0]
    texts = rec.get("mdr_text") or []
    row = {
        "report_number": rec.get("report_number"),
        "event_type": rec.get("event_type"),
        "date_received": rec.get("date_received"),
        "date_of_event": rec.get("date_of_event"),
        "manufacturer_d_name": rec.get("manufacturer_d_name"),
        "report_source_code": rec.get("report_source_code"),
        "initial_report_to_fda": rec.get("initial_report_to_fda"),
        "type_of_report": str(rec.get("type_of_report")) if rec.get("type_of_report") is not None else None,
        "patient_problem_code": None,
        "product_problems": str(list(dict.fromkeys(rec["product_problems"]))) if rec.get("product_problems") else None,
    }
    for f in DEVICE_FIELDS:
        row[f"device_0_{f}"] = dev.get(f)
    row["mdr_text_concat"] = " | ".join(t.get("text", "") for t in texts)
    return row

def main():
    first = get({"search": QUERY, "limit": 1})
    total = first["meta"]["results"]["total"]
    step = 1000 if os.environ.get("OPENFDA_API_KEY") else 100   # requests without a key are limited to 100 records
    rows, skip = [], 0
    while skip < total:
        page = get({"search": QUERY, "limit": step, "skip": skip})
        rows += [flatten(r) for r in page["results"]]
        skip += step
    df = pd.DataFrame(rows).drop_duplicates("report_number")
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW / "maude_ttfields_masterfile.csv", index=False)
    meta = {"endpoint": ENDPOINT, "query": QUERY, "n_records": len(df),
            "download_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(meta, open(RAW / "retrieval_meta.json", "w"), indent=1)
    print(f"{len(df)} records written to {RAW / 'maude_ttfields_masterfile.csv'}")

if __name__ == "__main__":
    main()
