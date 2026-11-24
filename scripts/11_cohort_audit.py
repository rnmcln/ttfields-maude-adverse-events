#!/usr/bin/env python3
"""Cohort ascertainment audit against the openFDA device event endpoint.
Runs independently constructed search arms (product code, approval number, brand name, generic name,
manufacturer name, narrative keywords), restricted to the study's date_received window, and compares
the report numbers retrieved with the 548 included reports. Also retrieves the current structured
record of every included report to audit the report unit (summary reports, number of events
summarised, report type, source) and reporter country.
Output: data/derived/cohort_audit/openfda_audit_records.csv (FDA record fields, not distributed), results/cohort_audit/audit_meta.json. Uses OPENFDA_API_KEY if set.
"""
import json, os, time, pathlib, urllib.request, urllib.parse
from paths import ROOT, DERIVED, ANN, RESULTS, TABLES, FIGURES, CODEBOOK
import pandas as pd

OUT = DERIVED / "cohort_audit"; OUT.mkdir(parents=True, exist_ok=True)
OUTR = RESULTS / "cohort_audit"; OUTR.mkdir(parents=True, exist_ok=True)
EP = "https://api.fda.gov/device/event.json"
KEY = os.environ.get("OPENFDA_API_KEY")
WINDOW = "date_received:[20120101 TO 20251017]"
ARMS = {
    "product_code_NZK": 'device.device_report_product_code:"NZK"',
    "product_code_SDA": 'device.device_report_product_code:"SDA"',
    "product_code_QGZ": 'device.device_report_product_code:"QGZ"',
    "approval_P100034": 'pma_pmn_number:"P100034"',
    "approval_H180002": 'pma_pmn_number:"H180002"',
    "approval_P230042": 'pma_pmn_number:"P230042"',
    "brand_optune": 'device.brand_name:optune',
    "brand_novottf": '(device.brand_name:novottf OR device.brand_name:"novo-ttf" OR device.brand_name:"novo ttf")',
    "generic_tumor_treating_fields": '(device.generic_name:"tumor treating fields" OR device.generic_name:"tumor treatment fields" OR device.generic_name:ttfields)',
    "manufacturer_novocure": '(device.manufacturer_d_name:novocure OR device.manufacturer_d_name:"novo cure" OR manufacturer_name:novocure)',
    "narrative_keywords": '(mdr_text.text:"tumor treating fields" OR mdr_text.text:"tumour treating fields" OR mdr_text.text:ttfields OR mdr_text.text:optune OR mdr_text.text:novottf OR mdr_text.text:novocure)',
}

def get(search, limit, skip=0):
    q = {"search": search, "limit": limit, "skip": skip}
    if KEY: q["api_key"] = KEY
    url = EP + "?" + urllib.parse.urlencode(q)
    for i in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404: return {"meta": {"results": {"total": 0}}, "results": []}
            time.sleep(3 * (i + 1))
        except Exception:
            time.sleep(3 * (i + 1))
    raise RuntimeError(url)

def fetch_all(search):
    lim = 1000 if KEY else 100
    first = get(search, lim); total = first["meta"]["results"]["total"]; rows = first["results"]
    while len(rows) < total:
        rows += get(search, lim, len(rows))["results"]; time.sleep(0.3)
    return rows

def flat(r):
    d0 = (r.get("device") or [{}])[0]
    return {"report_number": r.get("report_number"), "mdr_report_key": r.get("mdr_report_key"),
            "date_received": r.get("date_received"), "date_changed": r.get("date_changed"),
            "event_type": r.get("event_type"), "brand_name": d0.get("brand_name"), "generic_name": d0.get("generic_name"),
            "manufacturer_d_name": d0.get("manufacturer_d_name"), "product_code": d0.get("device_report_product_code"),
            "pma_pmn_number": r.get("pma_pmn_number"), "n_devices": len(r.get("device") or []),
            "summary_report_flag": r.get("summary_report_flag"), "noe_summarized": r.get("noe_summarized"),
            "number_patients_in_event": r.get("number_patients_in_event"), "n_patient_entries": len(r.get("patient") or []),
            "type_of_report": "|".join(r.get("type_of_report") or []), "mfr_report_type": r.get("mfr_report_type"),
            "report_source_code": r.get("report_source_code"), "source_type": "|".join(r.get("source_type") or []),
            "reporter_country_code": r.get("reporter_country_code"), "reporter_occupation_code": r.get("reporter_occupation_code"),
            "health_professional": r.get("health_professional"), "suppl_dates_fda_received": r.get("suppl_dates_fda_received"),
            "date_of_event": r.get("date_of_event"), "date_report": r.get("date_report"),
            "n_text_segments": len(r.get("mdr_text") or [])}

if __name__ == "__main__":
    inc = pd.read_csv(DERIVED / "analysis_dataset.csv", usecols=["row_id", "report_number"])
    incl = set(inc.report_number)
    arm_sets, recs = {}, {}
    for name, s in ARMS.items():
        rows = fetch_all(f"({s}) AND {WINDOW}")
        arm_sets[name] = {r["report_number"] for r in rows}
        for r in rows: recs[r["report_number"]] = flat(r)
        print(name, len(rows), "in study:", len(arm_sets[name] & incl), flush=True)
    # current records of all included reports (report_number lookup, batches)
    missing = [rn for rn in incl if rn not in recs]
    for i in range(0, len(missing), 40):
        s = " OR ".join(f'report_number:"{rn}"' for rn in missing[i:i + 40])
        for r in fetch_all(f"({s})"): recs[r["report_number"]] = flat(r)
    allr = pd.DataFrame(recs.values())
    for name, st in arm_sets.items():
        allr["arm_" + name] = allr.report_number.isin(st)
    allr["in_study"] = allr.report_number.isin(incl)
    allr.to_csv(OUT / "openfda_audit_records.csv", index=False)
    json.dump({"window": WINDOW, "arms": ARMS, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "n_by_arm": {k: len(v) for k, v in arm_sets.items()}}, open(OUTR / "audit_meta.json", "w"), indent=1)
    print("records", len(allr), "study reports found now:", int(allr.in_study.sum()))
