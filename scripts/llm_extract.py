#!/usr/bin/env python3
"""Automated cross-check extraction of clinical attributes (Ollama, local). Standard library only.
Usage: python3 llm_extract.py MODEL OUTFILE.jsonl [ORDER_FILE]
ORDER_FILE: optional text file with one row_id per line giving the processing order; remaining
reports follow in row order. Resumable. Settings as for the classification arms: temperature 0,
seed 20251017, num_predict 512, num_ctx 8192, JSON-constrained output; one resubmission if the
response cannot be parsed as JSON. Field values outside the permitted set are recorded as missing here and
normalised in the analysis script (13_extraction_compare.py) from the stored raw response.
"""
import json, sys, time, urllib.request, pathlib, re, hashlib

HERE = pathlib.Path(__file__).resolve().parent
import os
DATA = pathlib.Path(os.environ.get("REPORTS_FILE", HERE.parent / "data" / "derived" / "reports_for_llm.jsonl"))
PROMPT = (HERE.parent / "codebook" / "llm_extract_prompt.txt").read_text()
URL = "http://localhost:11434/api/chat"
OPTIONS = {"temperature": 0.0, "seed": 20251017, "num_predict": 512, "num_ctx": 8192}
RUNTIME = {"use_mmap": False}
YN = {"yes", "no"}
SCHEMA = {
    "skin_or_wound_event": YN,
    "surgical_site_involved": {"yes", "no", "not_applicable"},
    "wound_dehiscence": YN,
    "exposed_implant_or_bone": YN,
    "infection_status": {"confirmed", "suspected", "none"},
    "bone_flap_or_osteomyelitis": YN,
    "surgery_for_event": YN,
    "surgery_types": ("list", {"wound_revision_or_closure", "debridement_or_washout", "incision_and_drainage",
                               "bone_flap_removal", "hardware_or_mesh_removal", "shunt_revision_or_removal",
                               "flap_or_graft_reconstruction", "other"}),
    "iv_antibiotics": YN,
    "hospitalised": YN,
    "ttfields_interruption": {"permanent", "temporary", "none_stated"},
    "fall": YN,
    "fall_mechanism": ("list", {"tripped_or_entangled_in_cable", "pulled_by_cable", "carrying_device",
                                "other_cause_stated", "not_stated"}),
    "fall_injuries": ("list", {"scalp_or_head_wound", "fracture", "intracranial_haemorrhage", "other_injury", "none_stated"}),
    "array_role_in_fall_injury": {"stated_contribution", "cannot_be_ruled_out", "not_mentioned", "not_applicable"},
    "death_reported": YN,
    "death_attribution": {"other_cause_stated", "device_contribution_not_excluded", "not_stated", "not_applicable"},
    "reporter_attribution": {"related", "cannot_be_ruled_out", "not_related", "not_stated"},
    "device_mechanism_described": YN,
    "bevacizumab": {"current_or_recent", "previous_only", "not_receiving", "not_mentioned"},
}

def call(model, text):
    payload = {"model": model, "stream": False, "format": "json", "keep_alive": "30m",
               "options": {**OPTIONS, **RUNTIME},
               "messages": [{"role": "system", "content": PROMPT},
                            {"role": "user", "content": "Extract the fields from this MAUDE report:\n\n" + text}]}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def validate(content):
    """Return (cleaned dict, n_invalid_fields) or (None, None) if not parseable."""
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        return None, None
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return None, None
    out, bad = {}, 0
    for k, allowed in SCHEMA.items():
        v = d.get(k)
        if isinstance(allowed, tuple):
            vals = v if isinstance(v, list) else ([] if v in (None, "", "none") else [v])
            vals = [str(x).strip().lower() for x in vals]
            ok = [x for x in vals if x in allowed[1]]
            bad += len(vals) - len(ok)
            out[k] = sorted(set(ok))
        else:
            s = str(v).strip().lower() if v is not None else None
            if s in allowed:
                out[k] = s
            else:
                out[k] = None; bad += 1
    return out, bad

def main(model, outfile, order_file=None):
    out = pathlib.Path(outfile)
    done = set()
    if out.exists():
        done = {json.loads(l)["row_id"] for l in out.read_text().splitlines() if l.strip()}
    reports = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    if order_file:
        order = [int(x) for x in pathlib.Path(order_file).read_text().split()]
        pos = {r: i for i, r in enumerate(order)}
        reports.sort(key=lambda r: (pos.get(r["row_id"], 10**6), r["row_id"]))
    prompt_sha = hashlib.sha256(PROMPT.encode()).hexdigest()
    t0 = time.time(); n_new = 0
    with out.open("a") as fh:
        for rep in reports:
            if rep["row_id"] in done:
                continue
            attempts, parsed, nbad = [], None, None
            for attempt in range(2):
                ts = time.time()
                try:
                    resp = call(model, rep["text"]); content = resp["message"]["content"]
                except Exception as e:
                    content, resp = f"__ERROR__ {e!r}", {}
                attempts.append({"content": content, "seconds": round(time.time() - ts, 2),
                                 "prompt_eval_count": resp.get("prompt_eval_count"), "eval_count": resp.get("eval_count")})
                parsed, nbad = validate(content)
                if parsed is not None:
                    break
            rec = {"row_id": rep["row_id"], "report_number": rep["report_number"], "text_sha256": rep["sha256"],
                   "model": model, "prompt_sha256": prompt_sha, "options": {**OPTIONS, **RUNTIME},
                   "parse_ok": parsed is not None, "n_invalid_fields": nbad, "n_attempts": len(attempts),
                   "parsed": parsed, "attempts": attempts, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            n_new += 1
            if n_new % 10 == 0:
                print(f"{model}: {len(done) + n_new}/{len(reports)} done, {(time.time() - t0) / n_new:.1f}s/report", flush=True)
    print(f"{model}: finished, {n_new} new records", flush=True)

if __name__ == "__main__":
    main(*sys.argv[1:4])
