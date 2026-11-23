#!/usr/bin/env python3
"""LLM classification arm (Ollama, local). Standard library only.
Usage: python3 llm_classify.py MODEL OUTFILE.jsonl   (input: data/derived/reports_for_llm.jsonl from 00b_select_reports.py)
Resumable: report row_ids already present in OUTFILE are skipped.
Settings: temperature 0, seed 20251017, num_predict 512, num_ctx 8192, JSON-constrained output.
Responses failing validation are retried once (protocol section 6.2); persistent failures are
recorded with parse_ok=false and coded UNCLASSIFIABLE downstream.
"""
import json, sys, time, urllib.request, pathlib, re, hashlib

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "derived" / "reports_for_llm.jsonl"
PROMPT = (HERE.parent / "codebook" / "llm_prompt.txt").read_text()
URL = "http://localhost:11434/api/chat"
PRIMARY = {"SKIN_NONINF","INFECTION","HARDWARE","FALL_ENTANGLE","NEURO","SYSTEMIC",
           "USER_ERROR","DISEASE_PROG","OTHER","UNCLASSIFIABLE"}
SEV = {"MILD","MODERATE","SEVERE","FATAL"}
CAUS = {"DEFINITE","PROBABLE","POSSIBLE","UNLIKELY","UNASSESSABLE"}
OPTIONS = {"temperature": 0.0, "seed": 20251017, "num_predict": 512, "num_ctx": 8192}
# Runtime-only setting (does not change model outputs): keep weights resident in memory to avoid paging.
RUNTIME = {"use_mmap": False}

def call(model, text):
    payload = {"model": model, "stream": False, "format": "json", "keep_alive": "30m",
               "options": {**OPTIONS, **RUNTIME},
               "messages": [{"role": "system", "content": PROMPT},
                            {"role": "user", "content": "Classify this MAUDE report:\n\n" + text}]}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def validate(content):
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if str(d.get("primary_category","")).strip().upper() not in PRIMARY:
        return None
    return d

def main(model, outfile):
    out = pathlib.Path(outfile)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["row_id"])
    reports = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    prompt_sha = hashlib.sha256(PROMPT.encode()).hexdigest()
    t0 = time.time(); n_new = 0
    with out.open("a") as fh:
        for rep in reports:
            if rep["row_id"] in done:
                continue
            attempts = []
            parsed = None
            for attempt in range(2):
                ts = time.time()
                try:
                    resp = call(model, rep["text"])
                    content = resp["message"]["content"]
                except Exception as e:
                    content, resp = f"__ERROR__ {e!r}", {}
                attempts.append({"content": content, "seconds": round(time.time()-ts, 2),
                                 "prompt_eval_count": resp.get("prompt_eval_count"),
                                 "eval_count": resp.get("eval_count")})
                parsed = validate(content)
                if parsed is not None:
                    break
            rec = {"row_id": rep["row_id"], "report_number": rep["report_number"],
                   "text_sha256": rep["sha256"], "model": model,
                   "model_digest": resp.get("model") if isinstance(resp, dict) else None,
                   "prompt_sha256": prompt_sha, "options": {**OPTIONS, **RUNTIME},
                   "parse_ok": parsed is not None, "n_attempts": len(attempts),
                   "parsed": parsed, "attempts": attempts,
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            n_new += 1
            if n_new % 10 == 0:
                el = time.time() - t0
                print(f"{model}: {len(done)+n_new}/{len(reports)} done, {el/n_new:.1f}s/report", flush=True)
    print(f"{model}: finished, {n_new} new records", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
