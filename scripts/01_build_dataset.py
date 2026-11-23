#!/usr/bin/env python3
"""Build the analysis dataset.
Inputs: data/derived/maude_ttfields_clean.csv (548 study reports, from 00b_select_reports.py),
data/annotations/rater1_full.csv (human reference), data/annotations/rule_based.csv (rule-based arm),
data/annotations/llm/*.jsonl (LLM arms).
Output: data/derived/analysis_dataset.csv, data/derived/event_text_segments.csv, results/descriptor_patterns.json
"""
import json, re, hashlib, pathlib, warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd, numpy as np

from paths import DERIVED, ANN, RESULTS
CATS = ["SKIN_NONINF","INFECTION","HARDWARE","FALL_ENTANGLE","NEURO","SYSTEMIC","USER_ERROR","DISEASE_PROG","OTHER","UNCLASSIFIABLE"]
SEV = ["MILD","MODERATE","SEVERE","FATAL"]
CAUS = ["DEFINITE","PROBABLE","POSSIBLE","UNLIKELY","UNASSESSABLE"]
COMP = ["ARRAY","CABLE","BATTERY","UNIT","SOFTWARE","NONE","UNSPECIFIED"]

def norm(x, allowed, default=None):
    if x is None or (isinstance(x, float) and np.isnan(x)): return default
    v = str(x).strip().upper().replace(" ", "_") if str(x).strip() else ""
    v = v.replace("-", "_")
    if v == "": return default
    return v if v in allowed else ("INVALID:" + v)

def norm_comp(x):
    if x is None or (isinstance(x, float) and np.isnan(x)) or not str(x).strip(): return "UNSPECIFIED"
    parts = sorted({p.strip().upper() for p in re.split(r"[|,;/ ]+", str(x)) if p.strip()})
    parts = [p for p in parts if p in COMP] or ["UNSPECIFIED"]
    if len(parts) > 1: parts = [p for p in parts if p not in ("NONE","UNSPECIFIED")] or parts
    return "|".join(parts)

clean = pd.read_csv(DERIVED / "maude_ttfields_clean.csv", dtype={"report_number": str}).sort_values("row_id").reset_index(drop=True)
regex = pd.read_csv(ANN / "rule_based.csv", dtype={"report_number": str}).set_index("row_id").loc[clean.row_id].reset_index()
human = pd.read_csv(ANN / "rater1_full.csv", dtype={"report_number": str}).set_index("row_id").loc[clean.row_id].reset_index()
for x in (regex, human):
    assert (x.report_number.values == clean.report_number.values).all()
regex.columns = ["row_id", "report_number"] + ["regex_" + c for c in regex.columns[2:]]
human.columns = ["row_id", "report_number"] + ["human_" + c for c in human.columns[2:]]

df = clean.copy()
df["text_sha256"] = [hashlib.sha256(str(t).encode()).hexdigest() for t in df.mdr_text_concat]
for arm, src in [("human", human), ("regex", regex)]:
    df[f"{arm}_primary"] = [norm(v, CATS, "UNCLASSIFIABLE") for v in src[f"{arm}_primary_category"]]
    df[f"{arm}_secondary"] = [norm(v, CATS, "") for v in src[f"{arm}_secondary_category"]]
    df[f"{arm}_severity"] = [norm(v, SEV, "") for v in src[f"{arm}_severity"]]
    df[f"{arm}_causality"] = [norm(v, CAUS, "") for v in src[f"{arm}_causality"]]
    df[f"{arm}_component"] = [norm_comp(v) for v in src[f"{arm}_device_component"]]

# LLM arms (protocol 6.2: persistent parse failures coded UNCLASSIFIABLE)
for arm, fn in [("qwen", "qwen2.5_7b.jsonl"), ("llama", "llama3.1_8b.jsonl")]:
    p = ANN / "llm" / fn
    recs = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line); recs[r["row_id"]] = r
    def g(rid, key):
        r = recs.get(rid)
        if r is None: return None
        d = r["parsed"] or {}
        return d.get(key)
    df[f"{arm}_available"] = [rid in recs for rid in df.row_id]
    df[f"{arm}_parse_ok"] = [bool(recs[rid]["parse_ok"]) if rid in recs else False for rid in df.row_id]
    df[f"{arm}_sha_ok"] = [recs[rid]["text_sha256"] == h if rid in recs else False for rid, h in zip(df.row_id, df.text_sha256)]
    df[f"{arm}_primary"] = [norm(g(r, "primary_category"), CATS, "UNCLASSIFIABLE") if r in recs else None for r in df.row_id]
    df[f"{arm}_secondary"] = [norm(g(r, "secondary_category"), CATS, "") if r in recs else None for r in df.row_id]
    df[f"{arm}_severity"] = [norm(g(r, "severity"), SEV, "") if r in recs else None for r in df.row_id]
    df[f"{arm}_causality"] = [norm(g(r, "causality"), CAUS, "") if r in recs else None for r in df.row_id]
    df[f"{arm}_component"] = [norm_comp(g(r, "device_component")) if r in recs else None for r in df.row_id]
    df[f"{arm}_reasoning"] = [g(r, "reasoning") if r in recs else None for r in df.row_id]

# ---------------- narrative-derived descriptors ----------------
BOILER = re.compile(r"TRIAL|COMMERCIAL PROGRAM|PRESCRIBING INFORMATION|EXPECTED EVENT|INSTRUCTIONS FOR USE|"
                    r"NOVOCURE('S)? (MEDICAL )?OPINION|NOVOCURE AGREES|MEDICAL DEVICE REPORT|HDE#|CONTRIBUTING FACTOR|RISK FACTOR")
NEG = re.compile(r"\b(NO|NOT|WITHOUT|NEGATIVE FOR|DENIED|RULED OUT|NOR)\b")
def event_text(t):
    sents = re.split(r"(?<=[.!?])\s+|\s\|\s", str(t).upper())
    return " ".join(s for s in sents if not BOILER.search(s))
def has(p, t, neg=False, win=40):
    for m in re.finditer(p, t):
        if neg and NEG.search(t[max(0, m.start()-win):m.start()]):
            continue
        return True
    return False

df["event_text"] = [event_text(t) for t in df.mdr_text_concat]
full = df.mdr_text_concat.astype(str).str.upper()
ev = df.event_text
age = full.str.extract(r"\b(\d{1,3})[- ]YEARS?[- ]OLD")[0]
df["age"] = pd.to_numeric(age, errors="coerce")
df["sex"] = np.select([full.str.contains(r"\bFEMALE\b"), full.str.contains(r"\bMALE\b")], ["Female","Male"], "Not stated")
df["indication"] = np.select(
    [full.str.contains("MESOTHELIOMA"), full.str.contains(r"NON-SMALL CELL|NSCLC|LUNG CANCER"),
     full.str.contains(r"NEWLY DIAGNOSED (GLIOBLASTOMA|GBM)|NEWLY-DIAGNOSED (GLIOBLASTOMA|GBM)|NEWLY DIAGNOSED.{0,20}(GLIOBLASTOMA|GBM)"),
     full.str.contains(r"RECURRENT.{0,25}(GLIOBLASTOMA|GBM)|(GLIOBLASTOMA|GBM).{0,15}RECURREN"),
     full.str.contains(r"GLIOBLASTOMA|\bGBM\b|GLIOMA|ASTROCYTOMA")],
    ["Pleural mesothelioma","NSCLC","Newly diagnosed GBM","Recurrent GBM","Glioma, phase not stated"], "Not stated")
df["thoracic"] = df.device_0_brand_name.str.contains("LUA") | df.indication.isin(["Pleural mesothelioma","NSCLC"])

F = {
 "surgical_site": (r"CRANIOTOMY|RESECTION (SITE|SCAR|CAVITY|INCISION|AREA)|SURGICAL (SITE|INCISION|WOUND|SCAR)|"
                   r"INCISION(?! AND DRAINAGE)|\bSCAR\b|BONE FLAP|SUTURE LINE|STAPLE LINE", False),
 "implant_exposure": (r"EXPOS\w*.{0,60}(HARDWARE|PLATE|SCREW|MESH|SHUNT|BONE|CATHETER|TUBING|CRANIUM|SKULL|DURA)|"
                      r"(HARDWARE|PLATE|SCREW|MESH|SHUNT|TUBING|BONE|SKULL)S? (WAS |WERE |IS |ARE )?(VISIBLY )?EXPOSED", True),
 "surgical_revision": (r"UNDERWENT[^.]{0,160}(DEBRID|WASH ?OUT|REVISION|REMOV|CRANIECTOMY|CRANIOPLASTY|GRAFT|\bFLAP\b|EXCISION|"
                       r"DRAINAGE|CLOSURE|EXPLANT|EXPLOR)|BONE FLAP (WAS |WERE )?(REMOV|EXCIS|EXPLANT)|REMOVAL OF (THE )?(INFECTED )?BONE FLAP|"
                       r"(DEBRIDEMENT|WASHOUT|WOUND REVISION|REVISION SURGERY) (WAS )?PERFORMED", False),
 "organism_or_culture": (r"STAPH|STREPT|PROPIONI|CUTIBACT|MRSA|PSEUDOMON|ENTEROCOC|E\.? ?COLI|SERRATIA|KLEBSIEL|"
                         r"CULTURES? (WAS |WERE |CAME BACK )?POSITIVE|POSITIVE CULTURE|CULTURES? (DID )?GR[EO]W", True),
 "iv_antibiotics": (r"\bIV ANTIBIOTIC|INTRAVENOUS (ANTIBIOTIC|VANCOMYCIN|CEF)|VANCOMYCIN|CEFAZOLIN|CEFTRIAXONE|MEROPENEM|"
                    r"CEFEPIME|PIPERACILLIN|NAFCILLIN|DAPTOMYCIN|\bPICC\b", True),
 "hospitalised": (r"HOSPITALI[SZ]|\bADMITTED\b|ADMISSION", True),
 "temporary_interruption": (r"TEMPORARILY (DISCONTINU|INTERRUPT|STOPP|PAUS|SUSPEND|HELD|HALT)|THERAPY BREAK|TREATMENT BREAK|"
                            r"\bPAUSED\b|PLANN?ED TO RESUME|RESUMED (OPTUNE|THERAPY|TREATMENT|DEVICE)", False),
 "permanent_discontinuation": (r"PERMANENTLY (DISCONTINU|STOPP)|DISCONTINUED [A-Z ]{0,30}PERMANENTLY|WILL NOT RESUME|"
                               r"DID NOT RESUME|DECIDED TO (STOP|DISCONTINUE|END)", False),
 "fracture": (r"FRACTUR", True),
 "intracranial_bleed": (r"SUBDURAL|EPIDURAL H|SUBARACHNOID|INTRACRANIAL (HEMORRHAGE|HAEMORRHAGE|BLEED)|INTRAPARENCHYMAL|"
                        r"INTRACEREBRAL|BRAIN (BLEED|HEMORRHAGE)|(INTRACRANIAL|CEREBRAL|SUBDURAL|EPIDURAL) (HEMATOMA|HAEMATOMA)", True),
 "head_wound_after_fall": (r"LACERATION|SUTURE[DS]|STAPLED|STAPLES|HEAD (INJURY|TRAUMA|WOUND)|CONTUSION", True),
 "cable_or_bag_mechanism": (r"TRIPP?\w* (OVER|ON)|\bCORD\b|CABLE|WIRE|\bBAG\b|BACKPACK|STRAP|ENTANGL|CAUGHT|PULLED (HIM|HER|THE PATIENT|THE SUBJECT)", False),
 "bevacizumab": (r"BEVACIZUMAB|AVASTIN", False),
 "shunt": (r"SHUNT", False),
}
for k, (p, neg) in F.items():
    src = full if k == "bevacizumab" else ev
    df[k] = [has(p, t, neg) for t in src]
json.dump({k: {"pattern": v[0], "negation_window": v[1], "source": "full narrative" if k == "bevacizumab" else "event narrative (manufacturer commentary removed)"} for k, v in F.items()}, open(RESULTS / "descriptor_patterns.json", "w"), indent=1)
df["death_in_narrative"] = [has(r"\bDIED\b|PASSED AWAY|\bEXPIRED\b|\bDEATH\b|DECEASED", t, True) for t in full]
df.drop(columns=["event_text"]).to_csv(DERIVED / "analysis_dataset.csv", index=False)
df[["row_id","event_text"]].to_csv(DERIVED / "event_text_segments.csv", index=False)
print("LLM input text identical to study snapshot:", {a: int(df[f"{a}_sha_ok"].sum()) for a in ["qwen", "llama"]})
print("rows", len(df))
for arm in ["human","regex","qwen","llama"]:
    if df[f"{arm}_primary"].notna().any():
        print(arm, df[f"{arm}_primary"].value_counts(dropna=False).to_dict())
print({k: int(df[k].sum()) for k in F})
print(df.indication.value_counts().to_dict(), df.sex.value_counts().to_dict(), df.age.describe().round(1).to_dict())
