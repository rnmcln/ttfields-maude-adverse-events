#!/usr/bin/env python3
"""Combine the revised rule-based descriptors (13_descriptors_v2.py) with the independent LLM
cross-check extraction (llm_extract.py) and quantify their agreement.
For each binary attribute: counts positive by each method, by both (conservative estimate) and by
either (liberal estimate), Cohen kappa and positive specific agreement, within the relevant
denominator. Field values outside the permitted set are normalised as follows: for yes/no fields,
"not_stated", "not_applicable", "unknown" and "none" are read as "no"; other invalid values are missing.
Outputs: analysis/data/attributes_combined.csv, analysis/outputs/results_extraction.json
"""
import json, re, pathlib
from paths import ROOT, DERIVED, ANN, RESULTS, TABLES, FIGURES, CODEBOOK
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score

import importlib.util, sys
spec = importlib.util.spec_from_file_location("lx", pathlib.Path(__file__).resolve().parent / "llm_extract.py"); lx = importlib.util.module_from_spec(spec)
sys.argv = sys.argv[:1]; spec.loader.exec_module(lx)
YN_FALSE = {"not_stated", "not_applicable", "unknown", "none", "n/a", "na", "false"}

def load_llm(path):
    recs = {}
    if not path.exists():
        return pd.DataFrame()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        raw = r.get("raw_fields") or {}
        out = {}
        for k, allowed in lx.SCHEMA.items():
            v = raw.get(k)
            if isinstance(allowed, tuple):
                vals = v if isinstance(v, list) else ([] if v in (None, "", "none") else [v])
                out[k] = sorted({str(x).strip().lower() for x in vals} & allowed[1])
            else:
                s = str(v).strip().lower() if v is not None else None
                if allowed == lx.YN and s in YN_FALSE:
                    s = "no"
                if allowed == lx.YN and s == "true":
                    s = "yes"
                out[k] = s if s in allowed else None
        out = {"llm_" + k: v for k, v in out.items()}
        out["row_id"] = r["row_id"]; out["llm_parse_ok"] = r["parse_ok"]; out["llm_seconds"] = sum(a["seconds"] for a in r["attempts"])
        recs[r["row_id"]] = out
    return pd.DataFrame(recs.values())

def agree(a, b):
    a = pd.Series(a).astype(bool).values; b = pd.Series(b).astype(bool).values
    n11 = int((a & b).sum()); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n00 = int((~a & ~b).sum())
    k = cohen_kappa_score(a, b) if (a.any() or b.any()) and not (a.all() and b.all()) and len(set(a)) + len(set(b)) > 2 else float("nan")
    pa = 2 * n11 / (2 * n11 + n10 + n01) if (2 * n11 + n10 + n01) else float("nan")
    return {"n": int(len(a)), "rule": int(a.sum()), "llm": int(b.sum()), "both": n11, "rule_only": n10, "llm_only": n01, "neither": n00,
            "either": n11 + n10 + n01, "kappa": None if np.isnan(k) else round(float(k), 3), "pos_agreement": None if np.isnan(pa) else round(pa, 3)}

if __name__ == "__main__":
    d = pd.read_csv(DERIVED / "analysis_dataset.csv")
    v = pd.read_csv(ANN / "attributes_rule_based_v2.csv")
    L = pd.concat([load_llm(ANN / "llm_extract" / "qwen2.5_7b_extract.jsonl").assign(set="study"),
                   load_llm(ANN / "llm_missed" / "qwen2.5_7b_extract.jsonl").assign(set="missed")], ignore_index=True)
    x = v.merge(L, on=["row_id", "set"], how="left")
    x["llm_available"] = x.llm_skin_or_wound_event.notna() | x.llm_fall.notna()
    x.to_csv(ANN / "attributes_combined.csv", index=False)
    s = x[(x.set == "study") & x.llm_available].merge(d[["row_id", "human_primary", "human_severity", "thoracic"]], on="row_id")
    wound = s[s.human_primary.isin(["SKIN_NONINF", "INFECTION"]) & ~s.thoracic]
    falls = s[(s.human_primary == "FALL_ENTANGLE") & ~s.thoracic]
    Y = lambda col, val="yes": s.loc[:, col].eq(val)
    res = {"n_llm_available_study": int(len(s)), "wound": {}, "falls": {}, "all": {}}
    def W(df, rule, llm): return agree(df[rule], llm(df))
    res["wound"]["n"] = int(len(wound))
    res["wound"]["surgical_site"] = W(wound, "surgical_site_v2", lambda q: q.llm_surgical_site_involved.eq("yes"))
    res["wound"]["dehiscence"] = W(wound, "dehiscence", lambda q: q.llm_wound_dehiscence.eq("yes"))
    res["wound"]["exposure"] = W(wound, "exposure_v2", lambda q: q.llm_exposed_implant_or_bone.eq("yes"))
    res["wound"]["infection_any"] = agree(wound.infection_status.ne("none"), wound.llm_infection_status.isin(["suspected", "confirmed"]))
    res["wound"]["infection_confirmed"] = agree(wound.infection_status.eq("confirmed"), wound.llm_infection_status.eq("confirmed"))
    hi = wound.human_primary.eq("INFECTION")
    res["wound"]["infection_rule_vs_human"] = agree(hi, wound.infection_status.ne("none"))
    res["wound"]["infection_llm_vs_human"] = agree(hi, wound.llm_infection_status.isin(["suspected", "confirmed"]))
    res["wound"]["bone_infection"] = W(wound, "bone_infection", lambda q: q.llm_bone_flap_or_osteomyelitis.eq("yes"))
    res["wound"]["surgery"] = W(wound, "surgery_v2", lambda q: q.llm_surgery_for_event.eq("yes"))
    for k in ["debridement_or_washout", "incision_and_drainage", "bone_flap_removal", "hardware_or_mesh_removal",
              "shunt_revision_or_removal", "flap_or_graft_reconstruction", "wound_revision_or_closure"]:
        res["wound"]["proc_" + k] = W(wound, "proc_" + k, lambda q, k=k: q.llm_surgery_types.apply(lambda l: k in (l if isinstance(l, list) else [])))
    res["wound"]["iv_antibiotics"] = W(wound, "iv_antibiotics_v2", lambda q: q.llm_iv_antibiotics.eq("yes"))
    res["wound"]["hospitalised"] = W(wound, "hospitalised_v2", lambda q: q.llm_hospitalised.eq("yes"))
    res["falls"]["n"] = int(len(falls))
    res["falls"]["cable_trip_or_entangle"] = W(falls, "fallmech_tripped_or_entangled_in_cable", lambda q: q.llm_fall_mechanism.apply(lambda l: "tripped_or_entangled_in_cable" in l))
    res["falls"]["pulled_by_cable"] = W(falls, "fallmech_pulled_by_cable", lambda q: q.llm_fall_mechanism.apply(lambda l: "pulled_by_cable" in l))
    res["falls"]["carrying_device"] = W(falls, "fallmech_carrying_device", lambda q: q.llm_fall_mechanism.apply(lambda l: "carrying_device" in l))
    res["falls"]["any_device_mechanism"] = W(falls, "fall_device_mechanism", lambda q: q.llm_fall_mechanism.apply(lambda l: bool(set(l) & {"tripped_or_entangled_in_cable", "pulled_by_cable", "carrying_device"})))
    res["falls"]["other_cause"] = W(falls, "fallmech_other_cause_stated", lambda q: q.llm_fall_mechanism.apply(lambda l: "other_cause_stated" in l))
    res["falls"]["head_wound"] = W(falls, "fallinj_scalp_or_head_wound", lambda q: q.llm_fall_injuries.apply(lambda l: "scalp_or_head_wound" in l))
    res["falls"]["fracture"] = W(falls, "fallinj_fracture", lambda q: q.llm_fall_injuries.apply(lambda l: "fracture" in l))
    res["falls"]["ich"] = W(falls, "fallinj_intracranial_haemorrhage", lambda q: q.llm_fall_injuries.apply(lambda l: "intracranial_haemorrhage" in l))
    res["falls"]["array_stated"] = W(falls, "array_injury_stated", lambda q: q.llm_array_role_in_fall_injury.eq("stated_contribution"))
    res["falls"]["array_crbo"] = W(falls, "array_injury_crbo", lambda q: q.llm_array_role_in_fall_injury.eq("cannot_be_ruled_out"))
    res["all"]["death"] = W(s, "death_v2", lambda q: q.llm_death_reported.eq("yes"))
    res["all"]["death_vs_human_fatal_rule"] = agree(s.human_severity.eq("FATAL"), s.death_v2)
    res["all"]["death_vs_human_fatal_llm"] = agree(s.human_severity.eq("FATAL"), s.llm_death_reported.eq("yes"))
    res["all"]["device_mechanism"] = W(s, "device_mechanism_v2", lambda q: q.llm_device_mechanism_described.eq("yes"))
    res["all"]["hospitalised"] = W(s, "hospitalised_v2", lambda q: q.llm_hospitalised.eq("yes"))
    res["all"]["reporter_attribution_crosstab"] = pd.crosstab(s.reporter_attribution, s.llm_reporter_attribution).to_dict()
    res["all"]["reporter_attribution_kappa"] = round(float(cohen_kappa_score(s.reporter_attribution, s.llm_reporter_attribution.fillna("missing"))), 3)
    res["wound"]["bevacizumab_crosstab"] = pd.crosstab(wound.bevacizumab_context, wound.llm_bevacizumab.fillna("missing")).to_dict()
    res["wound"]["bevacizumab_current_both"] = int((wound.bevacizumab_context.eq("current_or_recent") & wound.llm_bevacizumab.eq("current_or_recent")).sum())
    res["wound"]["bevacizumab_current_either"] = int((wound.bevacizumab_context.eq("current_or_recent") | wound.llm_bevacizumab.eq("current_or_recent")).sum())
    res["llm"] = {"n_records": int(L.shape[0]), "n_parse_fail": int((~L.llm_parse_ok).sum()) if len(L) else 0,
                  "median_seconds": float(L.llm_seconds.median()) if len(L) else None}
    json.dump(res, open(RESULTS / "results_extraction.json", "w"), indent=1, default=str)
    print(json.dumps({k: {kk: (vv["kappa"], vv["rule"], vv["llm"], vv["both"]) for kk, vv in res[k].items() if isinstance(vv, dict) and "kappa" in vv} for k in ["wound", "falls", "all"]}, indent=0))
