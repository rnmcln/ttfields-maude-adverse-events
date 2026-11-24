#!/usr/bin/env python3
"""Revised clinical analysis.
Primary cohort: reports concerning devices used for intracranial tumours (thoracic-device reports
excluded; the complete set is a sensitivity analysis). Unit of analysis: one MAUDE report.
Clinical attributes: rule-based descriptors version 2 (main estimate) with the independent LLM
extraction as a cross-check (counts positive by both methods and by either method).
Inputs: analysis_dataset.csv, attributes_combined.csv (14), outputs/cohort_audit/*, missed_reports*.
Output: analysis/outputs/results_clinical_v2.json, tables/*.csv
"""
import json, pathlib, re
from paths import ROOT, DERIVED, ANN, RESULTS, TABLES, FIGURES, CODEBOOK
import numpy as np, pandas as pd
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUT = RESULTS; TAB = TABLES; AUD = DERIVED / "cohort_audit"
rng = np.random.default_rng(20251017)
CATS = ["SKIN_NONINF", "INFECTION", "FALL_ENTANGLE", "NEURO", "SYSTEMIC", "DISEASE_PROG", "USER_ERROR", "OTHER", "UNCLASSIFIABLE", "HARDWARE"]
SEV = ["MILD", "MODERATE", "SEVERE", "FATAL"]; CAUS = ["DEFINITE", "PROBABLE", "POSSIBLE", "UNLIKELY", "UNASSESSABLE"]

def ci(k, n):
    lo, hi = proportion_confint(k, n, method="wilson"); return [round(100 * lo, 1), round(100 * hi, 1)]
def pc(k, n):
    k, n = int(k), int(n)
    return {"n": k, "N": n, "pct": round(100 * k / n, 1) if n else None, "ci95": ci(k, n) if n else None}
def diff(k1, n1, k2, n2):
    lo, hi = confint_proportions_2indep(k2, n2, k1, n1, method="newcomb", compare="diff")
    return {"p1": pc(k1, n1), "p2": pc(k2, n2), "diff_pp": round(100 * (k2 / n2 - k1 / n1), 1), "ci95": [round(100 * lo, 1), round(100 * hi, 1)]}
def listcol(s):
    return s.apply(lambda v: v if isinstance(v, list) else (eval(v) if isinstance(v, str) and v.startswith("[") else []))

d = pd.read_csv(DERIVED / "analysis_dataset.csv")
ev = pd.read_csv(DERIVED / "event_text_segments.csv")
att = pd.read_csv(ANN / "attributes_combined.csv")
for c in ["llm_surgery_types", "llm_fall_mechanism", "llm_fall_injuries"]:
    if c in att: att[c] = listcol(att[c])
st = att[att.set == "study"].drop(columns=["set"])
x = d.merge(st, on=["row_id", "report_number"], how="left").merge(ev, on="row_id")
x["recent"] = x.year_received >= 2024
x["llm"] = x.llm_available.fillna(False).astype(bool)
R = {}

# ---------------- 1. cohort ascertainment audit ----------------
aud = pd.read_csv(AUD / "openfda_audit_records.csv", dtype=str)
meta = json.load(open(OUT / "cohort_audit" / "audit_meta.json"))
nov = aud.manufacturer_d_name.fillna("").str.upper().str.contains("NOVO") | aud.product_code.isin(["NZK", "SDA", "QGZ"])
win = aud.date_received <= "20250930"
target = aud[nov & win]
arms = [c for c in aud.columns if c.startswith("arm_")]
R["audit"] = {"retrieved_utc": meta["retrieved_utc"], "n_by_arm_window_to_20251017": meta["n_by_arm"],
              "novocure_reports_to_20250930": int(len(target)), "in_study": int(target.in_study.eq("True").sum()),
              "missed": int(target.in_study.eq("False").sum()),
              "arm_recall_of_target": {a[4:]: pc(target[a].eq("True").sum(), len(target)) for a in arms},
              "study_reports_not_in_target": int(aud[aud.in_study.eq("True") & ~(nov & win)].shape[0])}
m = target[target.in_study.eq("False")]
R["missed"] = {"n": int(len(m)), "years": m.date_received.str[:4].value_counts().sort_index().to_dict(),
               "event_type": m.event_type.value_counts().to_dict(), "brand": m.brand_name.fillna("(blank)").value_counts().to_dict(),
               "date_added_before_extraction": None}
mrx = pd.read_csv(ANN / "missed_reports_rule_based.csv")
R["missed"]["regex_primary"] = mrx.regex_primary_category.value_counts().to_dict()
mdesc = att[att.set == "missed"]
R["missed"]["narrative_death_rule"] = int(mdesc.death_v2.sum())
R["missed"]["narrative_death_llm"] = int(mdesc.llm_death_reported.eq("yes").sum()) if "llm_death_reported" in mdesc else None
R["missed"]["death_attribution_rule"] = mdesc[mdesc.death_v2].death_attribution.value_counts().to_dict()
for arm in ["qwen2.5_7b", "llama3.1_8b"]:
    p = ANN / "llm_missed" / f"{arm}.jsonl"
    if p.exists():
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        R["missed"][f"{arm}_primary"] = pd.Series([(r["parsed"] or {}).get("primary_category", "UNCLASSIFIABLE") for r in recs]).str.upper().value_counts().to_dict()
bulk = AUD / "bulk_device_matches.txt"
if bulk.exists():
    # column layouts differ between annual device files; fields are identified by content
    recs = []
    for l in open(bulk, encoding="latin-1"):
        r = l.rstrip("\n").split("|")
        date = next((v for v in r[2:] if re.fullmatch(r"\d{4}/\d{2}/\d{2}", v)), None)
        recs.append({"file": r[0], "key": r[1].strip(), "date": date, "pcode": next((v for v in r if v in ("NZK", "SDA", "QGZ")), ""),
                     "novo": any("NOVOCURE" in v.upper() or "NOVO CURE" in v.upper() for v in r)})
    bd = pd.DataFrame(recs); bd = bd[(bd.pcode != "") | bd.novo]
    bd["dr"] = pd.to_datetime(bd.date, format="%Y/%m/%d", errors="coerce")
    keys = set(bd[bd.dr <= "2025-09-30"].key)
    akeys = set(target.mdr_report_key.dropna()); allk = set(aud.mdr_report_key.dropna())
    R["bulk"] = {"device_rows_matched": int(len(bd)), "distinct_mdr_report_keys": int(len(keys)), "files": sorted(bd.file.unique().tolist()),
                 "target_keys_found_in_bulk": int(len(akeys & keys)), "target_keys": int(len(akeys)),
                 "n_bulk_keys_not_in_target": int(len(keys - akeys)), "n_bulk_keys_not_in_openfda_audit": int(len(keys - allk))}

# ---------------- 2. report unit ----------------
st_aud = aud[aud.in_study.eq("True")]
R["report_unit"] = {"summary_report_flag_N": int(st_aud.summary_report_flag.eq("N").sum()), "noe_summarized_1": int(st_aud.noe_summarized.eq("1").sum()),
                    "single_device": int(st_aud.n_devices.eq("1").sum()), "single_patient_entry": int(st_aud.n_patient_entries.eq("1").sum()),
                    "N": int(len(st_aud)), "followup": pc(d.type_of_report.astype(str).str.contains("Followup").sum(), len(d))}
t = x.event_text.fillna("").str.replace(r"\(B\)\(\d\)|\d+", " ", regex=True)
S = cosine_similarity(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit_transform(t)); np.fill_diagonal(S, 0)
iu = np.triu_indices_from(S, 1); sims = S[iu]
same = lambda col, i, j: pd.notna(x.loc[i, col]) and x.loc[i, col] == x.loc[j, col]
cand = [(int(x.row_id[i]), int(x.row_id[j]), round(float(S[i, j]), 3)) for i, j in zip(*iu) if S[i, j] >= 0.5]
key = x.assign(doe=x.date_of_event.astype(str)).dropna(subset=["age"])
grp = key[key.doe.ne("nan")].groupby(["doe", "age", "sex"]).row_id.apply(list)
dup_struct = [g for g in grp if len(g) > 1]
R["duplicates"] = {"max_cosine": round(float(sims.max()), 3), "pairs_cosine_ge_0.5": cand, "n_pairs_ge_0.7": int((sims >= 0.7).sum()),
                   "groups_same_event_date_age_sex": dup_struct}
excl_dup = {b for g in dup_struct for b in g[1:]}

# ---------------- 3. cohort description ----------------
N = len(x); ic = x[~x.thoracic]; NI = len(ic)
R["N"] = N; R["N_intracranial"] = NI; R["N_thoracic"] = int(x.thoracic.sum())
R["years"] = {int(k): int(v) for k, v in x.year_received.value_counts().sort_index().items()}
R["recent"] = pc(x.recent.sum(), N)
R["primary_ic"] = {c: pc((ic.human_primary == c).sum(), NI) for c in CATS}
R["wound_ic"] = pc(ic.human_primary.isin(["SKIN_NONINF", "INFECTION"]).sum(), NI)
R["severity_ic"] = {s: pc((ic.human_severity == s).sum(), NI) for s in SEV}
R["causality_ic"] = {c: pc((ic.human_causality == c).sum(), NI) for c in CAUS}
R["source"] = {k: pc(v, N) for k, v in x.report_source_code.value_counts().items()}
R["event_type"] = {k: pc(v, N) for k, v in x.event_type.value_counts().items()}
R["indication"] = {k: pc(v, N) for k, v in x.indication.value_counts().items()}
age = x.age.dropna(); R["age"] = {"n": int(len(age)), "median": float(age.median()), "q1": float(age.quantile(.25)), "q3": float(age.quantile(.75))}
R["sex"] = {k: pc(v, N) for k, v in x.sex.value_counts().items()}
R["thoracic_primary"] = x[x.thoracic].human_primary.value_counts().to_dict()

# factual endpoints (intracranial), rule v2 with LLM bounds
def endpoint(df, rule, llmexpr=None):
    o = {"rule": pc(df[rule].sum(), len(df))}
    sub = df[df.llm]
    if llmexpr is not None and len(sub):
        r_ = sub[rule].astype(bool); l_ = llmexpr(sub)
        o.update({"N_llm": int(len(sub)), "llm": pc(l_.sum(), len(sub)), "both": pc((r_ & l_).sum(), len(sub)), "either": pc((r_ | l_).sum(), len(sub))})
    return o
ic = ic.assign(death_flag=ic.death_v2.astype(bool))
R["endpoints_ic"] = {
    "hospitalised": endpoint(ic, "hospitalised_v2", lambda q: q.llm_hospitalised.eq("yes")),
    "surgery": endpoint(ic, "surgery_v2", lambda q: q.llm_surgery_for_event.eq("yes")),
    "iv_antibiotics": endpoint(ic, "iv_antibiotics_v2", lambda q: q.llm_iv_antibiotics.eq("yes")),
    "temporary_interruption": endpoint(ic, "temporary_interruption_v2", lambda q: q.llm_ttfields_interruption.eq("temporary")),
    "permanent_discontinuation": endpoint(ic, "permanent_discontinuation_v2", lambda q: q.llm_ttfields_interruption.eq("permanent")),
    "death": endpoint(ic, "death_flag", lambda q: q.llm_death_reported.eq("yes")),
}
none_doc = ~(ic.hospitalised_v2 | ic.surgery_v2 | ic.iv_antibiotics_v2 | ic.temporary_interruption_v2 | ic.permanent_discontinuation_v2 | ic.death_flag)
R["endpoints_ic"]["none_documented"] = {"rule": pc(none_doc.sum(), NI)}
R["severity_components"] = pd.crosstab(ic.human_severity, ic.hospitalised_v2).to_dict()

# ---------------- 4. wound and surgical-site attributes ----------------
w = ic[ic.human_primary.isin(["SKIN_NONINF", "INFECTION"])]; NW = len(w)
WA = {"surgical_site": ("surgical_site_v2", lambda q: q.llm_surgical_site_involved.eq("yes")),
      "dehiscence": ("dehiscence", lambda q: q.llm_wound_dehiscence.eq("yes")),
      "exposure": ("exposure_v2", lambda q: q.llm_exposed_implant_or_bone.eq("yes")),
      "infection_any": ("inf_any", lambda q: q.llm_infection_status.isin(["suspected", "confirmed"])),
      "infection_confirmed": ("inf_conf", lambda q: q.llm_infection_status.eq("confirmed")),
      "bone_infection": ("bone_infection", lambda q: q.llm_bone_flap_or_osteomyelitis.eq("yes")),
      "surgery": ("surgery_v2", lambda q: q.llm_surgery_for_event.eq("yes")),
      "iv_antibiotics": ("iv_antibiotics_v2", lambda q: q.llm_iv_antibiotics.eq("yes")),
      "hospitalised": ("hospitalised_v2", lambda q: q.llm_hospitalised.eq("yes")),
      "temporary_interruption": ("temporary_interruption_v2", lambda q: q.llm_ttfields_interruption.eq("temporary")),
      "permanent_discontinuation": ("permanent_discontinuation_v2", lambda q: q.llm_ttfields_interruption.eq("permanent"))}
w = w.assign(inf_any=w.infection_status.ne("none"), inf_conf=w.infection_status.eq("confirmed"))
PROC = ["wound_revision_or_closure", "debridement_or_washout", "incision_and_drainage", "bone_flap_removal", "hardware_or_mesh_removal",
        "shunt_revision_or_removal", "flap_or_graft_reconstruction"]
for k in PROC:
    WA["proc_" + k] = ("proc_" + k, lambda q, k=k: q.llm_surgery_types.apply(lambda l: k in l))
R["wound"] = {"N": NW, **{k: endpoint(w, r, f) for k, (r, f) in WA.items()}}
for c in ["SKIN_NONINF", "INFECTION"]:
    s = w[w.human_primary == c]
    R["wound"][c] = {"N": int(len(s)), **{k: pc(s[r].sum(), len(s)) for k, (r, _) in WA.items() if not k.startswith("proc_")}}
if "llm_skin_or_wound_event" in w and w.llm.any():
    ro = w.surgical_site_v2 & ~w.llm_surgical_site_involved.eq("yes")
    R["wound"]["llm_no_skin_event"] = int(w.llm_skin_or_wound_event.eq("no").sum())
    R["wound"]["site_rule_only"] = {"n": int(ro.sum()), "llm_no_skin_event": int((ro & w.llm_skin_or_wound_event.eq("no")).sum())}
R["wound"]["site_and_surgery"] = pc((w.surgical_site_v2 & w.surgery_v2).sum(), NW)
R["wound"]["exposure_and_surgery"] = pc((w.exposure_v2 & w.surgery_v2).sum(), NW)
R["wound"]["exposure_with_site"] = pc((w.exposure_v2 & w.surgical_site_v2).sum(), int(w.exposure_v2.sum()))
R["wound"]["surgery_with_site"] = pc((w.surgery_v2 & w.surgical_site_v2).sum(), int(w.surgery_v2.sum()))
R["wound"]["surgery_with_infection"] = pc((w.surgery_v2 & w.inf_any).sum(), int(w.surgery_v2.sum()))
R["wound"]["exposure_with_infection"] = pc((w.exposure_v2 & w.inf_any).sum(), int(w.exposure_v2.sum()))
R["wound"]["exposure_without_infection"] = int((w.exposure_v2 & ~w.inf_any).sum())
R["wound"]["surgery_types_per_surgery_report"] = {"median": float(w[w.surgery_v2][["proc_" + k for k in PROC]].sum(axis=1).median())}
R["wound"]["proc_unspecified_only"] = int((w.surgery_v2 & ~w[["proc_" + k for k in PROC]].any(axis=1)).sum())
R["wound"]["shunt_mentioned"] = pc(w.shunt.sum(), NW)
combo = w.assign(S=w.surgical_site_v2, E=w.exposure_v2, I=w.inf_any, O=w.surgery_v2).groupby(["S", "E", "I", "O"]).size()
R["wound"]["combinations"] = [{"site": bool(k[0]), "exposure": bool(k[1]), "infection": bool(k[2]), "surgery": bool(k[3]), "n": int(v)} for k, v in combo.items()]
bev = w.bevacizumab_context.value_counts()
R["wound"]["bevacizumab_rule"] = {k: pc(v, NW) for k, v in bev.items()}
if w.llm.any():
    sw = w[w.llm]
    R["wound"]["bevacizumab_llm"] = {k: pc(v, len(sw)) for k, v in sw.llm_bevacizumab.value_counts().items()}
    R["wound"]["bevacizumab_current_both"] = pc((sw.bevacizumab_context.eq("current_or_recent") & sw.llm_bevacizumab.eq("current_or_recent")).sum(), len(sw))
    R["wound"]["bevacizumab_current_either"] = pc((sw.bevacizumab_context.eq("current_or_recent") | sw.llm_bevacizumab.eq("current_or_recent")).sum(), len(sw))
R["wound"]["bevacizumab_v1_any_mention"] = pc(w.bevacizumab.sum(), NW)
pd.DataFrame([{"attribute": k, **{f"{m}_{f}": (v[m][f] if m in v else None) for m in ["rule", "llm", "both", "either"] for f in ["n", "N", "pct"]}}
              for k, v in R["wound"].items() if isinstance(v, dict) and "rule" in v]).to_csv(TAB / "wound_attributes_v2.csv", index=False)

# ---------------- 5. falls ----------------
f = ic[ic.human_primary == "FALL_ENTANGLE"]; NF = len(f)
FA = {"cable_trip_or_entangle": ("fallmech_tripped_or_entangled_in_cable", lambda q: q.llm_fall_mechanism.apply(lambda l: "tripped_or_entangled_in_cable" in l)),
      "pulled_by_cable": ("fallmech_pulled_by_cable", lambda q: q.llm_fall_mechanism.apply(lambda l: "pulled_by_cable" in l)),
      "carrying_device": ("fallmech_carrying_device", lambda q: q.llm_fall_mechanism.apply(lambda l: "carrying_device" in l)),
      "any_device_mechanism": ("fall_device_mechanism", lambda q: q.llm_fall_mechanism.apply(lambda l: bool(set(l) & {"tripped_or_entangled_in_cable", "pulled_by_cable", "carrying_device"}))),
      "other_cause": ("fallmech_other_cause_stated", lambda q: q.llm_fall_mechanism.apply(lambda l: "other_cause_stated" in l)),
      "head_wound": ("fallinj_scalp_or_head_wound", lambda q: q.llm_fall_injuries.apply(lambda l: "scalp_or_head_wound" in l)),
      "fracture": ("fallinj_fracture", lambda q: q.llm_fall_injuries.apply(lambda l: "fracture" in l)),
      "ich": ("fallinj_intracranial_haemorrhage", lambda q: q.llm_fall_injuries.apply(lambda l: "intracranial_haemorrhage" in l)),
      "array_stated": ("array_injury_stated", lambda q: q.llm_array_role_in_fall_injury.eq("stated_contribution")),
      "array_crbo": ("array_injury_crbo", lambda q: q.llm_array_role_in_fall_injury.eq("cannot_be_ruled_out")),
      "unrelated_stated": ("fall_unrelated_stated", None)}
R["falls"] = {"N": NF, **{k: endpoint(f, r, fn) for k, (r, fn) in FA.items()}}
mech = np.select([f.fall_device_mechanism, f.fallmech_other_cause_stated], ["device mechanism described", "other cause stated"], "mechanism not stated")
R["falls"]["mechanism_hierarchy_rule"] = {k: pc(v, NF) for k, v in pd.Series(mech).value_counts().items()}
R["falls"]["device_mech_and_head_wound"] = int((f.fall_device_mechanism & f.fallinj_scalp_or_head_wound).sum())
R["falls"]["unrelated_and_crbo"] = int((f.fall_unrelated_stated & f.array_injury_crbo).sum())
R["falls"]["causality_human"] = {c: pc((f.human_causality == c).sum(), NF) for c in CAUS}
R["falls"]["severity_human"] = {c: pc((f.human_severity == c).sum(), NF) for c in SEV}
R["falls"]["hospitalised"] = pc(f.hospitalised_v2.sum(), NF)
lm_dev = f.llm_fall_mechanism.apply(lambda l: bool(set(l) & {"tripped_or_entangled_in_cable", "pulled_by_cable", "carrying_device"})) if "llm_fall_mechanism" in f else pd.Series(False, index=f.index)
neg_txt = f.mdr_text_concat.str.upper().str.contains(r"DID NOT CONTRIBUTE|NOT A CONTRIBUTING FACTOR|NOT CONTRIBUTE TO THE FALL")
only_llm = lm_dev & ~f.fall_device_mechanism
R["falls"]["llm_only_mechanism"] = {"n": int(only_llm.sum()), "with_unrelated_or_no_contribution_statement": int((only_llm & (f.fall_unrelated_stated | neg_txt)).sum())}
st_llm = f.llm_array_role_in_fall_injury.eq("stated_contribution") if "llm_array_role_in_fall_injury" in f else pd.Series(False, index=f.index)
R["falls"]["llm_only_array_stated"] = {"n": int((st_llm & ~f.array_injury_stated).sum()), "with_not_ruled_out_statement": int((st_llm & ~f.array_injury_stated & f.array_injury_crbo).sum())}
R["falls_secondary"] = int(((ic.human_primary != "FALL_ENTANGLE") & (ic.human_secondary == "FALL_ENTANGLE")).sum())

# ---------------- 6. deaths ----------------
dd = x[x.death_v2.astype(bool)]
R["deaths"] = {"narrative_rule": int(len(dd)), "human_fatal": int((x.human_severity == "FATAL").sum()),
               "rule_vs_human_fatal_both": int((x.death_v2.astype(bool) & (x.human_severity == "FATAL")).sum()),
               "event_type_death": int((x.event_type == "Death").sum()),
               "event_type_of_narrative_deaths": dd.event_type.value_counts().to_dict(),
               "attribution_rule": dd.death_attribution.value_counts().to_dict(),
               "attribution_llm": dd.llm_death_attribution.value_counts().to_dict() if "llm_death_attribution" in dd else None,
               "human_causality": dd.human_causality.value_counts().to_dict(),
               "human_primary": dd.human_primary.value_counts().to_dict(),
               "event_type_death_attribution_rule": dd[dd.event_type == "Death"].death_attribution.value_counts().to_dict(),
               "injury_coded_other_cause": int(((dd.event_type != "Death") & (dd.death_attribution == "other_cause_stated")).sum()),
               "injury_coded_with_possible_contribution": int(((dd.event_type != "Death") & dd.human_causality.isin(["DEFINITE", "PROBABLE", "POSSIBLE"])).sum())}
ddl = dd[dd.llm_death_attribution.eq("device_contribution_not_excluded")] if "llm_death_attribution" in dd else dd.iloc[0:0]
R["deaths"]["llm_not_excluded"] = {"n": int(len(ddl)), "rule_other_cause": int(ddl.death_attribution.eq("other_cause_stated").sum()),
                                  "rule_device_not_excluded": int(ddl.death_attribution.eq("device_contribution_not_excluded").sum())}
R["deaths"]["llm_narrative"] = int(x.llm_death_reported.eq("yes").sum()) if "llm_death_reported" in x else None

# ---------------- 7. causality components ----------------
R["reporter_attribution_ic"] = {k: pc(v, NI) for k, v in ic.reporter_attribution.value_counts().items()}
if ic.llm.any():
    R["reporter_attribution_llm_ic"] = {k: pc(v, int(ic.llm.sum())) for k, v in ic[ic.llm].llm_reporter_attribution.value_counts().items()}
R["device_mechanism_ic"] = endpoint(ic, "device_mechanism_v2", lambda q: q.llm_device_mechanism_described.eq("yes"))
R["attribution_by_human_causality"] = pd.crosstab(ic.human_causality, ic.reporter_attribution).to_dict()
neu = ic[ic.human_primary == "NEURO"]
mech_any = neu.device_mechanism_v2.astype(bool)  # rule-based only: the language-model mechanism field agreed poorly (see results_extraction)
recode = neu.human_causality.eq("POSSIBLE") & ~mech_any & neu.reporter_attribution.ne("related")
R["neuro_causality_sensitivity"] = {"N": int(len(neu)), "possible": int(neu.human_causality.eq("POSSIBLE").sum()), "recoded_to_unassessable": int(recode.sum())}
caus2 = ic.human_causality.copy(); caus2.loc[recode[recode].index] = "UNASSESSABLE"
R["causality_ic_sensitivity"] = {c: pc((caus2 == c).sum(), NI) for c in CAUS}

# ---------------- 8. reporting discontinuity ----------------
x["q"] = pd.to_datetime(x.date_received.astype(str)).dt.to_period("Q").astype(str)
x["sevfat"] = x.human_severity.isin(["SEVERE", "FATAL"])
x["age_stated"] = x.age.notna()
x["gio"] = x.device_0_brand_name.astype(str).str.contains("GIO")
x["us"] = x.report_number.map(aud.set_index("report_number").reporter_country_code).eq("US")
x["hp_source"] = x.report_number.map(aud.set_index("report_number").source_type).fillna("").str.contains("Health Professional")
x["unrel_template"] = x.mdr_text_concat.str.upper().str.contains(r"UNRELATED TO (?:DEVICE USE|OPTUNE)")
qs = x.groupby("q").agg(n=("row_id", "size"), sevfat=("sevfat", "mean"), hosp=("hospitalised_v2", "mean"), age=("age_stated", "mean"),
                         us=("us", "mean"), wound=("human_primary", lambda s: s.isin(["SKIN_NONINF", "INFECTION"]).mean()),
                         fall=("human_primary", lambda s: (s == "FALL_ENTANGLE").mean()))
qs.to_csv(TAB / "quarterly_series.csv")
R["quarterly"] = {k: {kk: (round(float(vv), 3) if kk != "n" else int(vv)) for kk, vv in v.items()} for k, v in qs.to_dict(orient="index").items()}
xs = x.sort_values(["date_received", "row_id"]).reset_index(drop=True)
def changepoint(b):
    b = np.asarray(b, float); n = len(b); c1 = np.cumsum(b)
    best, arg = -np.inf, None
    for k in range(20, n - 20):
        a1, n1 = c1[k - 1], k; a2, n2 = c1[-1] - a1, n - k
        ll = 0
        for a, nn in ((a1, n1), (a2, n2)):
            p = min(max(a / nn, 1e-9), 1 - 1e-9); ll += a * np.log(p) + (nn - a) * np.log(1 - p)
        if ll > best: best, arg = ll, k
    return arg
R["changepoints"] = {}
for col in ["sevfat", "hospitalised_v2", "age_stated", "gio", "hp_source", "unrel_template"]:
    k = changepoint(xs[col].astype(bool)); boots = []
    for _ in range(500):
        idx = np.sort(rng.integers(0, len(xs), len(xs))); boots.append(str(xs.date_received.iloc[idx[changepoint(xs[col].astype(bool).values[idx])]]))
    R["changepoints"][col] = {"date": str(xs.date_received.iloc[k]), "before": pc(xs[col].iloc[:k].sum(), k), "after": pc(xs[col].iloc[k:].sum(), len(xs) - k),
                              "boot_ci": [sorted(boots)[12], sorted(boots)[487]]}
mo = x.assign(m=pd.to_datetime(x.date_received.astype(str)).dt.to_period("M")).groupby("m").size()
mo = mo.reindex(pd.period_range(mo.index.min(), mo.index.max(), freq="M"), fill_value=0)
best, karg = -np.inf, None
for k in range(6, len(mo) - 6):
    l1, l2 = mo.iloc[:k].mean(), mo.iloc[k:].mean()
    ll = (mo.iloc[:k] * np.log(max(l1, 1e-9)) - l1).sum() + (mo.iloc[k:] * np.log(max(l2, 1e-9)) - l2).sum()
    if ll > best: best, karg = ll, k
R["volume_changepoint"] = {"month": str(mo.index[karg]), "mean_before": round(float(mo.iloc[:karg].mean()), 1), "mean_after": round(float(mo.iloc[karg:].mean()), 1)}
P0, P1 = x[~x.recent], x[x.recent]
def pdiff(mask0, mask1): return diff(int(mask0.sum()), len(mask0), int(mask1.sum()), len(mask1))
R["period"] = {"n": [int(len(P0)), int(len(P1))],
               "sevfat": pdiff(P0.sevfat, P1.sevfat), "hosp": pdiff(P0.hospitalised_v2, P1.hospitalised_v2),
               "age_stated": pdiff(P0.age_stated, P1.age_stated), "gio": pdiff(P0.gio, P1.gio), "us": pdiff(P0.us, P1.us),
               "hp_source": pdiff(P0.hp_source, P1.hp_source), "unrel_template": pdiff(P0.unrel_template, P1.unrel_template),
               "manufacturer_gmbh": pdiff(P0.device_0_manufacturer_d_name.str.contains("GMBH"), P1.device_0_manufacturer_d_name.str.contains("GMBH")),
               "text_length_median": [int(P0.mdr_text_concat.str.len().median()), int(P1.mdr_text_concat.str.len().median())],
               "wound": pdiff(P0.human_primary.isin(["SKIN_NONINF", "INFECTION"]), P1.human_primary.isin(["SKIN_NONINF", "INFECTION"])),
               "fall": pdiff(P0.human_primary.eq("FALL_ENTANGLE"), P1.human_primary.eq("FALL_ENTANGLE")),
               "neuro": pdiff(P0.human_primary.eq("NEURO"), P1.human_primary.eq("NEURO"))}
R["period"]["country"] = pd.crosstab(x.report_number.map(aud.set_index("report_number").reporter_country_code).fillna("NA"), x.recent).sort_values(True, ascending=False).head(8).to_dict()
w0, w1 = w[~w.recent], w[w.recent]
R["period_wound"] = {"n": [int(len(w0)), int(len(w1))], **{k: pdiff(w0[r].astype(bool), w1[r].astype(bool)) for k, (r, _) in WA.items() if not k.startswith("proc_")}}
R["period_wound"]["sevfat"] = pdiff(w0.human_severity.isin(["SEVERE", "FATAL"]), w1.human_severity.isin(["SEVERE", "FATAL"]))
f0, f1 = f[~f.recent], f[f.recent]
R["period_falls"] = {"n": [int(len(f0)), int(len(f1))], "device_mech": pdiff(f0.fall_device_mechanism, f1.fall_device_mechanism),
                     "head_wound": pdiff(f0.fallinj_scalp_or_head_wound, f1.fallinj_scalp_or_head_wound), "fracture": pdiff(f0.fallinj_fracture, f1.fallinj_fracture)}
# prespecified comparison, reported descriptively
x["period_pre"] = np.where(x.year_received <= 2017, "2012-2017", "2018-2025")
grp_ = {"SKIN_NONINF": "Wound or infection", "INFECTION": "Wound or infection", "FALL_ENTANGLE": "Fall", "NEURO": "Neurological"}
x["cg"] = x.human_primary.map(grp_).fillna("Other")
pp0, pp1 = x[x.period_pre == "2012-2017"], x[x.period_pre == "2018-2025"]
R["prespecified_period"] = {"n": [int(len(pp0)), int(len(pp1))], **{g: pdiff(pp0.cg.eq(g), pp1.cg.eq(g)) for g in ["Wound or infection", "Fall", "Neurological", "Other"]}}

# ---------------- 9. robustness table ----------------
def est(df, kind):
    ww = df[df.human_primary.isin(["SKIN_NONINF", "INFECTION"])]; ff = df[df.human_primary == "FALL_ENTANGLE"]
    if kind == "site": return pc(ww.surgical_site_v2.sum(), len(ww))
    if kind == "exposure": return pc(ww.exposure_v2.sum(), len(ww))
    if kind == "surgery": return pc(ww.surgery_v2.sum(), len(ww))
    if kind == "fallmech": return pc(ff.fall_device_mechanism.sum(), len(ff))
    if kind == "sevfat": return pc(df.human_severity.isin(["SEVERE", "FATAL"]).sum(), len(df))
def est_v1(df, kind):
    ww = df[df.human_primary.isin(["SKIN_NONINF", "INFECTION"])]; ff = df[df.human_primary == "FALL_ENTANGLE"]
    m_ = {"site": ("surgical_site", ww), "exposure": ("implant_exposure", ww), "surgery": ("surgical_revision", ww), "fallmech": ("cable_or_bag_mechanism", ff)}
    if kind == "sevfat": return pc(df.human_severity.isin(["SEVERE", "FATAL"]).sum(), len(df))
    c, g = m_[kind]; return pc(g[c].sum(), len(g))
def est_llm(df, kind, how):
    df = df[df.llm]
    ww = df[df.human_primary.isin(["SKIN_NONINF", "INFECTION"])]; ff = df[df.human_primary == "FALL_ENTANGLE"]
    if kind == "sevfat": return None
    mp = {"site": (ww, "surgical_site_v2", ww.llm_surgical_site_involved.eq("yes")), "exposure": (ww, "exposure_v2", ww.llm_exposed_implant_or_bone.eq("yes")),
          "surgery": (ww, "surgery_v2", ww.llm_surgery_for_event.eq("yes")),
          "fallmech": (ff, "fall_device_mechanism", ff.llm_fall_mechanism.apply(lambda l: bool(set(l) & {"tripped_or_entangled_in_cable", "pulled_by_cable", "carrying_device"})))}
    g, c, l_ = mp[kind]
    if len(g) == 0: return None
    v = (g[c].astype(bool) & l_) if how == "both" else (g[c].astype(bool) | l_) if how == "either" else l_
    return pc(v.sum(), len(g))
rows = []
for kind, lbl in [("site", "Surgical-site involvement (wound reports)"), ("exposure", "Exposed bone or implant (wound reports)"),
                  ("surgery", "Operative procedure for the event (wound reports)"), ("fallmech", "Device mechanism of fall (fall reports)"),
                  ("sevfat", "Severe or fatal (all reports)")]:
    rows.append({"estimate": lbl, "main": est(ic, kind), "all_548": est(x, kind), "rule_v1": est_v1(ic, kind),
                 "llm": est_llm(ic, kind, "llm"), "both_methods": est_llm(ic, kind, "both"), "either_method": est_llm(ic, kind, "either"),
                 "2012_2023": est(ic[~ic.recent], kind), "2024_2025": est(ic[ic.recent], kind),
                 "excl_candidate_duplicates": est(ic[~ic.row_id.isin(excl_dup)], kind)})
R["robustness"] = rows
pd.DataFrame([{k: (f"{v['n']}/{v['N']} ({v['pct']})" if isinstance(v, dict) and v else (v if not isinstance(v, dict) else "")) for k, v in r.items()} for r in rows]).to_csv(TAB / "robustness.csv", index=False)

json.dump(R, open(OUT / "results_clinical_v2.json", "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ["audit", "missed", "report_unit", "endpoints_ic", "deaths", "changepoints", "neuro_causality_sensitivity"] if k in R}, indent=0, default=str)[:7000])
