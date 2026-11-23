#!/usr/bin/env python3
"""Clinical descriptive analysis based on the human (reference) classification.
All proportions are shares of reports (no exposure denominator). 95% CIs: Wilson score.
Output: analysis/outputs/results_clinical.json and tables/*.csv"""
import json, pathlib
import numpy as np, pandas as pd
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import chi2_contingency

from paths import DERIVED, RESULTS, TABLES
OUT = RESULTS; TAB = TABLES
rng = np.random.default_rng(20251017)
d = pd.read_csv(DERIVED / "analysis_dataset.csv")
N = len(d)
CATS = ["SKIN_NONINF","INFECTION","FALL_ENTANGLE","NEURO","SYSTEMIC","DISEASE_PROG","USER_ERROR","OTHER","UNCLASSIFIABLE","HARDWARE"]
SEV = ["MILD","MODERATE","SEVERE","FATAL"]; CAUS = ["DEFINITE","PROBABLE","POSSIBLE","UNLIKELY","UNASSESSABLE"]
def ci(k, n):
    lo, hi = proportion_confint(k, n, method="wilson"); return [round(100*lo, 1), round(100*hi, 1)]
def pc(k, n): return {"n": int(k), "N": int(n), "pct": round(100*k/n, 1) if n else None, "ci95": ci(k, n) if n else None}
R = {"N": N}

# ---- report characteristics
R["event_type"] = {k: pc(v, N) for k, v in d.event_type.value_counts().items()}
R["source"] = {k: pc(v, N) for k, v in d.report_source_code.value_counts().items()}
R["followup_any"] = pc(d.type_of_report.astype(str).str.contains("Followup").sum(), N)
R["brand"] = {k: pc(v, N) for k, v in d.device_0_brand_name.value_counts().items()}
R["model"] = {k: pc(v, N) for k, v in d.device_model_std.value_counts().items()}
R["thoracic"] = pc(d.thoracic.sum(), N)
R["operator_lay"] = pc((d.device_0_device_operator == "LAY USER/PATIENT").sum(), N)
pp = d.product_problems.astype(str)
R["product_problem_no_device_problem"] = pc(pp.str.contains("Without Identified Device or Use Problem").sum(), N)
R["years"] = {int(k): int(v) for k, v in d.year_received.value_counts().sort_index().items()}
R["first_received"] = str(int(d.date_received.min())); R["last_received"] = str(int(d.date_received.max()))
R["received_2024_2025"] = pc(d.year_received.isin([2024, 2025]).sum(), N)
L = d.mdr_text_concat.astype(str).str.len()
R["text_length"] = {"median": int(L.median()), "q1": int(L.quantile(.25)), "q3": int(L.quantile(.75)), "min": int(L.min()), "max": int(L.max())}
age = d.age.dropna()
R["age"] = {"n_stated": int(len(age)), "median": float(age.median()), "q1": float(age.quantile(.25)), "q3": float(age.quantile(.75)), "min": int(age.min()), "max": int(age.max()), "n_under18": int((age < 18).sum())}
R["sex"] = {k: pc(v, N) for k, v in d.sex.value_counts().items()}
R["indication"] = {k: pc(v, N) for k, v in d.indication.value_counts().items()}
R["bevacizumab_mentioned"] = pc(d.bevacizumab.sum(), N)

# ---- primary category distribution (human reference)
R["primary"] = {c: pc((d.human_primary == c).sum(), N) for c in CATS}
R["secondary_any"] = pc((d.human_secondary.fillna("") != "").sum(), N)
R["secondary"] = {k: int(v) for k, v in d.human_secondary.fillna("").replace("", np.nan).value_counts().items()}
R["skin_or_infection"] = pc(d.human_primary.isin(["SKIN_NONINF","INFECTION"]).sum(), N)
inf_skin2 = ((d.human_primary == "INFECTION") & (d.human_secondary == "SKIN_NONINF")).sum()
R["infection_with_skin_secondary"] = pc(inf_skin2, (d.human_primary == "INFECTION").sum())
R["severity"] = {s: pc((d.human_severity == s).sum(), N) for s in SEV}
R["causality"] = {c: pc((d.human_causality == c).sum(), N) for c in CAUS}
R["causality_def_prob"] = pc(d.human_causality.isin(["DEFINITE","PROBABLE"]).sum(), N)
comp = d.human_component.str.split("|").explode()
R["component"] = {k: pc(v, N) for k, v in comp.value_counts().items()}

# cross-tabulations (Table 2)
rows = []
for c in CATS:
    s = d[d.human_primary == c]; n = len(s)
    if n == 0: continue
    row = {"category": c, "n": n, "pct": round(100*n/N, 1)}
    for v in SEV: row[f"sev_{v}"] = int((s.human_severity == v).sum())
    for v in CAUS: row[f"caus_{v}"] = int((s.human_causality == v).sum())
    row["hospitalised_mention"] = int(s.hospitalised.sum())
    row["temporary_interruption_mention"] = int(s.temporary_interruption.sum())
    row["permanent_discontinuation_mention"] = int(s.permanent_discontinuation.sum())
    rows.append(row)
t2 = pd.DataFrame(rows); t2.to_csv(TAB / "table2_category_by_severity_causality.csv", index=False)
R["severe_or_fatal_by_cat"] = {r.category: pc(r.sev_SEVERE + r.sev_FATAL, r.n) for r in t2.itertuples()}
R["defprob_by_cat"] = {r.category: pc(r.caus_DEFINITE + r.caus_PROBABLE, r.n) for r in t2.itertuples()}
R["severe_or_fatal"] = pc(d.human_severity.isin(["SEVERE","FATAL"]).sum(), N)
R["temporary_interruption"] = pc(d.temporary_interruption.sum(), N)
R["permanent_discontinuation"] = pc(d.permanent_discontinuation.sum(), N)
R["hospitalised"] = pc(d.hospitalised.sum(), N)

# ---- scalp wound and infection reports (SKIN_NONINF + INFECTION)
w = d[d.human_primary.isin(["SKIN_NONINF","INFECTION"])]; nw = len(w)
R["wound"] = {"N": nw}
for f in ["surgical_site","implant_exposure","surgical_revision","organism_or_culture","iv_antibiotics","bevacizumab","shunt","hospitalised"]:
    R["wound"][f] = pc(w[f].sum(), nw)
    for c in ["SKIN_NONINF","INFECTION"]:
        s = w[w.human_primary == c]; R["wound"][f + "_" + c] = pc(s[f].sum(), len(s))
R["wound"]["surgical_site_and_revision"] = pc((w.surgical_site & w.surgical_revision).sum(), nw)
wound_tab = pd.DataFrame([{"descriptor": f, **{c: f"{int(w[w.human_primary==c][f].sum())}/{(w.human_primary==c).sum()}" for c in ["SKIN_NONINF","INFECTION"]},
                           "all": f"{int(w[f].sum())}/{nw}"} for f in ["surgical_site","implant_exposure","surgical_revision","organism_or_culture","iv_antibiotics","bevacizumab","shunt","hospitalised"]])
wound_tab.to_csv(TAB / "wound_descriptors.csv", index=False)

# ---- falls
f = d[d.human_primary == "FALL_ENTANGLE"]; nf = len(f)
R["falls"] = {"N": nf}
for x in ["fracture","intracranial_bleed","head_wound_after_fall","cable_or_bag_mechanism","hospitalised","temporary_interruption"]:
    R["falls"][x] = pc(f[x].sum(), nf)
R["falls"]["causality"] = {c: pc((f.human_causality == c).sum(), nf) for c in CAUS}
R["falls"]["severity"] = {c: pc((f.human_severity == c).sum(), nf) for c in SEV}
R["falls_any_mention_primary_or_secondary"] = pc(((d.human_primary == "FALL_ENTANGLE") | (d.human_secondary == "FALL_ENTANGLE")).sum(), N)

# ---- neurological
n_ = d[d.human_primary == "NEURO"]
R["neuro"] = {"N": len(n_), "causality": {c: pc((n_.human_causality == c).sum(), len(n_)) for c in CAUS},
              "seizure_mention": pc(n_.mdr_text_concat.str.upper().str.contains("SEIZURE|CONVULS|EPILEP").sum(), len(n_)),
              "headache_mention": pc(n_.mdr_text_concat.str.upper().str.contains("HEADACHE").sum(), len(n_))}

# ---- fatal reports and structured-field concordance
fat = d[d.human_severity == "FATAL"]
R["fatal"] = {"N": len(fat), "event_type_death": int((fat.event_type == "Death").sum()),
              "event_type_injury": int((fat.event_type == "Injury").sum()),
              "primary": {k: int(v) for k, v in fat.human_primary.value_counts().items()},
              "causality": {k: int(v) for k, v in fat.human_causality.value_counts().items()},
              "def_or_prob": int(fat.human_causality.isin(["DEFINITE","PROBABLE"]).sum())}
R["death_event_type_total"] = int((d.event_type == "Death").sum())
R["death_event_type_all_fatal_by_human"] = bool((d[d.event_type == "Death"].human_severity == "FATAL").all())
R["hardware_human"] = int((d.human_primary == "HARDWARE").sum())
R["malfunction_event_type"] = int((d.event_type == "Malfunction").sum())

# ---- temporal: prespecified periods 2012-2017 vs 2018-2025
d["period"] = np.where(d.year_received <= 2017, "2012-2017", "2018-2025")
main = ["SKIN_NONINF","INFECTION","FALL_ENTANGLE","NEURO"]
d["cat_grouped"] = np.where(d.human_primary.isin(main), d.human_primary, "OTHER_GROUPED")
ct = pd.crosstab(d.cat_grouped, d.period)
chi2, p_asym, dof, exp = chi2_contingency(ct)
# Monte Carlo permutation p-value (fixed seed)
obs = chi2; B = 10000; y = d.cat_grouped.values; g = d.period.values; ge = 0
for _ in range(B):
    gp = rng.permutation(g); c2 = chi2_contingency(pd.crosstab(y, gp))[0]; ge += c2 >= obs - 1e-12
R["period_test"] = {"table": {p: {k: int(v) for k, v in ct[p].items()} for p in ct.columns}, "chi2": round(chi2, 2), "dof": int(dof),
                    "p_permutation": round((ge + 1)/(B + 1), 4), "B": B, "min_expected": round(float(exp.min()), 1)}
R["period_share"] = {p: {c: pc((d[d.period == p].cat_grouped == c).sum(), (d.period == p).sum()) for c in main + ["OTHER_GROUPED"]} for p in ["2012-2017","2018-2025"]}
yr = pd.crosstab(d.year_received, d.human_primary); yr.to_csv(TAB / "annual_counts_by_category.csv")
# 2024-2025 vs earlier
d["recent"] = d.year_received >= 2024
R["recent_share"] = {k: {c: pc((d[d.recent == k].human_primary == c).sum(), (d.recent == k).sum()) for c in main} for k in [False, True]}
R["recent_share"] = {"2012-2023": R["recent_share"][False], "2024-2025": R["recent_share"][True]}


# ---- reporting composition before and after 2024 (post hoc, prompted by the 2024 increase)
from scipy.stats import fisher_exact
d["sev_or_fatal"] = d.human_severity.isin(["SEVERE","FATAL"])
d["gio_label"] = d.device_0_brand_name.str.contains("GIO")
d["age_stated"] = d.age.notna()
comp_ = {}
for var in ["sev_or_fatal","hospitalised","gio_label","age_stated","temporary_interruption"]:
    a = d[~d.recent][var]; b = d[d.recent][var]
    orr, p = fisher_exact([[int(b.sum()), int((~b).sum())], [int(a.sum()), int((~a).sum())]])
    comp_[var] = {"2012-2023": pc(int(a.sum()), len(a)), "2024-2025": pc(int(b.sum()), len(b)), "fisher_p": float(f"{p:.2e}")}
R["reporting_shift"] = comp_
R["by_year_share"] = {int(y): {"n": int(len(g)), "sev_or_fatal_pct": round(100*g.sev_or_fatal.mean(), 1), "hosp_pct": round(100*g.hospitalised.mean(), 1)}
                      for y, g in d.groupby("year_received")}
R["n_2012_2023"] = int((~d.recent).sum()); R["n_2024_2025"] = int(d.recent.sum())

# ---- sensitivity: CNS-only reports
cns = d[~d.thoracic]
R["cns_only"] = {"N": len(cns), "primary": {c: pc((cns.human_primary == c).sum(), len(cns)) for c in CATS}}
th = d[d.thoracic]
R["thoracic_primary"] = {k: int(v) for k, v in th.human_primary.value_counts().items()}

# Table 1
t1 = []
def add(lbl, k, n=N): t1.append({"characteristic": lbl, "value": f"{k} ({100*k/n:.1f})"})
for k, v in d.event_type.value_counts().items(): add(f"MAUDE event type: {k}", v)
for k, v in d.report_source_code.value_counts().items(): add(f"Report source: {k}", v)
add("Report with at least one follow-up", R["followup_any"]["n"])
for k, v in d.device_0_brand_name.value_counts().items(): add(f"Brand: {k}", v)
for k, v in d.indication.value_counts().items(): add(f"Indication (narrative): {k}", v)
for k, v in d.sex.value_counts().items(): add(f"Sex (narrative): {k}", v)
t1.append({"characteristic": f"Age, years (stated in {R['age']['n_stated']} narratives), median (IQR)", "value": f"{R['age']['median']:.0f} ({R['age']['q1']:.0f} to {R['age']['q3']:.0f})"})
add("Concomitant or prior bevacizumab mentioned", R["bevacizumab_mentioned"]["n"])
add("Product problem field: no identified device or use problem", R["product_problem_no_device_problem"]["n"])
pd.DataFrame(t1).to_csv(TAB / "table1_report_characteristics.csv", index=False)


# ---- additional verified narrative counts used in text
Tall = d.mdr_text_concat.astype(str).str.upper()
EV = pd.read_csv(DERIVED / "event_text_segments.csv").set_index("row_id").event_text.reindex(d.row_id).fillna("").str.upper().values
EV = pd.Series(EV, index=d.index)
sysm = d.human_primary == "SYSTEMIC"
R["systemic_vte"] = int((sysm & EV.str.contains(r"PULMONARY EMBOL|DEEP VEIN THROMB|\bDVT\b")).sum())
R["systemic_ich"] = int((sysm & EV.str.contains(r"BRAIN HEMORRHAGE|INTRACRANIAL HEMORRHAGE|CEREBRAL HEMORRHAGE")).sum())
fm = d.human_primary == "FALL_ENTANGLE"
unrel = Tall.str.contains(r"FALL WAS (?:UNRELATED|NOT RELATED)")
R["falls_unrelated_statement"] = int((fm & unrel).sum())
R["falls_unrelated_and_array_not_ruled_out"] = int((fm & unrel & Tall.str.contains("CANNOT BE RULED OUT") & Tall.str.contains("LACERATION")).sum())
R["falls_array_disc"] = int((fm & Tall.str.contains(r"ARRAY DIS[KC]|DIS[KC]S? OF THE (?:TRANSDUCER )?ARRAY|TRANSDUCER ARRAY DIS[KC]")).sum())
R["fatal_injury_with_followup"] = int(((d.human_severity == "FATAL") & (d.event_type == "Injury") & d.type_of_report.astype(str).str.contains("Followup")).sum())
wm = d.human_primary.isin(["SKIN_NONINF","INFECTION"])
R["wound_temp_interruption"] = pc(int((wm & d.temporary_interruption).sum()), int(wm.sum()))

json.dump(R, open(OUT / "results_clinical.json", "w"), indent=1, default=int)
print(json.dumps({k: R[k] for k in ["primary","severity","causality","wound","falls","fatal","period_test","age","received_2024_2025"]}, indent=0)[:6000])
