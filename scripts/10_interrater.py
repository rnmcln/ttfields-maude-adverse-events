#!/usr/bin/env python3
"""Inter-rater agreement between the first rater and the second rater on the double-coded random sample
(data/annotations/rater2_sample.csv). Output: results/results_interrater.json"""
import json, pathlib, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
from paths import DERIVED, ANN, RESULTS, TABLES
CATS = ["SKIN_NONINF","INFECTION","HARDWARE","FALL_ENTANGLE","NEURO","SYSTEMIC","USER_ERROR","DISEASE_PROG","OTHER","UNCLASSIFIABLE"]
SEV = ["MILD","MODERATE","SEVERE","FATAL"]
x = pd.read_csv(ANN / "rater2_sample.csv", dtype={"report_number": str})
x.columns = ["sample_id", "row_id", "report_number"] + ["alm_" + c for c in x.columns[3:]]
key = pd.read_csv(ANN / "double_coding_sample_key.csv")
assert list(x.row_id) == list(key.row_id), "row order changed"
for c in ["alm_primary_category","alm_severity","alm_causality"]:
    x[c] = x[c].astype(str).str.strip().str.upper().where(x[c].notna(), np.nan)
done = x.alm_primary_category.notna().sum()
if done < len(x):
    print(f"{done}/{len(x)} coded; analysis not run"); raise SystemExit(0)
d = pd.read_csv(DERIVED / "analysis_dataset.csv").set_index("row_id").loc[x.row_id]
rng = np.random.default_rng(20251123); B = 2000
def boot(f, a, b):
    a = np.asarray(a); b = np.asarray(b); est = f(a, b); v = []
    for _ in range(B):
        i = rng.integers(0, len(a), len(a))
        try: v.append(f(a[i], b[i]))
        except Exception: pass
    lo, hi = np.nanpercentile(v, [2.5, 97.5]); return {"est": round(float(est), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3)}
sm = {s: i for i, s in enumerate(SEV)}
R = {"n": int(len(x)), "seed_sample": 20251123, 
     "primary_agreement": boot(lambda a, b: float(np.mean(a == b)), d.human_primary.values, x.alm_primary_category.values),
     "primary_kappa": boot(lambda a, b: cohen_kappa_score(a, b, labels=CATS), d.human_primary.values, x.alm_primary_category.values),
     "severity_weighted_kappa": boot(lambda a, b: cohen_kappa_score(a, b, weights="linear"), d.human_severity.map(sm).values, x.alm_severity.map(sm).values),
     "causality_kappa": boot(cohen_kappa_score, d.human_causality.values, x.alm_causality.values)}
ha = d.human_component.str.contains("ARRAY").astype(int).values; aa = x.alm_device_component.astype(str).str.upper().str.contains("ARRAY").astype(int).values
R["array_component_kappa"] = boot(cohen_kappa_score, ha, aa)
R["severity_exact_agreement"] = boot(lambda a, b: float(np.mean(a == b)), d.human_severity.values, x.alm_severity.values)
R["causality_agreement"] = boot(lambda a, b: float(np.mean(a == b)), d.human_causality.values, x.alm_causality.values)
GRP = {"SKIN_NONINF": "WOUND", "INFECTION": "WOUND", "FALL_ENTANGLE": "FALL", "NEURO": "NEURO"}
g = lambda v: np.array([GRP.get(c, "OTHER_GROUPED") for c in v])
R["grouped_kappa"] = boot(cohen_kappa_score, g(d.human_primary.values), g(x.alm_primary_category.values))
R["grouped_agreement"] = boot(lambda a, b: float(np.mean(a == b)), g(d.human_primary.values), g(x.alm_primary_category.values))
ws, wa = g(d.human_primary.values) == "WOUND", g(x.alm_primary_category.values) == "WOUND"
R["wound_group"] = {"sn_n": int(ws.sum()), "alm_n": int(wa.sum()), "both": int((ws & wa).sum()), "kappa": boot(cohen_kappa_score, ws.astype(int), wa.astype(int))}
fs, fa = d.human_primary.values == "FALL_ENTANGLE", x.alm_primary_category.values == "FALL_ENTANGLE"
R["falls"] = {"sn_n": int(fs.sum()), "alm_n": int(fa.sum()), "both": int((fs & fa).sum()),
              "sn_fall_alm_other_or_dp": int((fs & np.isin(x.alm_primary_category.values, ["OTHER", "DISEASE_PROG"])).sum())}
ns, na_ = d.human_primary.values == "NEURO", x.alm_primary_category.values == "NEURO"
R["neuro"] = {"sn_n": int(ns.sum()), "alm_n": int(na_.sum()), "both": int((ns & na_).sum()), "sn_neuro_alm_dp": int((ns & (x.alm_primary_category.values == "DISEASE_PROG")).sum())}
R["infection_skin_swaps"] = {"sn_inf_alm_skin": int(((d.human_primary.values == "INFECTION") & (x.alm_primary_category.values == "SKIN_NONINF")).sum()),
                             "sn_skin_alm_inf": int(((d.human_primary.values == "SKIN_NONINF") & (x.alm_primary_category.values == "INFECTION")).sum())}
R["causality_unassessable_vs_possible"] = int(((d.human_causality.values == "UNASSESSABLE") & (x.alm_causality.values == "POSSIBLE")).sum())
R["causality_possible_vs_unlikely"] = int(((d.human_causality.values == "POSSIBLE") & (x.alm_causality.values == "UNLIKELY")).sum())
# reports whose coding queries were discussed with the supervising investigator (first rater's notes)
for arm in ["regex","qwen","llama"]:
    col = f"{arm}_primary"
    if col in d and d[col].notna().all():
        R[f"{arm}_kappa_vs_sn_sample"] = boot(lambda a, b: cohen_kappa_score(a, b, labels=CATS), d.human_primary.values, d[col].values)
for arm in ["regex","qwen","llama"]:
    col = f"{arm}_primary"
    if col in d and d[col].notna().all():
        R[f"{arm}_kappa_vs_alm"] = boot(lambda a, b: cohen_kappa_score(a, b, labels=CATS), x.alm_primary_category.values, d[col].values)
pd.crosstab(d.human_primary.values, x.alm_primary_category.values, rownames=["rater1"], colnames=["rater2"]).to_csv(TABLES / "interrater_confusion.csv")
json.dump(R, open(RESULTS / "results_interrater.json", "w"), indent=1); print(json.dumps(R, indent=1))
