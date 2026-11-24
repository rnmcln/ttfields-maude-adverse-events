#!/usr/bin/env python3
"""Secondary analysis of automated classification, revised.
(1) paired comparisons on the double-coded sample: every automated arm against the first rater, the
    second rater and the subset on which both raters agreed;
(2) paired bootstrap differences between arms on all reports (kappa and macro-F1);
(3) screening performance for three clinically defined endpoints (wound or infection report, fall
    report, reported death), for each arm and for the union of arms, with workload (share flagged);
(4) positive and negative specific agreement per category;
(5) rule-based classifier applied to the event narrative (manufacturer commentary removed).
Reference: first rater's classification unless stated otherwise. Bootstrap 2000, seed 20251017.
Output: analysis/outputs/results_concordance_v2.json
"""
import json, pathlib, glob, sys
from paths import ROOT, DERIVED, ANN, RESULTS, TABLES, FIGURES, CODEBOOK
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
from statsmodels.stats.proportion import proportion_confint

OUT = RESULTS
CATS = ["SKIN_NONINF", "INFECTION", "HARDWARE", "FALL_ENTANGLE", "NEURO", "SYSTEMIC", "USER_ERROR", "DISEASE_PROG", "OTHER", "UNCLASSIFIABLE"]
ARMS = ["regex", "qwen", "llama"]
d = pd.read_csv(DERIVED / "analysis_dataset.csv")
ev = pd.read_csv(DERIVED / "event_text_segments.csv")
import rule_based_classifier as ps
tmp = d[["row_id", "event_type"]].merge(ev, on="row_id").rename(columns={"event_text": "mdr_text_concat"})
d["regex_ev_primary"] = tmp.apply(ps.classify_regex, axis=1, result_type="expand")["regex_primary_category"].values
ARMS_ALL = ARMS + ["regex_ev"]
B = 2000; rng = np.random.default_rng(20251017); IDX = [rng.integers(0, len(d), len(d)) for _ in range(B)]
kap = lambda a, b: cohen_kappa_score(a, b, labels=CATS)
mf1 = lambda a, b: f1_score(a, b, labels=[c for c in CATS if c != "HARDWARE"], average="macro", zero_division=0)
def wil(k, n):
    lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
    return {"k": int(k), "n": int(n), "est": round(k / n, 3) if n else None, "ci": [round(lo, 3), round(hi, 3)] if n else None}
R = {}

# (1) double-coded sample
x = pd.read_csv(ANN / "rater2_sample.csv")[["row_id", "primary_category"]]
x["alm"] = x.primary_category.astype(str).str.strip().str.upper()
s = d.merge(x, on="row_id")
cons = s[s.human_primary == s.alm]
rng2 = np.random.default_rng(20251123)
def bk(a, b, fn=kap, n_b=2000):
    a, b = np.asarray(a), np.asarray(b); est = fn(a, b); v = []
    for _ in range(n_b):
        i = rng2.integers(0, len(a), len(a))
        try: v.append(fn(a[i], b[i]))
        except Exception: pass
    return {"est": round(float(est), 3), "lo": round(float(np.nanpercentile(v, 2.5)), 3), "hi": round(float(np.nanpercentile(v, 97.5)), 3)}
R["sample"] = {"n": int(len(s)), "n_consensus": int(len(cons)),
               "rater2_vs_rater1": bk(s.human_primary, s.alm)}
for a in ARMS_ALL:
    R["sample"][a] = {"vs_rater1": bk(s.human_primary, s[f"{a}_primary"]), "vs_rater2": bk(s.alm, s[f"{a}_primary"]),
                      "vs_consensus_kappa": bk(cons.human_primary, cons[f"{a}_primary"]),
                      "vs_consensus_accuracy": wil(int((cons.human_primary == cons[f"{a}_primary"]).sum()), len(cons)),
                      "agrees_with_either_rater": wil(int(((s[f"{a}_primary"] == s.human_primary) | (s[f"{a}_primary"] == s.alm)).sum()), len(s))}
R["sample"]["rater2_accuracy_vs_rater1"] = wil(int((s.human_primary == s.alm).sum()), len(s))

# (2) paired bootstrap differences on all reports
def pdiff(a1, a2, fn):
    y = d.human_primary.values; p1 = d[f"{a1}_primary"].values; p2 = d[f"{a2}_primary"].values
    est = fn(y, p2) - fn(y, p1); v = [fn(y[i], p2[i]) - fn(y[i], p1[i]) for i in IDX]
    return {"est": round(float(est), 3), "lo": round(float(np.percentile(v, 2.5)), 3), "hi": round(float(np.percentile(v, 97.5)), 3)}
R["paired_differences"] = {f"{b}_minus_{a}": {"kappa": pdiff(a, b, kap), "macro_f1": pdiff(a, b, mf1)}
                           for a, b in [("regex", "qwen"), ("regex", "llama"), ("llama", "qwen"), ("regex", "regex_ev")]}
R["overall"] = {a: {"kappa": round(float(kap(d.human_primary, d[f"{a}_primary"])), 3), "macro_f1": round(float(mf1(d.human_primary, d[f"{a}_primary"])), 3),
                    "accuracy": round(float((d.human_primary == d[f"{a}_primary"]).mean()), 3)} for a in ARMS_ALL}

# (3) screening endpoints
ENDP = {"wound_or_infection": lambda p: p.isin(["SKIN_NONINF", "INFECTION"]), "fall": lambda p: p.eq("FALL_ENTANGLE")}
truth = {"wound_or_infection": ENDP["wound_or_infection"](d.human_primary), "fall": ENDP["fall"](d.human_primary), "death": d.human_severity.eq("FATAL")}
pred = {}
for a in ARMS:
    pred[a] = {"wound_or_infection": ENDP["wound_or_infection"](d[f"{a}_primary"]), "fall": ENDP["fall"](d[f"{a}_primary"]), "death": d[f"{a}_severity"].eq("FATAL")}
pred["any_arm"] = {k: pred["regex"][k] | pred["qwen"][k] | pred["llama"][k] for k in truth}
pred["both_llms"] = {k: pred["qwen"][k] & pred["llama"][k] for k in truth}
def screen(t, p):
    t, p = t.values.astype(bool), p.values.astype(bool)
    tp, fp, fn, tn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum()
    return {"sensitivity": wil(tp, tp + fn), "specificity": wil(tn, tn + fp), "ppv": wil(tp, tp + fp), "npv": wil(tn, tn + fn),
            "flagged": wil(tp + fp, len(t)), "missed": int(fn)}
R["screening"] = {a: {k: screen(truth[k], pred[a][k]) for k in truth} for a in pred}
R["screening_prevalence"] = {k: wil(int(v.sum()), len(v)) for k, v in truth.items()}

# (4) positive and negative specific agreement per category
psa = {}
for a in ARMS_ALL:
    psa[a] = {}
    for c in CATS:
        t = d.human_primary.eq(c).values; p = d[f"{a}_primary"].eq(c).values
        n11, n10, n01, n00 = (t & p).sum(), (t & ~p).sum(), (~t & p).sum(), (~t & ~p).sum()
        psa[a][c] = {"n_ref": int(t.sum()), "positive": round(2 * n11 / (2 * n11 + n10 + n01), 3) if (2 * n11 + n10 + n01) else None,
                     "negative": round(2 * n00 / (2 * n00 + n10 + n01), 3) if (2 * n00 + n10 + n01) else None}
R["specific_agreement"] = psa
R["regex_unclassifiable"] = {"n": int(d.regex_primary.eq("UNCLASSIFIABLE").sum()), "human_unclassifiable_among": int((d.regex_primary.eq("UNCLASSIFIABLE") & d.human_primary.eq("UNCLASSIFIABLE")).sum()),
                             "human_category": d[d.regex_primary.eq("UNCLASSIFIABLE")].human_primary.value_counts().to_dict(),
                             "regex_ev_n": int(d.regex_ev_primary.eq("UNCLASSIFIABLE").sum())}
json.dump(R, open(OUT / "results_concordance_v2.json", "w"), indent=1)
print(json.dumps({"sample": R["sample"], "paired": R["paired_differences"], "overall": R["overall"]}, indent=0)[:4000])
print({a: {k: (v["sensitivity"]["est"], v["ppv"]["est"], v["flagged"]["est"], v["missed"]) for k, v in R["screening"][a].items()} for a in R["screening"]})
