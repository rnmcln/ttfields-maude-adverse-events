#!/usr/bin/env python3
"""Agreement between classification arms. Human classification is the reference standard.
Bootstrap: 2000 resamples of reports, seed 20251017, percentile 95% CIs.
Output: analysis/outputs/results_concordance.json and tables/concordance_*.csv"""
import json, pathlib, itertools, importlib.util, re
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import krippendorff

from paths import DERIVED, ANN, RESULTS, TABLES
OUT = RESULTS; TAB = TABLES
B = 2000; SEED = 20251017
d = pd.read_csv(DERIVED / "analysis_dataset.csv", keep_default_na=False, na_values=[""])
CATS = ["SKIN_NONINF","INFECTION","HARDWARE","FALL_ENTANGLE","NEURO","SYSTEMIC","USER_ERROR","DISEASE_PROG","OTHER","UNCLASSIFIABLE"]
SEV = ["MILD","MODERATE","SEVERE","FATAL"]
arms = ["human", "regex"] + [a for a in ["qwen", "llama"] if d.get(f"{a}_available", pd.Series([False])).astype(bool).all()]
LABEL = {"human": "Human", "regex": "Rule-based", "qwen": "Qwen 2.5 7B", "llama": "Llama 3.1 8B"}
R = {"arms": arms, "N": len(d), "B": B, "seed": SEED}
for a in ["qwen", "llama"]:
    if f"{a}_available" in d:
        R[f"{a}_n_available"] = int(d[f"{a}_available"].astype(bool).sum())
        R[f"{a}_parse_failures"] = int((d[f"{a}_available"].astype(bool) & ~d[f"{a}_parse_ok"].astype(bool)).sum())
        raw = ANN / "llm" / {"qwen": "qwen2.5_7b.jsonl", "llama": "llama3.1_8b.jsonl"}[a]
        if raw.exists():
            recs = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
            R[f"{a}_n_retried"] = int(sum(r["n_attempts"] > 1 for r in recs))
            R[f"{a}_median_seconds"] = float(np.median([r["attempts"][-1]["seconds"] for r in recs]))
rng = np.random.default_rng(SEED)
idx_boot = [rng.integers(0, len(d), len(d)) for _ in range(B)]
def boot(fn, *cols):
    est = fn(*cols); vals = []
    for ix in idx_boot:
        try: vals.append(fn(*[c[ix] for c in cols]))
        except Exception: pass
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return {"est": round(float(est), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3)}
def kappa(a, b): return cohen_kappa_score(a, b, labels=CATS)
def pa(a, b): return float(np.mean(a == b))
sev_map = {s: i for i, s in enumerate(SEV)}

# pairwise primary-category agreement
pairs = []
for a, b in itertools.combinations(arms, 2):
    x = d[f"{a}_primary"].values; y = d[f"{b}_primary"].values
    pairs.append({"pair": f"{LABEL[a]} vs {LABEL[b]}", "a": a, "b": b, "agreement": boot(pa, x, y), "kappa": boot(kappa, x, y)})
R["pairwise_primary"] = pairs
pd.DataFrame([{"pair": p["pair"], "agreement_%": f"{100*p['agreement']['est']:.1f} ({100*p['agreement']['lo']:.1f} to {100*p['agreement']['hi']:.1f})",
               "kappa": f"{p['kappa']['est']:.2f} ({p['kappa']['lo']:.2f} to {p['kappa']['hi']:.2f})"} for p in pairs]).to_csv(TAB / "concordance_pairwise_primary.csv", index=False)

# multi-rater
if len(arms) >= 3:
    M = d[[f"{a}_primary" for a in arms]].values
    tab, _ = aggregate_raters(M); R["fleiss_kappa_all"] = round(float(fleiss_kappa(tab)), 3)
    code = {c: i for i, c in enumerate(CATS)}
    R["krippendorff_alpha_all"] = round(float(krippendorff.alpha(reliability_data=np.vectorize(code.get)(M).T.astype(float), level_of_measurement="nominal")), 3)

# per-category performance vs human
perf = []; summ = {}
yt = d.human_primary.values
for a in arms[1:]:
    yp = d[f"{a}_primary"].values
    P, Rc, F, S = precision_recall_fscore_support(yt, yp, labels=CATS, zero_division=0)
    for c, p_, r_, f_, s_ in zip(CATS, P, Rc, F, S):
        mcc = matthews_corrcoef(yt == c, yp == c) if (yt == c).any() or (yp == c).any() else np.nan
        perf.append({"arm": LABEL[a], "category": c, "support_human": int(s_), "n_predicted": int((yp == c).sum()),
                     "precision": round(p_, 3), "recall": round(r_, 3), "f1": round(f_, 3), "mcc": round(float(mcc), 3) if mcc == mcc else None})
    present = [c for c in CATS if (yt == c).any()]
    def macro_f1(t, p): return precision_recall_fscore_support(t, p, labels=present, average="macro", zero_division=0)[2]
    def weighted_f1(t, p): return precision_recall_fscore_support(t, p, labels=present, average="weighted", zero_division=0)[2]
    summ[a] = {"accuracy": boot(pa, yt, yp), "kappa": boot(kappa, yt, yp), "macro_f1": boot(macro_f1, yt, yp), "weighted_f1": boot(weighted_f1, yt, yp),
               "macro_mcc": round(float(np.nanmean([matthews_corrcoef(yt == c, yp == c) for c in present])), 3)}
    cm = pd.DataFrame(confusion_matrix(yt, yp, labels=CATS), index=[f"human_{c}" for c in CATS], columns=[f"{a}_{c}" for c in CATS])
    cm.to_csv(TAB / f"confusion_human_vs_{a}.csv")
R["vs_human"] = summ
pd.DataFrame(perf).to_csv(TAB / "concordance_per_category.csv", index=False)
R["per_category"] = perf


# sensitivity: exclude the 20 calibration reports (row_id 1 to 20, coded jointly with the supervising investigator)
cal = d.row_id.values > 20
R["calibration_excluded"] = {"n": int(cal.sum())}
for a in arms[1:]:
    yt2 = d.human_primary.values[cal]; yp2 = d[f"{a}_primary"].values[cal]
    R["calibration_excluded"][a] = {"accuracy": round(float(np.mean(yt2 == yp2)), 3), "kappa": round(float(cohen_kappa_score(yt2, yp2, labels=CATS)), 3)}

# secondary dimensions
sec = {}
for a in arms[1:]:
    hs = d.human_severity.map(sev_map).values; xs = d[f"{a}_severity"].map(sev_map)
    ok = ~xs.isna().values
    sec[a] = {
        "severity_n_valid": int(ok.sum()),
        "severity_linear_weighted_kappa": boot(lambda u, v: cohen_kappa_score(u, v, weights="linear"), hs[ok], xs.values[ok].astype(int)),
        "severity_exact_agreement": boot(pa, hs[ok], xs.values[ok].astype(int)),
        "causality_kappa": boot(lambda u, v: cohen_kappa_score(u, v), d.human_causality.values, d[f"{a}_causality"].fillna("NA").values),
        "fatal_sensitivity": None, "fatal_ppv": None,
    }
    hf = d.human_severity.values == "FATAL"; af = d[f"{a}_severity"].values == "FATAL"
    sec[a]["fatal_sensitivity"] = {"k": int((hf & af).sum()), "n": int(hf.sum())}
    sec[a]["fatal_ppv"] = {"k": int((hf & af).sum()), "n": int(af.sum())}
    miss = hf & ~af
    sec[a]["fatal_missed"] = {"n": int(miss.sum()), "event_type_injury": int((miss & (d.event_type.values == "Injury")).sum())}
    ha = d.human_component.str.contains("ARRAY").values; aa = d[f"{a}_component"].fillna("").str.contains("ARRAY").values
    sec[a]["array_component_kappa"] = boot(cohen_kappa_score, ha.astype(int), aa.astype(int))
R["secondary_dimensions"] = sec
hf = d.human_severity.values == "FATAL"; ef = d.event_type.values == "Death"
R["event_type_death_vs_human_fatal"] = {"k": int((hf & ef).sum()), "n_human_fatal": int(hf.sum()), "n_event_type_death": int(ef.sum())}

# regex error analysis: infection over-calls attributable to manufacturer commentary
import rule_based_classifier as P
ev = pd.read_csv(DERIVED / "event_text_segments.csv").set_index("row_id").event_text.reindex(d.row_id).fillna("").values
fp_inf = (d.regex_primary == "INFECTION") & (d.human_primary != "INFECTION")
still = [P._match_with_negation(t, P.INFECTION_PATTERNS) for t in ev]
R["regex_infection_overcall"] = {"n_fp": int(fp_inf.sum()), "n_regex_infection": int((d.regex_primary == "INFECTION").sum()),
    "fp_human_category": {k: int(v) for k, v in d[fp_inf].human_primary.value_counts().items()},
    "fp_no_infection_term_after_removing_commentary": int((fp_inf & ~np.array(still)).sum())}
R["regex_unclassifiable"] = {"n": int((d.regex_primary == "UNCLASSIFIABLE").sum()),
    "human_category": {k: int(v) for k, v in d[d.regex_primary == "UNCLASSIFIABLE"].human_primary.value_counts().items()}}
R["regex_hardware"] = {"n": int((d.regex_primary == "HARDWARE").sum()),
    "human_category": {k: int(v) for k, v in d[d.regex_primary == "HARDWARE"].human_primary.value_counts().items()}}
# post hoc: regex on event text only (commentary removed)
d2 = d.copy(); d2["mdr_text_concat"] = ev
rx2 = d2.apply(P.classify_regex, axis=1, result_type="expand")["regex_primary_category"].values
R["regex_event_text_only_posthoc"] = {"accuracy": boot(pa, yt, rx2), "kappa": boot(kappa, yt, rx2),
                                       "n_infection": int((rx2 == "INFECTION").sum())}
json.dump(R, open(OUT / "results_concordance.json", "w"), indent=1, default=float)
print(json.dumps({k: R[k] for k in R if k not in ("per_category",)}, indent=0)[:5000])
