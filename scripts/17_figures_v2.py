#!/usr/bin/env python3
"""Figures for the revised manuscript (1-4) from results_clinical_v2.json and the analysis data."""
import json, pathlib
from paths import ROOT, DERIVED, ANN, RESULTS, TABLES, FIGURES, CODEBOOK
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
import fig_style as S
S.apply()
FIG = FIGURES
R = json.load(open(RESULTS / "results_clinical_v2.json"))
d = pd.read_csv(DERIVED / "analysis_dataset.csv")
aud = pd.read_csv(DERIVED / "cohort_audit" / "openfda_audit_records.csv", dtype=str)
def wil(k, n): lo, hi = proportion_confint(k, n, method="wilson"); return 100 * lo, 100 * hi
def show_bounds(v): return "both" in v and v["N_llm"] >= 0.95 * v["rule"]["N"]

# ---------------- Figure 1: reports over time and report composition ----------------
d["dt"] = pd.to_datetime(d.date_received.astype(str))
d["half"] = d.dt.dt.year + (d.dt.dt.month > 6) * 0.5
miss = aud[(aud.in_study == "False") & (aud.manufacturer_d_name.fillna("").str.upper().str.contains("NOVO") | aud.product_code.isin(["NZK", "SDA", "QGZ"])) & (aud.date_received <= "20250930")].copy()
miss["dt"] = pd.to_datetime(miss.date_received); miss["half"] = miss.dt.dt.year + (miss.dt.dt.month > 6) * 0.5
d["sevfat"] = d.human_severity.isin(["SEVERE", "FATAL"])
d["age_stated"] = d.age.notna(); d["gio"] = d.device_0_brand_name.astype(str).str.contains("GIO")
d["hp"] = d.report_number.map(aud.set_index("report_number").source_type).fillna("").str.contains("Health Professional")
att = pd.read_csv(ANN / "attributes_rule_based_v2.csv"); att = att[att.set == "study"][["row_id", "hospitalised_v2"]]
d = d.merge(att, on="row_id")
halves = np.arange(2012.0, 2025.51, 0.5)
fig = plt.figure(figsize=(7.1, 3.9))
gs = fig.add_gridspec(2, 5, height_ratios=[1.15, 1], hspace=0.62, wspace=0.42)
ax = fig.add_subplot(gs[0, :])
cnt = d.groupby("half").size().reindex(halves, fill_value=0); mc = miss.groupby("half").size().reindex(halves, fill_value=0)
ax.plot(halves[:-1] + 0.25, cnt.values[:-1], color=S.INK, lw=0.7, marker="o", ms=2.6, mfc="white", mew=0.6)
ax.plot(halves[-2:] + 0.25, cnt.values[-2:], color=S.INK, lw=0.7, ls=(0, (2, 1.5)))
ax.plot(halves[-1] + 0.25, cnt.values[-1], "o", ms=2.6, mfc=S.INK, mec=S.INK, mew=0.6)
ax.plot(halves[mc.values > 0] + 0.25, mc.values[mc.values > 0], ls="none", marker="s", ms=2.6, mfc=S.LIGHT, mec=S.GREY, mew=0.6)
ax.text(2013.6, 22, "Novocure reports not returned\nby the original query", fontsize=6, color=S.GREY, va="bottom")
ax.text(2019.0, 24, "Included reports", fontsize=6, color=S.INK, ha="center")
cps = {"Optune Gio brand": R["changepoints"]["gio"]["date"], "Severity": R["changepoints"]["sevfat"]["date"]}
for i, (lbl, dt) in enumerate(cps.items()):
    t = pd.Timestamp(dt); xv = t.year + (t.dayofyear - 1) / 365.25
    ax.axvline(xv, color=S.MID, lw=0.5, ls=(0, (2, 2)), zorder=0)
ax.text(2023.75, 140, "Estimated change points\n(brand; severity)", fontsize=6, color=S.GREY, va="top", ha="right")
ax.set_xlim(2011.8, 2026.2); ax.set_ylim(0, 150); ax.set_ylabel("Reports per half-year"); ax.set_xticks(range(2012, 2026))
ax.set_xticklabels([str(y) if y % 2 == 0 else "" for y in range(2012, 2026)])
S.panel(ax, "A")
SM = [("sevfat", "Graded severe or fatal"), ("hospitalised_v2", "Hospital admission\nmentioned"), ("age_stated", "Age stated in narrative"),
      ("gio", "Optune Gio brand"), ("hp", "Health professional\namong sources")]
for j, (col, lbl) in enumerate(SM):
    ax = fig.add_subplot(gs[1, j])
    yb = d.assign(yr=d.dt.dt.year).groupby("yr")[col].agg(["sum", "size"]).reindex(range(2012, 2026), fill_value=0)
    for yr, r in yb.iterrows():
        if r["size"] == 0: continue
        lo, hi = wil(r["sum"], r["size"]); ax.plot([yr, yr], [lo, hi], color=S.LIGHT, lw=0.8, solid_capstyle="butt")
    v = 100 * yb["sum"] / yb["size"].replace(0, np.nan)
    ax.plot(yb.index, v, color=S.INK, lw=0.6, marker="o", ms=2, mfc="white", mew=0.5)
    ax.set_ylim(0, 100); ax.set_yticks([0, 50, 100]); ax.set_xticks([2012, 2018, 2025]); ax.set_xticklabels(["2012", "2018", "2025"])
    ax.set_xlim(2011.3, 2025.7)
    ax.text(0, 1.06, lbl, transform=ax.transAxes, fontsize=6.2, va="bottom", color=S.INK)
    if j == 0:
        ax.set_ylabel("% of reports"); ax.text(-0.42, 1.06, "B", transform=ax.transAxes, fontsize=8, va="bottom", color=S.INK)
S.save(fig, FIG / "Figure1_reporting_over_time")

# ---------------- dot-plot helper for Figures 2 and 3 ----------------
def dotrows(ax, rows, xmax, xlabel, bounds=True):
    y = np.arange(len(rows))[::-1]
    for yi, (lbl, v) in zip(y, rows):
        r = v["rule"]; lo, hi = r["ci95"]
        if bounds and show_bounds(v):
            ax.plot([v["both"]["pct"], v["either"]["pct"]], [yi, yi], color=S.LIGHT, lw=3.2, solid_capstyle="butt", zorder=1)
        ax.plot([lo, hi], [yi, yi], color=S.GREY, lw=0.7, solid_capstyle="butt", zorder=2)
        ax.plot(r["pct"], yi, "o", ms=3.2, mfc="white", mec=S.INK, mew=0.7, zorder=3)
        ax.text(xmax * 0.995, yi, f"{r['n']}", va="center", ha="right", fontsize=6.2, color=S.GREY)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0, xmax); ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)

W = R["wound"]
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={"wspace": 1.05, "width_ratios": [1, 0.85]})
rows = [("Operative-site involvement", W["surgical_site"]), ("Wound dehiscence", W["dehiscence"]), ("Exposed bone or implant", W["exposure"]),
        ("Infection described", W["infection_any"]), ("Infection confirmed", W["infection_confirmed"]), ("Bone flap or bone infection", W["bone_infection"]),
        ("Operative procedure", W["surgery"]), ("Intravenous antibiotics", W["iv_antibiotics"]), ("Hospital admission", W["hospitalised"]),
        ("Temporary interruption", W["temporary_interruption"]), ("Permanent discontinuation", W["permanent_discontinuation"])]
dotrows(axes[0], rows, 100, f"% of wound and infection reports (n = {W['N']})"); S.panel(axes[0], "A")
pr = [("Wound revision or closure", W["proc_wound_revision_or_closure"]), ("Debridement or washout", W["proc_debridement_or_washout"]),
      ("Bone flap or bone removal", W["proc_bone_flap_removal"]), ("Hardware or mesh removal", W["proc_hardware_or_mesh_removal"]),
      ("Flap or graft reconstruction", W["proc_flap_or_graft_reconstruction"]), ("Incision and drainage", W["proc_incision_and_drainage"]),
      ("Shunt revision or removal", W["proc_shunt_revision_or_removal"])]
dotrows(axes[1], pr, 30, f"% of wound and infection reports", bounds=False); S.panel(axes[1], "B")
S.save(fig, FIG / "Figure2_wound_attributes")

# ---------------- Figure 3: falls ----------------
F = R["falls"]
fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.9), gridspec_kw={"wspace": 1.25})
dotrows(axes[0], [("Tripped or entangled\nin cable", F["cable_trip_or_entangle"]), ("Pulled by cable", F["pulled_by_cable"]),
                  ("Carrying the device", F["carrying_device"]), ("Other cause stated", F["other_cause"])], 100, "% of fall reports", bounds=False); S.panel(axes[0], "A")
dotrows(axes[1], [("Scalp or head wound", F["head_wound"]), ("Fracture", F["fracture"]), ("Intracranial haemorrhage", F["ich"])], 100, "% of fall reports"); S.panel(axes[1], "B")
dotrows(axes[2], [("Array contribution\nto injury stated", F["array_stated"]), ("Array contribution\nnot ruled out", F["array_crbo"]),
                  ("Fall stated unrelated\nto device", F["unrelated_stated"])], 100, "% of fall reports", bounds=False); S.panel(axes[2], "C")
S.save(fig, FIG / "Figure3_falls")

# ---------------- Figure 4: period differences ----------------
P, PW = R["period"], R["period_wound"]
items = [("Report characteristics", None), ("Optune Gio brand", P["gio"]), ("Age stated in narrative", P["age_stated"]),
         ("Health professional among sources", P["hp_source"]), ("Manufacturer site Novocure GmbH", P["manufacturer_gmbh"]), ("Reporter in the United States", P["us"]),
         ("Classification", None), ("Wound or infection category", P["wound"]), ("Fall category", P["fall"]), ("Neurological category", P["neuro"]),
         ("Graded severe or fatal", P["sevfat"]), ("Hospital admission mentioned", P["hosp"]),
         ("Wound and infection reports", None), ("Operative-site involvement", PW["surgical_site"]), ("Exposed bone or implant", PW["exposure"]),
         ("Infection described", PW["infection_any"]), ("Operative procedure", PW["surgery"]), ("Hospital admission", PW["hospitalised"]),
         ("Graded severe or fatal ", PW["sevfat"])]
fig, ax = plt.subplots(figsize=(7.1, 3.9))
y = np.arange(len(items))[::-1]
for yi, (lbl, v) in zip(y, items):
    if v is None:
        ax.text(-101, yi, lbl, fontsize=6.5, va="center", ha="left", color=S.INK, style="italic"); continue
    ax.plot(v["ci95"], [yi, yi], color=S.GREY, lw=0.7, solid_capstyle="butt")
    ax.plot(v["diff_pp"], yi, "o", ms=3, mfc="white", mec=S.INK, mew=0.7)
    ax.text(-99, yi, lbl, fontsize=6.3, va="center", ha="left")
    ax.text(104, yi, f"{v['p1']['pct']:.0f} → {v['p2']['pct']:.0f}", fontsize=6.2, va="center", ha="left", color=S.GREY)
ax.axvline(0, color=S.MID, lw=0.5)
ax.set_xlim(-100, 100); ax.set_xticks([-60, -40, -20, 0, 20, 40, 60, 80]); ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Difference, 2024–2025 minus 2012–2023, percentage points (95% CI)")
ax.text(104, y[0] + 0.9, "% 2012–23 → 2024–25", fontsize=6, color=S.GREY, ha="left")
S.save(fig, FIG / "Figure4_period_comparison")
print("figures written")
