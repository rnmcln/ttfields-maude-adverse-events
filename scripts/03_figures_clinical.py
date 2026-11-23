#!/usr/bin/env python3
import json, pathlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
import fig_style as S
S.apply()
from paths import DERIVED, FIGURES as FIG
d = pd.read_csv(DERIVED / "analysis_dataset.csv"); N = len(d)
ORDER = ["INFECTION","SKIN_NONINF","FALL_ENTANGLE","NEURO","OTHER","DISEASE_PROG","SYSTEMIC","UNCLASSIFIABLE","USER_ERROR","HARDWARE"]
SEV = ["MILD","MODERATE","SEVERE","FATAL"]
def wil(k, n): lo, hi = proportion_confint(k, n, method="wilson"); return 100*lo, 100*hi

# ---------- Figure 1: category distribution (A) and severity within category (B)
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.08})
ax = axes[0]; y = np.arange(len(ORDER))[::-1]
for yi, c in zip(y, ORDER):
    k = (d.human_primary == c).sum(); p = 100*k/N; lo, hi = wil(k, N)
    ax.plot([lo, hi], [yi, yi], color=S.GREY, lw=0.7, solid_capstyle="butt")
    ax.plot(p, yi, "o", ms=3.2, mfc="white", mec=S.INK, mew=0.7)
    ax.text(hi + 1.2, yi, f"{k}", va="center", fontsize=6.5, color=S.GREY)
ax.set_yticks(y); ax.set_yticklabels([S.LABELS[c] for c in ORDER])
ax.set_xlim(0, 45); ax.set_xlabel("Share of reports, % (95% CI)"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
S.panel(ax, "A")
ax = axes[1]
shown = [c for c in ORDER if (d.human_primary == c).sum() >= 10]
yy = np.array([y[ORDER.index(c)] for c in shown])
xs = {"MILD": 0, "MODERATE": 1, "SEVERE": 2, "FATAL": 3}
for c in shown:
    s = d[d.human_primary == c]; n = len(s); yi = y[ORDER.index(c)]
    for v in SEV:
        k = (s.human_severity == v).sum(); p = k/n
        if k == 0: continue
        col = S.ACCENT2 if v == "FATAL" else S.INK
        ax.scatter(xs[v], yi, s=2 + 90*p, facecolors="none", edgecolors=col, linewidths=0.6, zorder=3)
        ax.text(xs[v] + 0.22, yi, f"{100*p:.0f}", va="center", fontsize=5.8, color=S.GREY)
ax.set_xticks(range(4)); ax.set_xticklabels(["Mild","Moderate","Severe","Fatal"])
ax.set_xlim(-0.4, 3.6); ax.set_ylim(axes[0].get_ylim()); ax.set_yticks([])
ax.spines["left"].set_visible(False); ax.set_xlabel("Severity within category, % of category")
S.panel(ax, "B")
S.save(fig, FIG / "Figure1_categories_severity")

# ---------- Figure 2: annual reports, small multiples
yrs = np.arange(2012, 2026)
groups = [("INFECTION", "Infection"), ("SKIN_NONINF", "Non-infectious skin reaction"), ("FALL_ENTANGLE", "Fall or entanglement"),
          ("NEURO", "Neurological event"), ("REST", "All other categories")]
fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.0), gridspec_kw={"wspace": 0.42, "hspace": 0.6})
axes = axes.ravel()
groups = [("INFECTION", "Infection"), ("SKIN_NONINF", "Non-infectious skin reaction"), ("FALL_ENTANGLE", "Fall or entanglement"),
          ("NEURO", "Neurological event"), ("REST", "All other categories")]
tot = d.year_received.value_counts().reindex(yrs, fill_value=0)
def draw(ax, v, col, ymax, ticks, lbl):
    ax.plot(yrs[:-1], v[:-1], color=col, lw=0.7, marker="o", ms=2, mfc="white", mew=0.6)
    ax.plot(yrs[-2:], v[-2:], color=col, lw=0.7, ls=(0, (2, 1.5)))
    ax.plot(yrs[-1], v[-1], marker="o", ms=2, mfc=col, mec=col, mew=0.6, ls="none")
    ax.set_ylim(0, ymax); ax.set_yticks(ticks)
    ax.text(0.0, 1.06, lbl, transform=ax.transAxes, fontsize=6.5, ha="left", va="bottom", color=S.INK)
    ax.set_xticks([2012, 2018, 2025]); ax.set_xlim(2011.3, 2025.7)
draw(axes[0], tot.values, S.INK, 190, [0, 50, 100, 150], "All reports")
for ax, (c, lbl) in zip(axes[1:], groups):
    s = d[d.human_primary == c] if c != "REST" else d[~d.human_primary.isin([g[0] for g in groups[:4]])]
    v = s.year_received.value_counts().reindex(yrs, fill_value=0)
    draw(ax, v.values, S.ACCENT, 70, [0, 20, 40, 60], lbl)
sv = d.groupby("year_received").human_severity.apply(lambda x: 100*x.isin(["SEVERE","FATAL"]).mean()).reindex(yrs)
hp = d.groupby("year_received").hospitalised.mean().mul(100).reindex(yrs)
for ax, ser, lbl in [(axes[6], sv, "Severe or fatal, % of reports"), (axes[7], hp, "Hospital admission, % of reports")]:
    m = ser.notna().values
    ax.plot(yrs[m][:-1], ser.values[m][:-1], color=S.INK, lw=0.7, marker="o", ms=2, mfc="white", mew=0.6)
    ax.plot(yrs[m][-2:], ser.values[m][-2:], color=S.INK, lw=0.7, ls=(0, (2, 1.5)))
    ax.plot(yrs[-1], ser.values[-1], marker="o", ms=2, mfc=S.INK, mec=S.INK, ls="none")
    ax.set_ylim(0, 105); ax.set_yticks([0, 50, 100]); ax.set_xticks([2012, 2018, 2025]); ax.set_xlim(2011.3, 2025.7)
    ax.text(0.0, 1.06, lbl, transform=ax.transAxes, fontsize=6.5, ha="left", va="bottom", color=S.INK)
axes[0].set_ylabel("Reports per year"); axes[4].set_ylabel("Reports per year"); axes[6].set_ylabel("%")
S.save(fig, FIG / "Figure2_annual_reports")

# ---------- Figure 3: narrative descriptors in wound/infection and fall reports
w = d[d.human_primary.isin(["SKIN_NONINF","INFECTION"])]; f = d[d.human_primary == "FALL_ENTANGLE"]
WD = [("surgical_site", "Prior surgical site involved"), ("surgical_revision", "Surgical wound revision"),
      ("implant_exposure", "Exposed implant or bone"), ("hospitalised", "Hospital admission"),
      ("iv_antibiotics", "Intravenous antibiotics"), ("organism_or_culture", "Organism or positive culture"),
      ("bevacizumab", "Bevacizumab mentioned"), ("temporary_interruption", "Temporary interruption")]
FD = [("head_wound_after_fall", "Scalp or head wound"), ("temporary_interruption", "Temporary interruption"),
      ("hospitalised", "Hospital admission"), ("cable_or_bag_mechanism", "Cable, cord or bag mentioned"),
      ("fracture", "Fracture"), ("intracranial_bleed", "Intracranial haemorrhage")]
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4), gridspec_kw={"wspace": 0.75, "width_ratios": [1, 1]})
for ax, data, items, letter in [(axes[0], w, WD, "A"), (axes[1], f, FD, "B")]:
    yv = np.arange(len(items))[::-1]
    for yi, (k, lbl) in zip(yv, items):
        if data is w:
            for sub, mk, off, col in [("INFECTION", "o", 0.17, S.INK), ("SKIN_NONINF", "s", -0.17, S.ACCENT)]:
                s = data[data.human_primary == sub]; kk = s[k].sum(); n = len(s); lo, hi = wil(kk, n)
                ax.plot([lo, hi], [yi + off]*2, color=col, lw=0.6); ax.plot(100*kk/n, yi + off, mk, ms=2.8, mfc="white", mec=col, mew=0.7)
        else:
            kk = data[k].sum(); n = len(data); lo, hi = wil(kk, n)
            ax.plot([lo, hi], [yi]*2, color=S.INK, lw=0.6); ax.plot(100*kk/n, yi, "o", ms=2.8, mfc="white", mec=S.INK, mew=0.7)
    ax.set_yticks(yv); ax.set_yticklabels([i[1] for i in items]); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("Reports with narrative mention, % (95% CI)")
    S.panel(ax, letter)
# direct labels for panel A series
yi0 = len(WD) - 3
s1 = w[w.human_primary == "INFECTION"]; s2 = w[w.human_primary == "SKIN_NONINF"]
axes[0].text(100, yi0 + 0.17, f"Infection (n = {len(s1)})", fontsize=6, va="center", ha="right", color=S.INK)
axes[0].text(100, yi0 - 0.17, f"Non-infectious skin (n = {len(s2)})", fontsize=6, va="center", ha="right", color=S.ACCENT)
axes[1].text(100, len(FD) - 0.45, f"Falls or entanglement (n = {len(f)})", fontsize=6, ha="right", va="center", color=S.INK)
S.save(fig, FIG / "Figure3_wound_fall_descriptors")
print("ok")
