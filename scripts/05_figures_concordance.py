#!/usr/bin/env python3
import json, pathlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import fig_style as S
S.apply()
from paths import RESULTS, FIGURES as FIG
R = json.load(open(RESULTS / "results_concordance.json"))
pc = pd.DataFrame(R["per_category"])
MK = {"Rule-based": ("s", S.GREY), "Qwen 2.5 7B": ("o", S.ACCENT), "Llama 3.1 8B": ("^", S.ACCENT2)}
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), gridspec_kw={"wspace": 0.85, "width_ratios": [1, 1.1]})
ax = axes[0]; P = list(R["pairwise_primary"])
irp = RESULTS / "results_interrater.json"
if irp.exists():
    IR = json.load(open(irp)); P = [{"pair": f"First vs second human rater (n = {IR['n']})", "a": "human", "b": "human2", "kappa": IR["primary_kappa"]}] + P
y = np.arange(len(P))[::-1]
for yi, p in zip(y, P):
    k = p["kappa"]; ref = p["a"] == "human"
    if p.get("b") == "human2":
        ax.plot([k["lo"], k["hi"]], [yi, yi], color=S.INK, lw=0.7); ax.plot(k["est"], yi, "s", ms=3, mfc=S.INK, mec=S.INK)
        ax.text(1.0, yi, f'{k["est"]:.2f}', va="center", ha="right", fontsize=6, color=S.GREY); continue
    col = S.INK if ref else S.MID
    ax.plot([k["lo"], k["hi"]], [yi, yi], color=col, lw=0.7)
    ax.plot(k["est"], yi, "o", ms=3, mfc="white" if ref else col, mec=col, mew=0.7)
    ax.text(1.0, yi, f'{k["est"]:.2f}', va="center", ha="right", fontsize=6, color=S.GREY)
ax.set_yticks(y); ax.set_yticklabels([p["pair"] for p in P]); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8]); ax.set_xlabel("Cohen's kappa, primary category (95% CI)")
S.panel(ax, "A")
ax = axes[1]
cats = [c for c in ["INFECTION","SKIN_NONINF","FALL_ENTANGLE","NEURO","OTHER","DISEASE_PROG","SYSTEMIC"]]
yy = np.arange(len(cats))[::-1]
arms = [a for a in MK if a in set(pc.arm)]
offs = {a: o for a, o in zip(arms, np.linspace(0.2, -0.2, len(arms)) if len(arms) > 1 else [0])}
for a in arms:
    m, col = MK[a]; s = pc[pc.arm == a].set_index("category")
    ax.plot([s.loc[c, "f1"] for c in cats], yy + offs[a], m, ls="none", ms=3, mfc="white", mec=col, mew=0.7)
    yo = yy[cats.index("OTHER")] + offs[a]; xo = s.loc["OTHER", "f1"]
    ax.text(xo + 0.035, yo, a, fontsize=6, va="center", ha="left", color=col)
sup = pc[pc.arm == arms[0]].set_index("category")["support_human"]
ax.set_yticks(yy); ax.set_yticklabels([f"{S.LABELS[c]} ({sup[c]})" for c in cats]); ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False); ax.set_xlim(-0.03, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("F1 score against human classification")
S.panel(ax, "B")
S.save(fig, FIG / "Figure4_classification_agreement")
print("ok", arms)
