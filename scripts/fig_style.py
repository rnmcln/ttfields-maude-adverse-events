import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
GREY = "#4C566A"; LIGHT = "#D8DEE9"; MID = "#8F9BB3"; ACCENT = "#5E81AC"; ACCENT2 = "#BF616A"; INK = "#2E3440"
def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 150, "savefig.dpi": 600, "legend.frameon": False})
def panel(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
def save(fig, path):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{path}.tif", bbox_inches="tight", pad_inches=0.02, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
LABELS = {"SKIN_NONINF": "Non-infectious skin reaction", "INFECTION": "Infection", "FALL_ENTANGLE": "Fall or entanglement",
          "NEURO": "Neurological event", "SYSTEMIC": "Systemic complication", "DISEASE_PROG": "Disease progression",
          "USER_ERROR": "Use error", "OTHER": "Other", "UNCLASSIFIABLE": "Unclassifiable", "HARDWARE": "Hardware malfunction"}
