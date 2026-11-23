#!/usr/bin/env python3
"""Draw a simple random sample of reports for independent re-coding by a second rater (ALM)
and write a blinded coding workbook. Calibration reports (row_id 1 to 20) are excluded.
Sample size 80: for kappa around 0.7 on the primary category this gives a 95% CI half-width of
roughly 0.10 to 0.12. Seed fixed for reproducibility."""
import pathlib, re, numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

SEED, N_SAMPLE = 20251123, 80
from paths import DERIVED, ANN, CODEBOOK
clean = pd.read_csv(DERIVED / "maude_ttfields_clean.csv", dtype={"report_number": str})
elig = clean[clean.row_id > 20].copy()
rng = np.random.default_rng(SEED)
idx = rng.choice(elig.index.values, size=N_SAMPLE, replace=False)
samp = elig.loc[idx].copy()
samp.insert(0, "sample_id", [f"S{i+1:03d}" for i in range(N_SAMPLE)])
samp[["sample_id", "row_id", "report_number"]].to_csv(DERIVED / "double_coding_sample_key_redrawn.csv", index=False)
ref = pd.read_csv(ANN / "double_coding_sample_key.csv")
print("Redrawn sample identical to archived sample:", list(ref.row_id) == list(samp.row_id))

CATS = ["SKIN_NONINF","INFECTION","HARDWARE","FALL_ENTANGLE","NEURO","SYSTEMIC","USER_ERROR","DISEASE_PROG","OTHER","UNCLASSIFIABLE"]
SEV = ["MILD","MODERATE","SEVERE","FATAL"]; CAUS = ["DEFINITE","PROBABLE","POSSIBLE","UNLIKELY","UNASSESSABLE"]
F = Font(name="Arial", size=10); FB = Font(name="Arial", size=10, bold=True); YEL = PatternFill("solid", start_color="FFFF00")
thin = Side(style="thin", color="BFBFBF"); BOX = Border(top=thin, bottom=thin, left=thin, right=thin)
wb = Workbook()

# ---- Instructions
ws = wb.active; ws.title = "Instructions"
lines = [
 ("Independent re-coding of MAUDE TTFields reports (second rater)", FB),
 ("", F),
 (f"Purpose: estimate agreement between two human raters on the codebook, to qualify the single-rater reference standard. {N_SAMPLE} reports drawn at random (seed {SEED}) from the 528 reports classified independently (calibration reports 1 to 20 excluded).", F),
 ("Code each report from the narrative alone, without consulting the first rater's coding, the automated outputs or the analysis results for individual reports. Do not discuss individual reports with the first rater before all 80 are coded.", F),
 ("Use the codebook (Codebook_MAUDE_TTFields_v1.docx) and its decision rules; a summary is on the sheet 'Codes'.", F),
 ("", F),
 ("Cells to edit: the yellow columns on sheet 'Coding' only (H to M). Columns H, I, J and L have drop-down lists. Column K (device component) takes one or more of ARRAY, CABLE, BATTERY, UNIT, SOFTWARE, NONE, UNSPECIFIED separated by |. Column L (secondary category) may be left empty. Column M is for notes.", F),
 ("Do not change columns A to G or the order of rows. Return the file under the same name.", F),
 ("", F),
 ("Example of the expected format (the worked example given in section 8 of the codebook; not one of the sampled reports):", FB),
]
for i, (t, f) in enumerate(lines, 1):
    c = ws.cell(row=i, column=1, value=t); c.font = f; c.alignment = Alignment(wrap_text=True, vertical="top")
hdr = ["Primary category", "Severity", "Causality", "Device component", "Secondary category", "Notes"]
ex = ["SKIN_NONINF", "MODERATE", "PROBABLE", "ARRAY", "", "Dermatitis under transducer arrays requiring topical treatment (worked example from codebook section 8)"]
r0 = len(lines) + 1
for j, (h, v) in enumerate(zip(hdr, ex)):
    ws.cell(row=r0, column=1 + j, value=h).font = FB; ws.cell(row=r0 + 1, column=1 + j, value=v).font = F
ws.column_dimensions["A"].width = 110
for col in "BCDEF": ws.column_dimensions[col].width = 18

# ---- Coding sheet
ws = wb.create_sheet("Coding")
cols = ["sample_id", "row_id", "report_number", "event_type", "date_received", "device_0_brand_name", "mdr_text_concat",
        "alm_primary_category", "alm_severity", "alm_causality", "alm_device_component", "alm_secondary_category", "alm_notes"]
for j, c in enumerate(cols, 1):
    cell = ws.cell(row=1, column=j, value=c); cell.font = FB; cell.border = BOX; cell.alignment = Alignment(wrap_text=True, vertical="top")
for i, (_, r) in enumerate(samp.iterrows(), 2):
    vals = [r.sample_id, int(r.row_id), str(r.report_number), r.event_type, str(int(r.date_received)), r.device_0_brand_name, str(r.mdr_text_concat)]
    for j, v in enumerate(vals, 1):
        cell = ws.cell(row=i, column=j, value=v); cell.font = F; cell.border = BOX; cell.alignment = Alignment(wrap_text=True, vertical="top")
    for j in range(8, 14):
        cell = ws.cell(row=i, column=j); cell.fill = YEL; cell.border = BOX; cell.font = F; cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[i].height = min(409, 15 * (1 + len(vals[6]) // 95))
last = N_SAMPLE + 1
for col, opts, allow_blank in [("H", CATS, False), ("I", SEV, False), ("J", CAUS, False), ("L", CATS, True)]:
    dv = DataValidation(type="list", formula1='"' + ",".join(opts) + '"', allow_blank=allow_blank, showErrorMessage=True,
                        errorTitle="Invalid code", error="Choose a code from the list")
    ws.add_data_validation(dv); dv.add(f"{col}2:{col}{last}")
widths = [9, 7, 16, 10, 11, 13, 95, 18, 12, 14, 16, 18, 30]
for j, w in enumerate(widths, 1): ws.column_dimensions[get_column_letter(j)].width = w
ws.freeze_panes = "H2"

# ---- Codes reference
ws = wb.create_sheet("Codes")
ref = [("Code", "Definition (codebook v1.0; inclusion and exclusion criteria in full in the codebook)")]
for line in (CODEBOOK / "llm_prompt.txt").read_text().splitlines():
    m = re.match(r"^([A-Z_]+|\d\.): (.*)$", line) if not line.startswith(("DIMENSION", "Output")) else None
    if line.startswith("DIMENSION") or line.startswith("DECISION RULES"):
        ref.append(("", "")); ref.append((line.split(":")[0].title() if line.startswith("DIMENSION") else "Decision rules", line.split(":", 1)[1].strip() if ":" in line else ""))
    elif re.match(r"^\d\. ", line):
        ref.append(("Rule " + line[0], line[3:]))
    elif m and not line.startswith("{"):
        ref.append((m.group(1), m.group(2)))
    elif line.startswith("ARRAY"):
        ref.append(("Components", line))
for i, (a_, b_) in enumerate(ref, 1):
    ws.cell(row=i, column=1, value=a_).font = FB if a_ in ("Code", "Decision rules") or a_.startswith("Dimension") else F
    c = ws.cell(row=i, column=2, value=b_); c.font = F; c.alignment = Alignment(wrap_text=True)
ws.column_dimensions["A"].width = 18; ws.column_dimensions["B"].width = 120
out = DERIVED / "Double_coding_80_reports.xlsx"
wb.save(out); print(out, len(samp))
