"""Repository paths."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"                 # openFDA download (not distributed)
DERIVED = ROOT / "data" / "derived"         # files containing narratives (not distributed)
ANN = ROOT / "data" / "annotations"         # classifications by all arms (distributed)
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
CODEBOOK = ROOT / "codebook"
for p in (RAW, DERIVED, TABLES, FIGURES):
    p.mkdir(parents=True, exist_ok=True)
