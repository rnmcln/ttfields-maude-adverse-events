#!/bin/bash
# Independent ascertainment arm from FDA bulk MAUDE device files (not openFDA).
# Downloads device<YEAR>.zip for 2012-2025 to a temporary folder, keeps only device rows whose
# product code is NZK, SDA or QGZ or whose brand, generic or manufacturer name contains a TTFields
# or Novocure term, and writes them to analysis/outputs/cohort_audit/bulk_device_matches.txt.
# The downloaded files are not retained.
set -u
OUT="$(cd "$(dirname "$0")/.." && pwd)/data/derived/cohort_audit"
mkdir -p "$OUT"; TMP=$(mktemp -d)
: > "$OUT/bulk_device_matches.txt"
for Y in $(seq 2012 2025); do
  F="device${Y}.zip"
  curl -s -f -A "Mozilla/5.0" -o "$TMP/$F" "https://www.accessdata.fda.gov/MAUDE/ftparea/$F" || { echo "$F download failed" >> "$OUT/bulk_scan.log"; continue; }
  if [ "$Y" = 2012 ]; then unzip -p "$TMP/$F" | head -1 > "$OUT/bulk_device_header.txt"; fi
  unzip -p "$TMP/$F" | LC_ALL=C grep -a -i -E '\|(NZK|SDA|QGZ)\||NOVOCURE|NOVO CURE|OPTUNE|NOVOTTF|NOVO-TTF|NOVOTFF|TUMOR TREATING FIELD' \
     | sed "s/^/${Y}|/" >> "$OUT/bulk_device_matches.txt"
  echo "$F $(stat -f%z "$TMP/$F") bytes, cumulative matches $(wc -l < "$OUT/bulk_device_matches.txt")" >> "$OUT/bulk_scan.log"
  rm -f "$TMP/$F"
done
rmdir "$TMP" 2>/dev/null
echo done >> "$OUT/bulk_scan.log"
