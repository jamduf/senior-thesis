#!/usr/bin/env bash
set -euo pipefail

LASSO_JSON="$1"
SOURCE_MEASUREMENTS="$2"
TARGET_MEASUREMENTS="$3"
CHECKPOINT="$4"
OUT_DIR="$5"

mkdir -p "$OUT_DIR"

PANEL_MAP="$OUT_DIR/panel_map_from_labels.json"

echo "Creating panel map from labels..."

python - "$LASSO_JSON" "$PANEL_MAP" <<'PY'
import json
import sys

infile = sys.argv[1]
outfile = sys.argv[2]

with open(infile, "r", encoding="utf-8") as f:
    data = json.load(f)

panel_map = {}

for p in data["panels"]:
    pid = str(p["id"])
    name = p.get("name")

    if not name or name.startswith("panel_"):
        raise ValueError(f"Panel {pid} missing semantic label")

    panel_map[pid] = name

with open(outfile, "w", encoding="utf-8") as f:
    json.dump(panel_map, f, indent=2)

print(json.dumps(panel_map, indent=2))
print("Wrote", outfile)
PY

echo "Running lasso_to_model_input.py..."

python ./lasso_to_model_input.py \
  --lasso_json "$LASSO_JSON" \
  --panel_map "$PANEL_MAP" \
  --source_measurements "$SOURCE_MEASUREMENTS" \
  --target_measurements "$TARGET_MEASUREMENTS" \
  --checkpoint "$CHECKPOINT" \
  --out_dir "$OUT_DIR"

echo "Done. Outputs written to: $OUT_DIR"
