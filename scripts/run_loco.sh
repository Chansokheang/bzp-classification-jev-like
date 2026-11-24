#!/usr/bin/env bash
# Leave-one-company-out: train 8 models, each never seeing one company, and test on that company.
# Usage: bash scripts/run_loco.sh [extra train.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."   # run from the project root
for co in A회사 B회사 C회사 D회사 E회사 F회사 G회사 H회사; do
  python train.py --run "loco_${co}" --holdout "${co}" "$@"
done
python - <<'EOF'
import json, glob
rows = []
for f in sorted(glob.glob("runs/loco_*/results.json")):
    t = json.load(open(f, encoding="utf-8"))["test"]["all"]
    rows.append((f.split("/")[-2], t["n"], t["top1"], t["top3"]))
print("| Held-out company | n | Top-1 | Top-3 |\n|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]:,} | {r[2]:.1%} | {r[3]:.1%} |")
if rows:
    print(f"| **Average** | | {sum(r[2] for r in rows)/len(rows):.1%} | {sum(r[3] for r in rows)/len(rows):.1%} |")
EOF
