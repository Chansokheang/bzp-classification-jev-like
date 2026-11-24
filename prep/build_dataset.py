"""Export the training data for the JEV-like model (run locally, then copy to the GPU server).

Reads data/prepared/*.parquet and writes data/jev/:
    companies.json     {company_id: {"name": ..., "codes": [...], "names": [...]}}  active codes only
    examples.jsonl     one usable transaction per line:
        id, company_id, company, split (train/val/test), state, label_code,
        merchant_seen_train   - merchant booked at this company before Dec 16 (train history)
        merchant_seen_trainval- merchant booked at this company before Jan 1 (train+val history)
        merchant_seen_other   - merchant booked at any other company
        hard_negs             - up to 63 confusable wrong codes for training (from the lookup
                                baseline using train-split history of the same company)
    meta.json          build settings and counts

Only merchant/industry text, purpose names and codes are exported: no user ids, amounts or pks.

Usage:  python prep/build_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import load, lookup_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "jev"
N_HARD = 63


def state_text(r) -> str:
    ind = r.industry_name if isinstance(r.industry_name, str) else "정보없음"
    code = r.industry_code if isinstance(r.industry_code, str) and not r.industry_placeholder else "-"
    merchant = r.merchant_name if isinstance(r.merchant_name, str) else "정보없음"
    return f"가맹점명: {merchant}\n업종: {ind} ({code})"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tx, cands = load()

    companies = {}
    for cid, c in cands.items():
        name = tx.loc[tx["company_id"] == cid, "company"]
        companies[cid] = {"name": name.iloc[0] if len(name) else cid,
                          "codes": c["code"].tolist(), "names": c["name"].tolist()}

    tx = tx.sort_values(["company_id", "approved_at"]).reset_index(drop=True)
    tx["hard_negs"] = None
    tx["merchant_seen_train"] = False
    tx["merchant_seen_trainval"] = False
    for cid, t in tx.groupby("company_id"):
        codes = companies[cid]["codes"]
        hist = t[t["time_split"] == "train"]
        tx.loc[t.index, "merchant_seen_train"] = t["merchant_key"].isin(set(hist["merchant_key"])).to_numpy()
        tv = t[t["time_split"].isin(["train", "val"])]
        tx.loc[t.index, "merchant_seen_trainval"] = t["merchant_key"].isin(set(tv["merchant_key"])).to_numpy()

        S = lookup_scores(hist, t, codes)
        order = np.argsort(-S, axis=1, kind="stable")
        codes_arr = np.array(codes)
        labels = t["label_code"].to_numpy()
        negs = []
        for i in range(len(t)):
            ranked = codes_arr[order[i]]
            negs.append([c for c in ranked[: N_HARD + 1] if c != labels[i]][:N_HARD])
        tx.loc[t.index, "hard_negs"] = pd.Series(negs, index=t.index)

    n = 0
    with open(OUT / "examples.jsonl", "w", encoding="utf-8") as f:
        for r in tx.itertuples(index=False):
            f.write(json.dumps({
                "id": n, "company_id": r.company_id, "company": r.company, "split": r.time_split,
                "state": state_text(r), "label_code": r.label_code,
                "merchant_seen_train": bool(r.merchant_seen_train),
                "merchant_seen_trainval": bool(r.merchant_seen_trainval),
                "merchant_seen_other": bool(r.merchant_seen_other_company),
                "hard_negs": list(r.hard_negs),
            }, ensure_ascii=False) + "\n")
            n += 1

    (OUT / "companies.json").write_text(json.dumps(companies, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {"n_examples": n, "splits": tx["time_split"].value_counts().to_dict(),
            "companies": {companies[c]["name"]: len(companies[c]["codes"]) for c in companies},
            "n_hard_negs": N_HARD, "state_fields": ["merchant_name", "industry_name", "industry_code"]}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
