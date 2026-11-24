"""Phase 1 data prep for the expense purpose-code classifier.

Reads every company workbook in the raw folder (sheet 1 = purpose codebook,
sheet 2 = card transactions, read by position, so sheet names don't matter),
cleans them into one table, adds evaluation splits, and writes:

    <out>/transactions.parquet   one row per card transaction
    <out>/codebook.parquet       one row per (company, purpose code)
    <out>/companies.parquet      one row per company
    <out>/data_report.md         summary for humans

New company files dropped into the raw folder are picked up automatically.

Usage:
    python prep/prepare_data.py
    python prep/prepare_data.py --raw data/raw --out data/prepared
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORG_INFO_PREFIX = "["  # files like "[공통]이용기관정보.xlsx" are shared info, not companies

# Transaction sheet columns by position (A회사 has "HOI" instead of 이용기관id in column 0).
TX_COLS = {
    0: "company_id", 1: "pk", 2: "date", 3: "time", 4: "merchant_name",
    5: "industry_code", 6: "industry_name", 7: "biz_no", 8: "supply_amount",
    9: "vat_amount", 10: "tax_type", 11: "label_code", 12: "label_name",
}
# Columns after these are matched by header name because B회사 has an extra column.
TX_NAMED = {"입력항목명1": "extra_input", "사용자id": "user_id", "작성자id": "writer_id"}

CODEBOOK_COLS = {0: "company_id", 1: "code", 2: "name", 3: "active"}

FREE_INPUT_PAT = re.compile(r"직접\s*입력")
PLACEHOLDER_INDUSTRY = {"AAAA"}


def clean_text(s: pd.Series) -> pd.Series:
    """Strip, remove space padding inside names ('택   시' -> '택시'), blanks -> NA."""
    s = s.astype("string").str.strip()
    s = s.str.replace(r"\s{2,}", "", regex=True)
    return s.mask(s == "")


def read_company(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    names = list(sheets)
    if len(names) < 2:
        raise ValueError(f"{path.name}: expected 2 sheets (codebook, transactions), got {names}")
    cb_raw, tx_raw = sheets[names[0]], sheets[names[1]]

    cb = cb_raw.iloc[:, list(CODEBOOK_COLS)].copy()
    cb.columns = list(CODEBOOK_COLS.values())
    for c in cb.columns:
        cb[c] = clean_text(cb[c])
    cb["active"] = cb["active"].str.upper().eq("Y")
    cb = cb.dropna(subset=["code"])

    tx = tx_raw.iloc[:, list(TX_COLS)].copy()
    tx.columns = list(TX_COLS.values())
    for src, dst in TX_NAMED.items():
        tx[dst] = tx_raw[src] if src in tx_raw.columns else pd.NA
    for c in tx.columns:
        tx[c] = clean_text(tx[c])

    tx["source_file"] = path.name
    cb["source_file"] = path.name
    return cb, tx


def normalize(tx: pd.DataFrame, cb: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["time"] = tx["time"].str.zfill(6)
    tx["approved_at"] = pd.to_datetime(tx["date"] + tx["time"], format="%Y%m%d%H%M%S", errors="coerce")
    for c in ["supply_amount", "vat_amount"]:
        tx[c] = pd.to_numeric(tx[c], errors="coerce")
    tx["total_amount"] = tx["supply_amount"].fillna(0) + tx["vat_amount"].fillna(0)

    tx["biz_no"] = tx["biz_no"].str.replace(r"\D", "", regex=True)
    tx["biz_no"] = tx["biz_no"].mask(tx["biz_no"] == "")
    tx["industry_placeholder"] = tx["industry_code"].isin(PLACEHOLDER_INDUSTRY)

    # Raw tax type mixes names ("일반") with numeric codes ("1", "4") in C/F/H; meaning of the
    # numeric codes is unconfirmed, so keep them tagged instead of guessing.
    tt = tx["tax_type"]
    tx["tax_type_norm"] = tt.where(~tt.str.fullmatch(r"\d+", na=False), "코드" + tt)

    # Merchant key: business registration number, else merchant name.
    tx["merchant_key"] = ("B:" + tx["biz_no"]).fillna("N:" + tx["merchant_name"].fillna("?"))

    # Label checks against the company's own codebook.
    book = cb.set_index(["company_id", "code"])
    key = pd.MultiIndex.from_frame(tx[["company_id", "label_code"]])
    tx["label_in_codebook"] = key.isin(book.index)
    tx["label_active"] = tx["label_in_codebook"] & pd.Series(
        book["active"].reindex(key).fillna(False).to_numpy(), index=tx.index
    ).astype(bool)
    # Use the codebook name (whitespace-cleaned) as the canonical label text.
    tx["label_name"] = pd.Series(book["name"].reindex(key).to_numpy(), index=tx.index).fillna(tx["label_name"])

    tx["is_labeled"] = tx["label_code"].notna()
    tx["is_free_input"] = tx["label_name"].str.contains(FREE_INPUT_PAT, na=False)
    tx["usable"] = tx["is_labeled"] & ~tx["is_free_input"] & tx["label_active"] & tx["approved_at"].notna()
    return tx


def add_splits(tx: pd.DataFrame, val_start: str, test_start: str) -> pd.DataFrame:
    """Time split within each company + merchant-novelty flags for both evaluation setups.

    time_split: train < val_start <= val < test_start <= test
    Leave-one-company-out needs no column: hold out by company_id.
    """
    tx = tx.copy()
    d = tx["approved_at"]
    tx["time_split"] = np.select(
        [d < pd.Timestamp(val_start), d < pd.Timestamp(test_start)], ["train", "val"], "test"
    )
    tx.loc[d.isna(), "time_split"] = pd.NA

    use = tx[tx["usable"]]
    train_keys = set(zip(use.loc[use["time_split"] == "train", "company_id"],
                         use.loc[use["time_split"] == "train", "merchant_key"]))
    tx["merchant_seen_in_train"] = [
        (c, m) in train_keys for c, m in zip(tx["company_id"], tx["merchant_key"])
    ]

    # For leave-one-company-out: is this merchant known from any *other* company?
    owners = use.groupby("merchant_key")["company_id"].agg(set)
    tx["merchant_seen_other_company"] = [
        bool(owners.get(m, set()) - {c}) for c, m in zip(tx["company_id"], tx["merchant_key"])
    ]
    return tx


def load_org_info(raw: Path) -> pd.DataFrame:
    files = [p for p in raw.glob("*.xlsx") if p.name.startswith(ORG_INFO_PREFIX)]
    if not files:
        return pd.DataFrame(columns=["company_id", "company_name", "industry", "category", "members"])
    df = pd.read_excel(files[0], dtype=str)
    df = df.iloc[:, :5]
    df.columns = ["company_name", "company_id", "industry", "category", "members"]
    for c in df.columns:
        df[c] = clean_text(df[c])
    df["members"] = pd.to_numeric(df["members"], errors="coerce")
    return df


def build(raw: Path, val_start: str, test_start: str):
    files = sorted(p for p in raw.glob("*.xlsx")
                   if not p.name.startswith(ORG_INFO_PREFIX) and not p.name.startswith("~$"))
    if not files:
        raise SystemExit(f"No company workbooks found in {raw}")
    cbs, txs = [], []
    for p in files:
        cb, tx = read_company(p)
        ids = tx["company_id"].dropna().unique()
        if len(ids) != 1:
            raise ValueError(f"{p.name}: expected one company id, found {list(ids)}")
        cbs.append(cb)
        txs.append(tx)
        print(f"  read {p.name}: {len(cb)} codes, {len(tx)} transactions ({ids[0]})")
    cb = pd.concat(cbs, ignore_index=True)
    tx = pd.concat(txs, ignore_index=True)

    dup = tx["pk"].duplicated().sum()
    if dup:
        print(f"  warning: dropping {dup} duplicate pk rows")
        tx = tx.drop_duplicates("pk")

    tx = normalize(tx, cb)
    tx = add_splits(tx, val_start, test_start)

    org = load_org_info(raw)
    comp = (tx.groupby(["company_id", "source_file"]).size().rename("n_tx").reset_index()
            .merge(org, on="company_id", how="left"))
    comp["company_name"] = comp["company_name"].fillna(comp["source_file"].str.replace(".xlsx", "", regex=False))
    name_map = comp.set_index("company_id")["company_name"]
    tx.insert(0, "company", tx["company_id"].map(name_map))
    cb.insert(0, "company", cb["company_id"].map(name_map))
    return tx, cb, comp


def pct(x: float) -> str:
    return f"{x:.0%}" if pd.notna(x) else "-"


def write_report(tx: pd.DataFrame, cb: pd.DataFrame, comp: pd.DataFrame, out: Path,
                 val_start: str, test_start: str) -> None:
    L = []
    L.append("# Phase 1 data report\n")
    L.append(f"Generated by `prep/prepare_data.py` from {comp['source_file'].nunique()} company workbooks. "
             f"{len(tx):,} transactions, {len(cb):,} purpose codes.\n")

    L.append("## Companies\n")
    L.append("| Company | Category | Industry | Members | Transactions | Usable | Codes (active) | Codes used | Date range |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, c in comp.sort_values("company_name").iterrows():
        t = tx[tx["company_id"] == c["company_id"]]
        b = cb[cb["company_id"] == c["company_id"]]
        u = t[t["usable"]]
        cat = "-" if pd.isna(c.get("category")) else c["category"]
        ind = "-" if pd.isna(c.get("industry")) else c["industry"]
        L.append(f"| {c['company_name']} | {cat} | {ind} | "
                 f"{'' if pd.isna(c.get('members')) else int(c['members'])} | {len(t):,} | {len(u):,} ({pct(len(u)/len(t))}) | "
                 f"{len(b)} ({int(b['active'].sum())}) | {u['label_code'].nunique()} | "
                 f"{t['approved_at'].min():%Y-%m-%d} ~ {t['approved_at'].max():%Y-%m-%d} |")

    L.append("\n## Why rows are excluded (`usable = False`)\n")
    L.append("| Company | Unlabeled | Free-input label (직접입력) | Label inactive / not in codebook | Bad date |")
    L.append("|---|---|---|---|---|")
    for name, t in tx.groupby("company"):
        lab = t["is_labeled"]
        L.append(f"| {name} | {(~lab).sum():,} | {t['is_free_input'].sum():,} | "
                 f"{(lab & ~t['label_active']).sum():,} | {t['approved_at'].isna().sum():,} |")

    L.append("\n## Input field quality\n")
    L.append("| Company | Merchant name missing | Industry code missing | Industry code = AAAA | Industry name missing | Biz no. missing |")
    L.append("|---|---|---|---|---|---|")
    for name, t in tx.groupby("company"):
        L.append(f"| {name} | {pct(t['merchant_name'].isna().mean())} | {pct(t['industry_code'].isna().mean())} | "
                 f"{pct(t['industry_placeholder'].mean())} | {pct(t['industry_name'].isna().mean())} | "
                 f"{pct(t['biz_no'].isna().mean())} |")

    L.append("\n## Splits (usable rows only)\n")
    L.append(f"Time split per company: train < {val_start} ≤ val < {test_start} ≤ test. "
             "Leave-one-company-out: hold out each `company_id` in turn.\n")
    L.append("| Company | Train | Val | Test | Test: merchant seen in train | Merchant also seen at another company |")
    L.append("|---|---|---|---|---|---|")
    u = tx[tx["usable"]]
    for name, t in u.groupby("company"):
        s = t["time_split"].value_counts()
        te = t[t["time_split"] == "test"]
        L.append(f"| {name} | {s.get('train', 0):,} | {s.get('val', 0):,} | {s.get('test', 0):,} | "
                 f"{pct(te['merchant_seen_in_train'].mean()) if len(te) else '-'} | "
                 f"{pct(t['merchant_seen_other_company'].mean())} |")
    s = u["time_split"].value_counts()
    L.append(f"| **All** | **{s.get('train', 0):,}** | **{s.get('val', 0):,}** | **{s.get('test', 0):,}** | "
             f"**{pct(u.loc[u['time_split']=='test','merchant_seen_in_train'].mean())}** | "
             f"**{pct(u['merchant_seen_other_company'].mean())}** |")

    L.append("\n## Tax type values (`tax_type_norm`)\n")
    tt = tx.groupby("company")["tax_type_norm"].value_counts(dropna=False).unstack(fill_value=0)
    L.append("| Company | " + " | ".join(str(c) for c in tt.columns) + " |")
    L.append("|---|" + "---|" * len(tt.columns))
    for name, r in tt.iterrows():
        L.append(f"| {name} | " + " | ".join(f"{v:,}" for v in r) + " |")

    L.append("\n## Cleaning applied\n")
    L.append("- Sheets read by position (sheet 1 = codebook, sheet 2 = transactions); sheet names ignored.")
    L.append("- Whitespace stripped everywhere and space padding removed inside names (`택   시` → `택시`). "
             "This fixes G회사 labels that differed from the codebook only by trailing spaces.")
    L.append("- Business registration number kept as digits only; blanks → missing.")
    L.append("- `merchant_key` = business registration number, falling back to merchant name.")
    L.append("- `total_amount` = supply amount + VAT. `approved_at` built from date + zero-padded time.")
    L.append("- Numeric tax-type codes (`1`, `4`, `7`, …) kept as `코드1`, `코드4`, … — their meaning is unconfirmed.")
    L.append("- `usable` = labeled, not a free-input (직접입력) code, label active in the company codebook, valid date.")

    L.append("\n## Open questions\n")
    L.append("1. What do numeric tax-type codes `0, 1, 2, 4, 7` mean (C, F, H)?")
    L.append("2. B회사's 직접입력 code is excluded. Its column `입력항목명1` holds only the field name `내용`, not what "
             "the user typed. Is the typed text available elsewhere?")
    L.append("3. Are the unlabeled rows (largest at F) still pending, or intentionally left uncoded?")

    (out / "data_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "prepared")
    ap.add_argument("--val-start", default="2025-12-16")
    ap.add_argument("--test-start", default="2026-01-01")
    args = ap.parse_args()

    print(f"Reading {args.raw}")
    tx, cb, comp = build(args.raw, args.val_start, args.test_start)
    args.out.mkdir(parents=True, exist_ok=True)
    tx.to_parquet(args.out / "transactions.parquet", index=False)
    cb.to_parquet(args.out / "codebook.parquet", index=False)
    comp.to_parquet(args.out / "companies.parquet", index=False)
    write_report(tx, cb, comp, args.out, args.val_start, args.test_start)
    print(f"Wrote {len(tx):,} transactions, {len(cb):,} codes, {len(comp)} companies to {args.out}")
    print(f"Usable rows: {tx['usable'].sum():,}")


if __name__ == "__main__":
    main()
