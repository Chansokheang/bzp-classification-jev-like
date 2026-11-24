"""Phase 2 baselines for purpose-code recommendation (top-1 / top-3).

Inputs: data/prepared/{transactions,codebook}.parquet from prep/prepare_data.py.
Features used (by design): merchant name, industry code/name, business registration number.

Baselines
  1. lookup   (known company, time split)
     Rank the company's active codes by: P(code | merchant) at this company, then
     P(code | industry name), then P(code | industry code), then the company prior.
     Val uses train history; test uses train+val history.
  2. transfer (new company, leave-one-company-out)
     The held-out company has no history. Take what *other* companies booked at the same
     merchant / industry (as purpose-name text), then map those names onto the held-out
     company's candidate names by character n-gram similarity.
  3. textsim  (new company, floor)
     Character n-gram similarity between merchant+industry text and candidate names only.

Also reports shortlist recall@K: how often the correct code is inside the top K
candidates. Phase 3 scores only a shortlist for long code lists, so this sets its ceiling.

Usage:  python prep/baselines.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "prepared"
OUT = ROOT / "results" / "baselines"

KS = (1, 3, 10, 20, 30)
W = dict(merchant=1e6, ind_name=1e3, ind_code=1e1, prior=1e-2, text=1e-4)  # lexicographic back-off


# ---------------------------------------------------------------- helpers
def load():
    tx = pd.read_parquet(DATA / "transactions.parquet")
    cb = pd.read_parquet(DATA / "codebook.parquet")
    tx = tx[tx["usable"]].copy()
    tx["ind_code_key"] = tx["industry_code"].where(~tx["industry_placeholder"])
    tx["text"] = (tx["merchant_name"].fillna("") + " " + tx["industry_name"].fillna("")).str.strip()
    cands = {c: g.loc[g["active"], ["code", "name"]].reset_index(drop=True)
             for c, g in cb.groupby("company_id")}
    return tx, cands


def cond_probs(hist: pd.DataFrame, key: str, target: str, columns) -> pd.DataFrame:
    """Row-normalized P(target | key) table with fixed columns."""
    h = hist.dropna(subset=[key])
    if h.empty:
        return pd.DataFrame(columns=columns, dtype=float)
    t = pd.crosstab(h[key], h[target]).reindex(columns=columns, fill_value=0).astype(float)
    s = t.sum(axis=1).replace(0, 1)
    return t.div(s, axis=0)


def lookup_rows(table: pd.DataFrame, keys: pd.Series) -> np.ndarray:
    return table.reindex(keys.to_numpy()).fillna(0.0).to_numpy()


def rank_metrics(scores: np.ndarray, labels_idx: np.ndarray) -> pd.DataFrame:
    """Rank of the true label (0 = first) with stable tie-breaking; -> hit@K columns."""
    order = np.argsort(-scores, axis=1, kind="stable")
    rank = np.argmax(order == labels_idx[:, None], axis=1)
    return pd.DataFrame({f"hit@{k}": rank < k for k in KS})


_PREFIX = re.compile(r"^(\((제조|일반|판관|판|제)\)|(제조|판|제)\))\s*")


def norm_name(s: str) -> str:
    """Drop accounting-section prefixes like '(제조)', '판)' so names compare by meaning."""
    return _PREFIX.sub("", str(s)).strip()


# ---------------------------------------------------------------- scorers (reused by build_jev_dataset.py)
def lookup_scores(hist: pd.DataFrame, ev: pd.DataFrame, codes: list) -> np.ndarray:
    """Known-company scores (n_ev x n_codes) from this company's own history."""
    S = (W["merchant"] * lookup_rows(cond_probs(hist, "merchant_key", "label_code", codes), ev["merchant_key"])
         + W["ind_name"] * lookup_rows(cond_probs(hist, "industry_name", "label_code", codes), ev["industry_name"])
         + W["ind_code"] * lookup_rows(cond_probs(hist, "ind_code_key", "label_code", codes), ev["ind_code_key"]))
    prior = hist["label_code"].value_counts(normalize=True).reindex(codes, fill_value=0).to_numpy()
    return S + W["prior"] * prior[None, :]


def fit_vectorizer(tx: pd.DataFrame, cands: dict) -> TfidfVectorizer:
    names = sorted({norm_name(n) for c in cands.values() for n in c["name"]} |
                   {norm_name(n) for n in tx["label_name"].unique()})
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True)
    return vec.fit(names + tx["text"].drop_duplicates().tolist())


def transfer_scores(others: pd.DataFrame, ev: pd.DataFrame, cand: pd.DataFrame, vec) -> tuple[np.ndarray, np.ndarray]:
    """New-company scores from *other* companies' history. Returns (transfer, textsim)."""
    others = others.assign(fname=others["label_name"].map(norm_name))
    fnames = sorted(others["fname"].unique())
    cvec = vec.transform(cand["name"].map(norm_name))
    M = (vec.transform(fnames) @ cvec.T).toarray() ** 3   # foreign name -> candidate, sharpened
    E = (W["merchant"] * lookup_rows(cond_probs(others, "merchant_key", "fname", fnames), ev["merchant_key"])
         + W["ind_name"] * lookup_rows(cond_probs(others, "industry_name", "fname", fnames), ev["industry_name"])
         + W["ind_code"] * lookup_rows(cond_probs(others, "ind_code_key", "fname", fnames), ev["ind_code_key"]))
    T = (vec.transform(ev["text"]) @ cvec.T).toarray()
    return E @ M + W["text"] * T, T


# ---------------------------------------------------------------- baseline 1: lookup
def run_lookup(tx: pd.DataFrame, cands: dict) -> pd.DataFrame:
    out = []
    for cid, t in tx.groupby("company_id"):
        codes = cands[cid]["code"].tolist()
        idx = {c: i for i, c in enumerate(codes)}
        for split, hist_splits in [("val", ["train"]), ("test", ["train", "val"])]:
            hist = t[t["time_split"].isin(hist_splits)]
            ev = t[t["time_split"] == split]
            if ev.empty:
                continue
            m = rank_metrics(lookup_scores(hist, ev, codes), ev["label_code"].map(idx).to_numpy())
            m["company"] = ev["company"].to_numpy()
            m["split"] = split
            m["segment"] = np.where(ev["merchant_key"].isin(set(hist["merchant_key"])).to_numpy(),
                                    "known merchant", "new merchant")
            m["n_cands"] = len(codes)
            out.append(m)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- baselines 2 & 3: new company
def run_new_company(tx: pd.DataFrame, cands: dict) -> pd.DataFrame:
    vec = fit_vectorizer(tx, cands)
    out = []
    for cid, ev in tx.groupby("company_id"):
        cand = cands[cid]
        S_tr, S_txt = transfer_scores(tx[tx["company_id"] != cid], ev, cand, vec)
        idx = {c: i for i, c in enumerate(cand["code"])}
        y = ev["label_code"].map(idx).to_numpy()

        for name, S in [("transfer", S_tr), ("textsim", S_txt)]:
            m = rank_metrics(S, y)
            m["company"] = ev["company"].to_numpy()
            m["split"] = "all"
            m["segment"] = np.where(ev["merchant_seen_other_company"].to_numpy(),
                                    "merchant seen elsewhere", "merchant unseen anywhere")
            m["n_cands"] = len(cand)
            m["baseline"] = name
            out.append(m)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- reporting
def summarize(m: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = m.groupby(by)
    s = g[[f"hit@{k}" for k in KS]].mean()
    s.insert(0, "n", g.size())
    return s


def fmt(df: pd.DataFrame, cols=("hit@1", "hit@3")) -> str:
    d = df.reset_index()
    head = "| " + " | ".join(d.columns[: len(d.columns) - df.shape[1]].tolist() + ["n"] + [c.replace("hit@", "top-") for c in cols]) + " |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for _, r in d.iterrows():
        keys = [str(r[c]) for c in d.columns[: len(d.columns) - df.shape[1]]]
        lines.append("| " + " | ".join(keys + [f"{int(r['n']):,}"] + [f"{r[c]:.1%}" for c in cols]) + " |")
    return "\n".join(lines)


def overall(m: pd.DataFrame, label: str) -> dict:
    per_co = m.groupby("company")[["hit@1", "hit@3"]].mean()
    return {"setting": label, "n": len(m),
            "top-1 (all rows)": m["hit@1"].mean(), "top-3 (all rows)": m["hit@3"].mean(),
            "top-3 (company avg)": per_co["hit@3"].mean()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tx, cands = load()
    print(f"{len(tx):,} usable rows, {tx['company_id'].nunique()} companies")

    lk = run_lookup(tx, cands)
    nc = run_new_company(tx, cands)
    lk.to_parquet(OUT / "lookup_rows.parquet", index=False)
    nc.to_parquet(OUT / "new_company_rows.parquet", index=False)

    test = lk[lk["split"] == "test"]
    tr = nc[nc["baseline"] == "transfer"]
    ts = nc[nc["baseline"] == "textsim"]

    L = ["# Phase 2 baseline report\n",
         "Inputs: merchant name, industry code/name, business registration number only. "
         "Candidates: each company's active purpose codes. Generated by `prep/baselines.py`.\n",
         "## Summary\n",
         "| Setting | Baseline | Rows | Top-1 | Top-3 | Top-3 (avg of companies) |",
         "|---|---|---|---|---|---|"]
    for lab, b, m in [("Known company (Jan test)", "lookup", test),
                      ("Known company, new merchant", "lookup", test[test["segment"] == "new merchant"]),
                      ("New company (leave-one-out)", "transfer", tr),
                      ("New company (leave-one-out)", "textsim", ts)]:
        o = overall(m, lab)
        L.append(f"| {lab} | {b} | {o['n']:,} | {o['top-1 (all rows)']:.1%} | {o['top-3 (all rows)']:.1%} | {o['top-3 (company avg)']:.1%} |")

    L += ["\n## 1. Lookup — known company, time split (test = January)\n",
          fmt(summarize(test, ["company", "segment"])),
          "\nValidation (Dec 16–31, history = before Dec 16):\n",
          fmt(summarize(lk[lk["split"] == "val"], ["segment"])),
          "\n## 2. Transfer — new company (leave-one-company-out, all rows of the held-out company)\n",
          fmt(summarize(tr, ["company"])),
          "\nBy whether another company has booked this merchant:\n",
          fmt(summarize(tr, ["segment"])),
          "\n## 3. Text similarity only — new company floor\n",
          fmt(summarize(ts, ["company"])),
          "\n## 4. Shortlist recall@K (is the right code inside the top K?)\n",
          "Phase 3 scores a shortlist of candidates; recall@K is the most it can reach.\n",
          "| Company | Active codes | Lookup R@10 | R@20 | R@30 | Transfer R@10 | R@20 | R@30 |",
          "|---|---|---|---|---|---|---|---|"]
    a = summarize(test, ["company"])
    b = summarize(tr, ["company"])
    ncand = test.groupby("company")["n_cands"].first()
    for co in a.index:
        L.append(f"| {co} | {ncand[co]} | " + " | ".join(f"{a.loc[co, f'hit@{k}']:.1%}" for k in (10, 20, 30)) + " | "
                 + " | ".join(f"{b.loc[co, f'hit@{k}']:.1%}" for k in (10, 20, 30)) + " |")
    L.append("\nCompanies with ≤ K active codes are trivially 100% at that K.")

    (OUT / "baseline_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
