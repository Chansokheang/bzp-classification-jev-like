"""Data loading, candidate sampling and batching for the JEV-like model."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

CAND_PREFIX = "\n용도: "


@dataclass
class Example:
    id: int
    company_id: str
    company: str
    split: str
    state: str
    label_code: str
    merchant_seen_train: bool
    merchant_seen_trainval: bool
    merchant_seen_other: bool
    hard_negs: list


def load_data(data_dir: str | Path):
    data_dir = Path(data_dir)
    companies = json.loads((data_dir / "companies.json").read_text(encoding="utf-8"))
    examples = []
    with open(data_dir / "examples.jsonl", encoding="utf-8") as f:
        for line in f:
            examples.append(Example(**json.loads(line)))
    return companies, examples


def resolve_company(companies: dict, key: str) -> str:
    for cid, c in companies.items():
        if key in (cid, c["name"]):
            return cid
    raise SystemExit(f"--holdout {key!r} not found; choose from {[c['name'] for c in companies.values()]}")


def select(examples, holdout_id: str | None):
    """Return (train, val, test) lists.

    holdout None : time split (train < Dec 16 <= val < Jan 1 <= test), all companies.
    holdout X    : company X is never trained on; train/val = other companies' train/val splits,
                   test = every row of company X (simulates a brand-new customer).
    """
    if holdout_id is None:
        return ([e for e in examples if e.split == "train"],
                [e for e in examples if e.split == "val"],
                [e for e in examples if e.split == "test"])
    return ([e for e in examples if e.split == "train" and e.company_id != holdout_id],
            [e for e in examples if e.split == "val" and e.company_id != holdout_id],
            [e for e in examples if e.company_id == holdout_id])


def segment(e: Example, split_name: str, holdout: bool) -> str:
    if holdout:
        return "merchant seen at other company" if e.merchant_seen_other else "merchant unseen anywhere"
    seen = e.merchant_seen_train if split_name == "val" else e.merchant_seen_trainval
    return "known merchant" if seen else "new merchant"


class Tokens:
    """Caches token ids for states and candidate names."""

    def __init__(self, tokenizer, companies: dict, max_state: int, max_cand: int):
        self.tok = tokenizer
        self.max_state = max_state
        self.cand = {}
        for cid, c in companies.items():
            enc = tokenizer([CAND_PREFIX + n for n in c["names"]], add_special_tokens=False)["input_ids"]
            self.cand[cid] = {code: ids[:max_cand] for code, ids in zip(c["codes"], enc)}
        self.state_cache: dict[int, list[int]] = {}

    def state(self, e: Example) -> list[int]:
        if e.id not in self.state_cache:
            ids = self.tok(e.state, add_special_tokens=False)["input_ids"]
            self.state_cache[e.id] = ids[: self.max_state]
        return self.state_cache[e.id]


def train_candidates(e: Example, codes: list[str], k: int, n_hard: int, rng: random.Random):
    """Label + up to n_hard confusable codes + random codes, shuffled. Returns (codes, label_idx)."""
    if len(codes) <= k:
        chosen = list(codes)
    else:
        chosen = [e.label_code] + [c for c in e.hard_negs[:n_hard] if c != e.label_code]
        rest = [c for c in codes if c not in set(chosen)]
        chosen += rng.sample(rest, k - len(chosen)) if len(chosen) < k else []
        chosen = chosen[:k]
    rng.shuffle(chosen)
    return chosen, chosen.index(e.label_code)


def eval_jobs(examples, companies: dict, chunk: int):
    """Split each example's full code list into chunks: [(example_pos, codes_chunk)]."""
    jobs = []
    for i, e in enumerate(examples):
        codes = companies[e.company_id]["codes"]
        for s in range(0, len(codes), chunk):
            jobs.append((i, codes[s: s + chunk]))
    return jobs
