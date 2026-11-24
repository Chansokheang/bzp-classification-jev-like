"""Train and evaluate the JEV-like purpose-code model.

Examples
  # time split (known companies), head only, then LoRA
  python train.py --run head_only  --lora-layers 0
  python train.py --run lora12     --lora-layers 12

  # new-company test: never train on H회사, test on all of its rows
  python train.py --run loco_H --holdout H회사 --lora-layers 12

  # quick check that everything runs (small subset)
  python train.py --run smoke --max-train 512 --max-eval 256 --epochs 1

See README.md for details.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.data import Tokens, eval_jobs, load_data, resolve_company, segment, select, train_candidates
from core.model import JEVModel, pack


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/jev")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--run", required=True, help="run name (output folder)")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--holdout", default=None, help="company name/id to exclude from training and test on")
    # architecture
    ap.add_argument("--keep-layers", type=int, default=20)
    ap.add_argument("--lora-layers", type=int, default=12, help="LoRA on the last N kept layers; 0 = head only")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32)
    ap.add_argument("--head-hidden", type=int, default=256)
    ap.add_argument("--max-state-tokens", type=int, default=64)
    ap.add_argument("--max-cand-tokens", type=int, default=24)
    # training
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=16, help="transactions per micro-batch")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--k-train", type=int, default=32, help="candidates per training example")
    ap.add_argument("--n-hard", type=int, default=16, help="confusable negatives among the k-train")
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--grad-checkpointing", action="store_true")
    # evaluation
    ap.add_argument("--eval-every", type=int, default=500, help="optimizer steps between val checks")
    ap.add_argument("--val-sample", type=int, default=3000, help="val rows used during training")
    ap.add_argument("--eval-chunk", type=int, default=64, help="candidates per sequence at eval")
    ap.add_argument("--eval-batch", type=int, default=32)
    # debugging
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, examples, companies, tokens, args, device, dtype, split_name, holdout):
    """Score every active code of each example's company (in chunks) and compute top-k."""
    model.eval()
    jobs = eval_jobs(examples, companies, args.eval_chunk)
    scores = [dict() for _ in examples]
    pad = tokens.tok.pad_token_id
    for s in range(0, len(jobs), args.eval_batch):
        batch = jobs[s: s + args.eval_batch]
        states = [tokens.state(examples[i]) for i, _ in batch]
        cands = [[tokens.cand[examples[i].company_id][c] for c in codes] for i, codes in batch]
        ids, pos, mask, cidx, n = pack(states, cands, pad, dtype)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            out = model(ids.to(device), pos.to(device), mask.to(device), cidx.to(device), int(n.max()))
        out = out.float().cpu().numpy()
        for row, (i, codes) in enumerate(batch):
            scores[i].update(zip(codes, out[row, : len(codes)]))

    rows, preds = [], []
    for e, sc in zip(examples, scores):
        codes = list(sc)
        v = np.array([sc[c] for c in codes])
        p = np.exp(v - v.max()); p /= p.sum()
        order = np.argsort(-p)
        top = [codes[j] for j in order[:3]]
        rank = top.index(e.label_code) if e.label_code in top else 99
        rows.append({"company": e.company, "segment": segment(e, split_name, holdout),
                     "top1": rank == 0, "top3": rank < 3, "nll": -math.log(max(p[codes.index(e.label_code)], 1e-12)),
                     "conf": float(p[order[0]])})
        preds.append({"id": e.id, "company": e.company, "label": e.label_code,
                      "top3": top, "probs": [round(float(p[j]), 4) for j in order[:3]]})
    model.train()
    return rows, preds


def summarize(rows):
    def agg(rs):
        return {"n": len(rs), "top1": float(np.mean([r["top1"] for r in rs])),
                "top3": float(np.mean([r["top3"] for r in rs])), "nll": float(np.mean([r["nll"] for r in rs]))}
    out = {"all": agg(rows), "by_company": {}, "by_segment": {}}
    for key, dst in [("company", "by_company"), ("segment", "by_segment")]:
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        out[dst] = {k: agg(v) for k, v in sorted(groups.items())}
    out["company_avg_top3"] = float(np.mean([v["top3"] for v in out["by_company"].values()]))
    # calibration: accuracy of top-1 when confidence >= threshold
    out["coverage"] = {}
    for t in (0.5, 0.7, 0.9):
        sel = [r for r in rows if r["conf"] >= t]
        out["coverage"][f"conf>={t}"] = {"share": len(sel) / max(1, len(rows)),
                                         "top1": float(np.mean([r["top1"] for r in sel])) if sel else None}
    return out


def report_md(name, s):
    L = [f"### {name}", "", f"All: n={s['all']['n']:,}  top-1 {s['all']['top1']:.1%}  top-3 {s['all']['top3']:.1%}  "
         f"(company avg top-3 {s['company_avg_top3']:.1%})", "", "| Group | n | Top-1 | Top-3 |", "|---|---|---|---|"]
    for title in ("by_company", "by_segment"):
        for k, v in s[title].items():
            L.append(f"| {k} | {v['n']:,} | {v['top1']:.1%} | {v['top3']:.1%} |")
    L += ["", "| Confidence | Share of rows | Top-1 among them |", "|---|---|---|"]
    for k, v in s["coverage"].items():
        acc = "-" if v["top1"] is None else f"{v['top1']:.1%}"
        L.append(f"| {k} | {v['share']:.1%} | {acc} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out = Path(args.out) / args.run
    (out / "best").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1, ensure_ascii=False), encoding="utf-8")

    companies, examples = load_data(args.data)
    holdout_id = resolve_company(companies, args.holdout) if args.holdout else None
    train, val, test = select(examples, holdout_id)
    rng = random.Random(args.seed)
    if args.max_train:
        train = rng.sample(train, min(args.max_train, len(train)))
    if args.max_eval:
        val, test = rng.sample(val, min(args.max_eval, len(val))), rng.sample(test, min(args.max_eval, len(test)))
    val_small = rng.sample(val, min(args.val_sample, len(val)))
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,} | holdout {args.holdout}")

    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tokens = Tokens(tok, companies, args.max_state_tokens, args.max_cand_tokens)

    model = JEVModel.from_pretrained(args.model, dtype=dtype, keep_layers=args.keep_layers,
                                     lora_layers=args.lora_layers, lora_r=args.lora_r,
                                     lora_alpha=args.lora_alpha, head_hidden=args.head_hidden).to(device)
    if args.grad_checkpointing and args.lora_layers > 0:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"params: {n_all/1e6:.0f}M total, {n_train/1e6:.2f}M trainable, LoRA modules {model.n_lora}")

    head_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    lora_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    groups = [{"params": head_p, "lr": args.lr_head}]
    if lora_p:
        groups.append({"params": lora_p, "lr": args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train) / (args.batch_size * args.grad_accum))
    total = max(1, int(steps_per_epoch * args.epochs))
    sched = get_cosine_schedule_with_warmup(opt, int(total * args.warmup), total)

    log = open(out / "metrics.jsonl", "a", encoding="utf-8")
    best, step, t0 = -1.0, 0, time.time()
    model.train()
    pad = tok.pad_token_id
    done, micro = False, 0
    while not done:
        order = rng.sample(train, len(train))
        for s in range(0, len(order), args.batch_size):
            micro += 1
            batch = order[s: s + args.batch_size]
            states, cands, labels = [], [], []
            for e in batch:
                codes, y = train_candidates(e, companies[e.company_id]["codes"], args.k_train, args.n_hard, rng)
                states.append(tokens.state(e))
                cands.append([tokens.cand[e.company_id][c] for c in codes])
                labels.append(y)
            ids, pos, mask, cidx, n = pack(states, cands, pad, dtype)
            k_max = int(n.max())
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                logits = model(ids.to(device), pos.to(device), mask.to(device), cidx.to(device), k_max)
            valid = torch.arange(k_max)[None, :] < n[:, None]
            logits = logits.float().masked_fill(~valid.to(device), float("-inf"))
            loss = F.cross_entropy(logits, torch.tensor(labels, device=device)) / args.grad_accum
            loss.backward()
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                print(f"step {step}/{total} loss {loss.item()*args.grad_accum:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
            if step % args.eval_every == 0 or step == total:
                rows, _ = evaluate(model, val_small, companies, tokens, args, device, dtype, "val", holdout_id is not None)
                sm = summarize(rows)
                rec = {"step": step, "val_top1": sm["all"]["top1"], "val_top3": sm["all"]["top3"],
                       "val_nll": sm["all"]["nll"], "time_s": round(time.time() - t0)}
                print(rec, flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
                if sm["all"]["top3"] + 1e-3 * sm["all"]["top1"] > best:
                    best = sm["all"]["top3"] + 1e-3 * sm["all"]["top1"]
                    torch.save(model.trainable_state_dict(), out / "best" / "trainable.pt")
            if step >= total:
                done = True
                break

    # final evaluation with the best checkpoint
    model.load_trainable_state_dict(torch.load(out / "best" / "trainable.pt", map_location="cpu"))
    model.to(device)
    results, md = {}, [f"# Run {args.run}\n", f"Holdout: {args.holdout or 'none (time split)'}\n"]
    for name, data in [("val", val), ("test", test)]:
        if not data:
            continue
        rows, preds = evaluate(model, data, companies, tokens, args, device, dtype,
                               "val" if name == "val" else "test", holdout_id is not None)
        results[name] = summarize(rows)
        md.append(report_md(name, results[name]))
        with open(out / f"predictions_{name}.jsonl", "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
