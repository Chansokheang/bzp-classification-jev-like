# JEV-like purpose-code recommender

Recommends the top-3 **purpose codes (용도코드)** for a corporate card transaction from
merchant information (merchant name, industry code/name, business registration number).

The model is a JEV-like decision model: a truncated **Qwen3-0.6B** (Apache 2.0) reads the
transaction once together with the company's own purpose names, and a small head gives each
purpose name a score. Candidates cannot see each other, so scores don't depend on candidate
order, and any company's code list, including a new company's, can be scored without retraining.

---

## Pipeline

| Step | Where | Command | Output |
|---|---|---|---|
| 1. Clean raw workbooks | local | `python prep/prepare_data.py` | `data/prepared/` + `data_report.md` |
| 2. Baselines | local | `python prep/baselines.py` | `results/baselines/baseline_report.md` |
| 3. Export training data | local | `python prep/build_dataset.py` | `data/jev/` |
| 4. Train / evaluate | **GPU server** | `python train.py ...` | `runs/<name>/` |

Steps 1–3 are already done. `data/jev/` is ready to copy.
Local steps need `pip install -r requirements.txt`; the raw workbooks go in `data/raw/`.

---

## Moving to the GPU server

### 1. Copy these files

```
core/                   # model + data code
tests/
scripts/
train.py
requirements-train.txt
README.md
data/jev/               # companies.json, examples.jsonl (78 MB), meta.json
```

The `prep/` folder, the raw workbooks (`data/raw/`) and `data/prepared/` are **not** needed on the server.
`data/jev/` contains merchant names, industry names, purpose codes and names only: no user IDs,
amounts or transaction keys.

From Windows, for example:
```powershell
scp -r core tests scripts train.py requirements-train.txt README.md user@server:~/jev-like/
scp -r data/jev user@server:~/jev-like/data/
```

### 2. Set up the environment

```bash
cd ~/jev-like
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA version
pip install -r requirements-train.txt
python -m pytest tests/test_model.py -q        # must pass: checks the attention mask
```

The first `train.py` run downloads `Qwen/Qwen3-0.6B` (~1.2 GB) from Hugging Face.
If the server has no internet, download it elsewhere and pass `--model /path/to/Qwen3-0.6B`.

### 3. Smoke test (a few minutes)

```bash
python train.py --run smoke --max-train 1024 --max-eval 512 --val-sample 256 --epochs 1 --eval-every 20
```

Check that the loss goes down and `runs/smoke/report.md` is written.

---

## Experiments to run

Run them in this order. Each writes `runs/<name>/report.md`.

```bash
# A. Known companies (time split: train < Dec 16 <= val < Jan 1 <= test = January)
python train.py --run head_only --lora-layers 0          # backbone frozen, head only
python train.py --run lora12    --lora-layers 12         # + LoRA on layers 8–19

# B. New company (leave-one-company-out, 8 runs): use whichever of A won
bash scripts/run_loco.sh --lora-layers 12
```

### Results to beat (Phase 2 baselines, same splits)

| Setting | Baseline | Top-1 | Top-3 |
|---|---|---|---|
| Known company, January test | merchant/industry lookup | 72.9% | 90.3% |
| Known company, **new merchant** | lookup | 58.7% | 84.4% |
| **New company** (leave-one-out, all rows) | name transfer | 12.7% | 28.0% |
| New company, average of companies | name transfer | – | 48.8% |

Full per-company numbers: `results/baselines/baseline_report.md` (local).
The model is worth adopting if it clearly beats lookup on **new merchants** and transfer on **new companies**.

---

## What `train.py` does

- **Input per transaction:**
  `가맹점명: <merchant>\n업종: <industry name> (<industry code>)`, plus each candidate as `\n용도: <purpose name>`.
- **Training:** each transaction gets `--k-train` (32) candidates: the correct code,
  `--n-hard` (16) confusable codes (from the lookup baseline), and random others. Loss is
  cross-entropy over the candidates.
- **Evaluation:** **all** active codes of the company are scored (in chunks of `--eval-chunk`),
  so there is no shortlist ceiling, even for G회사's 306 codes.
- **Trainable parts:** the choice head (~0.3M params) and, if `--lora-layers > 0`, LoRA
  adapters on the last N of the `--keep-layers` (20) kept layers. Everything else is frozen.
- **Checkpoint:** `runs/<name>/best/trainable.pt` holds only the head + LoRA weights (a few MB).
  The base Qwen3 weights are loaded from Hugging Face.

### Outputs in `runs/<name>/`

| File | Content |
|---|---|
| `config.json` | all arguments |
| `metrics.jsonl` | validation top-1/top-3/NLL during training |
| `best/trainable.pt` | best checkpoint (by validation top-3) |
| `report.md` | val + test results by company, by merchant segment, and by confidence |
| `results.json` | the same, machine-readable |
| `predictions_{val,test}.jsonl` | top-3 codes and probabilities per transaction |

The confidence table shows how accurate top-1 is when the model's confidence is ≥ 0.5 / 0.7 / 0.9.
Use it in Phase 4 to choose the threshold for sending a transaction to human review.

### Main options

| Option | Default | Notes |
|---|---|---|
| `--holdout` | none | company name (e.g. `H회사`) or ID: excluded from training, used as test |
| `--lora-layers` | 12 | 0 = head only |
| `--keep-layers` | 20 | Qwen3-0.6B has 28 layers |
| `--epochs` | 2 | ~51K training transactions per epoch (time split) |
| `--batch-size` / `--grad-accum` | 16 / 2 | effective batch 32 transactions |
| `--lr` / `--lr-head` | 2e-4 / 1e-3 | LoRA / head learning rates |
| `--grad-checkpointing` | off | turn on if GPU memory is tight |
| `--eval-every` / `--val-sample` | 500 / 3000 | validation during training |

### GPU memory

A training micro-batch is 16 transactions × ~32 candidates ≈ 16 sequences of ~500 tokens.
With bf16 this should fit in 16 GB with LoRA; for 8 GB GPUs use `--batch-size 8 --grad-accum 4`
or `--grad-checkpointing`. Head-only training needs much less memory.

---

## Notes and troubleshooting

- **Attention implementation:** the model uses `sdpa` because it needs a custom 4D attention mask.
  Flash-attention 2 does not support that mask; don't switch to it.
- **transformers version:** tested with 4.56.2. If you change it, run `pytest tests/test_model.py`
  first. It checks that packed scoring equals scoring each candidate separately.
- **Tested locally** on CPU with a tiny random Qwen3 (tests + full `train.py` runs for the
  time split and holdout modes). It has **not** been run with the real Qwen3-0.6B weights or on GPU yet.
- **LoRA** is implemented in `core/model.py` (no `peft` dependency).

## Layout

```
README.md
requirements.txt          local prep + baselines (pandas, scikit-learn, ...)
requirements-train.txt    GPU server (torch, transformers)
.gitignore                excludes all company data, runs and checkpoints

prep/                     LOCAL ONLY: steps 1-3
  prepare_data.py           data/raw/*.xlsx -> data/prepared/*.parquet + data_report.md
  baselines.py              lookup / transfer / text-similarity baselines
  build_dataset.py          data/prepared -> data/jev (training export)

core/                     model package (used by train.py)
  model.py                  packing + 4D mask, LoRA, choice head, JEVModel
  data.py                   loading, splits, candidate sampling
train.py                  training + evaluation (GPU server)
scripts/run_loco.sh       8-fold leave-one-company-out
tests/test_model.py       mask / order-invariance / LoRA tests

data/                     (git-ignored)
  raw/                      original workbooks A-H회사.xlsx, [공통]이용기관정보.xlsx
  prepared/                 cleaned parquet + data_report.md
  jev/                      training export: companies.json, examples.jsonl, meta.json
results/baselines/        baseline_report.md (+ row-level parquet, git-ignored)
runs/<name>/              training outputs (git-ignored)
```

Everything runs from the project root: `python prep/...`, `python train.py`, `bash scripts/run_loco.sh`.
