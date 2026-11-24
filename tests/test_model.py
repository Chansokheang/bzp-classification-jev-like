"""Checks for the packing / attention-mask logic, using a tiny random Qwen3 (no download).

    python -m pytest tests/test_model.py -q      (or)      python tests/test_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoModel, Qwen3Config  # noqa: E402

from core.model import JEVModel, pack  # noqa: E402

STATE = [5, 6, 7, 8, 9]
CANDS = [[10, 11], [12, 13, 14], [15], [16, 17, 18, 19]]


def tiny_backbone(attn: str):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256)
    return AutoModel.from_config(cfg, attn_implementation=attn).eval()


def cand_hidden_packed(bb, states, cands):
    ids, pos, mask, cidx, _ = pack(states, cands, pad_id=0, dtype=torch.float32)
    with torch.no_grad():
        h = bb(input_ids=ids, position_ids=pos, attention_mask=mask, use_cache=False).last_hidden_state
    return [[h[b][cidx[b] == k] for k in range(len(cands[b]))] for b in range(len(cands))]


def cand_hidden_separate(bb, state, cand):
    ids = torch.tensor([state + cand])
    with torch.no_grad():
        h = bb(input_ids=ids, use_cache=False).last_hidden_state
    return h[0, len(state):]


def test_packed_equals_separate():
    for attn in ("sdpa", "eager"):
        bb = tiny_backbone(attn)
        # batch of two examples with different lengths -> exercises padding too
        packed = cand_hidden_packed(bb, [STATE, STATE[:3]], [CANDS, CANDS[:2]])
        for b, (s, cs) in enumerate([(STATE, CANDS), (STATE[:3], CANDS[:2])]):
            for k, c in enumerate(cs):
                ref = cand_hidden_separate(bb, s, c)
                assert torch.allclose(packed[b][k], ref, atol=1e-4), (attn, b, k)


def test_order_invariance_and_chunking():
    bb = tiny_backbone("sdpa")
    model = JEVModel(bb, keep_layers=3, lora_layers=0).eval()

    def scores(cands):
        ids, pos, mask, cidx, n = pack([STATE], [cands], 0, torch.float32)
        with torch.no_grad():
            return model(ids, pos, mask, cidx, int(n.max()))[0]

    full = scores(CANDS)
    perm = [3, 1, 0, 2]
    permuted = scores([CANDS[i] for i in perm])
    assert torch.allclose(full[perm], permuted, atol=1e-5)
    chunked = torch.cat([scores(CANDS[:2]), scores(CANDS[2:])])
    assert torch.allclose(full, chunked, atol=1e-5)


def test_lora_trains_only_adapters_and_head():
    bb = tiny_backbone("sdpa")
    model = JEVModel(bb, keep_layers=2, lora_layers=1, lora_r=4).train()
    assert len(model.backbone.layers) == 2 and model.n_lora == 7
    ids, pos, mask, cidx, n = pack([STATE, STATE], [CANDS, CANDS[:3]], 0, torch.float32)
    logits = model(ids, pos, mask, cidx, int(n.max()))
    logits = logits.masked_fill(torch.arange(4)[None] >= n[:, None], float("-inf"))
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2])).backward()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(("lora_" in n) or n.startswith("head.") for n in trainable)
    assert all(p.grad is not None for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n)
    assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)


if __name__ == "__main__":
    test_packed_equals_separate()
    test_order_invariance_and_chunking()
    test_lora_trains_only_adapters_and_head()
    print("all tests passed")
