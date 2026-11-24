"""JEV-like decision model: Qwen3 backbone + LoRA + order-invariant choice head.

One sequence = [state tokens][cand 1][cand 2]...[cand K].
  * every candidate restarts at the same position id (right after the state)
  * a 4D attention mask lets each candidate see the state and itself, never other candidates
So each candidate's score is independent of the other candidates and of their order, and a
long code list can be scored in chunks with identical results.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ------------------------------------------------------------------ packing
def pack(states: list[list[int]], cands: list[list[list[int]]], pad_id: int, dtype: torch.dtype):
    """Build model inputs for a batch.

    states[b]   token ids of example b's state
    cands[b][k] token ids of candidate k of example b
    Returns input_ids (B,L), position_ids (B,L), attention_mask (B,1,L,L additive float),
            cand_index (B,L) = candidate number for candidate tokens, -1 otherwise,
            n_cands (B,) number of candidates per example.
    """
    B = len(states)
    ids, pos, grp = [], [], []
    for s, cs in zip(states, cands):
        i, p, g = list(s), list(range(len(s))), [-1] * len(s)
        for k, c in enumerate(cs):
            i += c
            p += range(len(s), len(s) + len(c))
            g += [k] * len(c)
        ids.append(i), pos.append(p), grp.append(g)
    L = max(len(x) for x in ids)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    position_ids = torch.zeros((B, L), dtype=torch.long)
    group = torch.full((B, L), -2, dtype=torch.long)          # -2 pad, -1 state, k candidate
    for b in range(B):
        n = len(ids[b])
        input_ids[b, :n] = torch.tensor(ids[b])
        position_ids[b, :n] = torch.tensor(pos[b])
        group[b, :n] = torch.tensor(grp[b])

    idx = torch.arange(L)
    causal = idx[None, :] <= idx[:, None]                       # key index <= query index
    q, k = group[:, :, None], group[:, None, :]
    allowed = causal & ((q == k) | (k == -1)) & (q != -2)
    allowed |= torch.eye(L, dtype=torch.bool)[None] & (q == -2)  # pad rows see themselves (no NaN)
    mask = torch.zeros((B, 1, L, L), dtype=dtype)
    mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)

    cand_index = group.clone()
    cand_index[cand_index < 0] = -1
    n_cands = torch.tensor([len(c) for c in cands], dtype=torch.long)
    return input_ids, position_ids, mask, cand_index, n_cands


# ------------------------------------------------------------------ LoRA
class LoRALinear(nn.Module):
    """y = W x + (alpha/r) * B A x, with W frozen. A, B kept in fp32."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        lora = F.linear(F.linear(self.drop(x), self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return self.base(x) + lora * self.scale


def add_lora(layers: nn.ModuleList, layer_ids, r: int, alpha: float, dropout: float) -> int:
    n = 0
    for i in layer_ids:
        for parent in layers[i].modules():
            for name, child in list(parent.named_children()):
                if name in LORA_TARGETS and isinstance(child, nn.Linear):
                    setattr(parent, name, LoRALinear(child, r, alpha, dropout))
                    n += 1
    return n


# ------------------------------------------------------------------ head
class ChoiceHead(nn.Module):
    """Attention-pool each candidate's tokens into one vector, then score it."""

    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.Linear(d_model, 1)
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor, cand_index: torch.Tensor, k_max: int) -> torch.Tensor:
        h = self.norm(h.float())                                       # (B,L,D)
        a = self.attn(h).squeeze(-1)                                   # (B,L)
        member = cand_index[:, None, :] == torch.arange(k_max, device=h.device)[None, :, None]  # (B,K,L)
        w = a[:, None, :].masked_fill(~member, float("-inf")).softmax(-1)
        w = torch.nan_to_num(w, nan=0.0)                               # absent candidates
        pooled = torch.einsum("bkl,bld->bkd", w, h)
        return self.mlp(pooled).squeeze(-1)                            # (B,K) scores


# ------------------------------------------------------------------ full model
def truncate_layers(backbone: nn.Module, keep: int) -> None:
    backbone.layers = nn.ModuleList(list(backbone.layers)[:keep])
    cfg = backbone.config
    cfg.num_hidden_layers = keep
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[:keep]


class JEVModel(nn.Module):
    def __init__(self, backbone: nn.Module, keep_layers: int, lora_layers: int,
                 lora_r: int = 16, lora_alpha: float = 32, lora_dropout: float = 0.05,
                 head_hidden: int = 256):
        super().__init__()
        truncate_layers(backbone, keep_layers)
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.n_lora = 0
        if lora_layers > 0:
            ids = range(max(0, keep_layers - lora_layers), keep_layers)
            self.n_lora = add_lora(backbone.layers, ids, lora_r, lora_alpha, lora_dropout)
        self.backbone = backbone
        self.head = ChoiceHead(backbone.config.hidden_size, head_hidden)
        self.backbone_trainable = lora_layers > 0

    @classmethod
    def from_pretrained(cls, name: str, dtype=torch.bfloat16, attn_implementation: str = "sdpa", **kw):
        from transformers import AutoModel
        bb = AutoModel.from_pretrained(name, torch_dtype=dtype, attn_implementation=attn_implementation)
        return cls(bb, **kw)

    def forward(self, input_ids, position_ids, attention_mask, cand_index, k_max: int):
        ctx = torch.enable_grad() if (self.backbone_trainable and self.training) else torch.no_grad()
        with ctx:
            h = self.backbone(input_ids=input_ids, position_ids=position_ids,
                              attention_mask=attention_mask, use_cache=False).last_hidden_state
        return self.head(h, cand_index, k_max)

    def trainable_state_dict(self) -> dict:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}

    def load_trainable_state_dict(self, sd: dict) -> None:
        missing = set(self.trainable_state_dict()) - set(sd)
        if missing:
            raise KeyError(f"checkpoint missing {len(missing)} tensors, e.g. {sorted(missing)[:3]}")
        self.load_state_dict(sd, strict=False)
