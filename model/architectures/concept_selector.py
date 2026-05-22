"""
ConceptSelector: Concept-based Learnable Token Selector for BreastGPT2.

Core idea: text tokens act as semantic "concepts" that query image tokens
via cross-attention. The resulting attention map IS the importance score —
visual patches attended by text concepts are retained, others pruned.

Two concept sources (can combine):
  1. Text concepts: from the instruction/question text embedding
  2. Learnable concepts: fixed learnable queries that capture
     domain-specific visual concepts (mass, calcification, etc.)

Selection pipeline:
  text/learned concepts → CrossAttn(Q=concepts, K=V=image) → attn map
  → aggregate across concepts → per-token importance → Gumbel top-k

References:
  - MADTP (CVPR 2024): learnable tokens for cross-modal alignment
  - LightVLA* (2025): learnable query-based token pruning
  - Q-Former (BLIP-2): learnable queries bridging modalities
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Gumbel top-k (same as before, included for self-containedness)
# ---------------------------------------------------------------------------

def _gumbel_noise(shape, device, dtype, eps=1e-20):
    u = torch.rand(shape, device=device, dtype=dtype).clamp(eps, 1 - eps)
    return -torch.log(-torch.log(u))


def gumbel_topk(
    logits: torch.Tensor,
    k: int,
    tau: float = 1.0,
    noise_scale: float = 1.0,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable top-k via Gumbel perturbation + Straight-Through.

    Returns:
        mask: (B, L) differentiable mask (hard forward, soft backward)
        idx:  (B, k) selected indices (sorted ascending)
    """
    if training and noise_scale > 0:
        perturbed = logits + _gumbel_noise(logits.shape, logits.device, logits.dtype) * noise_scale
    else:
        perturbed = logits

    _, topk_idx = perturbed.topk(k, dim=-1)
    topk_idx, _ = topk_idx.sort(dim=-1)  # positional order

    hard_mask = torch.zeros_like(logits)
    hard_mask.scatter_(1, topk_idx, 1.0)

    if training:
        soft_mask = torch.sigmoid(perturbed / tau)
        mask = hard_mask + (soft_mask - soft_mask.detach())  # ST
    else:
        mask = hard_mask

    return mask, topk_idx


# ---------------------------------------------------------------------------
# Coverage diversity loss (regularizer)
# ---------------------------------------------------------------------------

def coverage_loss(
    selected: torch.Tensor,    # (B, k, D)
    full_set: torch.Tensor,    # (B, L, D)
    text: Optional[torch.Tensor] = None,  # (B, m, D)
    alpha: float = 0.5,
    vv_tau: float = 0.2,
    tv_tau: float = 0.02,
) -> torch.Tensor:
    """Negative coverage: minimize to maximize coverage over full set + text."""
    sel_n = F.normalize(selected, dim=-1)
    all_n = F.normalize(full_set.detach(), dim=-1)

    # vision coverage: each full-set token's max sim to any selected token
    sim_vv = torch.bmm(all_n, sel_n.transpose(1, 2))  # (B, L, k)
    vv_cov = F.softmax(sim_vv / vv_tau, dim=-1).max(dim=-1).values.mean(dim=-1)

    loss = -alpha * vv_cov

    if text is not None:
        txt_n = F.normalize(text.detach(), dim=-1)
        sim_tv = torch.bmm(txt_n, sel_n.transpose(1, 2))  # (B, m, k)
        tv_cov = F.softmax(sim_tv / tv_tau, dim=-1).max(dim=-1).values.mean(dim=-1)
        loss = loss - (1.0 - alpha) * tv_cov

    return loss.mean()

# ---------------------------------------------------------------------------
# Concept-based cross-attention scorer
# ---------------------------------------------------------------------------

class ConceptCrossAttention(nn.Module):
    """
    Multi-head cross-attention: concepts query image tokens.
    Returns per-image-token importance scores aggregated from attention map.

    Q = concepts (text tokens / learnable queries)
    K = V = image tokens

    Attention map shape: (B, heads, m, L)
    → aggregate over concept dim (m) and heads → (B, L) importance
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        qk_dim: Optional[int] = None,
        dropout: float = 0.0,
        aggregate: str = 'max',   # 'max' | 'mean' | 'sum'
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qk_dim = qk_dim or (hidden_dim // num_heads)
        self.head_dim = self.qk_dim
        self.scale = self.head_dim ** -0.5
        self.aggregate = aggregate

        inner_dim = num_heads * self.head_dim

        self.q_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, inner_dim, bias=False)

        # optional: value path for producing refined features (not scores)
        # omitted here — we only need the attention map for scoring

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj]:
            nn.init.xavier_uniform_(m.weight)

    def forward(
        self,
        concepts: torch.Tensor,    # (B, m, D)   — text or learnable queries
        image_tokens: torch.Tensor, # (B, L, D)   — vision tokens
    ) -> torch.Tensor:
        """
        Returns:
            scores: (B, L) per-token importance logits
        """
        B, m, _ = concepts.shape
        L = image_tokens.shape[1]
        H = self.num_heads
        d = self.head_dim
        q = self.q_proj(concepts).reshape(B, m, H, d).transpose(1, 2)   # (B, H, m, d)
        k = self.k_proj(image_tokens).reshape(B, L, H, d).transpose(1, 2)  # (B, H, L, d)

        # attention: (B, H, m, L)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # aggregate: (B, H, m, L) → (B, L)
        if self.aggregate == 'max':
            # max over concepts → (B, H, L) → mean over heads → (B, L)
            scores = attn.max(dim=2).values.mean(dim=1)
        elif self.aggregate == 'mean':
            scores = attn.mean(dim=2).mean(dim=1)
        elif self.aggregate == 'sum':
            scores = attn.sum(dim=2).mean(dim=1)
        else:
            raise ValueError(f"Unknown aggregate: {self.aggregate}")

        # convert to logits (log-space for Gumbel compatibility)
        scores = torch.log(scores.clamp(min=1e-8))

        return scores


# ---------------------------------------------------------------------------
# ConceptSelector — main module
# ---------------------------------------------------------------------------

class ConceptSelector(nn.Module):
    """
    Concept-based Learnable Token Selector.

    Text concepts (and/or learnable domain concepts) cross-attend to
    image tokens. The attention weights define token importance.
    Gumbel top-k selects tokens differentiably.

    Args:
        hidden_dim: model hidden dimension (e.g. 4096)
        num_heads: heads for cross-attention scoring
        num_learnable_concepts: number of learnable concept queries
            (set 0 to use text-only concepts)
        aggregate: how to reduce (concepts, heads) → per-token score
        gumbel_tau: Gumbel-Softmax temperature
        gumbel_noise_init: initial Gumbel noise scale
        noise_anneal_steps: steps to anneal noise → noise_min
        noise_min: final noise scale
        cov_alpha: coverage loss weight (vv vs tv)

    Forward:
        visual_feats:  (B, L, D)
        text_feats:    (B, m, D) or None
        k:             int, target token count

    Returns:
        selected_feats: (B, k, D)
        selected_idx:   (B, k)
        aux_loss:       scalar (coverage regularization)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_learnable_concepts: int = 16,
        aggregate: str = 'max',
        gumbel_tau: float = 1.0,
        gumbel_noise_init: float = 1.0,
        noise_anneal_steps: int = 10000,
        noise_min: float = 0.1,
        cov_alpha: float = 0.5,
        cov_vv_tau: float = 0.2,
        cov_tv_tau: float = 0.02,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_learnable_concepts = num_learnable_concepts
        self.gumbel_tau = gumbel_tau
        self.gumbel_noise_init = gumbel_noise_init
        self.noise_anneal_steps = noise_anneal_steps
        self.noise_min = noise_min
        self.cov_alpha = cov_alpha
        self.cov_vv_tau = cov_vv_tau
        self.cov_tv_tau = cov_tv_tau

        # cross-attention scorer
        self.cross_attn = ConceptCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            aggregate=aggregate,
        )

        # learnable concept queries (domain-specific visual concepts)
        if num_learnable_concepts > 0:
            self.concept_queries = nn.Parameter(
                torch.randn(1, num_learnable_concepts, hidden_dim) * 0.02
            )
        else:
            self.concept_queries = None

        # gate to balance text concepts vs learnable concepts
        if num_learnable_concepts > 0:
            self.concept_gate = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))

    @property
    def current_noise_scale(self) -> float:
        if self.noise_anneal_steps <= 0:
            return self.gumbel_noise_init
        t = min(self._step.item() / self.noise_anneal_steps, 1.0)
        return self.gumbel_noise_init * (1.0 - t) + self.noise_min * t

    def _build_concepts(
        self,
        text_feats: Optional[torch.Tensor],
        B: int,
    ) -> torch.Tensor:
        """
        Assemble concept queries from text and/or learnable queries.

        Three modes:
          1. text_feats only (num_learnable_concepts=0)
          2. learnable only  (text_feats=None)
          3. both: concat with learned gating
        """
        parts = []

        if self.concept_queries is not None:
            lq = self.concept_queries.expand(B, -1, -1)  # (B, n_lq, D)
            parts.append(lq)

        if text_feats is not None:
            parts.append(text_feats)

        if len(parts) == 0:
            raise ValueError(
                "ConceptSelector needs at least one of: "
                "text_feats or num_learnable_concepts > 0"
            )

        if len(parts) == 1:
            out = parts[0]
            if hasattr(self, 'concept_gate') and text_feats is not None:
                gate_dummy = self.concept_gate(text_feats.mean(dim=1, keepdim=True))
                out = out + gate_dummy * 0.0
            return out

        # gate: per-token weight between learnable vs text concepts
        # use mean-pooled text as conditioning
        gate_input = text_feats.mean(dim=1, keepdim=True)  # (B, 1, D)
        gate_val = self.concept_gate(gate_input)  # (B, 1, 1)

        # scale learnable concepts by gate, text by (1-gate)
        lq_scaled = parts[0] * gate_val
        txt_scaled = parts[1] * (1.0 - gate_val)

        return torch.cat([lq_scaled, txt_scaled], dim=1)  # (B, n_lq + m, D)

    def forward(
        self,
        visual_feats: torch.Tensor,
        k: Optional[int] = 64,
        text_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = visual_feats.shape
        device = visual_feats.device

        # 1. build concept queries
        concepts = self._build_concepts(text_feats, B)  # (B, n_concepts, D)
        # 2. cross-attention scoring
        logits = self.cross_attn(concepts, visual_feats)  # (B, L)

        if L <= k:
            weight = F.softmax(logits, dim=-1).unsqueeze(-1)       # (B, L, 1)
            out = visual_feats * weight * 0.0 + visual_feats       # 值不变，梯度通
            idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            return out, idx, torch.tensor(0.0, device=device)

        # 3. Gumbel top-k
        if self.training:
            self._step += 1

        mask, topk_idx = gumbel_topk(
            logits, k,
            tau=self.gumbel_tau,
            noise_scale=self.current_noise_scale,
            training=self.training,
        )

        # 4. gather with ST gradient
        if self.training:
            weighted = visual_feats * mask.unsqueeze(-1)
            selected = torch.gather(
                weighted, 1,
                topk_idx.unsqueeze(-1).expand(-1, -1, D),
            )
        else:
            selected = torch.gather(
                visual_feats, 1,
                topk_idx.unsqueeze(-1).expand(-1, -1, D),
            )

        # 5. coverage regularization
        aux_loss = coverage_loss(
            selected, visual_feats,
            text=text_feats,
            alpha=self.cov_alpha,
            vv_tau=self.cov_vv_tau,
            tv_tau=self.cov_tv_tau,
        )

        return selected, topk_idx, aux_loss


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, L, D, m = 2, 576, 4096, 12
    k = 64

    v = torch.randn(B, L, D, device=device)
    t = torch.randn(B, m, D, device=device)

    print("=" * 60)
    print("ConceptSelector (text + 16 learnable concepts)")
    print("=" * 60)

    sel = ConceptSelector(
        hidden_dim=D, num_heads=8,
        num_learnable_concepts=16,
    ).to(device)

    sel.train()
    out, idx, loss = sel(v, 64, text_feats=t)
    print(f"  [train] out={out.shape}, idx={idx.shape}, loss={loss.item():.4f}")
    loss.backward()
    grad_n = sum(p.grad.norm().item() for p in sel.parameters() if p.grad is not None)
    print(f"  [train] grad norm: {grad_n:.4f}")

    sel.eval()
    with torch.no_grad():
        out, idx, loss = sel(v, text_feats=t)
    print(f"  [eval]  out={out.shape}, idx={idx.shape}, loss={loss.item():.4f}")

    n_params = sum(p.numel() for p in sel.parameters())
    print(f"  params: {n_params:,} ({n_params / 1e6:.2f}M)")

    print()
    print("=" * 60)
    print("ConceptSelector (learnable concepts only, no text)")
    print("=" * 60)

    sel2 = ConceptSelector(
        hidden_dim=D, num_heads=8,
        num_learnable_concepts=32,
    ).to(device)

    sel2.train()
    out, idx, loss = sel2(v, 64, text_feats=None)
    print(f"  [train] out={out.shape}, loss={loss.item():.4f}")