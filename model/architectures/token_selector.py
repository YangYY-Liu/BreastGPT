import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple

class CoverageTokenSelector(nn.Module):
    """
        Coverage-based vision token selector (MMTok algorithm).

        Selects a subset of vision tokens via greedy submodular maximization
        over a coverage matrix Combined = [P; α·Q], where:
        - P: text-vision coverage — each text token's attention over vision patches
        - Q: vision-vision coverage — inter-patch similarity for diversity

        Gradient: Straight-Through surrogate — forward uses hard-indexed features,
        backward flows through soft attention-weighted features to ViT/projector.

        No learnable parameters. For learnable selection, use ConceptSelector.

        Args:
            alpha:    Q 的缩放权重。P 不缩放，α 缩放 Q。
                    0 = 纯 text-vision, 0.5 = MMTok 默认, 1.0 = text 与 vision 等权。
            tv_tau:   text-vision softmax 温度。
                    低 → 每个 text token 只关注少数 patch 默认 0.02。
            vv_tau:   vision-vision softmax 温度。
                    高于 tv_tau，因为相邻 patch 本身就相似 默认 0.2。
            st_tau:   ST surrogate 的 soft weights 温度。
                    低(0.3) → 梯度集中在被选 token；高(3.0) → 梯度分散。
                    训练不稳定调高，loss 下降慢调低。建议从 1.0 开始。
            use_st:   启用 Straight-Through 梯度。全参微调必须 True，纯推理设 False。
            use_txt:  启用 text-vision coverage（P 矩阵）。
                    True = MMTok 多模态选择；False = 纯 vision-vision 覆盖。

        Forward:
            visual_feats: (B, L, D) vision tokens
            k:            int, 目标保留 token 数
            text_feats:   (B, m, D) text embeddings, use_txt=True 时生效

        Returns:
            selected_feats: (B, k, D)
            selected_idx:   (B, k) LongTensor
            aux_loss:       tensor(0.0), 无可训练 loss

        Reference:
            MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs (ICLR 2026)
            https://arxiv.org/abs/2508.18264
    """
    def __init__(
        self,
        alpha: float = 0.5,
        tv_tau: float = 0.02,
        vv_tau: float = 0.2,
        st_tau: float = 1.0,
        use_st: bool = True,
        use_txt: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.tv_tau = tv_tau
        self.vv_tau = vv_tau
        self.st_tau = st_tau
        self.use_st = use_st
        self.use_txt = use_txt

    # ------------------------------------------------------------------
    # greedy coverage maximization
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _greedy_max_coverage(Combined: torch.Tensor, k: int) -> torch.Tensor:
        """
        Greedy submodular maximization over Combined = [P; α·Q].

        Each step picks the column with largest marginal coverage gain:
            delta_j = Σ_i max(0, Combined[i,j] - best[i])

        Returns:
            selected: (k,) LongTensor of selected column indices
        """
        R, n = Combined.shape
        device = Combined.device
        dtype = Combined.dtype

        best = torch.zeros(R, device=device, dtype=dtype)
        score_mask = torch.zeros(n, device=device, dtype=dtype)
        selected = torch.empty(k, dtype=torch.long, device=device)
        neg_inf = float('-inf')

        for i in range(k):
            delta = (Combined - best.unsqueeze(1)).clamp_min_(0).sum(dim=0)
            delta = delta + score_mask
            best_idx = delta.argmax()
            selected[i] = best_idx
            best = torch.maximum(best, Combined[:, best_idx])
            score_mask[best_idx] = neg_inf

        return selected

    # ------------------------------------------------------------------
    # build coverage matrices (detached, for greedy selection only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _build_coverage(
        self,
        v_norm: torch.Tensor,
        t_norm: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build Combined = [P; α·Q].

        P: (m, L) text-vision, row-softmax / m
        Q: (L, L) vision-vision, row-softmax / L
        Combined: (m+L, L) or (L, L) if no text

        Returns:
            Combined: (R, L) coverage matrix for greedy selection
        """
        L = v_norm.shape[0]

        # Q: vision-vision coverage
        sim_vv = v_norm @ v_norm.t()
        Q = F.softmax(sim_vv / self.vv_tau, dim=1) / L

        # P: text-vision coverage
        if t_norm is not None and self.use_txt:
            m = t_norm.shape[0]
            sim_tv = t_norm @ v_norm.t()
            P = F.softmax(sim_tv / self.tv_tau, dim=1) / m
            Combined = torch.cat([P, Q * self.alpha], dim=0)
        else:
            Combined = Q

        return Combined

    # ------------------------------------------------------------------
    # ST soft surrogate (WITH gradient through ViT)
    # ------------------------------------------------------------------

    def _build_soft_surrogate(
        self,
        visual_feats: torch.Tensor,      # (L, D) WITH grad
        selected_idx: torch.Tensor,       # (k,) LongTensor
        text_feats: Optional[torch.Tensor] = None,  # (m, D)
    ) -> torch.Tensor:
        """
        Differentiable surrogate for selected features.

        sim = vision-vision similarity + α · text relevance
        soft_s = Σ_j softmax(sim[s,j] / τ) · visual_feats[j]

        Returns:
            soft: (k, D) WITH grad → ViT receives gradient
        """
        v_norm = F.normalize(visual_feats, dim=-1)      # (L, D) WITH grad
        sim_vv = v_norm @ v_norm.t()                             # (L, L) WITH grad

        sim = sim_vv[selected_idx]                               # (k, L)

        if text_feats is not None and self.use_txt:
            t_norm = F.normalize(text_feats, dim=-1)     # (m, D)
            sim_tv = t_norm @ v_norm.t()                         # (m, L) WITH grad
            text_relevance = sim_tv.max(dim=0).values            # (L,)
            sim = self.alpha * sim + text_relevance.unsqueeze(0) # (k, L)

        weights = F.softmax(sim / self.st_tau, dim=-1)           # (k, L)
        soft = weights @ visual_feats                            # (k, D)
        return soft

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        visual_feats: torch.Tensor,             # (B, L, D)
        k: int,
        text_feats: Optional[torch.Tensor] = None,  # (B, m, D)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, L, D = visual_feats.shape
        device = visual_feats.device
        zero_loss = torch.tensor(0.0, device=device)

        # bypass: nothing to prune
        if L <= k:
            idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            return visual_feats, idx, zero_loss

        k = min(k, L)
        need_st = self.training and self.use_st

        all_selected_idx = []
        all_feats = []

        for b in range(B):
            v = visual_feats[b]                                  # (L, D)

            # --- detached features for greedy selection ---
            v_norm = F.normalize(v.detach(), dim=-1)

            t_norm = None
            if text_feats is not None:
                t_norm = F.normalize(text_feats[b].detach(), dim=-1)

            # --- build coverage & greedy select (no grad) ---
            Combined = self._build_coverage(
                v_norm, t_norm,
            )
            sel_idx = self._greedy_max_coverage(Combined, k)
            sel_idx, _ = sel_idx.sort()

            # --- gather features ---
            hard = v[sel_idx]                                    # (k, D)

            if need_st:
                t_b = text_feats[b] if text_feats is not None else None
                soft = self._build_soft_surrogate(v, sel_idx, t_b)
                out = hard + (soft - soft.detach())              # ST trick
            else:
                out = hard

            all_selected_idx.append(sel_idx)
            all_feats.append(out)

        selected_idx = torch.stack(all_selected_idx, dim=0)      # (B, k)
        selected_feats = torch.stack(all_feats, dim=0)           # (B, k, D)

        return selected_feats, selected_idx, zero_loss

if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
    B, L, D, m = 2, 576, 4096, 10
    k = 64
 
    v = torch.randn(B, L, D, device=device, requires_grad=True)
    t = torch.randn(B, m, D, device=device)
 
    sel = ConceptSelector(alpha=0.5, use_st=True)
    # --- train mode: ST gradient ---
    out, idx, loss = sel(v, k, text_feats=t, training=True)
    print(f"[train] out={out.shape}, idx={idx.shape}")
 
    fake_loss = out.sum()
    fake_loss.backward()
    print(f"[train] v.grad is None: {v.grad is None}")
    print(f"[train] v.grad norm:    {v.grad.norm().item():.4f}")
    print()
 
    # --- eval mode: no ST, pure hard ---
    v2 = torch.randn(B, L, D, device=device)
    out2, idx2, _ = sel(v2, k, text_feats=t, training=False)
    print(f"[eval]  out={out2.shape}, idx={idx2.shape}")
    print()
 
    # --- verify ST correctness: forward value == hard ---
    v3 = torch.randn(B, L, D, device=device, requires_grad=True)
    out3, idx3, _ = sel(v3, k, text_feats=t, training=True)
    hard_check = torch.stack([v3[b][idx3[b]] for b in range(B)])
    diff = (out3 - hard_check).abs().max().item()
    print(f"[ST check] max |out - hard|: {diff:.2e} (should be ~0)")