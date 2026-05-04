import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32768
    max_seq_len: int = 300
    model_dim: int = 640
    n_layers: int = 12
    n_heads: int = 10
    n_kv_heads: int = 5
    head_dim: int = 64
    # MoE
    n_shared_experts: int = 2
    n_routed_experts: int = 12
    n_activated_experts: int = 4
    expert_hidden_dim: int = 640
    moe_score_func: str = "softmax"
    moe_n_groups: int = 1
    moe_topk_groups: int = 1
    moe_route_scale: float = 1.0
    moe_balance_alpha: float = 0.001


class FFN(nn.Module):
    def __init__(self, model_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(model_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(model_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, model_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(
        self,
        model_dim,
        n_routed_experts,
        n_activated,
        score_func: str = 'softmax',
        n_groups: int = 1,
        topk_groups: int = 1,
        route_scale: float = 1.0,
        use_bias: bool = False,
    ):
        super().__init__()
        self.n_activated = n_activated
        self.score_func = score_func
        self.n_groups = n_groups
        self.topk_groups = topk_groups
        self.route_scale = route_scale

        self.weight = nn.Parameter(torch.empty(n_routed_experts, model_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bias = (
            nn.Parameter(torch.zeros(n_routed_experts, dtype=torch.float32))
            if use_bias
            else None
        )

    def forward(self, x):
        # x: [B*T, D]
        logits = F.linear(x, self.weight) # [B*T, N_r]

        if self.score_func == 'softmax':
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = logits.sigmoid().to(torch.float32)

        original_scores = scores # [B*T, N_r]

        routing = scores
        if self.bias is not None:
            routing = routing + self.bias

        if self.n_groups > 1:
            # [T, n_groups, experts_per_group]
            g = routing.view(x.shape[0], self.n_groups, -1)
            if self.bias is None:
                group_scores = g.amax(dim=-1) # [T, n_groups]
            else:
                group_scores = g.topk(2, dim=-1)[0].sum(dim=-1)
            _, top_groups = group_scores.topk(self.topk_groups, dim=-1) # [T, topk_g]
            mask = torch.ones(
                x.shape[0], self.n_groups, dtype=torch.bool, device=x.device
            ).scatter_(
                1, top_groups, False
            )
            routing = g.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)

        _, indices = routing.topk(self.n_activated, dim=-1) # [B*T, N_a]

        weights = original_scores.gather(1, indices)

        if self.score_func == 'sigmoid':
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        weights = (weights * self.route_scale).to(x.dtype)
        return weights, indices, original_scores


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        self.moe_balance_alpha = config.moe_balance_alpha

        # shared experts
        shared_hidden_dim = config.n_shared_experts * config.expert_hidden_dim
        self.shared_experts = FFN(config.model_dim, shared_hidden_dim)

        # routed experts
        self.gate = Gate(
            model_dim=config.model_dim,
            n_routed_experts=config.n_routed_experts,
            n_activated=config.n_activated_experts,
            score_func=config.moe_score_func,
            n_groups=config.moe_n_groups,
            topk_groups=config.moe_topk_groups,
            route_scale=config.moe_route_scale,
            use_bias=False,
        )
        self.experts = nn.ModuleList(
            [
                FFN(config.model_dim, config.expert_hidden_dim * 2)
                for _ in range(config.n_routed_experts)
            ]
        )

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D) # [B*T, D]
        n_tokens = x_flat.shape[0] # B*T

        # shared experts
        z = self.shared_experts(x_flat) # [B*T, D]

        # routed experts
        weights, indices, scores = self.gate(z)

        y = torch.zeros_like(x_flat)

        print(indices)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
        for i, expert in enumerate(self.experts):
            if counts[i].item() == 0:
                continue
            token_idx, rank_in_k = torch.where(indices == i)
            y[token_idx] += expert(x_flat[token_idx]) * weights[token_idx, rank_in_k, None]

        output = (y + z).view(B, T, D)

        # balance
        balance_loss = None
        if self.training and self.moe_balance_alpha > 0.0:
            balance_loss = self._balance_loss(indices, scores, n_tokens)

        return output, balance_loss

    def _balance_loss(self, indices, scores, n_tokens):
        N_r, K = self.n_routed_experts, self.n_activated_experts

        # Routing counts per expert (non-differentiable)
        counts = torch.zeros(N_r, dtype=torch.float32, device=indices.device)
        counts.scatter_add_(
            0,
            indices.flatten(),
            torch.ones(indices.numel(), dtype=torch.float32, device=indices.device),
        )
        f = counts * (N_r / (K * n_tokens))  # normalised frequency [N_r]

        # Mean soft gate score per expert (differentiable through softmax/sigmoid)
        P = scores.mean(dim=0)  # [N_r]

        # f is derived from hard top-K → no gradient; gradient flows through P only
        return (f * P).sum()


if __name__ == "__main__":
    torch.manual_seed(42)

    config = GPTConfig(
        model_dim=640,
        n_shared_experts=2,
        n_routed_experts=12,
        n_activated_experts=4,
        expert_hidden_dim=640,
        moe_score_func="softmax",
        moe_n_groups=1,
        moe_topk_groups=1,
        moe_route_scale=1.0,
        moe_balance_alpha=0.001,
    )

    moe = MoE(config)
    # total_params = sum(p.numel() for p in moe.parameters())
    # print("total params:", total_params)
    moe.train()

    B, T, D = 2, 4, config.model_dim
    x = torch.randn(B, T, D)

    output, balance_loss = moe(x)

    print("input shape:", x.shape)
    print("output shape:", output.shape)
    print("balance_loss:", balance_loss)
    print("output mean:", output.mean().item())
    print("output std:", output.std().item())

    loss = output.mean()

    if balance_loss is not None:
        loss = loss + balance_loss

    loss.backward()

    print("loss:", loss.item())
    print("x grad is None:", x.grad is None)