import torch
import torch.nn as nn
import torch.nn.functional as F


class Gemma4FFNMinimal(nn.Module):
    """
    Gemma 4 风格的最小可用 FFN

    结构:
        x -> gate_up_proj -> split(gate, up)
          -> GELU(gate) * up
          -> down_proj
    """

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        bias: bool = False,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim

        # 一次性投影到 2 * hidden_dim，然后拆成两路
        self.gate_up_proj = nn.Linear(model_dim, hidden_dim * 2, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, model_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        return: [B, T, D]
        """
        gate_up = self.gate_up_proj(x)  # [B, T, 2H]
        gate, up = gate_up.chunk(2, dim=-1)  # 各自 [B, T, H]

        hidden = F.gelu(gate, approximate="tanh") * up
        out = self.down_proj(hidden)
        return out


def demo():
    torch.manual_seed(42)

    b, t, d = 2, 8, 256
    hidden_dim = 1024

    x = torch.randn(b, t, d)
    ffn = Gemma4FFNMinimal(model_dim=d, hidden_dim=hidden_dim)

    y = ffn(x)

    print("x.shape =", x.shape)
    print("y.shape =", y.shape)


if __name__ == "__main__":
    demo()