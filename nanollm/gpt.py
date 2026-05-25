from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50286
    max_seq_len: int = 1024
    model_dim: int = 1024
    n_layers: int = 16
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 128


def norm(x):
    return F.rms_norm(x, (x.shape[-1],))

def apply_rotary_embedding(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)


class KVCache(nn.Module):
    def __init__(self, batch_size, n_kv_heads, max_seq_len, head_dim, n_layers, device, dtype):
        super().__init__()
        self.k_cache = torch.zeros(n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device=device, dtype=dtype)
        self.cache_seqlen = 0

    def get_kv_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.model_dim = config.model_dim
        self.head_dim = config.head_dim
        assert self.n_heads * self.head_dim == self.model_dim, (
            f"model_dim must equal n_heads * head_dim"
        )
        assert self.n_kv_heads <= self.n_heads and self.n_heads % self.n_kv_heads == 0, (
            f"n_kv_heads must be <= n_heads and n_heads must be divisible by n_kv_heads"
        )
        self.to_q = nn.Linear(self.model_dim, self.n_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(self.model_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(self.model_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.Linear(self.model_dim, self.model_dim, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, D = x.shape

        q = self.to_q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.to_k(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.to_v(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Rotary Embeddings
        cos, sin = cos_sin
        q, k = apply_rotary_embedding(q, cos, sin), apply_rotary_embedding(k, cos, sin)
        q, k = norm(q), norm(k)

        # Flash Attention
        if kv_cache is None:
            # training
            q = q.transpose(1, 2)  # [B, N, T, D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            enable_gqa = q.shape[1] != k.shape[1]
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        else:
            # single-token generation
            k_cache, v_cache = kv_cache.get_kv_cache(self.layer_idx)
            pos = kv_cache.cache_seqlen
            if k_cache is not None and v_cache is not None:
                k_cache[:, pos:pos+T, :, :] = k
                v_cache[:, pos:pos+T, :, :] = v

            k_full = k_cache[:, :pos+T]
            v_full = v_cache[:, :pos+T]
            # (B, T, H, D) -> (B, H, T, D)
            q = q.transpose(1, 2)
            k_full = k_full.transpose(1, 2)
            v_full = v_full.transpose(1, 2)
            enable_gqa = q.shape[1] != k_full.shape[1]
            causal = True if T != 1 else False
            y = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=causal, enable_gqa=enable_gqa)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.to_out(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.model_dim, 4 * config.model_dim, bias=False)
        self.linear2 = nn.Linear(4 * config.model_dim, config.model_dim, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x).square()
        x = self.linear2(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, dtype=torch.bfloat16, device=None):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device

        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.transformer = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layers)])
        self.to_out = nn.Linear(config.model_dim, config.vocab_size, bias=False)

        cos, sin = self._precompute_rotary_embeddings(config.max_seq_len, config.head_dim, device=device, dtype=dtype)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.to_out.weight, mean=0.0, std=0.001)

        # transformer
        model_dim = self.config.model_dim
        s = 3**0.5 * model_dim**-0.5
        for block in self.transformer:
            nn.init.uniform_(block.attn.to_q.weight, -s, s)
            nn.init.uniform_(block.attn.to_k.weight, -s, s)
            nn.init.uniform_(block.attn.to_v.weight, -s, s)
            nn.init.zeros_(block.attn.to_out.weight)
            nn.init.uniform_(block.mlp.linear1.weight, -s, s)
            nn.init.zeros_(block.mlp.linear2.weight)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None, dtype=torch.bfloat16):
        if device is None:
            device = self.get_device()
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos = cos.to(device=device, dtype=dtype)
        sin = sin.to(device=device, dtype=dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def get_device(self):
        return self.token_embedding.weight.device

    def forward(self, x, kv_cache=None):
        _, T = x.shape

        T0 = 0 if kv_cache is None else kv_cache.cache_seqlen
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # forward
        x = self.token_embedding(x)
        x = norm(x)
        for i, block in enumerate(self.transformer):
            x = block(x, cos_sin, kv_cache)
        x = norm(x)
        if kv_cache is not None:
            kv_cache.cache_seqlen += T

        logits = self.to_out(x)
        softcap = 20
        logits = softcap * torch.tanh(logits / softcap)
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, top_p=None,
                 frequency_penalty=0.0, penalize_prompt=True, seed=42):
        device = self.get_device()

        kv_cache = KVCache(
            batch_size=1,
            n_kv_heads=self.config.n_kv_heads,
            max_seq_len=self.config.max_seq_len,
            head_dim=self.config.head_dim,
            n_layers=self.config.n_layers,
            device=device,
            dtype=next(self.parameters()).dtype
        )

        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # Prefill prompt
        logits = self.forward(ids, kv_cache=kv_cache)
        logits = logits[:, -1, :]

        token_counts = None
        if frequency_penalty > 0.0:
            token_counts = torch.zeros(1, logits.shape[-1], dtype=logits.dtype, device=device)

            # Optionally count prompt tokens too.
            # This discourages the model from repeatedly copying the prompt.
            if penalize_prompt:
                token_counts.scatter_add_(dim=-1, index=ids, src=torch.ones_like(ids, dtype=token_counts.dtype))

        for step in range(max_tokens):
            sample_logits = logits.clone()

            # Frequency penalty
            if token_counts is not None:
                sample_logits = sample_logits - frequency_penalty * token_counts

            if temperature > 0:
                sample_logits = sample_logits / temperature

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(sample_logits, min(top_k, sample_logits.shape[-1]))
                    sample_logits[sample_logits < v[:, [-1]]] = -float('Inf')

                if top_p is not None and top_p > 0:
                    sorted_logits, sorted_indices = torch.sort(sample_logits, descending=True, dim=-1)

                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Remove tokens whose cumulative probability is above top_p.
                    sorted_remove_mask = cumulative_probs > top_p

                    # Keep at least the first token above the threshold.
                    sorted_remove_mask[..., 1:] = sorted_remove_mask[..., :-1].clone()
                    sorted_remove_mask[..., 0] = False

                    remove_mask = torch.zeros_like(sample_logits, dtype=torch.bool)
                    remove_mask.scatter_(
                        dim=-1,
                        index=sorted_indices,
                        src=sorted_remove_mask,
                    )

                    sample_logits = sample_logits.masked_fill_(
                        remove_mask,
                        -float("Inf")
                    )

                probs = F.softmax(sample_logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(sample_logits, dim=-1, keepdim=True)

            # Update token frequency after generating the new token.
            if token_counts is not None:
                token_counts.scatter_add_(
                    dim=-1,
                    index=next_ids,
                    src=torch.ones_like(next_ids, dtype=token_counts.dtype)
                )

            token = next_ids.item()
            yield token

            # No need to run another forward after the last generated token.
            if step == max_tokens - 1:
                break

            # Decode one new token and update KV cache.
            logits = self.forward(next_ids, kv_cache=kv_cache)
            logits = logits[:, -1, :]