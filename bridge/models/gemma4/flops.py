"""Gemma 4 forward FLOPs per token, from the HF text_config, so every dense size (E4B, 12B, ...) is counted the same way.

per layer   Q / K / V / O projections (K/V skipped on the KV-shared layers),
            causal attention (sliding layers see at most sliding_window keys, full layers use global_head_dim),
            gated MLP, Per-Layer-Embedding gate + projection
per model   PLE model projection and LM head
"""


def avg_context(seq_len, window=None):
    """Average number of keys a query attends to under a causal, optionally windowed, mask."""
    if window is None or window >= seq_len:
        return (seq_len + 1) / 2
    return (window * (window + 1) / 2 + (seq_len - window) * window) / seq_len


def forward_flops_per_token(c, seq_len):
    if c.get("num_experts"):
        raise NotImplementedError("Gemma 4 MoE FLOPs are not counted yet")
    hidden, heads, layers = (c["hidden_size"], c["num_attention_heads"], c["num_hidden_layers"])
    ple, shared = (c.get("hidden_size_per_layer_input") or 0, c.get("num_kv_shared_layers") or 0)
    flops = 0.0
    for i, kind in enumerate(c["layer_types"]):
        full = kind == "full_attention"
        hd = (c.get("global_head_dim") or c["head_dim"]) if full else c["head_dim"]
        kv = (c.get("num_global_key_value_heads") if full else None) or c["num_key_value_heads"]
        flops += 2 * 2 * hidden * heads * hd  # Q, O
        if i < layers - shared:
            flops += 2 * 2 * hidden * kv * hd  # K, V
        flops += (
            2 * 2 * heads * hd * avg_context(seq_len, None if full else c["sliding_window"])
        )  # QK^T, PV
        flops += 3 * 2 * hidden * c["intermediate_size"]  # gate, up, down
        flops += 2 * 2 * hidden * ple  # PLE gate + projection
    flops += 2 * hidden * layers * ple  # PLE model projection
    flops += 2 * hidden * c["vocab_size"]  # LM head
    return flops
