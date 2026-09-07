"""Graph-bucket scheduling hint, never an expert-capacity limit."""


def expected_m_for_tokens(local_tokens: int, ep_size: int, topk: int, experts: int) -> int:
    if local_tokens < 0 or min(ep_size, topk, experts) <= 0:
        raise ValueError("invalid MoE scheduling geometry")
    return max(1, (local_tokens * ep_size * topk + experts - 1) // experts)
