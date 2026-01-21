# This module provides utilities for deterministic routing that matches SGLang.
#
# The main routing logic is now in router.py (_sglang_router_forward),
# which directly implements the same flow as SGLang's Qwen3MoE when
# rl_on_policy_target is set:
#   1. gate(x) -> router_logits
#   2. Pure PyTorch: F.softmax -> torch.topk -> renormalize

from typing import Tuple

import torch


# =============================================================================
# Format Conversion: SGLang -> Megatron
# =============================================================================

def convert_topk_to_megatron_format(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert SGLang format to Megatron format.
    
    Args:
        topk_weights: [num_tokens, topk] - weights for selected experts
        topk_ids: [num_tokens, topk] - indices of selected experts
        num_experts: total number of experts
        dtype: output dtype (defaults to topk_weights.dtype)
    
    Returns:
        routing_probs: [num_tokens, num_experts] - sparse routing probabilities
        routing_map: [num_tokens, num_experts] - boolean mask of selected experts
    """
    num_tokens, topk = topk_weights.shape
    device = topk_weights.device
    dtype = dtype or topk_weights.dtype

    routing_probs = torch.zeros(
        (num_tokens, num_experts), dtype=dtype, device=device
    )
    routing_map = torch.zeros(
        (num_tokens, num_experts), dtype=torch.bool, device=device
    )

    routing_probs.scatter_(1, topk_ids.long(), topk_weights.to(dtype))
    routing_map.scatter_(
        1, topk_ids.long(), torch.ones_like(topk_weights, dtype=torch.bool)
    )

    return routing_probs, routing_map


# =============================================================================
# Utility
# =============================================================================

def is_sglang_router_available() -> bool:
    """
    Check if deterministic routing is available.

    Since we now use pure PyTorch (matching SGLang's on-policy patch),
    this is always available.
    """
    return True
