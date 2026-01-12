# This module provides deterministic routing by directly using SGLang's implementations:
# - fused_moe_router_cudacore: Fused GEMM + Softcap + TopK kernel (CUDA)
# - fused_topk_torch_native: PyTorch fallback for CPU/non-CUDA
#
# Usage: Set use_sglang_router=True in TransformerConfig
#
# Reference: sglang/python/sglang/srt/layers/moe/router.py
#            sglang/python/sglang/srt/layers/moe/topk.py

from typing import Optional, Tuple

import torch

# =============================================================================
# Import SGLang functions (use directly when available)
# =============================================================================

# Try to import SGLang's fused router
try:
    from sglang.srt.layers.moe.router import fused_moe_router_cudacore
    HAVE_SGLANG_FUSED_ROUTER = True
except ImportError:
    HAVE_SGLANG_FUSED_ROUTER = False
    fused_moe_router_cudacore = None

# Try to import SGLang's topk functions (for fallback)
try:
    from sglang.srt.layers.moe.topk import fused_topk_torch_native
    HAVE_SGLANG_TOPK = True
except ImportError:
    HAVE_SGLANG_TOPK = False
    fused_topk_torch_native = None


# =============================================================================
# Main API: fused_moe_router_deterministic
# =============================================================================

def fused_moe_router_deterministic(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(hidden_states.shape) == 2, f"Expected 2D input, got {hidden_states.shape}"
    assert hidden_states.shape[1] == router_weight.shape[1], "Hidden dim mismatch"
    
    # Use SGLang's fused kernel when available and on CUDA
    if HAVE_SGLANG_FUSED_ROUTER and hidden_states.is_cuda:
        return fused_moe_router_cudacore(
            x=hidden_states,
            router_weight=router_weight,
            topk=topk,
            moe_softcapping=moe_softcapping,
            correction_bias=correction_bias,
        )
    
    # Fallback: compute GEMM + softcap manually, then use SGLang's topk
    logits = hidden_states.float() @ router_weight.float().t()
    
    # Apply softcapping (SGLang formula)
    if moe_softcapping != 0:
        logits = torch.tanh(logits / moe_softcapping) * moe_softcapping
    
    # Add bias after softcapping
    if correction_bias is not None:
        logits = logits + correction_bias.float()
    
    # Use SGLang's torch_native topk if available
    if HAVE_SGLANG_TOPK:
        topk_weights, topk_ids = fused_topk_torch_native(
            hidden_states=hidden_states,
            gating_output=logits,
            topk=topk,
            renormalize=False,  # SGLang's router doesn't renormalize
            correction_bias=None,  # Already applied above
            scoring_func="softmax",
        )
        return topk_weights.float(), topk_ids.int()
    
    # Minimal fallback when SGLang is completely unavailable
    scores = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)
    return topk_weights.float(), topk_ids.int()


# =============================================================================
# Format Conversion: SGLang -> Megatron
# =============================================================================

def convert_topk_to_megatron_format(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens, topk = topk_weights.shape
    device = topk_weights.device
    dtype = dtype or topk_weights.dtype
    
    routing_probs = torch.zeros((num_tokens, num_experts), dtype=dtype, device=device)
    routing_map = torch.zeros((num_tokens, num_experts), dtype=torch.bool, device=device)
    
    routing_probs.scatter_(1, topk_ids.long(), topk_weights.to(dtype))
    routing_map.scatter_(1, topk_ids.long(), torch.ones_like(topk_weights, dtype=torch.bool))
    
    return routing_probs, routing_map


# =============================================================================
# Utility
# =============================================================================

def is_sglang_router_available() -> bool:
    """Check if SGLang's fused router is available."""
    return HAVE_SGLANG_FUSED_ROUTER
