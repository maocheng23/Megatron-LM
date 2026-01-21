# This module provides deterministic routing that matches SGLang's Qwen3MoE
# when rl_on_policy_target is set.
#
# SGLang's patched Qwen3MoE flow (when rl_on_policy_target is set):
#   1. gate(x) -> router_logits (ReplicatedLinear)
#   2. Pure PyTorch: F.softmax -> torch.topk -> renormalize
#
# NOTE: SGLang's patch explicitly bypasses fused_topk for on-policy training,
# using pure PyTorch instead for deterministic behavior. We match this exactly.
#
# Usage: Set use_sglang_router=True in TransformerConfig

from typing import Optional, Tuple

import torch

from .true_on_policy_config import TrueOnPolicyConfig


# =============================================================================
# Router Implementation for Qwen3-MoE
# =============================================================================

def router_qwen3_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk: int,
    renormalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Router matching SGLang's Qwen3MoE when rl_on_policy_target is set.

    This matches the SGLang patch behavior exactly:
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

    NOTE: We do NOT use fused_topk here because SGLang's patch explicitly
    bypasses fused_topk when rl_on_policy_target is set, using pure PyTorch
    instead for deterministic behavior.

    Args:
        hidden_states: [num_tokens, hidden_size]
        router_logits: [num_tokens, num_experts] (pre-computed)
        topk: Number of experts to select
        renormalize: Whether to renormalize topk weights

    Returns:
        topk_weights: [num_tokens, topk]
        topk_ids: [num_tokens, topk]
    """
    # Match SGLang patch: use pure PyTorch for deterministic on-policy routing
    # SGLang patch does:
    #   routing_weights = F.softmax(..., dtype=torch.float)  # float32
    #   routing_weights, selected_experts = torch.topk(
    #       routing_weights, self.top_k, dim=-1)
    #   routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    #   routing_weights = routing_weights.to(hidden_states.dtype)  # FusedMoE
    #
    # We keep everything in float32 for Megatron since
    # convert_topk_to_megatron_format handles dtype conversion as needed.

    # Step 1: Softmax with float32
    routing_weights = torch.softmax(router_logits.float(), dim=-1)

    # Step 2: TopK selection
    topk_weights, topk_ids = torch.topk(routing_weights, k=topk, dim=-1)

    # Step 3: Renormalize
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Return float32 weights (convert_topk_to_megatron_format handles dtype)
    return topk_weights, topk_ids.int()


# =============================================================================
# Main API
# =============================================================================

def fused_moe_router_deterministic(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
    config: Optional[TrueOnPolicyConfig] = None,
    layer_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic router for Qwen3-MoE that matches SGLang's on-policy patch.

    Flow (matching SGLang's patched Qwen3MoE when rl_on_policy_target is set):
        1. Compute logits: hidden_states @ router_weight.T
        2. Pure PyTorch routing: softmax -> topk -> renormalize

    This uses pure PyTorch to match SGLang's deterministic on-policy
    behavior, NOT the fused_topk kernel which is bypassed when
    rl_on_policy_target is set.

    Args:
        hidden_states: [num_tokens, hidden_size]
        router_weight: [num_experts, hidden_size]
        topk: Number of experts to select
        moe_softcapping: Softcapping value (0 = disabled, Qwen3 doesn't use it)
        correction_bias: Optional bias (Qwen3 doesn't use it)
        config: TrueOnPolicyConfig (required)
        layer_id: Optional layer ID for debugging

    Returns:
        topk_weights: [num_tokens, topk]
        topk_ids: [num_tokens, topk]
    """
    if config is None:
        raise ValueError(
            "TrueOnPolicyConfig is required. "
            "Please set --true-on-policy-model (e.g., 'qwen3_moe')."
        )

    # Check that we're using a supported model
    supported_models = ["qwen3_moe", "qwen3-moe", "qwen3moe"]
    model_name = config.model_name.lower().replace("-", "_")
    if model_name not in [m.replace("-", "_") for m in supported_models]:
        raise NotImplementedError(
            f"Model '{config.model_name}' not implemented. "
            f"Currently only Qwen3-MoE is supported."
        )

    # Step 1: Compute router logits (like SGLang's gate)
    # NOTE: SGLang computes gate in model's native dtype (bfloat16/float16),
    # then topk_softmax kernel converts to float32 internally.
    # We must NOT convert to float32 here to match SGLang's behavior.
    router_logits = hidden_states @ router_weight.t()

    # Step 2: Apply softcapping if needed (Qwen3 doesn't use this)
    if moe_softcapping != 0:
        router_logits = torch.tanh(
            router_logits / moe_softcapping
        ) * moe_softcapping

    # Step 3: Apply correction bias if provided (Qwen3 doesn't use this)
    if correction_bias is not None:
        router_logits = router_logits + correction_bias.float()

    # Step 4: Apply pure PyTorch routing (matching SGLang's on-policy patch)
    topk_weights, topk_ids = router_qwen3_moe(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk=topk,
        renormalize=config.renormalize,
    )

    # Debug: print router results for comparison with SGLang (only first 2 layers, rank 1)
    import os
    import sys
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    should_log = (
        os.environ.get("SLIME_DEBUG_ROUTER", "0") == "1"
        and (layer_id is None or layer_id < 2)
        and rank == 1
    )
    if should_log:
        idx = 91  # Position to debug
        if router_logits.shape[0] > idx:
            layer_str = f"layer_id={layer_id}" if layer_id is not None else ""
            print(f"[Megatron Router] {layer_str}, pos={idx}, rank={rank}",
                  file=sys.stderr, flush=True)
            print(f"  hidden_states[{idx},:5]: {hidden_states[idx, :5].tolist()}",
                  file=sys.stderr, flush=True)
            print(f"  router_weight[:2,:5]: {router_weight[:2, :5].tolist()}",
                  file=sys.stderr, flush=True)
            print(f"  router_logits[{idx}]: {router_logits[idx, :].tolist()}",
                  file=sys.stderr, flush=True)
            print(f"  topk_weights[{idx}]: {topk_weights[idx, :].tolist()}",
                  file=sys.stderr, flush=True)
            print(f"  topk_ids[{idx}]: {topk_ids[idx, :].tolist()}",
                  file=sys.stderr, flush=True)

    return topk_weights, topk_ids


# =============================================================================
# Format Conversion: SGLang -> Megatron
# =============================================================================

def convert_topk_to_megatron_format(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert SGLang format to Megatron format."""
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
