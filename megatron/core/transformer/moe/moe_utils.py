# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import functools
import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

logger = logging.getLogger(__name__)

from megatron.core import parallel_state
from megatron.core.fp4_utils import get_fp4_align_size
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from megatron.core.tensor_parallel.mappings import _tree_all_reduce_sum, _tree_all_reduce_sum_impl
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import internal_api

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import (
        fused_compute_score_for_moe_aux_loss,
        fused_moe_aux_loss,
        fused_permute,
        fused_permute_with_probs,
        fused_sort_chunks_by_index,
        fused_sort_chunks_by_index_with_probs,
        fused_topk_with_score_function,
        fused_unpermute,
        te_general_gemm,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

# Try to import SGLang's fused_experts for deterministic expert computation
try:
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts_impl
    HAVE_SGLANG_FUSED_EXPERTS = True

    # Initialize SGLang server args if not already set (needed for fused_experts_impl)
    # Only set deterministic mode if MEGATRON_TRUE_ON_POLICY=1 is set
    from sglang.srt.server_args import get_global_server_args, set_global_server_args_for_scheduler
    try:
        get_global_server_args()
    except ValueError:
        # Server args not set - create minimal mock for Megatron usage
        use_deterministic = os.environ.get("MEGATRON_USE_DETERMINISTIC_ALLREDUCE", "0") == "1"
        print("MEGATRON_USE_DETERMINISTIC_ALLREDUCE: ", use_deterministic)
        class _MinimalServerArgs:
            enable_deterministic_inference = use_deterministic
            rl_on_policy_target = "fsdp_tp" if use_deterministic else None
        set_global_server_args_for_scheduler(_MinimalServerArgs())
except ImportError:
    HAVE_SGLANG_FUSED_EXPERTS = False
    fused_experts_impl = None

# MOE logging
_MOE_LAYER_WISE_LOGGING_TRACKER = {}


def switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
    fused: bool = False,
):
    """Calculate the auxiliary loss for load balancing.
    Refer to the Switch Transformer (https://arxiv.org/abs/2101.03961)
    and Global Load Balancing Loss(https://arxiv.org/abs/2501.11873) for details.

    ### Detailed explanation of the auxiliary loss #######

    The formula for the auxiliary loss is:
        loss = E * Σ_{i=1}^{E} (f_i * P_i)
    where:
        f_i = 1 / (T * topk) * Σ_{x∈B} routing_map(x, i)
             (fraction of tokens dispatched to expert i)
        P_i = 1 / T * Σ_{x∈B} probs(x, i)
             (averaged router probability allocated for expert i)
        E is the number of experts
        T is the total number of tokens in the batch B

    For distributed training with sequence or context parallelism, each rank can
    process a subset of the batch.
        loss = E * Σ_{i=1}^{E} (f_i * Σ_{j=1}^{N} P_ij)
             = E * Σ_{i=1}^{E} Σ_{j=1}^{N} (f_i * P_ij)
             = Σ_{j=1}^{N} E * (Σ_{i=1}^{E} f_i * P_ij)

    where:
        f_i = 1 / (T * topk) * Σ_{x∈B} routing_map(x, i)
             (fraction of tokens dispatched to expert i in the global batch)
        P_ij = 1 / T * Σ_{x∈B_j} probs(x, i)
              (averaged router probability allocated for expert i in local batch of the j-th rank)
        N is the number of ranks
        B_j is the batch of tokens in the j-th rank
        T is the total number of tokens in the global batch B

    Note:
    To calculate the auxiliary loss at different levels (micro-batch or global batch):
    - probs: Should always be from the local batch being processed
    - tokens_per_expert: Should represent token counts at the desired level
      (either micro-batch or global batch)
    - total_num_tokens: Should match the total token count at the same level as tokens_per_expert

    #########################################################

    Args:
        probs (torch.Tensor): Softmax probabilities output by the router for each token.
                              Shape in [num_tokens, num_experts].
        tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert in the batch.
                                          Shape in [num_experts]
        total_num_tokens (int): Total number of tokens in the batch.
        topk (int): The number of experts selected for each token.
        num_experts (int): The number of experts.
        moe_aux_loss_coeff (float): The coefficient for the auxiliary loss.
    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    if fused:
        if not HAVE_TE or fused_moe_aux_loss is None:
            raise ValueError("fused_moe_aux_loss is not available. Please install TE >= 2.7.0.")
        return fused_moe_aux_loss(
            probs=probs,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=topk,
            num_experts=num_experts,
            coeff=moe_aux_loss_coeff,
        )

    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )
    return aux_loss


def z_loss_func(logits, z_loss_coeff):
    """Encourages the router's logits to remain small to enhance stability.
    Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

    Args:
        logits (torch.Tensor): The logits of the router.

    Returns:
        torch.Tensor: The logits after applying the z-loss.
    """

    z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * z_loss_coeff
    return z_loss


def sinkhorn(cost: torch.Tensor, tol: float = 0.0001):
    """Sinkhorn based MoE routing function"""
    cost = torch.exp(cost)
    d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
    d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)

    eps = 0.00000001
    error = 1e9
    d1_old = d1
    while error > tol:
        d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
        d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
        error = torch.mean(torch.abs(d1_old - d1))
        d1_old = d1
    return d1 * cost * d0.unsqueeze(1)


def get_capacity(num_tokens: int, num_experts: int, capacity_factor: float, min_capacity=None):
    """
    Calculate the capacity of each expert.

    Args:
        num_tokens (int): num of the input tokens.
        num_experts (int): num of the experts.
        capacity_factor (float): Capacity factor.
        min_capacity (int, optional): Minimum capacity. Defaults to None.

    Returns:
        Tensor: Capacity of each expert.
    """
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    return capacity


class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for auxiliary loss."""

    main_loss_backward_scale: Optional[torch.Tensor] = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss
                                               gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=aux_loss.device
            )
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the aux loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            MoEAuxLossAutoScaler.main_loss_backward_scale = scale
        else:
            MoEAuxLossAutoScaler.main_loss_backward_scale.copy_(scale)


def permute(
    tokens,
    routing_map,
    probs: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    When drop_and_pad=True, in routing_map, the number of non-zeros in each column equals to
    expert capacity. This function exploits this feature to use ops that support cuda graph.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        probs (torch.Tensor, optional): The probs tensor, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
        fused (bool, optional): Whether use the fused permute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
                                       If set to true, routing_map has a fixed number of non-zeros
                                       in each column.
    """
    if fused and probs is None:
        if not HAVE_TE or fused_permute is None:
            raise ValueError("fused_permute is not available. Please install TE >= 2.1.0.")
        permuted_input, sorted_indices = fused_permute(
            tokens, routing_map, num_out_tokens=num_out_tokens
        )
        return permuted_input, None, sorted_indices

    if fused and probs is not None:
        if not HAVE_TE or fused_permute_with_probs is None:
            raise ValueError(
                "fused_permute_with_probs is not available. Please install TE >= 2.1.0."
            )
        return fused_permute_with_probs(tokens, probs, routing_map, num_out_tokens=num_out_tokens)

    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    permuted_probs = None
    if drop_and_pad and not (num_out_tokens is None):
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        # use argsort to put indices of all non-zeros in the beginning of list
        # and keep the first `capacity` number of indices
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[
            :, :capacity
        ].contiguous()
        # flatten from [num_experts, capacity] to 1D
        sorted_indices = sorted_indices.view(-1)

        if probs is not None:
            # [num_tokens, num_experts] -> num_experts * num_tokens
            probs_T_1D = probs.T.contiguous().view(-1)
            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_tokens + indices_dim1).view(-1)
            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
    else:
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.bool().T.contiguous()

        # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
        token_indices = (
            torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        )
        sorted_indices = token_indices.masked_select(routing_map)

        if probs is not None:
            permuted_probs = probs.T.contiguous().masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, sorted_indices


class FusedExpertsFunction(torch.autograd.Function):
    """Custom autograd function for fused experts.
    
    Forward: Uses SGLang's fused_experts_impl (triton kernel) for bitwise identical results
    Backward: Uses pure PyTorch operations for gradient computation
    
    IMPORTANT: In EP mode, grad_topk_weights is all-reduced across EP ranks so the router
    receives complete gradients from all experts.
    """
    
    @staticmethod
    def forward(ctx, hidden_states, w1, w2, topk_weights, topk_ids, activation, layer_id, ep_group):
        """Forward pass using SGLang's triton kernel.
        
        Args:
            ep_group: Expert parallel process group for all-reducing grad_topk_weights.
                      Can be None if EP size is 1.
        """
        # Save tensors for backward
        ctx.save_for_backward(hidden_states, w1, w2, topk_weights, topk_ids)
        ctx.activation = activation
        ctx.ep_group = ep_group
        ctx.layer_id = layer_id
        
        # Use SGLang's fused_experts_impl for bitwise identical forward
        with torch.no_grad():
            output = fused_experts_impl(
                hidden_states=hidden_states.contiguous(),
                w1=w1.contiguous(),
                w2=w2.contiguous(),
                topk_weights=topk_weights.contiguous(),
                topk_ids=topk_ids.contiguous(),
                inplace=False,
                activation=activation,
                is_gated=True,
                apply_router_weight_on_input=False,
                filter_expert=True,
                layer_id=layer_id,
            )
        
        # Mark output as requiring grad if inputs do
        return output.clone().requires_grad_(hidden_states.requires_grad or w1.requires_grad or w2.requires_grad or topk_weights.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using pure PyTorch operations.
        
        IMPORTANT: In EP mode, each rank only computes grad_topk_weights for its local experts.
        We all-reduce grad_topk_weights across EP ranks so the router receives complete gradients.
        """
        hidden_states, w1, w2, topk_weights, topk_ids = ctx.saved_tensors
        activation = ctx.activation
        ep_group = ctx.ep_group
        layer_id = ctx.layer_id
        
        # Debug: verify backward is being called and EP group info
        if os.environ.get("DEBUG_GRAD_ALLREDUCE", "0") == "1":
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if ep_group is not None:
                ep_size = torch.distributed.get_world_size(ep_group)
                ep_rank = torch.distributed.get_rank(ep_group)
                print(f"[FusedExpertsFunction.backward][Rank {rank}][Layer {layer_id}] CALLED! "
                      f"ep_group={ep_group}, ep_size={ep_size}, ep_rank={ep_rank}, "
                      f"grad_output.shape={grad_output.shape}, "
                      f"topk_weights.requires_grad={topk_weights.requires_grad}, "
                      f"hidden_states.requires_grad={hidden_states.requires_grad}")
            else:
                print(f"[FusedExpertsFunction.backward][Rank {rank}][Layer {layer_id}] CALLED! "
                      f"ep_group=None, grad_output.shape={grad_output.shape}, "
                      f"topk_weights.requires_grad={topk_weights.requires_grad}")

        # DEBUG: Log grad_output magnitude for diagnosis
        if os.environ.get("DEBUG_EXPERT_GRAD_MAGNITUDE", "0") == "1" and layer_id >= 46:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            grad_out_norm = grad_output.float().norm().item()
            grad_out_sum = grad_output.float().sum().item()
            print(f"[FusedExpertsFunction.backward][Rank {rank}][Layer {layer_id}] "
                  f"grad_output: norm={grad_out_norm:.6e}, sum={grad_out_sum:.6e}, "
                  f"shape={grad_output.shape}, dtype={grad_output.dtype}")
        
        num_tokens, hidden_size = hidden_states.shape
        num_local_experts, ffn_hidden_size, _ = w1.shape  # In EP mode, w1 only has local experts!
        topk = topk_ids.shape[1]
        
        # Get EP info to compute global expert IDs
        if ep_group is not None:
            ep_world_size = torch.distributed.get_world_size(ep_group)
            ep_rank = torch.distributed.get_rank(ep_group)
        else:
            ep_world_size = 1
            ep_rank = 0
        
        # Initialize gradients
        grad_hidden_states = torch.zeros_like(hidden_states) if hidden_states.requires_grad else None
        grad_w1 = torch.zeros_like(w1) if w1.requires_grad else None
        grad_w2 = torch.zeros_like(w2) if w2.requires_grad else None
        # Initialize grad_topk_weights for router gradient
        # NOTE: Set DISABLE_ROUTER_GRAD=1 to disable router gradient computation for debugging
        # This helps isolate whether the router gradient computation is causing divergence
        if os.environ.get("DISABLE_ROUTER_GRAD", "0") == "1":
            grad_topk_weights = None  # Disable router gradient for debugging
        else:
            grad_topk_weights = torch.zeros_like(topk_weights) if topk_weights.requires_grad else None
        
        # Store expert outputs for grad_topk_weights calculation
        # expert_outputs[expert_id] = dict mapping token_index -> expert_output_vector
        expert_outputs_cache = {}
        
        # Process each LOCAL expert
        # IMPORTANT: topk_ids saved in ctx.save_for_backward is LOCAL expert IDs (from topk_ids_local)!
        # In EP mode, topk_ids_local contains:
        # - 0, 1, ..., num_local_experts-1 for tokens that selected this rank's experts
        # - -1 for tokens that selected other ranks' experts
        # So we match against LOCAL expert ID, not global!
        
        # DEBUG: Check topk_ids distribution in backward
        if os.environ.get("DEBUG_EXPERT_GRAD", "0") == "1":
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            unique_ids, counts = topk_ids.unique(return_counts=True)
            print(f"[FusedExpertsFunction.backward][Rank {rank}][Layer {layer_id}] topk_ids distribution:")
            print(f"  unique_ids: {unique_ids.tolist()}, counts: {counts.tolist()}")
            print(f"  num_local_experts: {num_local_experts}, topk_ids.shape: {topk_ids.shape}")
            local_expert_count = (topk_ids >= 0).sum().item()
            remote_expert_count = (topk_ids == -1).sum().item()
            print(f"  local_expert_selections: {local_expert_count}, remote_expert_selections: {remote_expert_count}")
        
        for local_expert_id in range(num_local_experts):
            # Match against LOCAL expert ID (topk_ids contains local IDs!)
            mask = (topk_ids == local_expert_id)
            if not mask.any():
                # DEBUG: Log skipped experts
                if os.environ.get("DEBUG_EXPERT_GRAD", "0") == "1":
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    if local_expert_id < 3:  # Only log first few to reduce noise
                        print(f"[FusedExpertsFunction.backward][Rank {rank}][Layer {layer_id}] "
                              f"Skipping local_expert_id={local_expert_id} (no tokens selected it)")
                continue
            
            token_indices = mask.any(dim=1).nonzero(as_tuple=True)[0]
            if len(token_indices) == 0:
                continue
            
            expert_input = hidden_states[token_indices]
            expert_w1 = w1[local_expert_id]  # Use local ID to index into local weights
            expert_w2 = w2[local_expert_id]
            
            # Forward pass (recompute for backward)
            gate_up = torch.nn.functional.linear(expert_input, expert_w1)
            gate_pre, up = gate_up.chunk(2, dim=-1)
            
            if activation == "silu":
                gate = torch.nn.functional.silu(gate_pre)
            elif activation == "gelu":
                gate = torch.nn.functional.gelu(gate_pre)
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            
            intermediate = gate * up
            
            # Compute expert output (before weighting) for grad_topk_weights
            # expert_output = linear(intermediate, expert_w2)
            expert_output = torch.nn.functional.linear(intermediate, expert_w2)
            
            # Cache for grad_topk_weights computation (use local ID as key)
            if grad_topk_weights is not None:
                expert_outputs_cache[local_expert_id] = (token_indices, expert_output)
            
            # Compute weighted grad_output for this expert
            expert_grad_output = torch.zeros(len(token_indices), hidden_size,
                                            dtype=grad_output.dtype, device=grad_output.device)
            for slot in range(topk):
                slot_mask = mask[token_indices, slot]
                if slot_mask.any():
                    slot_token_indices = token_indices[slot_mask]
                    slot_weights = topk_weights[slot_token_indices, slot].unsqueeze(-1)
                    # Map back to expert_grad_output indices
                    local_indices = slot_mask.nonzero(as_tuple=True)[0]
                    expert_grad_output[local_indices] += grad_output[slot_token_indices] * slot_weights

            # DEBUG: Compare gradient computation inputs between on-policy and off-policy
            if os.environ.get("DEBUG_GRAD_COMPARE", "0") == "1" and layer_id >= 46 and local_expert_id == 0:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                if rank == 0:
                    print(f"\n[DEBUG_GRAD_COMPARE][Layer {layer_id}][Expert {local_expert_id}] ===== GRADIENT INPUTS =====")
                    print(f"  num_tokens_for_expert: {len(token_indices)}")
                    print(f"  topk (num slots): {topk}")
                    print(f"  topk_weights.shape: {topk_weights.shape}, dtype: {topk_weights.dtype}")
                    print(f"  topk_weights[token_indices].shape: {topk_weights[token_indices].shape}")
                    # Show sample routing weights for first 3 selected tokens
                    sample_weights = topk_weights[token_indices][:min(3, len(token_indices))]
                    print(f"  topk_weights sample (first 3 tokens): {sample_weights.tolist()}")
                    print(f"  topk_weights per-token sum (first 3): {sample_weights.sum(dim=-1).tolist()}")
                    print(f"  topk_weights (all selected) mean: {topk_weights[token_indices].float().mean().item():.6f}")
                    print(f"  topk_weights (all selected) sum: {topk_weights[token_indices].float().sum().item():.6f}")
                    print(f"  grad_output.shape: {grad_output.shape}")
                    print(f"  grad_output norm (all): {grad_output.float().norm().item():.6e}")
                    print(f"  grad_output norm (selected): {grad_output[token_indices].float().norm().item():.6e}")
                    print(f"  expert_grad_output norm: {expert_grad_output.float().norm().item():.6e}")
                    print(f"  intermediate norm: {intermediate.float().norm().item():.6e}")
                    print(f"  expert_input norm: {expert_input.float().norm().item():.6e}")
                    # The actual gradient contribution
                    grad_w2_contrib = expert_grad_output.T @ intermediate
                    print(f"  grad_w2 contribution norm: {grad_w2_contrib.float().norm().item():.6e}")
            
            # Backward through down projection: expert_output = linear(intermediate, expert_w2)
            # linear computes: intermediate @ expert_w2.T
            # expert_w2 shape: [hidden_size, ffn_hidden_size//2] = [2048, 768]
            # grad_intermediate = grad_expert_output @ expert_w2 (not w2.T!)
            # grad_w2 = grad_expert_output.T @ intermediate
            grad_intermediate = expert_grad_output @ expert_w2.to(expert_grad_output.dtype)
            if grad_w2 is not None:
                grad_w2[local_expert_id] += expert_grad_output.T.to(grad_w2.dtype) @ intermediate.to(grad_w2.dtype)

            # DEBUG: Log intermediate gradient magnitudes for first expert of last layer
            if os.environ.get("DEBUG_EXPERT_GRAD_MAGNITUDE", "0") == "1" and layer_id >= 46 and local_expert_id == 0:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                print(f"[FusedExpertsFunction.backward][Rank {rank}][Layer {layer_id}][Expert {local_expert_id}] "
                      f"num_tokens={len(token_indices)}, "
                      f"expert_grad_output_norm={expert_grad_output.float().norm().item():.6e}, "
                      f"intermediate_norm={intermediate.float().norm().item():.6e}, "
                      f"grad_w2_contrib_norm={(expert_grad_output.T @ intermediate).float().norm().item():.6e}")
            
            # Backward through element-wise multiply: intermediate = gate * up
            grad_gate = grad_intermediate * up
            grad_up = grad_intermediate * gate
            
            # Backward through activation
            if activation == "silu":
                sigmoid_gate_pre = torch.sigmoid(gate_pre)
                grad_gate_pre = grad_gate * sigmoid_gate_pre * (1 + gate_pre * (1 - sigmoid_gate_pre))
            elif activation == "gelu":
                # Use autograd for gelu backward
                gate_pre_detached = gate_pre.detach().requires_grad_(True)
                with torch.enable_grad():
                    gate_recompute = torch.nn.functional.gelu(gate_pre_detached)
                    grad_gate_pre = torch.autograd.grad(
                        gate_recompute, gate_pre_detached, grad_gate
                    )[0]
            
            # Combine grad_gate_pre and grad_up back to grad_gate_up
            grad_gate_up = torch.cat([grad_gate_pre, grad_up], dim=-1)
            
            # Backward through gate_up projection: gate_up = linear(expert_input, expert_w1)
            # linear computes: expert_input @ expert_w1.T
            # expert_w1 shape: [ffn_hidden_size, hidden_size]
            # grad_expert_input = grad_gate_up @ expert_w1
            # grad_w1 = grad_gate_up.T @ expert_input
            if grad_hidden_states is not None:
                grad_expert_input = grad_gate_up @ expert_w1.to(grad_gate_up.dtype)
                grad_hidden_states[token_indices] += grad_expert_input.to(grad_hidden_states.dtype)
            
            if grad_w1 is not None:
                grad_w1[local_expert_id] += grad_gate_up.T.to(grad_w1.dtype) @ expert_input.to(grad_w1.dtype)
        
        # Compute grad_topk_weights
        # output[i] = sum_k(topk_weights[i,k] * expert_k(input[i]))
        # d(output[i]) / d(topk_weights[i,k]) = expert_k(input[i])
        # grad_topk_weights[i,k] = sum_j(grad_output[i,j] * expert_output_k[i,j])
        #                       = (grad_output[i] · expert_output_k[i])
        if grad_topk_weights is not None:
            for local_expert_id, (token_indices, expert_output) in expert_outputs_cache.items():
                mask = (topk_ids == local_expert_id)  # Match against LOCAL expert ID!
                for slot in range(topk):
                    slot_mask = mask[token_indices, slot]
                    if slot_mask.any():
                        slot_token_indices = token_indices[slot_mask]
                        local_indices = slot_mask.nonzero(as_tuple=True)[0]
                        # grad_topk_weights[token_idx, slot] = dot(grad_output[token_idx], expert_output[local_idx])
                        expert_out_selected = expert_output[local_indices]  # [num_selected, hidden_size]
                        grad_out_selected = grad_output[slot_token_indices]  # [num_selected, hidden_size]
                        # Element-wise multiply and sum over hidden dimension
                        grad_weights = (grad_out_selected * expert_out_selected).sum(dim=-1)  # [num_selected]
                        grad_topk_weights[slot_token_indices, slot] += grad_weights.to(grad_topk_weights.dtype)
            
            # CRITICAL: All-reduce grad_topk_weights across EP ranks
            # Each rank only computes gradients for its local experts. Without all-reduce,
            # the router would receive partial gradients and weights would diverge across ranks.
            # NOTE: We use SUM all-reduce because each rank computes a PARTIAL gradient
            # (only for tokens that selected its local experts). The sum gives the total gradient.
            # This is NOT like data parallel where we average - here we sum because each rank
            # has a disjoint contribution.
            if ep_group is not None:
                ep_world_size = torch.distributed.get_world_size(ep_group)
                ep_rank_in_group = torch.distributed.get_rank(ep_group)
                
                # ALWAYS log all-reduce status (not just with debug flag)
                if os.environ.get("DEBUG_GRAD_SYNC", "0") == "1":
                    rank = torch.distributed.get_rank()
                    print(f"[FusedExpertsFunction][Rank {rank}] ep_group info: "
                          f"ep_world_size={ep_world_size}, ep_rank_in_group={ep_rank_in_group}, "
                          f"will_allreduce={ep_world_size > 1}")
                
                if ep_world_size > 1:
                    # Debug: log before all-reduce
                    if os.environ.get("DEBUG_GRAD_ALLREDUCE", "0") == "1":
                        rank = torch.distributed.get_rank()
                        print(f"[FusedExpertsFunction][Rank {rank}] BEFORE grad_topk_weights all-reduce: "
                              f"sum={grad_topk_weights.sum().item():.6e}, norm={grad_topk_weights.norm().item():.6e}")
                    
                    # CRITICAL: Use deterministic tree all-reduce to match forward pass
                    # Standard NCCL all-reduce can cause non-determinism due to floating-point
                    # accumulation order differences, leading to router weight divergence across ranks
                    # NOTE: Use _tree_all_reduce_sum_impl directly (not _tree_all_reduce_sum) because
                    # we're in backward context and don't need autograd support
                    if os.environ.get("MEGATRON_USE_DETERMINISTIC_ALLREDUCE", "0") == "1":
                        grad_topk_weights_reduced = _tree_all_reduce_sum_impl(grad_topk_weights, ep_group)
                        grad_topk_weights.copy_(grad_topk_weights_reduced)
                    else:
                        torch.distributed.all_reduce(grad_topk_weights, group=ep_group)
                    
                    # Debug: log after all-reduce
                    if os.environ.get("DEBUG_GRAD_ALLREDUCE", "0") == "1" and layer_id <= 1:
                        print(f"[FusedExpertsFunction][Rank {rank}] AFTER grad_topk_weights all-reduce: "
                              f"sum={grad_topk_weights.sum().item():.6e}, norm={grad_topk_weights.norm().item():.6e}")
                    
                    # CRITICAL DEBUG: Verify all ranks have identical grad_topk_weights after all-reduce
                    # NOTE: Sums can be all zeros on some steps (e.g. pipeline fill/drain, or when
                    # grad_output into this layer is zero); non-zero when gradient flows normally.
                    if os.environ.get("DEBUG_GRAD_SYNC", "0") == "1" and layer_id <= 1:
                        # All-gather the sum from all ranks to verify they're identical
                        local_sum = torch.tensor([grad_topk_weights.sum().item()], device=grad_topk_weights.device)
                        all_sums = [torch.zeros_like(local_sum) for _ in range(ep_world_size)]
                        torch.distributed.all_gather(all_sums, local_sum, group=ep_group)
                        sums = [s.item() for s in all_sums]
                        max_diff = max(sums) - min(sums)
                        if rank == 0:
                            print(f"[FusedExpertsFunction][DEBUG_GRAD_SYNC][Layer {layer_id}] grad_topk_weights sums across ranks: {sums}")
                            print(f"[FusedExpertsFunction][DEBUG_GRAD_SYNC][Layer {layer_id}] max_diff={max_diff:.6e}")
                            if max_diff > 1e-6:
                                print(f"[FusedExpertsFunction][DEBUG_GRAD_SYNC][Layer {layer_id}] WARNING: grad_topk_weights DIFFERS across ranks!")
            else:
                if os.environ.get("DEBUG_GRAD_ALLREDUCE", "0") == "1":
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    print(f"[FusedExpertsFunction][Rank {rank}] WARNING: ep_group is None, skipping grad_topk_weights all-reduce!")
        
        # CRITICAL: All-reduce grad_hidden_states across EP ranks
        # Each rank only computes gradients from its local experts. The full gradient
        # is the sum of contributions from all experts across all ranks.
        # NOTE: Set DISABLE_HIDDEN_GRAD_ALLREDUCE=1 to disable this all-reduce for debugging.
        # This will cause INCORRECT upstream gradients but helps isolate divergence sources.
        if grad_hidden_states is not None:
            if os.environ.get("DISABLE_HIDDEN_GRAD_ALLREDUCE", "0") == "1":
                # Skip all-reduce for debugging - upstream weights will get wrong gradients!
                if os.environ.get("DEBUG_GRAD_ALLREDUCE", "0") == "1":
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    print(f"[FusedExpertsFunction][Rank {rank}] SKIP grad_hidden_states all-reduce (DISABLE_HIDDEN_GRAD_ALLREDUCE=1)")
            elif ep_group is not None:
                ep_world_size = torch.distributed.get_world_size(ep_group)
                if ep_world_size > 1:
                    # Debug: log before all-reduce
                    if os.environ.get("DEBUG_GRAD_ALLREDUCE", "0") == "1":
                        rank = torch.distributed.get_rank()
                        print(f"[FusedExpertsFunction][Rank {rank}] BEFORE grad_hidden_states all-reduce: "
                              f"sum={grad_hidden_states.sum().item():.6e}, norm={grad_hidden_states.norm().item():.6e}")
                    
                    # CRITICAL: Use deterministic tree all-reduce to match forward pass
                    # This ensures gradient accumulation order is consistent across ranks
                    # NOTE: Use _tree_all_reduce_sum_impl directly (not _tree_all_reduce_sum) because
                    # we're in backward context and don't need autograd support
                    if os.environ.get("MEGATRON_USE_DETERMINISTIC_ALLREDUCE", "0") == "1":
                        grad_hidden_states_reduced = _tree_all_reduce_sum_impl(grad_hidden_states, ep_group)
                        grad_hidden_states.copy_(grad_hidden_states_reduced)
                    else:
                        torch.distributed.all_reduce(grad_hidden_states, group=ep_group)
                    
                    # Debug: log after all-reduce
                    if os.environ.get("DEBUG_GRAD_ALLREDUCE", "0") == "1":
                        print(f"[FusedExpertsFunction][Rank {rank}] AFTER grad_hidden_states all-reduce: "
                              f"sum={grad_hidden_states.sum().item():.6e}, norm={grad_hidden_states.norm().item():.6e}")
                    
                    # CRITICAL DEBUG: Verify all ranks have identical grad_hidden_states after all-reduce
                    if os.environ.get("DEBUG_GRAD_SYNC", "0") == "1" and layer_id in [46, 47]:
                        local_sum = torch.tensor([grad_hidden_states.sum().item()], device=grad_hidden_states.device)
                        all_sums = [torch.zeros_like(local_sum) for _ in range(ep_world_size)]
                        torch.distributed.all_gather(all_sums, local_sum, group=ep_group)
                        sums = [s.item() for s in all_sums]
                        max_diff = max(sums) - min(sums)
                        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                        if rank == 0:
                            print(f"[FusedExpertsFunction][DEBUG_GRAD_SYNC][Layer {layer_id}] grad_hidden_states sums across ranks: {sums}")
                            print(f"[FusedExpertsFunction][DEBUG_GRAD_SYNC][Layer {layer_id}] max_diff={max_diff:.6e}")
                            if max_diff > 1e-6:
                                print(f"[FusedExpertsFunction][DEBUG_GRAD_SYNC][Layer {layer_id}] WARNING: grad_hidden_states DIFFERS across ranks!")
            else:
                if os.environ.get("DEBUG_GRAD_ALLREDUCE", "0") == "1":
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    print(f"[FusedExpertsFunction][Rank {rank}] WARNING: ep_group is None, skipping grad_hidden_states all-reduce!")
        
        # DEBUG: Only update experts for a specific layer
        # Set ONLY_UPDATE_LAYER_EXPERTS=<layer_id> to only update that layer's experts
        # This helps isolate expert weight sync issues by testing with a single layer
        # For last layer: set to (num_layers - 1), e.g., ONLY_UPDATE_LAYER_EXPERTS=27 for 28 layers
        #
        # Combined with DISABLE_ROUTER_GRAD=1, this ensures:
        # - Only the specified layer's experts update
        # - Router doesn't update
        # - Upstream layers don't update (grad_hidden_states zeroed)
        #
        # ONLY_UPDATE_ENTIRE_LAYER=<layer_spec>: Updates ENTIRE layers (attention + MoE + layernorm)
        # Supports: single "47", comma-separated "46,47", or range "40-47"
        # - For target layers: grad_hidden_states is NOT zeroed (so attention gets gradient)
        # - For non-target layers: grad_hidden_states is zeroed
        
        only_update_entire_layer = os.environ.get("ONLY_UPDATE_ENTIRE_LAYER", None)
        only_update_layer = os.environ.get("ONLY_UPDATE_LAYER_EXPERTS", None)
        
        def parse_layer_spec(spec_str):
            """Parse layer specification: '47', '46,47', '40-47', or '40-47,0'"""
            layers = set()
            for part in spec_str.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = part.split('-')
                    layers.update(range(int(start), int(end) + 1))
                else:
                    layers.add(int(part))
            return layers
        
        # Determine target layers and mode
        target_layers_set = None
        entire_layer_mode = False
        if only_update_entire_layer is not None:
            target_layers_set = parse_layer_spec(only_update_entire_layer)
            entire_layer_mode = True
        elif only_update_layer is not None:
            target_layers_set = {int(only_update_layer)}
            entire_layer_mode = False
        
        if target_layers_set is not None:
            is_target_layer = (layer_id in target_layers_set)
            
            if not is_target_layer:
                # Zero out expert gradients for non-target layers
                if grad_w1 is not None:
                    grad_w1.zero_()
                if grad_w2 is not None:
                    grad_w2.zero_()
                if os.environ.get("DEBUG_GRAD_ALLREDUCE", "0") == "1":
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    mode_str = "ONLY_UPDATE_ENTIRE_LAYER" if entire_layer_mode else "ONLY_UPDATE_LAYER_EXPERTS"
                    print(f"[FusedExpertsFunction][Rank {rank}][Layer {layer_id}] "
                          f"ZEROING expert gradients ({mode_str}={sorted(target_layers_set)})")

            # Decide whether to zero grad_hidden_states
            # - MoE-only mode: zero for ALL layers (prevent upstream layers from updating)
            # - Entire-layer mode: zero only for NON-target layers (target layer's attention needs gradient)
            #   Also zero for the LOWEST target layer to prevent gradient flowing to upstream layers
            should_zero_hidden_grad = False
            if entire_layer_mode:
                min_target_layer = min(target_layers_set)
                # Zero for non-target layers OR for the lowest target layer (to truncate upstream)
                should_zero_hidden_grad = (not is_target_layer) or (layer_id == min_target_layer)
            else:
                # MoE-only mode: zero for all layers
                should_zero_hidden_grad = True
            
            if should_zero_hidden_grad and grad_hidden_states is not None:
                grad_hidden_states.zero_()
                if os.environ.get("DEBUG_GRAD_ALLREDUCE", "0") == "1":
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    print(f"[FusedExpertsFunction][Rank {rank}][Layer {layer_id}] "
                          f"ZEROING grad_hidden_states (isolating layer update)")

        # DEBUG: Check final gradient values before returning
        if os.environ.get("DEBUG_EXPERT_GRAD", "0") == "1" or (os.environ.get("DEBUG_EXPERT_GRAD_MAGNITUDE", "0") == "1" and layer_id >= 46):
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            grad_w1_norm = grad_w1.float().norm().item() if grad_w1 is not None else 0
            grad_w2_norm = grad_w2.float().norm().item() if grad_w2 is not None else 0
            grad_topk_norm = grad_topk_weights.float().norm().item() if grad_topk_weights is not None else 0
            grad_hidden_norm = grad_hidden_states.float().norm().item() if grad_hidden_states is not None else 0
            print(f"[FusedExpertsFunction.backward][Rank {rank}][Layer {layer_id}] FINAL gradients:")
            print(f"  grad_w1_norm: {grad_w1_norm:.6e}, grad_w2_norm: {grad_w2_norm:.6e}")
            print(f"  grad_topk_weights_norm: {grad_topk_norm:.6e} (ROUTER)")
            print(f"  grad_hidden_states_norm: {grad_hidden_norm:.6e} (upstream)")
            print(f"  RATIO grad_topk/grad_w2: {grad_topk_norm/grad_w2_norm if grad_w2_norm > 0 else 0:.4f}")
            if grad_w1 is not None and grad_w1_norm > 0:
                # Print per-expert gradient norms
                for i in range(min(3, grad_w1.shape[0])):
                    print(f"  grad_w1[{i}] norm: {grad_w1[i].float().norm().item():.6e}")
        
        # DEBUG: Detailed comparison logging for gradient analysis
        # Enable with DEBUG_COMPARE_GRAD_DISTRIBUTION=1
        if os.environ.get("DEBUG_COMPARE_GRAD_DISTRIBUTION", "0") == "1" and layer_id in [0, 47]:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                grad_w1_norm = grad_w1.float().norm().item() if grad_w1 is not None else 0
                grad_w2_norm = grad_w2.float().norm().item() if grad_w2 is not None else 0
                grad_topk_norm = grad_topk_weights.float().norm().item() if grad_topk_weights is not None else 0
                grad_hidden_norm = grad_hidden_states.float().norm().item() if grad_hidden_states is not None else 0
                grad_output_norm = grad_output.float().norm().item()
                
                total_expert_grad = (grad_w1_norm**2 + grad_w2_norm**2)**0.5
                
                print(f"\n[GRAD_DISTRIBUTION][Layer {layer_id}] ========================================")
                print(f"  grad_output (incoming): {grad_output_norm:.6e}")
                print(f"  grad_w1 (expert):       {grad_w1_norm:.6e}")
                print(f"  grad_w2 (expert):       {grad_w2_norm:.6e}")
                print(f"  TOTAL expert grad:      {total_expert_grad:.6e}")
                print(f"  grad_topk (router):     {grad_topk_norm:.6e}")
                print(f"  grad_hidden (upstream): {grad_hidden_norm:.6e}")
                print(f"  RATIO expert/router:    {total_expert_grad/grad_topk_norm if grad_topk_norm > 0 else 0:.4f}")
                print(f"  RATIO expert/incoming:  {total_expert_grad/grad_output_norm if grad_output_norm > 0 else 0:.4f}")
                print(f"========================================\n")

        # Return gradients for all inputs (None for non-tensor inputs)
        return grad_hidden_states, grad_w1, grad_w2, grad_topk_weights, None, None, None, None


def _pytorch_fused_experts_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    layer_number: int,
    ep_group,
) -> torch.Tensor:
    """Pure PyTorch forward for MoE expert computation with natural autograd.
    
    This function computes the same result as FusedExpertsFunction but uses
    standard PyTorch ops that enable natural autograd gradient computation.
    Use this for debugging gradient issues by comparing with the custom backward.
    
    Enable with USE_PYTORCH_MOE_FORWARD=1 environment variable.
    
    Args:
        hidden_states: Input tensor [num_tokens, hidden_size]
        w1: Gate/up projection weights [num_local_experts, ffn_hidden_size, hidden_size]
        w2: Down projection weights [num_local_experts, hidden_size, ffn_hidden_size//2]
        topk_weights: Router weights [num_tokens, topk]
        topk_ids: Expert indices [num_tokens, topk] (LOCAL expert IDs, -1 for remote)
        activation: Activation function name ("silu" or "gelu")
        layer_number: Layer index for debugging
        ep_group: Expert parallel process group for all-reducing output
        
    Returns:
        output: Output tensor [num_tokens, hidden_size]
    """
    import torch.distributed as dist
    
    num_tokens, hidden_size = hidden_states.shape
    num_local_experts = w1.shape[0]
    topk = topk_ids.shape[1]
    
    # Initialize output
    output = torch.zeros(num_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)
    
    # Process each LOCAL expert
    for local_expert_id in range(num_local_experts):
        # Find tokens that selected this expert
        mask = (topk_ids == local_expert_id)
        if not mask.any():
            continue
        
        token_indices = mask.any(dim=1).nonzero(as_tuple=True)[0]
        if len(token_indices) == 0:
            continue
        
        # Get expert input
        expert_input = hidden_states[token_indices]  # [num_tokens_for_expert, hidden_size]
        expert_w1 = w1[local_expert_id]  # [ffn_hidden_size, hidden_size]
        expert_w2 = w2[local_expert_id]  # [hidden_size, ffn_hidden_size//2]
        
        # Forward through expert
        gate_up = torch.nn.functional.linear(expert_input, expert_w1)
        gate_pre, up = gate_up.chunk(2, dim=-1)
        
        if activation == "silu":
            gate = torch.nn.functional.silu(gate_pre)
        elif activation == "gelu":
            gate = torch.nn.functional.gelu(gate_pre)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        intermediate = gate * up
        expert_output = torch.nn.functional.linear(intermediate, expert_w2)
        
        # Apply router weights and accumulate output
        for slot in range(topk):
            slot_mask = mask[token_indices, slot]
            if slot_mask.any():
                slot_token_indices = token_indices[slot_mask]
                slot_weights = topk_weights[slot_token_indices, slot].unsqueeze(-1)
                local_indices = slot_mask.nonzero(as_tuple=True)[0]
                weighted_output = expert_output[local_indices] * slot_weights
                output[slot_token_indices] += weighted_output.to(output.dtype)
    
    # All-reduce output across EP ranks
    # IMPORTANT: Use autograd-aware all-reduce so gradients flow correctly
    if ep_group is not None:
        ep_world_size = dist.get_world_size(ep_group)
        if ep_world_size > 1:
            # Use the autograd-aware _tree_all_reduce_sum for proper backward support
            from megatron.core.tensor_parallel.mappings import _tree_all_reduce_sum
            output = _tree_all_reduce_sum(output, ep_group, layer_id=layer_number)
    
    # Debug logging
    if os.environ.get("DEBUG_PYTORCH_MOE_FORWARD", "0") == "1" and layer_number in [0, 47]:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"[_pytorch_fused_experts_forward][Layer {layer_number}] "
                  f"output.shape={output.shape}, "
                  f"output.norm()={output.float().norm().item():.6e}, "
                  f"output.sum()={output.float().sum().item():.6e}")
    
    return output


def sglang_fused_experts(
    layer_number: int,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    # EP parameters (new)
    num_experts: Optional[int] = None,
    num_local_experts: Optional[int] = None,
    ep_rank: int = 0,
    ep_size: int = 1,
    ep_group = None,
):
    """Call fused experts implementation for deterministic MoE computation.

    Forward: Uses SGLang's fused_experts_impl (triton kernel) for bitwise identical results
    Backward: Uses custom autograd function for gradient computation

    Args:
        hidden_states: Input tensor [num_tokens, hidden_size]
        w1: Gate/up projection weights [num_local_experts, ffn_hidden_size, hidden_size]
        w2: Down projection weights [num_local_experts, hidden_size, ffn_hidden_size//2]
        topk_weights: Router weights [num_tokens, topk]
        topk_ids: Expert indices [num_tokens, topk] (global expert ids)
        activation: Activation function name ("silu" or "gelu")
        apply_router_weight_on_input: Whether to apply router weight on input
        num_experts: Total number of experts (global)
        num_local_experts: Number of local experts on this rank
        ep_rank: Expert parallel rank
        ep_size: Expert parallel world size
        ep_group: Expert parallel process group for all-reducing grad_topk_weights

    Returns:
        output: Output tensor [num_tokens, hidden_size]
    """
    if not HAVE_SGLANG_FUSED_EXPERTS:
        raise ImportError(
            "SGLang's fused_experts_impl is required. Please install sglang."
        )

    # EP mode: convert global expert ids to local expert ids
    # This matches SGLang's StandardDispatcher behavior
    local_expert_mapping = None
    if ep_size > 1 and num_experts is not None and num_local_experts is not None:
        # Create global -> local expert mapping
        # Non-local experts are mapped to -1 (will be skipped)
        local_expert_mapping = torch.full(
            (num_experts,), -1, dtype=torch.int32, device=topk_ids.device
        )
        local_start = ep_rank * num_local_experts
        local_expert_mapping[local_start : local_start + num_local_experts] = torch.arange(
            0, num_local_experts, dtype=torch.int32, device=topk_ids.device
        )
        
        # Convert topk_ids from global to local
        topk_ids_local = local_expert_mapping[topk_ids.long()]
    else:
        topk_ids_local = topk_ids.to(torch.int32)

    # Debug print for EP token mapping
    if os.environ.get("DEBUG_MEGATRON_EP_MAPPING", "0") == "1" and layer_number <= 1:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        num_tokens = topk_ids.shape[0]
        topk = topk_ids.shape[1]
        
        print(f"[moe_utils.py][Megatron EP Mapping][Rank {rank}][Layer {layer_number}] EP config: ep_size={ep_size}, ep_rank={ep_rank}")
        print(f"[moe_utils.py][Megatron EP Mapping][Rank {rank}][Layer {layer_number}] num_experts={num_experts}, num_local_experts={num_local_experts}")
        print(f"[moe_utils.py][Megatron EP Mapping][Rank {rank}][Layer {layer_number}] num_tokens={num_tokens}, topk={topk}")
        
        if local_expert_mapping is not None:
            local_start = ep_rank * num_local_experts
            local_end = local_start + num_local_experts
            print(f"[moe_utils.py][Megatron EP Mapping][Rank {rank}][Layer {layer_number}] Local expert range: [{local_start}, {local_end})")
            print(f"[moe_utils.py][Megatron EP Mapping][Rank {rank}][Layer {layer_number}] local_expert_mapping: {local_expert_mapping.tolist()}")

    # Debug: compare expert input
    if os.environ.get("DEBUG_MEGATRON_EP_MAPPING", "0") == "1" and layer_number <= 1:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        position = 91 if hidden_states.shape[0] > 91 else 0
        print(f"[moe_utils.py][Megatron Expert Input][Rank {rank}][Layer {layer_number}] hidden_states.shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
        print(f"[moe_utils.py][Megatron Expert Input][Rank {rank}][Layer {layer_number}] hidden_states[{position}, :5]: {hidden_states[position, :5].tolist()}")
        print(f"[moe_utils.py][Megatron Expert Input][Rank {rank}][Layer {layer_number}] w1.shape: {w1.shape}, w2.shape: {w2.shape}")
        print(f"[moe_utils.py][Megatron Expert Input][Rank {rank}][Layer {layer_number}] topk_ids_local.shape: {topk_ids_local.shape}, topk_ids_local[{position}]: {topk_ids_local[position].tolist()}")
        print(f"[moe_utils.py][Megatron Expert Input][Rank {rank}][Layer {layer_number}] topk_weights.shape: {topk_weights.shape}, topk_weights[{position}]: {topk_weights[position].tolist()}")

    # Debug: check if topk_weights requires grad
    if os.environ.get("DEBUG_GRAD_ALLREDUCE", "0") == "1" and layer_number <= 1:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[sglang_fused_experts][Rank {rank}][Layer {layer_number}] "
              f"topk_weights.requires_grad={topk_weights.requires_grad}, "
              f"hidden_states.requires_grad={hidden_states.requires_grad}, "
              f"ep_group={ep_group}, ep_size={ep_size}")
    
    # DEBUG: Use pure PyTorch forward for gradient comparison
    # Enable with USE_PYTORCH_MOE_FORWARD=1 to bypass triton kernel and use PyTorch ops
    # This allows autograd to compute gradients naturally, helping diagnose if the
    # custom backward in FusedExpertsFunction is causing gradient magnitude issues
    if os.environ.get("USE_PYTORCH_MOE_FORWARD", "0") == "1":
        output = _pytorch_fused_experts_forward(
            hidden_states.contiguous(),
            w1.contiguous(),
            w2.contiguous(),
            topk_weights.contiguous(),
            topk_ids_local.contiguous(),
            activation,
            layer_number,
            ep_group,
        )
    else:
        # Use custom autograd function: forward uses triton kernel (bitwise identical),
        # backward uses PyTorch operations for gradient computation
        # Pass ep_group so grad_topk_weights can be all-reduced across EP ranks
        output = FusedExpertsFunction.apply(
            hidden_states.contiguous(),
            w1.contiguous(),
            w2.contiguous(),
            topk_weights.contiguous(),
            topk_ids_local.contiguous(),
            activation,
            layer_number,
            ep_group,
        )

    # Debug: compare expert output
    if os.environ.get("DEBUG_MEGATRON_EP_MAPPING", "0") == "1" and layer_number <= 1:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        position = 91 if output.shape[0] > 91 else 0
        print(f"[moe_utils.py][Megatron Expert Output][Rank {rank}][Layer {layer_number}] output.shape: {output.shape}, dtype: {output.dtype}")
        print(f"[moe_utils.py][Megatron Expert Output][Rank {rank}][Layer {layer_number}] output[{position}, :5]: {output[position, :5].tolist()}")
        print(f"[moe_utils.py][Megatron Expert Output][Rank {rank}][Layer {layer_number}] output[{position}].norm(): {output[position].norm().item():.6f}")
        print(f"[moe_utils.py][Megatron Expert Output][Rank {rank}][Layer {layer_number}] output[{position}].sum(): {output[position].sum().item():.6f}")

    return output


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: Optional[torch.Tensor] = None,
    routing_map: Optional[torch.Tensor] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    When drop_and_pad=True, the tensors will have the following properties:
      - In routing_map, the number of non-zeros in each column equals to expert capacity
      - The size of sorted_indices equals to num_experts * capacity, each split of `capacity`
        contains the indices of tokens routed to an expert.
    This function exploits these features to use ops that support cuda graph.

    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The indices used to sort the tokens.
        restore_shape (torch.Size): The shape of the unpermuted tensor.
        probs (torch.Tensor, optional): The unpermuted probs tensor,
        routing_map (torch.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].
        fused (bool, optional): Whether use the fused unpermute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.

    Returns:
        torch.Tensor: The tokens restored to their original order.
    """
    if fused:
        if not HAVE_TE or fused_unpermute is None:
            raise ValueError("fused_unpermute is not available. Please install TE >= 2.1.0.")
        return fused_unpermute(
            permuted_tokens, sorted_indices, merging_probs=probs, restore_shape=restore_shape
        )

    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        if drop_and_pad:
            num_experts = routing_map.size(1)
            num_permuted_tokens = sorted_indices.size(0)
            capacity = num_permuted_tokens // num_experts
            num_unpermuted_tokens = probs.size(0)

            # [num_unpermuted_tokens, num_experts] -> num_experts * num_unpermuted_tokens
            probs_T_1D = probs.T.contiguous().view(-1)

            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)

            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
        else:
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        # Here may promote permuted_tokens to higher precision (fp32/fp64) if probs is in
        # higher precision due to moe_router_dtype being enabled. This can lead to
        # additional GPU memory usage. Use --moe-permute-fusion flag to avoid this extra memory
        # allocation.
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = torch.zeros(
        restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
    )
    if torch.are_deterministic_algorithms_enabled():
        # Use index_add which is deterministic when deterministic algorithms are enabled
        # and is CUDA graph compatible
        output_tokens = torch.zeros(
            restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
        )
        # index_add is deterministic when torch.use_deterministic_algorithms(True) is set
        # and is CUDA graph compatible unlike scatter_add
        output_tokens.index_add_(0, sorted_indices, permuted_tokens)
    else:
        # Scatter add the permuted_input back to the original positions
        output_tokens.scatter_add_(
            0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens
        )
    return output_tokens.to(dtype=input_dtype)


def sort_chunks_by_idxs(
    input: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    fused: bool = False,
):
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    if fused and probs is None:
        if not HAVE_TE or fused_sort_chunks_by_index is None:
            raise ValueError(
                "fused_sort_chunks_by_index is not available. Please install TE >= 2.1.0."
            )
        return fused_sort_chunks_by_index(input, split_sizes, sorted_idxs), None

    if fused and probs is not None:
        if not HAVE_TE or fused_sort_chunks_by_index_with_probs is None:
            raise ValueError(
                "fused_sort_chunks_by_index_with_probs is not available. "
                "Please install TE >= 2.1.0."
            )
        return fused_sort_chunks_by_index_with_probs(input, probs, split_sizes, sorted_idxs)

    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs.tolist()], dim=0)
    if probs is not None:
        probs = torch.split(probs, split_sizes.tolist(), dim=0)
        permuted_probs = torch.cat([probs[i] for i in sorted_idxs.tolist()], dim=0)
    else:
        permuted_probs = None
    return output, permuted_probs


def group_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    group_topk: int,
):
    """Perform top-k routing on a subset of expert groups.

    When using group-limited routing:
    1. Experts are divided into 'moe_router_num_groups' equal-sized groups
    2. For each token, 'moe_router_group_topk' groups are selected based on routing scores
       (specifically, the sum of top-2 expert scores within each group)
    3. From these selected groups, 'moe_router_topk' individual experts are chosen

    Two common use cases:
    - Device-limited routing: Set 'moe_router_num_groups' equal to expert parallel size (EP)
      to limit each token to experts on a subset of devices
      (See DeepSeek-V2: https://arxiv.org/pdf/2405.04434)

    - Node-limited routing: Set 'moe_router_num_groups' equal to number of nodes in EP group
      to limit each token to experts on a subset of nodes
      (See DeepSeek-V3: https://arxiv.org/pdf/2412.19437)

    Args:
        scores (torch.Tensor): Softmax scores generated by the router.
        topk (int): The number of experts to select for each token.
        num_tokens (int): The number of tokens.
        num_experts (int): The number of experts.
        num_groups (int): Number of groups for routed experts.
        group_topk (int): Number of groups selected for each token.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Probs and indices tensor.
    """
    # Organize the experts into groups
    # Select groups based on sum of top-(topk/group_topk) routing scores within each group
    group_scores = (
        scores.view(num_tokens, num_groups, -1).topk(topk // group_topk, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    # Mask the experts based on selection groups
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )

    masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
    probs, top_indices = torch.topk(masked_scores, k=topk, dim=-1)

    return probs, top_indices


def pad_routing_map(routing_map: torch.Tensor, pad_multiple: int) -> torch.Tensor:
    """Pad the routing map to ensure each expert has a multiple of pad_multiple tokens.

    This function ensures that each expert has a number of tokens that is a multiple of
    pad_multiple by converting some 0s to 1s in the routing map. The padding is done by
    selecting the first N zero elements in each row, where N is the number needed to reach
    the next multiple of pad_multiple.

    Args:
        routing_map (torch.Tensor): A boolean or integer tensor of shape [num_tokens,
            num_experts] indicating which tokens are routed to which experts.
        pad_multiple (int): The multiple to pad each expert's token count to.

    Returns:
        torch.Tensor: The padded routing map of shape [num_tokens, num_experts].
    """
    # Transpose to [num_experts, num_tokens] for easier row-wise operations
    routing_map = routing_map.transpose(0, 1)  # [num_experts, num_tokens]

    # Calculate how many tokens need to be padded for each expert
    num_ones = routing_map.sum(dim=1)
    num_to_pad = (-num_ones) % pad_multiple

    # Find the positions of zeros in each row and their ranks
    is_zero = routing_map == 0
    zero_ranks = torch.cumsum(is_zero.int(), dim=1)

    # Create mask for elements that need to be padded (converted from 0 to 1)
    mask = zero_ranks <= num_to_pad.unsqueeze(1)
    routing_map[mask] = 1

    routing_map = routing_map.transpose(0, 1)
    return routing_map


def topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
    fused: bool = False,
):
    """Compute the routing probabilities and map for top-k selection with score function.
    Args:
        logits (torch.Tensor): Logits tensor.
        topk (int): The number of experts to select for each token.
        use_pre_softmax (bool): Whether to apply softmax or sigmoid before top-k selection.
        num_groups (int): Number of groups for routed experts.
        group_topk (int): Number of selected groups for each token.
        scaling_factor (float): Scaling factor of routing score in top-k selection.
        score_function (str): The score function to use. Can be either "softmax" or "sigmoid".
        expert_bias (torch.Tensor): The bias added to logits for expert routing.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - routing_probs (torch.Tensor): A tensor of shape [num_tokens, num_experts] containing
              the routing probabilities for each token to each expert.
            - routing_map (torch.Tensor): A mask tensor of shape [num_tokens, num_experts]
              indicating which experts were selected for each token. True values represent
              the selected experts.
    """
    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
    num_tokens, num_experts = logits.shape
    if fused:
        if not HAVE_TE or fused_topk_with_score_function is None:
            raise ValueError(
                "fused_topk_with_score_function is not available. Please install TE >= 2.6.0."
            )
        return fused_topk_with_score_function(
            logits=logits,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            expert_bias=expert_bias,
        )

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        else:
            return torch.topk(scores, k=topk, dim=1)

    from slime.utils.routing_replay import get_routing_replay_compute_topk
    compute_topk = get_routing_replay_compute_topk(compute_topk)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    if torch.are_deterministic_algorithms_enabled():
        # build [num_tokens, num_experts] from [num_tokens, topk]
        routing_probs = torch.zeros_like(logits)
        rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
        routing_probs.index_put_((rows, top_indices), probs, accumulate=False)

        routing_map = torch.zeros_like(logits, dtype=logits.dtype)
        routing_map.index_put_(
            (rows, top_indices), torch.ones_like(probs, dtype=routing_map.dtype), accumulate=False
        )
        routing_map = routing_map.bool()
    else:
        # TODO Try using element-wise operations instead of scatter?
        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

    return routing_probs, routing_map


def compute_routing_scores_for_aux_loss(
    logits: torch.Tensor, topk: int, score_function: str, fused: bool = False
):
    """Compute routing scores based on the score function.

    Args:
        logits (torch.Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].

    Returns:
        torch.Tensor: The normalized routing scores.
    """
    if fused:
        if not HAVE_TE or fused_compute_score_for_moe_aux_loss is None:
            raise ValueError(
                "fused_compute_score_for_moe_aux_loss is not available. Please install TE >= 2.6.0."
            )
        return fused_compute_score_for_moe_aux_loss(
            logits=logits, topk=topk, score_function=score_function
        )

    if score_function == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits)
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    _, top_indices = torch.topk(scores, k=topk, dim=1)
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return routing_map, scores


def apply_router_token_dropping(
    routing_probs: torch.Tensor,
    routing_map: torch.Tensor,
    router_topk: int,
    capacity_factor: float,
    drop_policy: str = "probs",
    pad_to_capacity: bool = False,
):
    """Apply token dropping to top-k expert selection.

    This function enforces expert capacity limits by dropping tokens that exceed
    the capacity and optionally padding to capacity.

    Args:
        routing_probs (torch.Tensor): Tensor of shape [num_tokens, num_experts]
            containing the routing probabilities for selected experts.
        routing_map (torch.Tensor): Boolean tensor of shape [num_tokens, num_experts]
            indicating which experts were selected for each token.
        router_topk (int): Number of experts selected per token.
        capacity_factor (float): The capacity factor of each expert.
        drop_policy (str): Policy to drop tokens - "probs" or "position".
        pad_to_capacity (bool): Whether to pad to capacity.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - final_probs: Routing probabilities after applying capacity constraints
            - final_map: Boolean mask after applying capacity constraints
    """
    assert routing_probs.ndim == 2 and routing_map.ndim == 2
    num_tokens, num_experts = routing_probs.shape
    # Calculate expert capacity
    expert_capacity = get_capacity(
        num_tokens=num_tokens * router_topk,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )

    # Create capacity mask based on drop policy
    if expert_capacity > num_tokens:
        # No need to drop tokens if capacity exceeds the number of tokens
        capacity_mask = torch.ones_like(routing_probs).bool()
    else:
        if drop_policy == "probs":
            _, capacity_indices = torch.topk(routing_probs, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(routing_probs).scatter(0, capacity_indices, 1).bool()
        elif drop_policy == "position":
            _, capacity_indices = torch.topk(
                routing_map.int(), k=expert_capacity, dim=0, sorted=False
            )
            capacity_mask = torch.zeros_like(routing_probs).scatter(0, capacity_indices, 1).bool()
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

    # Apply capacity constraints
    if pad_to_capacity:
        final_map = capacity_mask
        final_probs = routing_probs * final_map
    else:
        # Get exceed mask and maskout exceeded probs and indices
        final_map = torch.logical_and(routing_map, capacity_mask)
        final_probs = routing_probs * final_map

    return final_probs, final_map


def save_to_aux_losses_tracker(
    name: str,
    loss: torch.Tensor,
    layer_number: int,
    num_layers: int,
    reduce_group: Optional[torch.distributed.ProcessGroup] = None,
    avg_group: Optional[torch.distributed.ProcessGroup] = None,
    reduce_group_has_dp: bool = False,
):
    """Save the auxiliary loss for logging.
    Args:
        name (str): The name of the loss.
        loss (torch.Tensor): The loss tensor.
        layer_number (int): Layer index of the loss.
        num_layers (int): The number of total layers.
        reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
        avg_group (torch.distributed.ProcessGroup): The group for averaging the loss.
        reduce_group_has_dp (bool): Whether the reduce group has data parallel ranks.
            Set this to True if the reduce group has data parallel ranks. This flag is used to
            ensure the correct reduction in aux loss tracking.
    """
    # Skip aux loss logging if layer_number is None.
    if layer_number is None:
        return

    tracker = get_moe_layer_wise_logging_tracker()
    if name not in tracker:
        tracker[name] = {}
        tracker[name]["values"] = torch.zeros(num_layers, device=loss.device)
    tracker[name]["values"][layer_number - 1] += loss.detach()  # Aggregate the loss for the layer.
    tracker[name]["reduce_group"] = reduce_group
    tracker[name]["avg_group"] = avg_group
    tracker[name]["reduce_group_has_dp"] = reduce_group_has_dp


def clear_aux_losses_tracker():
    """Clear the auxiliary losses."""
    tracker = get_moe_layer_wise_logging_tracker()
    for name in tracker:
        tracker[name]["values"].zero_()


def reduce_aux_losses_tracker_across_ranks(track_names: Optional[List[str]] = None):
    """Collect and reduce the auxiliary losses across ranks."""
    tracker = get_moe_layer_wise_logging_tracker()
    if track_names is None:
        track_names = tracker.keys()
    for name in track_names:
        values = tracker[name]["values"]
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Collect aux losses across PP.
        torch.distributed.all_reduce(
            values, group=parallel_state.get_pipeline_model_parallel_group()
        )
        # Reduce aux losses across ranks.
        if tracker[name].get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker[name].get('reduce_group'))
            # Need to conduct reduction across data parallel ranks. When the reduce_group
            # does not have 'dp' attribute, do it manually.
            if not tracker[name].get('reduce_group_has_dp', False):
                torch.distributed.all_reduce(
                    values,
                    group=parallel_state.get_data_parallel_group(with_context_parallel=False),
                    op=torch.distributed.ReduceOp.AVG,
                )
        if tracker[name].get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker[name]['avg_group'], op=torch.distributed.ReduceOp.AVG
            )


def track_moe_metrics(
    loss_scale: float,
    iteration: int,
    writer,
    wandb_writer=None,
    total_loss_dict=None,
    per_layer_logging=False,
    force_initialize: bool = False,
    track_names: Optional[List[str]] = None,
    num_layers: Optional[int] = None,
    moe_layer_freq: Optional[Union[int, List[int]]] = None,
    mtp_num_layers: Optional[int] = None,
):
    """Track the MoE metrics for logging."""
    # Aux loss logging
    tracker = get_moe_layer_wise_logging_tracker()
    # Initialize the tracker if force_initialize is True
    if force_initialize:
        if track_names is not None:
            for key in track_names:
                if key not in tracker:
                    tracker[key] = {}
                    tracker[key]["values"] = torch.zeros(num_layers, device="cuda")
                    tracker[key]["reduce_group"] = None
                    tracker[key]["avg_group"] = None
                    tracker[key]["reduce_group_has_dp"] = False
    reduce_aux_losses_tracker_across_ranks(track_names)

    # Get number of MoE layers
    if moe_layer_freq is None:
        num_moe_layers = num_layers
    elif isinstance(moe_layer_freq, int):
        assert isinstance(num_layers, int)
        moe_layer_pattern = [1 if (i % moe_layer_freq == 0) else 0 for i in range(num_layers)]
        num_moe_layers = sum(moe_layer_pattern)
    elif isinstance(moe_layer_freq, list):
        num_moe_layers = sum(moe_layer_freq)
    else:
        raise ValueError(f"Invalid moe_layer_freq: {moe_layer_freq}")

    if mtp_num_layers is not None:
        num_moe_layers += mtp_num_layers

    aux_losses = {k: v['values'].float() * loss_scale for k, v in tracker.items()}
    for name, loss_list in aux_losses.items():
        if total_loss_dict is not None:
            if name not in total_loss_dict:
                total_loss_dict[name] = loss_list.sum() / num_moe_layers
            else:
                total_loss_dict[name] += loss_list.sum() / num_moe_layers
        if writer is not None:
            # currently when using add_scalars,
            # torch.utils.add_scalars makes each timer its own run, which
            # polutes the runs list, so we just add each as a scalar
            writer.add_scalar(name, loss_list.sum() / num_moe_layers, iteration)
            if per_layer_logging:
                for i, loss in enumerate(loss_list.tolist()):
                    writer.add_scalar(f"moe/{name}_layer_{i}", loss, iteration)

            # W&B logging lacks support for logging multiple scalars simultaneously.
            # As a workaround, we log each scalar individually first, then we can create
            # a custom panel to manually group them to a single plot.
            if wandb_writer:
                wandb_writer.log({f"{name}": loss_list.sum() / num_moe_layers}, iteration)
                if per_layer_logging:
                    wandb_writer.log(
                        {
                            f"moe/{name}_layer_{i}": loss
                            for i, loss in enumerate(loss_list.tolist())
                        },
                        iteration,
                    )

    clear_aux_losses_tracker()


def get_updated_expert_bias(tokens_per_expert, expert_bias, expert_bias_update_rate):
    """Update expert bias for biased expert routing. See https://arxiv.org/abs/2408.15664v1#

    Args:
        tokens_per_expert (torch.Tensor): The number of tokens assigned to each expert.
        expert_bias (torch.Tensor): The bias for each expert.
        expert_bias_udpate_rate (float): The update rate for the expert bias.
    """
    with torch.no_grad():
        # All Reduce Across TPxCPxDP group
        torch.distributed.all_reduce(
            tokens_per_expert,
            # TODO(Hepteract): delete the usage of the global parallel_state.
            group=parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
        )
        average_tokens = tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]
        offset = average_tokens - tokens_per_expert
        updated_expert_bias = expert_bias + torch.sign(offset) * expert_bias_update_rate
        return updated_expert_bias


def maybe_move_tensor_to_cpu(tensor, as_numpy=False, record_stream=False):
    """Move a tensor to CPU if it is on GPU.
    Args:
        tensor (torch.Tensor or None): The tensor to move to CPU.
        as_numpy (bool): Whether to convert the tensor to a numpy array.
        record_stream (bool): Whether to record the stream of the tensor, to prevent memory leak
                              when the DtoH data transfer is on a side stream.
    """
    if torch.is_tensor(tensor) and tensor.is_cuda:
        cpu_tensor = tensor.to(torch.device("cpu"), non_blocking=True)
        if as_numpy:
            cpu_tensor = cpu_tensor.numpy()
        if record_stream:
            tensor.record_stream(torch.cuda.current_stream())
        tensor = cpu_tensor
    return tensor


def get_moe_layer_wise_logging_tracker():
    """Return the moe layer wise tracker."""
    global _MOE_LAYER_WISE_LOGGING_TRACKER
    return _MOE_LAYER_WISE_LOGGING_TRACKER


@internal_api
class RandomSTE(torch.autograd.Function):
    """
    Straight-Through Estimator(STE) function that returns random values
    with different seed for each rank.

    This is used to generate random logits of router for load-balanced benchmark.
    """

    @staticmethod
    def forward(ctx, logits):
        """
        Forward pass returns random logits with rank-specific seed.
        """
        with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
            random_logits = logits.clone().normal_()
        return random_logits

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass propagates the gradient for logits.
        """
        return grad_output


def apply_random_logits(logits):
    """
    Apply the RandomSTE function to the logits.
    """
    return RandomSTE.apply(logits)


class RouterGatingLinearFunction(torch.autograd.Function):
    """
    Autograd function for router gating linear.
    """

    @staticmethod
    def forward(
        ctx, inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, router_dtype: torch.dtype
    ):
        """
        Forward pass of the RouterGatingLinearFunction function.
        """
        ctx.save_for_backward(inp, weight, bias)
        ctx.router_dtype = router_dtype
        ctx.input_dtype = inp.dtype
        ctx.weight_dtype = weight.dtype
        inp_shape = inp.shape
        inp = inp.view(-1, inp_shape[-1])

        if te_general_gemm is not None and router_dtype != torch.float64:
            output = te_general_gemm(weight, inp, router_dtype, layout="TN", bias=bias)
            output = output[0]
        elif bias is None:
            output = torch.mm(inp.to(router_dtype), weight.to(router_dtype).t())
        else:
            output = torch.addmm(
                bias.to(router_dtype), inp.to(router_dtype), weight.to(router_dtype).t()
            )

        output = output.view(*inp_shape[:-1], -1)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass of the RouterGatingLinearFunction function.
        """
        inp, weight, bias = ctx.saved_tensors
        inp_shape = inp.shape
        grad_shape = grad_output.shape
        inp = inp.view(-1, inp_shape[-1])
        grad_output = grad_output.view(-1, grad_shape[-1])

        if te_general_gemm is not None and ctx.router_dtype != torch.float64:
            grad_input = te_general_gemm(
                weight.to(ctx.router_dtype), grad_output, ctx.router_dtype, layout="NN", grad=True
            )
            grad_weight = te_general_gemm(
                inp.to(ctx.router_dtype), grad_output, ctx.router_dtype, layout="NT", grad=True
            )
            grad_input = grad_input[0].to(ctx.input_dtype)
            grad_weight = grad_weight[0].to(ctx.weight_dtype)
        else:
            grad_input = torch.mm(grad_output, weight.to(ctx.router_dtype)).to(ctx.input_dtype)
            grad_weight = torch.mm(grad_output.t(), inp.to(ctx.router_dtype)).to(ctx.weight_dtype)

        grad_bias = grad_output.sum(dim=0).to(ctx.weight_dtype) if bias is not None else None
        grad_input = grad_input.view(*inp_shape)
        return grad_input, grad_weight, grad_bias, None


def router_gating_linear(
    inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, router_dtype: torch.dtype
):
    """
    Customized linear layer for router gating.
    This linear layer accepts bfloat16 input and weight, and can return output with router_dtype.
    It can reduce the memory usage by avoiding saving the intermediate high precision tensors.
    """
    return RouterGatingLinearFunction.apply(inp, weight, bias, router_dtype)


def get_align_size_for_quantization(config: TransformerConfig):
    """Get the alignment size for quantization."""
    if config.fp8:
        return get_fp8_align_size(config.fp8_recipe)
    elif config.fp4:
        return get_fp4_align_size(config.fp4_recipe)
    return 16


# TODO(Hepteract): delete the usage of the global parallel_state.
# Initialize process groups with the global parallel_state.
def get_default_pg_collection():
    """Get the default process groups for MoE.

    Returns:
        ProcessGroupCollection: The default process groups for MoE.
    """
    pg_collection = ProcessGroupCollection()
    pg_collection.ep = parallel_state.get_expert_model_parallel_group()
    pg_collection.tp = parallel_state.get_tensor_model_parallel_group()
    pg_collection.cp = parallel_state.get_context_parallel_group()
    pg_collection.expt_tp = parallel_state.get_expert_tensor_parallel_group()
    pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()
    pg_collection.tp_ep = parallel_state.get_expert_tensor_and_model_parallel_group()
    pg_collection.tp_cp = parallel_state.get_tensor_and_context_parallel_group()
    pg_collection.tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group(
        with_context_parallel=True
    )
    return pg_collection


class MoECudaGraphPartialCaptureSignal(Exception):
    """
    Used to early-return from a MoE layer forward pass in CUDA graph capture.
    This signal is raised when we are partially capturing the CUDA graph of the MoE layer,
    and the related intermediate tensors are recorded in self.kwargs.
    Call self.get_early_return_outputs() to collect the CUDA graph outputs.
    """

    def __init__(self, moe_layer, return_step: str, **kwargs):
        self.moe_layer = moe_layer
        self.return_step = return_step
        self.kwargs = kwargs

    def get_early_return_outputs(
        self, hidden_states: torch.Tensor, shared_expert_output: torch.Tensor
    ):
        """
        Get the CUDA graph early return outputs for the MoE layer, including the intermediate
        tensors and the intermediate attributes of the token dispatcher.

        The returned output tensors are in the order of:
        - routed experts path outputs
          - hidden states, probs, and routing map for capturing router
          - hidden states and probs for capturing router and preprocess
        - intermediate attributes of the token dispatcher (if capturing the preprocess step)
        - shared expert path output (if exists)
        """
        if self.return_step == "route":
            # Capturing the router step returns three intermediate tensors:
            # hidden states, routing probabilities, and routing map.
            outputs = [hidden_states, self.kwargs['probs'], self.kwargs['routing_map']]
        elif self.return_step == "preprocess":
            # Capturing the preprocess step returns two intermediate tensors:
            # hidden states and routing probabilities.
            # It also returns the intermediate attributes of the token dispatcher, recorded in
            # "token_dispatcher.cudagraph_attrs".
            outputs = [self.kwargs['hidden_states'], self.kwargs['probs']]
            valid_cudagraph_attrs = []
            for attr_name in self.moe_layer.token_dispatcher.cudagraph_attrs:
                hier_attr_name = attr_name.split('.')
                attr = self.moe_layer.token_dispatcher
                for name in hier_attr_name:
                    attr = getattr(attr, name, None)
                    if attr is None:
                        break
                if isinstance(attr, torch.Tensor):
                    outputs.append(attr)
                    valid_cudagraph_attrs.append(attr_name)
            if self.moe_layer.token_dispatcher.valid_cudagraph_attrs is None:
                self.moe_layer.token_dispatcher.valid_cudagraph_attrs = valid_cudagraph_attrs
            else:
                assert (
                    self.moe_layer.token_dispatcher.valid_cudagraph_attrs == valid_cudagraph_attrs
                ), (
                    "valid_cudagraph_attrs mismatch: "
                    f"{self.moe_layer.token_dispatcher.valid_cudagraph_attrs} != "
                    f"{valid_cudagraph_attrs}"
                )
        # Also return the shared expert output, if it is not None.
        if shared_expert_output is not None:
            outputs.append(shared_expert_output)
        return outputs


@internal_api
@dataclass
class MoECudaGraphTensorStore:
    """Storage for tensors used in CUDA graph replay for MoE layers.

    This dataclass stores intermediate tensors computed during CUDA graph replay
    that need to be resumed from the end of the CUDA graph scope to skip redundant computations.

    Attributes:
        hidden_states (Optional[torch.Tensor]): The hidden states output from the CUDA graph replay.
        probs (Optional[torch.Tensor]): The routing probabilities for each token-expert pair.
        routing_map (Optional[torch.Tensor]): The sparse mapping indicating which experts
            were selected for each token. Used to skip the normal router step.
        shared_expert_output (Optional[torch.Tensor]): The output from shared experts
            computation. Used to skip the normal shared expert computation step.
    """

    hidden_states: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None
    routing_map: Optional[torch.Tensor] = None
    shared_expert_output: Optional[torch.Tensor] = None

    def is_empty(self) -> bool:
        """Check if the store has any non-None tensors.

        Returns:
            bool: True if all fields are None, False otherwise.
        """
        return all(
            getattr(self, field_name) is None
            for field_name in ['hidden_states', 'probs', 'routing_map', 'shared_expert_output']
        )

    def set(self, **kwargs):
        """Set the tensors in the store from keyword arguments."""
        for field_name, value in kwargs.items():
            assert field_name in [
                'hidden_states',
                'probs',
                'routing_map',
                'shared_expert_output',
            ], f"Invalid field name: {field_name}"
            if value is not None:
                assert isinstance(
                    value, torch.Tensor
                ), f"Value must be a torch.Tensor, got {type(value)} for field {field_name}"
                setattr(self, field_name, value)

    def clear(self):
        """Reset all stored tensors to None."""
        for field_name in ['hidden_states', 'probs', 'routing_map', 'shared_expert_output']:
            setattr(self, field_name, None)


def maybe_skip_or_early_return_by_cudagraph(step_condition):
    """
    Decorator to skip certain codepaths in the MoE layer forward pass in CUDA graph replay,
    or early return from the MoE layer forward pass in CUDA graph capture.

    Args:
        step_condition: The step condition to check. Can be "shared_experts_compute", "route",
        or "preprocess". If "shared_experts_compute", the shared experts computation will be
        skipped in replay if it is in the CUDA graph scope. If "route" or "preprocess", the
        router or preprocess will be skipped in replay if it is in the CUDA graph scope, or
        early return from the MoE layer forward pass if it is in CUDA graph capturing mode.

    Returns:
        A decorator function that wraps the MoE layer forward pass.
    """

    def maybe_raise_signal(moe_layer, **kwargs):
        """
        Check if the MoE layer should early return for CUDA graph capture.
        If so, raise a MoECudaGraphPartialCaptureSignal.
        """
        if (
            moe_layer.config.cuda_graph_impl == "transformer_engine"
            and moe_layer.training
            and is_graph_capturing()
        ):
            if (
                step_condition == "route"
                and CudaGraphScope.moe_router in moe_layer.config.cuda_graph_scope
                and CudaGraphScope.moe_preprocess not in moe_layer.config.cuda_graph_scope
            ):
                raise MoECudaGraphPartialCaptureSignal(moe_layer, "route", **kwargs)
            elif (
                step_condition == "preprocess"
                and CudaGraphScope.moe_preprocess in moe_layer.config.cuda_graph_scope
            ):
                raise MoECudaGraphPartialCaptureSignal(moe_layer, "preprocess", **kwargs)

    def decorator(func):

        @functools.wraps(func)
        def wrapped_func(moe_layer, *args, **kwargs):
            """
            Check if we should skip executing the original function based on the current
            step condition and the tensor store status. If the tensor can be found in the store,
            it indicates that it is already computed by the CUDA graph replay, so we can skip it.
            Otherwise, we execute the original function and check if we should raise a signal to
            early return in CUDA graph capture.
            """
            # The non-cudagraph codepath just calls the original function.
            if not is_graph_capturing() and moe_layer.cudagraph_tensor_store.is_empty():
                return func(moe_layer, *args, **kwargs)

            assert (
                not is_graph_capturing() or moe_layer.cudagraph_tensor_store.is_empty()
            ), "cudagraph_tensor_store cannot be used when it is capturing cuda graph."
            if step_condition == "shared_experts_compute":
                if moe_layer.cudagraph_tensor_store.shared_expert_output is None:
                    # Don't skip the shared expert computation.
                    shared_expert_output = func(moe_layer, *args, **kwargs)
                else:
                    # Skip the shared expert computation and get value from store.
                    shared_expert_output = moe_layer.cudagraph_tensor_store.shared_expert_output
                return shared_expert_output
            elif step_condition == "route":
                if moe_layer.cudagraph_tensor_store.probs is None:
                    # Don't skip the router.
                    assert (
                        moe_layer.cudagraph_tensor_store.routing_map is None
                    ), "routing_map must be None if probs is None"
                    probs, routing_map = func(moe_layer, *args, **kwargs)

                    # Maybe early return after the router.
                    maybe_raise_signal(moe_layer, probs=probs, routing_map=routing_map)
                else:
                    # Skip the router and get value from store.
                    probs, routing_map = (
                        moe_layer.cudagraph_tensor_store.probs,
                        moe_layer.cudagraph_tensor_store.routing_map,
                    )
                return probs, routing_map
            elif step_condition == "preprocess":
                if (
                    moe_layer.cudagraph_tensor_store.is_empty()
                    or moe_layer.cudagraph_tensor_store.routing_map is not None
                ):
                    # Don't skip the preprocess.
                    hidden_states, probs = func(moe_layer, *args, **kwargs)

                    # Maybe early return after the preprocess.
                    maybe_raise_signal(moe_layer, hidden_states=hidden_states, probs=probs)
                else:
                    # Skip the preprocess and get value from store.
                    assert (
                        moe_layer.cudagraph_tensor_store.hidden_states is not None
                        and moe_layer.cudagraph_tensor_store.probs is not None
                    ), "hidden_states and probs must be given in moe_preprocess cudagraph replay"
                    hidden_states, probs = (
                        moe_layer.cudagraph_tensor_store.hidden_states,
                        moe_layer.cudagraph_tensor_store.probs,
                    )
                return hidden_states, probs

        return wrapped_func

    return decorator
