# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core import parallel_state, tensor_parallel, utils
from megatron.core.tensor_parallel.mappings import _tree_all_reduce_sum
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoECudaGraphPartialCaptureSignal,
    MoECudaGraphTensorStore,
    get_default_pg_collection,
    maybe_skip_or_early_return_by_cudagraph,
    sglang_fused_experts,
    HAVE_SGLANG_FUSED_EXPERTS,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import internal_api

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TELinear, te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

import os
import logging
logger = logging.getLogger(__name__)

@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.attn_tp_group = pg_collection.tp
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)
        assert ep_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % ep_size == 0
        self.num_local_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        self.submodules = submodules
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super(MoELayer, self).__init__(
            config=config, layer_number=layer_number, pg_collection=pg_collection
        )
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )

        self.tp_group = pg_collection.tp

        # Initialize router.
        self.router = TopKRouter(config=self.config, pg_collection=pg_collection)

        # Initialize latent projections.
        if self.config.moe_latent_size:
            assert HAVE_TE, "TransformerEngine is required for MoE latent projections."
            self.fc1_latent_proj = TELinear(
                self.config.hidden_size,
                self.config.moe_latent_size,
                parallel_mode="duplicated",
                config=self.config,
                init_method=self.config.init_method,
                bias=self.config.add_bias_linear,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                is_expert=False,
            )
            self.fc2_latent_proj = TELinear(
                self.config.moe_latent_size,
                self.config.hidden_size,
                parallel_mode="duplicated",
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                is_expert=False,
            )

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
            pg_collection=pg_collection,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts,
                config=self.config,
                pg_collection=pg_collection,
                gate=self.config.moe_shared_expert_gate,
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

        # Cudagraph tensor store for resuming the forward pass from the end of the cudagraph.
        self.cudagraph_tensor_store = MoECudaGraphTensorStore()

    @maybe_skip_or_early_return_by_cudagraph("route")
    def route(self, hidden_states: torch.Tensor):
        """Compute token routing for preprocessing.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping.
        """
        probs, routing_map = self.router(hidden_states)
        return probs, routing_map

    @maybe_skip_or_early_return_by_cudagraph("preprocess")
    def preprocess(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ):
        """Preprocess token routing for dispatch.

        This method preprocesses the hidden states and routing probabilities for the token
        dispatcher.
        """
        # Project the hidden_states from hidden dimension down to latent dimenion.
        if self.config.moe_latent_size:
            assert (
                not self.shared_expert_overlap
            ), "Shared expert overlap not supported when MoE latent projections are used."
            hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        return hidden_states, probs

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to assigned expert ranks via communication.

        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    @maybe_skip_or_early_return_by_cudagraph("shared_experts_compute")
    def shared_experts_compute(self, hidden_states: torch.Tensor):
        """Computes the output of the shared experts.

        If a shared expert is configured and not overlapped with communication,
        it is computed here.
        """
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            if self.shared_experts_recompute:
                if self.config.fp8 or self.config.fp4:
                    shared_expert_output = te_checkpoint(
                        self.shared_experts,
                        False,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        parallel_state.get_tensor_model_parallel_group(),
                        hidden_states,
                    )
                else:
                    shared_expert_output = tensor_parallel.checkpoint(
                        self.shared_experts, False, hidden_states
                    )
            else:
                shared_expert_output = self.shared_experts(hidden_states)

        return shared_expert_output

    @internal_api
    def routed_experts_compute(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Computes the output of the routed experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        The output from the experts is preprocessed for the combine step.
        """
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        output = self.token_dispatcher.combine_preprocess(expert_output)

        return output, mlp_bias

    def combine(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication). It then adds the output
        from the shared expert if it exists.
        """
        output = self.token_dispatcher.token_combine(output)
        output = self.token_dispatcher.combine_postprocess(output)
        # Project the output back from latent dimension to hidden dimension after combine
        # in latent dimension.
        if self.config.moe_latent_size:
            output, _ = self.fc2_latent_proj(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def router_and_preprocess(self, hidden_states: torch.Tensor):
        """This method is a combined method of route and preprocess. Deprecated."""

        probs, routing_map = self.route(hidden_states)
        hidden_states, probs, residual = self.preprocess(hidden_states, probs, routing_map)
        return hidden_states, probs, residual

    def _get_expert_weights_for_sglang(self):
        """Extract expert weights in SGLang format [num_experts, out_features, in_features]."""
        # Get weights from TEGroupedMLP
        # linear_fc1: [num_experts, ffn_hidden_size, hidden_size]
        # linear_fc2: [num_experts, hidden_size, ffn_hidden_size // 2] (for gated)

        # Access weights - TEGroupedLinear stores weights as individual parameters
        w1_list = []
        w2_list = []
        num_experts = self.num_local_experts
        
        # Debug: track which path was used
        w1_path = None
        w2_path = None

        for i in range(num_experts):
            # Try different ways to access weights depending on implementation
            if hasattr(self.experts, 'linear_fc1'):
                if hasattr(self.experts.linear_fc1, f'weight{i}'):
                    w1_list.append(getattr(self.experts.linear_fc1, f'weight{i}'))
                    w1_path = f'linear_fc1.weight{i}'
                elif hasattr(self.experts.linear_fc1, 'weights'):
                    w1_list.append(self.experts.linear_fc1.weights[i])
                    w1_path = 'linear_fc1.weights[i]'
                elif hasattr(self.experts.linear_fc1, 'weight'):
                    # Single weight tensor for all experts
                    w1_list.append(self.experts.linear_fc1.weight[i])
                    w1_path = 'linear_fc1.weight[i]'
            elif hasattr(self.experts, 'weight1'):
                # GroupedMLP uses weight1/weight2 directly
                # weight1: [hidden_size, ffn_hidden_size * num_experts] -> reshape to [num_experts, ffn_hidden_size, hidden_size]
                w1_reshaped = self.experts.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
                w1_list.append(w1_reshaped[i])
                w1_path = 'weight1 (GroupedMLP)'

            if hasattr(self.experts, 'linear_fc2'):
                if hasattr(self.experts.linear_fc2, f'weight{i}'):
                    w2_list.append(getattr(self.experts.linear_fc2, f'weight{i}'))
                    w2_path = f'linear_fc2.weight{i}'
                elif hasattr(self.experts.linear_fc2, 'weights'):
                    w2_list.append(self.experts.linear_fc2.weights[i])
                    w2_path = 'linear_fc2.weights[i]'
                elif hasattr(self.experts.linear_fc2, 'weight'):
                    w2_list.append(self.experts.linear_fc2.weight[i])
                    w2_path = 'linear_fc2.weight[i]'
            elif hasattr(self.experts, 'weight2'):
                # GroupedMLP uses weight1/weight2 directly
                w2_reshaped = self.experts.weight2.view(self.num_local_experts, -1, self.config.hidden_size)
                w2_list.append(w2_reshaped[i])
                w2_path = 'weight2 (GroupedMLP)'

        # Stack into [num_experts, out_features, in_features]
        w1 = torch.stack(w1_list, dim=0)
        w2 = torch.stack(w2_list, dim=0)
        
        # Debug: verify gradient connectivity
        if os.environ.get("DEBUG_EXPERT_WEIGHTS", "0") == "1" and self.layer_number in [1, 48]:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(f"[_get_expert_weights_for_sglang][Layer {self.layer_number}]")
                print(f"  w1_path: {w1_path}")
                print(f"  w2_path: {w2_path}")
                print(f"  w1.shape: {w1.shape}, w1.requires_grad: {w1.requires_grad}, w1.grad_fn: {w1.grad_fn}")
                print(f"  w2.shape: {w2.shape}, w2.requires_grad: {w2.requires_grad}, w2.grad_fn: {w2.grad_fn}")
                if w1_list:
                    print(f"  w1_list[0].requires_grad: {w1_list[0].requires_grad}, w1_list[0].grad_fn: {w1_list[0].grad_fn}")
                    print(f"  w1_list[0].is_leaf: {w1_list[0].is_leaf}")

        return w1, w2

    def _sglang_forward(self, hidden_states: torch.Tensor):
        """Forward using SGLang's fused experts for true on-policy computation.
        
        This implements SGLang's EP backend=None mode:
        - All tokens are processed on each EP rank
        - Router computes topk for all tokens (same result on each rank)
        - Expert computation only processes local experts (non-local experts skipped)
        - Results are all-reduced across EP ranks
        """
        # Compute shared experts
        shared_expert_output = self.shared_experts_compute(hidden_states)

        # Get routing (this also stores topk_weights and topk_ids in router)
        probs, routing_map = self.route(hidden_states)

        # Get topk values from router (stored during _sglang_router_forward)
        topk_weights = self.router._sglang_topk_weights
        topk_ids = self.router._sglang_topk_ids
        
        # Get EP info
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)
        
        # Debug: verify EP group and TP group consistency
        if os.environ.get("DEBUG_GRAD_ALLREDUCE", "0") == "1" and self.layer_number <= 1:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            tp_size = utils.get_pg_size(self.attn_tp_group)
            tp_rank = utils.get_pg_rank(self.attn_tp_group)
            print(f"[moe_layer._sglang_forward][Rank {rank}][Layer {self.layer_number}] "
                  f"ep_group={self.ep_group}, ep_size={ep_size}, ep_rank={ep_rank}, "
                  f"attn_tp_group={self.attn_tp_group}, tp_size={tp_size}, tp_rank={tp_rank}, "
                  f"(EP_size==TP_size: {ep_size == tp_size}), "
                  f"USING attn_tp_group for BOTH forward and backward all-reduce")

        # Debug print
        if os.environ.get("DEBUG_MEGATRON_EP_MAPPING", "0") == "1" and self.layer_number <= 1:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[moe_layer.py][Megatron _sglang_forward][Rank {rank}][Layer {self.layer_number}] EP config: ep_size={ep_size}, ep_rank={ep_rank}")
            print(f"[moe_layer.py][Megatron _sglang_forward][Rank {rank}][Layer {self.layer_number}] num_local_experts={self.num_local_experts}, "
                       f"num_moe_experts={self.config.num_moe_experts}")
            print(f"[moe_layer.py][Megatron _sglang_forward][Rank {rank}][Layer {self.layer_number}] topk_ids shape: {topk_ids.shape}, "
                       f"topk_weights shape: {topk_weights.shape}")
            print(f"[moe_layer.py][Megatron _sglang_forward][Rank {rank}][Layer {self.layer_number}] topk_ids (first 5 tokens): {topk_ids[:5].tolist()}")

        # Reshape hidden_states if needed
        original_shape = hidden_states.shape
        if len(original_shape) == 3:
            hidden_states_2d = hidden_states.view(-1, original_shape[-1])
        else:
            hidden_states_2d = hidden_states

        # Get expert weights (only local experts)
        w1, w2 = self._get_expert_weights_for_sglang()

        # Debug: verify weight shapes and values
        # Note: Megatron layer_number=1 corresponds to SGLang layer_id=0
        debug_experts = os.environ.get("SLIME_DEBUG_ATTN", "0") == "1" and self.layer_number == 1 and utils.get_pg_rank(self.tp_group) == 1
        if debug_experts:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            tp_rank = utils.get_pg_rank(self.tp_group)
            tp_size = utils.get_pg_size(self.tp_group)
            pos = 91  # Match SGLang's pos
            prefix = f"[moe_layer.py][Megatron MoE][TP {tp_rank}/{tp_size}][Layer {self.layer_number}]"
            hs = hidden_states_2d
            tw, ti = topk_weights, topk_ids
            print(f"{prefix} ===== EXPERT INPUT ===== shape={hs.shape}, dtype={hs.dtype}, "
                  f"hs[{pos},:5]={hs[pos, :5].tolist()}, hs[{pos}] sum={hs[pos].float().sum().item():.6f}, std={hs[pos].float().std().item():.6f}, "
                  f"ALL sum={hs.float().sum().item():.6f}, ALL std={hs.float().std().item():.6f}, "
                  f"topk_weights.dtype={tw.dtype}, topk_ids.dtype={ti.dtype}, "
                  f"weights[{pos}]={tw[pos, :].tolist()}, ids[{pos}]={ti[pos, :].tolist()}")
            print(f"{prefix} ===== EXPERT WEIGHTS ===== w1.shape={w1.shape}, w1.dtype={w1.dtype}, w1[0,0,:5]={w1[0, 0, :5].tolist()}, "
                  f"w1 sum={w1.float().sum().item():.6f}, w1 std={w1.float().std().item():.6f}, "
                  f"w2.shape={w2.shape}, w2[0,0,:5]={w2[0, 0, :5].tolist()}, "
                  f"w2 sum={w2.float().sum().item():.6f}, w2 std={w2.float().std().item():.6f}")

        # Call SGLang's fused experts with EP parameters
        output = sglang_fused_experts(
            layer_number=self.layer_number,
            hidden_states=hidden_states_2d,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="silu",  # Qwen uses SwiGLU
            # EP parameters
            num_experts=self.config.num_moe_experts,
            num_local_experts=self.num_local_experts,
            ep_rank=ep_rank,
            ep_size=ep_size,
            ep_group=self.attn_tp_group,  # Use TP group for gradient all-reduce to match forward all-reduce
        )

        # DEBUG: Expert output before all-reduce
        if debug_experts:
            out = output
            print(f"{prefix} ===== EXPERT OUTPUT (BEFORE all-reduce) ===== shape={out.shape}, dtype={out.dtype}, "
                  f"out[{pos},:5]={out[pos, :5].tolist()}, out[{pos},-5:]={out[pos, -5:].tolist()}, "
                  f"sum={out[pos].float().sum().item():.6f}, mean={out[pos].float().mean().item():.6f}, "
                  f"std={out[pos].float().std().item():.6f}, max={out[pos].float().max().item():.6f}, "
                  f"min={out[pos].float().min().item():.6f}, nonzero={int((out[pos] != 0).sum().item())}, "
                  f"ALL sum={out.float().sum().item():.6f}, ALL std={out.float().std().item():.6f}")

        # MoE all-reduce: use TP group to match SGLang's tensor_model_parallel_tree_all_reduce
        # SGLang uses get_tp_group() for MoE all-reduce, so we use attn_tp_group (which is pg_collection.tp)
        tp_size = utils.get_pg_size(self.attn_tp_group)
        if tp_size > 1:
            output = _tree_all_reduce_sum(output, self.attn_tp_group, layer_id=self.layer_number)
            
            # DEBUG: Expert output after TP all-reduce
            if debug_experts:
                print(f"{prefix} ===== EXPERT OUTPUT (AFTER all-reduce) ===== out[{pos},:5]={output[pos, :5].tolist()}, sum={output[pos].float().sum().item():.6f}")

        # Reshape output if needed
        if len(original_shape) == 3:
            output = output.view(original_shape[0], original_shape[1], -1)

        # Add shared expert output
        if shared_expert_output is not None:
            output = output + shared_expert_output

        return output, None  # mlp_bias is None

    def forward(self, hidden_states: torch.Tensor):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        # DEBUG: MoE input
        debug_moe = os.environ.get("SLIME_DEBUG_ATTN", "0") == "1" and self.layer_number == 1
        if debug_moe:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            tp_rank = utils.get_pg_rank(self.tp_group)
            tp_size = utils.get_pg_size(self.tp_group)
            pos = 91
            # Flatten to 2D for consistent logging
            hs_2d = hidden_states.view(-1, hidden_states.shape[-1]) if len(hidden_states.shape) == 3 else hidden_states
            prefix = f"[moe_layer.py][Megatron MoE][Rank {rank}][TP {tp_rank}/{tp_size}][Layer {self.layer_number}]"
            print(f"{prefix} MoE INPUT hidden_states[{pos},:5]: {hs_2d[pos, :5].tolist()}")
            print(f"{prefix} MoE INPUT hidden_states sum: {hs_2d[pos].float().sum().item():.6f}")

        # if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
        #     raise ValueError(
        #         "During training, performance may degrade if MoE and tensor parallelism"
        #         "are enabled without also enabling sequence parallelism."
        #     )

        # Use SGLang fused experts when use_sglang_router is enabled
        use_sglang_experts = (
            getattr(self.config, 'use_sglang_router', False)
            and HAVE_SGLANG_FUSED_EXPERTS
        )

        if use_sglang_experts:
            # Use SGLang's fused expert computation for true on-policy
            def custom_forward(hidden_states):
                return self._sglang_forward(hidden_states)
        else:
            # MoE forward: route -> dispatch -> compute -> combine
            def custom_forward(hidden_states):
                try:
                    shared_expert_output = self.shared_experts_compute(hidden_states)
                    probs, routing_map = self.route(hidden_states)
                    hidden_states, probs = self.preprocess(hidden_states, probs, routing_map)
                except MoECudaGraphPartialCaptureSignal as e:
                    # This signal is raised from the maybe_skip_or_early_return_by_cudagraph decorator.
                    # It means we should early-return from the MoE layer forward pass.
                    # This happens when we are partially capturing the CUDA graph of the MoE layer,
                    # like cuda_graph_scope=["moe_router", "moe_preprocess"].
                    # We need to return the intermediate tensors as CUDA graph outputs.
                    return e.get_early_return_outputs(hidden_states, shared_expert_output)

                dispatched_input, probs = self.dispatch(hidden_states, probs)
                output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)
                assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
                output = self.combine(output, shared_expert_output)

                return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8 or self.config.fp4:
                outputs = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            else:
                outputs = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            outputs = custom_forward(hidden_states)

        # DEBUG: MoE output
        if debug_moe:
            output_tensor = outputs[0] if isinstance(outputs, tuple) else outputs
            out_2d = output_tensor.view(-1, output_tensor.shape[-1]) if len(output_tensor.shape) == 3 else output_tensor
            print(f"{prefix} MoE OUTPUT[{pos},:5]: {out_2d[pos, :5].tolist()}")
            print(f"{prefix} MoE OUTPUT sum: {out_2d[pos].float().sum().item():.6f}")

        return outputs

    def backward_dw(self):
        """Compute weight gradients for experts and shared experts."""
        self.experts.backward_dw()
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()

    def set_for_recompute_pre_mlp_layernorm(self):
        """Set the MoE layer for recompute pre_mlp_layernorm. Only needed for fp8/fp4."""
        # If shared_experts_recompute is used, nothing needs to be done because the checkpoint
        # function will save the original input tensors.
        if self.shared_experts is not None and not self.shared_experts_recompute:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.shared_experts.linear_fc1)
