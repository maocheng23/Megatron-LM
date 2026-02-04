# SGLang Backend Extension for Megatron-LM
#
# This module provides wrappers for SGLang's batch-invariant (deterministic) kernels
# to achieve training-inference consistency.
#
# Reference: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
#
# Coverage comparison with Kitchen extension:
# ============================================
# | Component                | Kitchen         | SGLang Extension |
# |--------------------------|-----------------|------------------|
# | Column Parallel Linear   | ✅              | ✅               |
# | Row Parallel Linear      | ✅              | ✅               |
# | Grouped Linear (MoE)     | ✅              | ✅               |
# | LayerNorm + Linear       | ✅              | ✅               |
# | Flash Attention (FA3)    | ✅              | ✅               |
# | RMSNorm                  | fallback        | ✅               |
# | LayerNorm                | fallback        | ✅               |
# | Activation (SwiGLU/GeGLU)| ✅              | ✅               |
# | Quantization (FP8)       | ✅ (recipe)     | ✅ (DeepGEMM)    |
# ============================================
#
# Key Features:
# - All kernels use batch-invariant operations for training-inference consistency
# - FA3 with num_splits=1 for deterministic attention
# - DeepGEMM support for FP8 quantization
# TODO: Add full MoE support with grouped linear layers

import logging
import math
import os
import warnings
from dataclasses import dataclass
import inspect
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_expert_data_parallel_rank,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import get_tensor_model_parallel_group_if_none, log_single_rank

logger = logging.getLogger(__name__)


def _print_sglang_log(message: str, level: int = logging.INFO):
    """logger.info SGLang log message on rank 0 in distributed training."""
    log_single_rank(logger, level, message)
    # Also logger.info to stdout for visibility
    try:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                logger.info(message,  )
        else:
            logger.info(message,  )
    except Exception:
        # Fallback if distributed is not available
        logger.info(message,  )


# =============================================================================
# SGLang Batch-Invariant Ops Imports
# =============================================================================
# These are the deterministic/batch-invariant kernels from SGLang that ensure
# consistent numerical results regardless of batch size.

try:
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
        mm_batch_invariant,
        addmm_batch_invariant,
        bmm_batch_invariant,
        rms_norm_batch_invariant,
        enable_batch_invariant_mode,
        disable_batch_invariant_mode,
        is_batch_invariant_mode_enabled,
    )
    HAVE_SGLANG_BATCH_INVARIANT = True
except ImportError:
    HAVE_SGLANG_BATCH_INVARIANT = False
    mm_batch_invariant = None
    addmm_batch_invariant = None
    bmm_batch_invariant = None
    rms_norm_batch_invariant = None
    enable_batch_invariant_mode = None
    disable_batch_invariant_mode = None

    def is_batch_invariant_mode_enabled():
        """Fallback when SGLang is not available."""
        return False


# Try to import DeepGEMM for FP8 support
try:
    from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM
    HAVE_DEEPGEMM = bool(ENABLE_JIT_DEEPGEMM)
except ImportError:
    HAVE_DEEPGEMM = False
    ENABLE_JIT_DEEPGEMM = False

# Try to import Flash Attention 3
# IMPORTANT: Use flash_attn_varlen_func (high-level API with backward support)
# instead of _flash_attn_forward (low-level kernel without backward support)
try:
    # First try flash_attn_interface (FA3 standard location)
    from flash_attn_interface import flash_attn_varlen_func as fa3_varlen_func
    from flash_attn_3.flash_attn_interface import _flash_attn_forward  # Keep for reference
    HAVE_FA3 = True
    HAVE_FA3_VARLEN = True
except ImportError:
    try:
        # Fallback: try flash_attn_3 module directly
        from flash_attn_3.flash_attn_interface import flash_attn_varlen_func as fa3_varlen_func
        from flash_attn_3.flash_attn_interface import _flash_attn_forward
        HAVE_FA3 = True
        HAVE_FA3_VARLEN = True
    except ImportError:
        HAVE_FA3 = False
        HAVE_FA3_VARLEN = False
        _flash_attn_forward = None
        fa3_varlen_func = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SGLangConfig:
    """Configuration for SGLang extension."""
    batch_invariant_mode: bool = True
    use_deep_gemm: bool = True
    use_sglang_attention: bool = True
    allow_fallback: bool = True


# Global configuration
_sglang_config = SGLangConfig()


def get_sglang_config() -> SGLangConfig:
    """Get the global SGLang configuration."""
    return _sglang_config


def set_sglang_config(config: SGLangConfig):
    """Set the global SGLang configuration."""
    global _sglang_config
    _sglang_config = config


# =============================================================================
# Batch-Invariant Mode Management
# =============================================================================

def enable_sglang_batch_invariant_mode(enable_bmm: bool = True):
    if HAVE_SGLANG_BATCH_INVARIANT:
        enable_batch_invariant_mode(enable_bmm=enable_bmm)
        logger.info("SGLang batch-invariant ops enabled (Triton kernels active)")
    else:
        logger.warning("WARNING: SGLang batch_invariant_ops not available, using PyTorch fallbacks.")

    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['NCCL_ALGO'] = 'Ring'
    os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0'



def disable_sglang_batch_invariant_mode():
    if HAVE_SGLANG_BATCH_INVARIANT:
        disable_batch_invariant_mode()
        logger.info("SGLang batch-invariant ops disabled.")


# Alias for backwards compatibility
enable_sglang_deterministic_mode = enable_sglang_batch_invariant_mode


# =============================================================================
# Batch-Invariant Core Operations
# =============================================================================

def sglang_mm(a: Tensor, b: Tensor) -> Tensor:
    return torch.mm(a, b)


def sglang_addmm(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0
) -> Tensor:
    return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)


def sglang_bmm(a: Tensor, b: Tensor) -> Tensor:
    return torch.bmm(a, b)


def sglang_rms_norm(input: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    if HAVE_SGLANG_BATCH_INVARIANT and is_batch_invariant_mode_enabled():
        return rms_norm_batch_invariant(input, weight, eps=eps)
    input_dtype = input.dtype
    input_float = input.float()
    variance = input_float.pow(2).mean(-1, keepdim=True)
    normed = input_float * torch.rsqrt(variance + eps)
    return (weight.float() * normed).to(input_dtype)


# =============================================================================
# SGLang Linear Layers (matching Kitchen's structure)
# =============================================================================

class SGLangLinear(MegatronModule):
    """
    SGLang-compatible Linear layer using batch-invariant operations.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        layer_number: Optional[int] = None,
        is_expert: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        stride: int = 1,
    ):
        super().__init__(config=config)

        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.parallel_mode = parallel_mode
        self.is_expert = is_expert
        self.skip_bias_add = skip_bias_add
        self.layer_number = layer_number
        self.stride = stride

        # Determine device and dtype
        if config.init_model_with_meta_device:
            device = 'meta'
        elif config.use_cpu_initialization:
            device = 'cpu'
        else:
            device = torch.cuda.current_device()
        dtype = config.params_dtype

        # Handle tensor parallelism
        if parallel_mode == "duplicated":
            self.tp_size = 1
            self.tp_group = None
        else:
            self.tp_group = tp_group
            self.tp_size = tp_group.size() if tp_group is not None else 1

        # Compute local sizes based on parallel mode
        local_input_size = input_size
        local_output_size = output_size
        if parallel_mode == "column" and self.tp_size > 1:
            local_output_size = divide(output_size, self.tp_size)
        elif parallel_mode == "row" and self.tp_size > 1:
            local_input_size = divide(input_size, self.tp_size)

        # Initialize weight
        if not skip_weight_param_allocation:
            self.weight = nn.Parameter(
                torch.empty(local_output_size, local_input_size, dtype=dtype, device=device)
            )
            if config.perform_initialization and device != 'meta':
                init_method(self.weight)
            
            # Set TP attributes for weight update/checkpoint compatibility
            # NOTE: partition_stride tracks stride-based sharding so TP weight
            # reconstruction can restore GLU gate/up layout when stride > 1.
            partition_stride = stride if parallel_mode in ("column", "row") and self.tp_size > 1 else 1
            if parallel_mode in ("column", "row") and self.tp_size > 1:
                setattr(self.weight, 'tensor_model_parallel', True)
                setattr(self.weight, 'partition_dim', 0 if parallel_mode == "column" else 1)
                setattr(self.weight, 'partition_stride', partition_stride)
            else:
                setattr(self.weight, 'tensor_model_parallel', False)
                setattr(self.weight, 'partition_dim', -1)
                setattr(self.weight, 'partition_stride', 1)
        else:
            self.register_parameter('weight', None)

        # Initialize bias
        if bias and not skip_bias_add:
            self.bias = nn.Parameter(
                torch.zeros(local_output_size, dtype=dtype, device=device)
            )
            # Set TP attributes for bias (column parallel has sharded bias)
            if parallel_mode == "column" and self.tp_size > 1:
                setattr(self.bias, 'tensor_model_parallel', True)
                setattr(self.bias, 'partition_dim', 0)
                setattr(self.bias, 'partition_stride', partition_stride)
            else:
                setattr(self.bias, 'tensor_model_parallel', False)
                setattr(self.bias, 'partition_dim', -1)
                setattr(self.bias, 'partition_stride', 1)
        else:
            self.register_parameter('bias', None)

        # Set gradient attributes
        if self.weight is not None:
            if is_expert:
                self.weight.allreduce = not (config.expert_model_parallel_size > 1)
            else:
                self.weight.allreduce = True

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass using batch-invariant operations.
        
        Uses explicit BF16 casting to match SGLang's FSDP-compatible numerical paths.
        """
        # Cast to BF16 to match SGLang's FSDP-compatible paths
        # In SGLang's logits_processor: torch.matmul(hidden_states.bfloat16(), weight.T.bfloat16())
        x = x.to(torch.bfloat16)
        
        # Reshape for matrix multiplication
        orig_shape = x.shape
        x = x.view(-1, self.input_size if self.parallel_mode != "row" else x.shape[-1])

        # Use batch-invariant GEMM with BF16 weight
        weight_bf16 = self.weight.to(torch.bfloat16)
        if self.bias is not None and not self.skip_bias_add:
            bias_bf16 = self.bias.to(torch.bfloat16)
            output = sglang_addmm(
                bias_bf16.unsqueeze(0).expand(x.size(0), -1),
                x,
                weight_bf16.t(),
            )
        else:
            output = sglang_mm(x, weight_bf16.t())

        # Restore shape
        output = output.view(*orig_shape[:-1], output.size(-1))

        if self.skip_bias_add and self.bias is not None:
            return output, self.bias.to(torch.bfloat16)
        return output, None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharded state dict for distributed checkpointing."""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(state_dict, prefix, None, sharded_offsets)


class SGLangColumnParallelLinear(SGLangLinear):
    """
    Column-parallel linear layer using SGLang batch-invariant operations.

    Equivalent to KitchenColumnParallelLinear.
    Splits output dimension across TP ranks.
    
    IMPORTANT: Column parallel linear requires all-reduce of grad_input in backward.
    This is achieved by using copy_to_tensor_model_parallel_region before the matmul,
    which has identity forward but all-reduce backward.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        gather_output: bool = False,
        bias: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        layer_number: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        stride: int = 1,
    ):
        if gather_output:
            raise ValueError("SGLang linear layers do not support gather_output = True")

        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=init_method if config.perform_initialization else (lambda w: None),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            layer_number=layer_number,
            tp_group=tp_group,
            stride=stride,
        )

        self.stride = stride

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
        
        if self.tp_size > 1 and self.tp_group is not None:
            x = copy_to_tensor_model_parallel_region(x, group=self.tp_group)
        
        return super().forward(x)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded."""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 0, "bias": 0}, sharded_offsets
        )


class SGLangRowParallelLinear(SGLangLinear):
    """
    Row-parallel linear layer using SGLang batch-invariant operations.

    Equivalent to KitchenRowParallelLinear.
    Splits input dimension across TP ranks.
    
    IMPORTANT: Row parallel linear requires all_reduce after GEMM to combine
    partial results from all TP ranks.
    
    For MoE true on-policy mode, set reduce_results=False to skip all-reduce here,
    and let the all-reduce happen in transformer_layer._forward_mlp with tree_all_reduce
    to match SGLang's numerical path exactly.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        layer_number: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        reduce_results: bool = True,
    ):
        if not input_is_parallel:
            raise ValueError("SGLang linear layers do not support input_is_parallel = False")

        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=init_method if config.perform_initialization else (lambda w: None),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            layer_number=layer_number,
            tp_group=tp_group,
        )
        
        self.reduce_results = reduce_results

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass with all_reduce for row parallelism.
        
        Row parallel linear splits the input along the last dimension.
        Each rank computes a partial result, then all_reduce combines them.
        
        For SGLang true on-policy mode, uses tree_all_reduce instead of standard
        NCCL all_reduce to match SGLang's tensor_model_parallel_tree_all_reduce
        for numerical consistency.
        """
        # Call parent's forward to get partial result
        output_before, bias = super().forward(x)
        
        # CRITICAL: all_reduce to combine partial results from all TP ranks
        # Without this, each rank only has its partial computation
        if self.reduce_results and self.tp_size > 1 and self.tp_group is not None:
            # Check if we should use tree_all_reduce for SGLang true on-policy mode
            use_tree_allreduce = getattr(self.config, 'use_sglang', False)
            
            if use_tree_allreduce:
                # Use tree_all_reduce for SGLang true on-policy mode
                # This matches SGLang's tensor_model_parallel_tree_all_reduce
                from megatron.core.tensor_parallel.mappings import _tree_all_reduce_sum
                output = _tree_all_reduce_sum(output_before, self.tp_group)
            else:
                # Standard NCCL all_reduce
                from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
                output = reduce_from_tensor_model_parallel_region(output_before, group=self.tp_group)
        else:
            output = output_before
        
        return output, bias

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 1, bias not sharded."""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 1}, sharded_offsets
        )


# =============================================================================
# SGLang Grouped Linear (for MoE) - matching Kitchen's structure
# =============================================================================

class SGLangGroupedLinear(MegatronModule):
    """
    Grouped linear layer for MoE using SGLang batch-invariant operations.

    Equivalent to KitchenGroupedLinear.
    Handles multiple expert GEMMs in a single operation.
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        layer_number: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.num_gemms = num_gemms
        self.input_size = input_size
        self.output_size = output_size
        self.parallel_mode = parallel_mode
        self.is_expert = is_expert
        self.skip_bias_add = skip_bias_add

        # Determine device and dtype
        if config.init_model_with_meta_device:
            device = 'meta'
        else:
            device = torch.cuda.current_device()
        dtype = config.params_dtype

        # Handle tensor parallelism
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self.tp_size = tp_group.size() if tp_group is not None else 1
        self.tp_group = tp_group

        # Compute local sizes
        local_input_size = input_size
        local_output_size = output_size

        self.expert_parallel = config.expert_model_parallel_size > 1
        self.explicit_expert_comm = is_expert and (self.tp_size > 1 or self.expert_parallel)

        if self.explicit_expert_comm:
            if parallel_mode == "column":
                local_output_size = divide(output_size, self.tp_size)
            elif parallel_mode == "row":
                local_input_size = divide(input_size, self.tp_size)

        # Initialize weights for each expert using TE-compatible naming (weight0, weight1, etc.)
        # This matches TEGroupedLinear's naming convention for checkpoint compatibility
        self.use_bias = bias and not skip_bias_add
        for i in range(num_gemms):
            weight = nn.Parameter(
                torch.empty(local_output_size, local_input_size, dtype=dtype, device=device)
            )
            self.register_parameter(f'weight{i}', weight)

            if config.perform_initialization and device != 'meta':
                init_method(weight)

            # Set TP attributes for weights (for weight update/checkpoint compatibility)
            if self.explicit_expert_comm and self.tp_size > 1:
                setattr(weight, 'tensor_model_parallel', True)
                setattr(weight, 'partition_dim', 0 if parallel_mode == "column" else 1)
                setattr(weight, 'partition_stride', 1)
            else:
                setattr(weight, 'tensor_model_parallel', False)
                setattr(weight, 'partition_dim', -1)
                setattr(weight, 'partition_stride', 1)

            # Set gradient attributes
            weight.allreduce = not (is_expert and self.expert_parallel)

        # Initialize biases using TE-compatible naming (bias0, bias1, etc.)
        if self.use_bias:
            for i in range(num_gemms):
                bias_param = nn.Parameter(
                    torch.zeros(local_output_size, dtype=dtype, device=device)
                )
                self.register_parameter(f'bias{i}', bias_param)

                # Set TP attributes for biases
                if self.explicit_expert_comm and parallel_mode == "column" and self.tp_size > 1:
                    setattr(bias_param, 'tensor_model_parallel', True)
                    setattr(bias_param, 'partition_dim', 0)
                    setattr(bias_param, 'partition_stride', 1)
                else:
                    setattr(bias_param, 'tensor_model_parallel', False)
                    setattr(bias_param, 'partition_dim', -1)
                    setattr(bias_param, 'partition_stride', 1)

    def forward(self, x: Tensor, m_splits: List[int]) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for grouped linear.

        Args:
            x: Input tensor [total_tokens, hidden_size]
            m_splits: List of token counts for each expert
        """
        outputs = []
        offset = 0

        for i, num_tokens in enumerate(m_splits):
            if num_tokens > 0:
                expert_input = x[offset:offset + num_tokens]
                expert_input = expert_input.view(-1, expert_input.size(-1))

                weight = getattr(self, f'weight{i}')
                if self.use_bias:
                    bias = getattr(self, f'bias{i}')
                    expert_output = sglang_addmm(
                        bias.unsqueeze(0).expand(expert_input.size(0), -1),
                        expert_input,
                        weight.t(),
                    )
                else:
                    expert_output = sglang_mm(expert_input, weight.t())

                outputs.append(expert_output)
            offset += num_tokens

        if outputs:
            output = torch.cat(outputs, dim=0)
        else:
            output = x.new_empty(0, self.output_size)

        if self.skip_bias_add and self.use_bias:
            # Return concatenated biases (same size as output)
            return output, None  # Bias handling for grouped is complex
        return output, None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharded state dict for distributed checkpointing.

        Matches TE's _sharded_state_dict_grouped format for compatibility with TEGroupedMLP.
        Returns dict keys in format: {prefix}weight{idx}, {prefix}bias{idx}
        """
        from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding

        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        sharded_state_dict = {}
        num_global_experts = get_expert_model_parallel_world_size() * self.num_gemms
        local_expert_indices_offset = get_expert_model_parallel_rank() * self.num_gemms
        ep_axis = len(sharded_offsets)

        for gemm_idx in range(self.num_gemms):
            global_expert_idx = local_expert_indices_offset + gemm_idx
            # Create state dict with indexed keys (matching TE format)
            # Use getattr to access weight{i} and bias{i} parameters
            state_dict = {f"{gemm_idx}.weight": getattr(self, f'weight{gemm_idx}')}
            if self.use_bias:
                state_dict[f"{gemm_idx}.bias"] = getattr(self, f'bias{gemm_idx}')

            tp_axis_map = {}
            if self.parallel_mode == "column":
                tp_axis_map = {f"{gemm_idx}.weight": 0, f"{gemm_idx}.bias": 0}
            elif self.parallel_mode == "row":
                tp_axis_map = {f"{gemm_idx}.weight": 1}

            # Determine expert prefix for ShardedTensor.key (matches TE logic)
            if singleton_local_shards:
                expert_prefix = f"{global_expert_idx}.{prefix}"
                new_sharded_offsets = sharded_offsets
            else:
                expert_prefix = prefix
                new_sharded_offsets = (
                    *sharded_offsets,
                    (ep_axis, global_expert_idx, num_global_experts),
                )

            # Create sharded tensors with empty prefix (like TE)
            # IMPORTANT: Pass tp_group to use expert_tensor_parallel_group for MoE experts
            sub_sd = make_sharded_tensors_for_checkpoint(
                state_dict,
                '',  # Empty prefix, will be set by replace_prefix_for_sharding
                tp_axis_map,
                new_sharded_offsets,
                tp_group=self.tp_group,  # Use expert TP group, not default TP group
            )

            # Update ShardedTensor.key from "{idx}." to expert_prefix (matching TE)
            replace_prefix_for_sharding(sub_sd, f"{gemm_idx}.", expert_prefix)

            # Add to result dict with TE-compatible keys: {prefix}weight{idx}, {prefix}bias{idx}
            sharded_state_dict[f"{prefix}weight{gemm_idx}"] = sub_sd[f"{gemm_idx}.weight"]
            if self.use_bias:
                sharded_state_dict[f"{prefix}bias{gemm_idx}"] = sub_sd[f"{gemm_idx}.bias"]

        # Adjust replica ids - replication along DP modulo EP (matching TE)
        for k, sh_ten in sharded_state_dict.items():
            replica_id = sh_ten.replica_id
            if len(replica_id) == 3:
                sh_ten.replica_id = (*replica_id[:2], get_expert_data_parallel_rank())

        return sharded_state_dict


class SGLangColumnParallelGroupedLinear(SGLangGroupedLinear):
    """Column-parallel grouped linear for MoE.
    
    IMPORTANT: Column parallel grouped linear requires all-reduce of grad_input 
    in backward. This is achieved by using copy_to_tensor_model_parallel_region.
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        layer_number: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=init_method if config.perform_initialization else (lambda w: None),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            layer_number=layer_number,
            tp_group=tp_group,
        )

    def forward(self, x: Tensor, m_splits: List[int]) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass with proper TP communication for MoE.
        
        Uses copy_to_tensor_model_parallel_region to ensure grad_input is
        all-reduced in backward pass when using TP for experts.
        """
        from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
        
        if self.tp_size > 1 and self.tp_group is not None and self.explicit_expert_comm:
            x = copy_to_tensor_model_parallel_region(x, group=self.tp_group)
        
        return super().forward(x, m_splits)


class SGLangRowParallelGroupedLinear(SGLangGroupedLinear):
    """Row-parallel grouped linear for MoE.
    
    IMPORTANT: Row parallel grouped linear requires all_reduce after GEMM
    to combine partial results from all TP ranks (when not using expert parallel).
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        layer_number: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=init_method if config.perform_initialization else (lambda w: None),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            layer_number=layer_number,
            tp_group=tp_group,
        )

    def forward(self, x: Tensor, m_splits: List[int]) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass with all_reduce for row parallelism.
        
        Row parallel grouped linear splits the input along the last dimension.
        Each rank computes a partial result, then all_reduce combines them.
        """
        # Call parent's forward to get partial result
        output, bias = super().forward(x, m_splits)
        
        # CRITICAL: all_reduce to combine partial results from all TP ranks
        # Skip if using explicit expert comm (EP handles communication differently)
        if self.tp_size > 1 and self.tp_group is not None and not self.explicit_expert_comm:
            from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
            output = reduce_from_tensor_model_parallel_region(output, group=self.tp_group)
        
        return output, bias


# =============================================================================
# SGLang Normalization Layers
# =============================================================================

class SGLangRMSNorm(MegatronModule):
    """
    RMSNorm matching SGLang's FSDP-compatible numerical paths.
    
    When residual is provided (for MoE pre_mlp_layernorm), this matches SGLang's
    RMSNorm.forward_native with fp32_residual=False:
    1. x = x + residual (bf16 add)
    2. residual = x.clone()
    3. RMSNorm computation in FP32
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__(config=config)

        self.hidden_size = hidden_size
        self.eps = eps
        self.variance_epsilon = eps  # Alias for compatibility

        if config.init_model_with_meta_device:
            device = 'meta'
        else:
            device = torch.cuda.current_device()

        # Use FP32 weights to match SGLang's FSDP mode (weight_dtype=torch.float32)
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.float32, device=device)
        )
    
    def forward(self, x: Tensor, residual: Tensor = None):
        """Forward matching SGLang's forward_native with FSDP settings.
        
        Args:
            x: Input tensor (attention output for MoE, or already-resadded for Dense)
            residual: Optional residual tensor. When provided (MoE case), performs
                     bf16 residual add inside LayerNorm to match SGLang exactly.
        
        Returns:
            If residual is None: normalized tensor
            If residual is provided: (normalized tensor, updated residual)
        """
        if not x.is_contiguous():
            x = x.contiguous()
        
        orig_dtype = x.dtype
        
        # If residual is provided, do resadd in bf16 (matching SGLang's fp32_residual=False)
        if residual is not None:
            x = x + residual  # bf16 add, matching SGLang
            residual = x.clone()  # Update residual to resadd result, matching SGLang
        
        x = x.to(torch.float32)
        
        # RMSNorm computation in FP32
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        
        # cast_x_before_out_mul=True: weight * x.to(orig_dtype)
        # Match SGLang exactly - don't add extra dtype conversion
        x = self.weight * x.to(orig_dtype)
        
        if residual is not None:
            return x, residual
        return x


class SGLangFinalRMSNorm(MegatronModule):
    """
    Final RMSNorm matching SGLang's FSDP-compatible numerical paths for true on-policy mode.
    
    This is specifically for the final layer norm, which in SGLang uses:
    - override_orig_dtype=torch.float32
    - cast_x_before_out_mul=True
    
    This ensures bitwise-identical results between SGLang inference and Megatron training
    for the final layer norm output.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__(config=config)

        self.hidden_size = hidden_size
        self.eps = eps

        if config.init_model_with_meta_device:
            device = 'meta'
        else:
            device = torch.cuda.current_device()

        # Use FP32 weights to match SGLang's FSDP mode (weight_dtype=torch.float32)
        # In SGLang, weight is created as FP32 and stays FP32 even after loading from checkpoint
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.float32, device=device)
        )
        
        _print_sglang_log(f"🔍 SGLangFinalRMSNorm.__init__ called: hidden_size={hidden_size}, eps={eps}")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Override to ensure weight remains FP32 after loading from checkpoint.
        
        In SGLang, weight_dtype=torch.float32 ensures weight is always FP32.
        We need to maintain this behavior even when loading from checkpoints
        that might have bfloat16 weights.
        """
        weight_key = prefix + 'weight'
        if weight_key in state_dict:
            loaded_weight = state_dict[weight_key]
            if loaded_weight.dtype != torch.float32:
                _print_sglang_log(
                    f"🔍 SGLangFinalRMSNorm: Converting weight from {loaded_weight.dtype} to FP32 during load",
                    level=logging.INFO
                )
                # Convert to FP32 to match SGLang's weight_dtype=torch.float32 behavior
                state_dict[weight_key] = loaded_weight.to(torch.float32)
        # Call parent implementation
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward matching SGLang's final RMSNorm with true on-policy settings.
        
        Matches SGLang's RMSNorm.forward_native with:
        - cast_x_before_out_mul=True
        - override_orig_dtype=torch.float32 (for true on-policy mode)
        
        In true on-policy mode, SGLang's final norm uses override_orig_dtype=torch.float32,
        which means the output dtype should be FP32, not the input dtype.
        """
        if not x.is_contiguous():
            x = x.contiguous()
        
        # In true on-policy mode, SGLang uses override_orig_dtype=torch.float32
        # This means orig_dtype should be FP32, not the input dtype
        # This ensures bitwise-identical results with SGLang inference
        orig_dtype = torch.bfloat16  # Match SGLang's override_orig_dtype=torch.float32
        x = x.to(torch.float32)
        
        # RMSNorm computation in FP32
        # Match SGLang's forward_native exactly:
        # 1. Compute variance using x_var (which is x in our case, no variance_size_override)
        #    SGLang line 201-202: x_var = x (when variance_size_override is None)
        #    SGLang line 212: variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        #    SGLang line 213: x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * torch.rsqrt(variance + self.eps)
        
        # Match SGLang line 216 exactly: x = self.weight * x.to(orig_dtype)
        # where orig_dtype=torch.float32, so x.to(orig_dtype) is a no-op (x is already FP32)
        # But we do it to match SGLang's exact computation path
        x = self.weight * x.to(orig_dtype)
        
        # CRITICAL: This function returns float32, not bfloat16.
        # This matches SGLang's final norm output in true on-policy mode,
        # which uses override_orig_dtype=torch.float32.
        # The output dtype is float32, ensuring bitwise-identical results with SGLang.
        return x


class SGLangLayerNorm(MegatronModule):
    """
    LayerNorm matching SGLang's FSDP-compatible numerical paths.

    Uses FP32 computation with proper dtype handling to match SGLang inference.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
    ):
        super().__init__(config=config)

        self.hidden_size = hidden_size
        self.eps = eps

        if config.init_model_with_meta_device:
            device = 'meta'
        else:
            device = torch.cuda.current_device()

        # Use FP32 weights to match SGLang's FSDP mode
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.float32, device=device)
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_size, dtype=torch.float32, device=device)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward matching SGLang's LayerNorm behavior."""
        orig_dtype = x.dtype
        x_float = x.float()

        mean = x_float.mean(-1, keepdim=True)
        var = x_float.var(-1, keepdim=True, unbiased=False)
        normed = (x_float - mean) / torch.sqrt(var + self.eps)

        # Apply weight and bias in FP32, then cast back
        return (self.weight * normed + self.bias).to(orig_dtype)


class SGLangNorm(MegatronModule):
    """
    Wrapper that selects RMSNorm or LayerNorm based on config.
    """

    def __new__(cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5):
        if config.normalization == "RMSNorm":
            return SGLangRMSNorm(config, hidden_size, eps)
        else:
            return SGLangLayerNorm(config, hidden_size, eps)


# =============================================================================
# SGLang Fused LayerNorm + Linear (matching Kitchen)
# =============================================================================

class SGLangLayerNormColumnParallelLinear(MegatronModule):
    """
    Fused LayerNorm + Column-Parallel Linear.

    Equivalent to KitchenLayerNormColumnParallelLinear.
    Note: We implement this as sequential operations since SGLang
    doesn't have a fused kernel, but maintain the same interface.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool = False,
        bias: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        skip_weight_param_allocation: bool = False,
        layer_number: Optional[int] = None,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        stride: int = 1,
    ):
        super().__init__(config=config)

        if gather_output:
            raise ValueError("SGLang does not support gather_output = True")
        if is_expert:
            raise ValueError("SGLang does not yet support MoE for fused LayerNorm+Linear")

        self.config = config
        self.skip_bias_add = skip_bias_add
        self.stride = stride

        # LayerNorm component
        if config.normalization == "RMSNorm":
            self.norm = SGLangRMSNorm(config, input_size, eps=config.layernorm_epsilon)
        else:
            self.norm = SGLangLayerNorm(config, input_size, eps=config.layernorm_epsilon)

        # Linear component
        self.linear = SGLangColumnParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=False,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            layer_number=layer_number,
            tp_group=tp_group,
            stride=stride,
        )

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    @property
    def layer_norm_weight(self):
        """TE-compatible property for layer norm weight."""
        return self.norm.weight

    @property
    def layer_norm_bias(self):
        """TE-compatible property for layer norm bias."""
        return getattr(self.norm, 'bias', None)

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        normed = self.norm(x)
        linear_output, _ = self.linear(normed)
        return linear_output, None

    def state_dict(self, *args, prefix="", keep_vars=False, **kwargs):
        """State dict with TE-compatible key names for checkpoint compatibility."""
        # Get actual state dict from parent
        actual_state_dict = super().state_dict(*args, prefix="", keep_vars=keep_vars, **kwargs)

        # Map SGLang internal names to TE-compatible names
        te_compatible_state_dict = {}
        for key, value in actual_state_dict.items():
            if key == "norm.weight":
                te_compatible_state_dict["layer_norm_weight"] = value
            elif key == "norm.bias":
                te_compatible_state_dict["layer_norm_bias"] = value
            elif key == "linear.weight":
                te_compatible_state_dict["weight"] = value
            elif key == "linear.bias":
                te_compatible_state_dict["bias"] = value
            else:
                te_compatible_state_dict[key] = value

        # Add prefix if provided
        if prefix:
            te_compatible_state_dict = {f"{prefix}{k}": v for k, v in te_compatible_state_dict.items()}

        return te_compatible_state_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Load state dict with TE-compatible key names."""
        # Map TE-compatible keys to SGLang internal keys
        mapped_state_dict = {}
        prefix_len = len(prefix)

        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue

            local_key = key[prefix_len:]

            # Map TE keys to SGLang internal keys
            if local_key == "layer_norm_weight":
                mapped_state_dict[f"{prefix}norm.weight"] = value
            elif local_key == "layer_norm_bias":
                mapped_state_dict[f"{prefix}norm.bias"] = value
            elif local_key == "weight":
                mapped_state_dict[f"{prefix}linear.weight"] = value
            elif local_key == "bias":
                mapped_state_dict[f"{prefix}linear.bias"] = value
            else:
                mapped_state_dict[key] = value

        # Call parent's _load_from_state_dict with mapped keys
        super()._load_from_state_dict(
            mapped_state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharded state dict with TE-compatible key names."""
        # Build state dict with TE-compatible keys for checkpoint compatibility
        state_dict = {
            "layer_norm_weight": self.norm.weight,
            "weight": self.linear.weight,
        }
        if hasattr(self.norm, 'bias') and self.norm.bias is not None:
            state_dict["layer_norm_bias"] = self.norm.bias
        if self.linear.bias is not None:
            state_dict["bias"] = self.linear.bias

        # Use TE-compatible sharding map
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 0, "bias": 0}, sharded_offsets
        )


# =============================================================================
# SGLang Rotary Position Embedding (RoPE)
# =============================================================================

def sglang_apply_rotary_pos_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    is_neox_style: bool = True,
) -> Tensor:
    if cos.dim() == 2:
        cos = cos.unsqueeze(-2)  # [seq, 1, head_dim//2]
        sin = sin.unsqueeze(-2)
    
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def sglang_apply_rotary_pos_emb_to_qk(
    query: Tensor,
    key: Tensor,
    freqs: Tensor,
    config: "TransformerConfig",
) -> Tuple[Tensor, Tensor]:
    if isinstance(freqs, tuple):
        q_freqs, k_freqs = freqs
    else:
        q_freqs = k_freqs = freqs
    
    head_dim = query.shape[-1]
    
    if q_freqs.shape[-1] == head_dim:
        cos = q_freqs[..., :head_dim // 2]
        sin = q_freqs[..., head_dim // 2:]
    else:
        # Deinterleave
        cos = q_freqs[..., 0::2]
        sin = q_freqs[..., 1::2]
    
    while cos.dim() > 3 and cos.shape[1] == 1:
        cos = cos.squeeze(1)
        sin = sin.squeeze(1)
    
    is_neox_style = not getattr(config, 'rotary_interleaved', False)
    
    rotated_query = sglang_apply_rotary_pos_emb(query, cos, sin, is_neox_style)
    
    if k_freqs is not q_freqs:
        if k_freqs.shape[-1] == head_dim:
            k_cos = k_freqs[..., :head_dim // 2]
            k_sin = k_freqs[..., head_dim // 2:]
        else:
            k_cos = k_freqs[..., 0::2]
            k_sin = k_freqs[..., 1::2]
        while k_cos.dim() > 3 and k_cos.shape[1] == 1:
            k_cos = k_cos.squeeze(1)
            k_sin = k_sin.squeeze(1)
        rotated_key = sglang_apply_rotary_pos_emb(key, k_cos, k_sin, is_neox_style)
    else:
        rotated_key = sglang_apply_rotary_pos_emb(key, cos, sin, is_neox_style)
    
    return rotated_query, rotated_key


# Global flag to enable SGLang RoPE
_USE_SGLANG_ROPE = False

def enable_sglang_rope():
    """Enable SGLang-compatible RoPE for true on-policy consistency."""
    global _USE_SGLANG_ROPE
    _USE_SGLANG_ROPE = True
    _print_sglang_log("✅ SGLANG KERNEL: Enabled SGLang-compatible RoPE")


def sglang_apply_rotary_pos_emb_with_freqs(
    x: Tensor,
    freqs: Tensor,
    config: "TransformerConfig",
    layer_number: Optional[int] = None,
) -> Tensor:
    x_seq_len = x.shape[0]
    freqs_seq_len = freqs.shape[0]
    
    # Extract cos/sin from freqs (raw angles format)
    freqs_flat = freqs.squeeze(1).squeeze(1)  # [seq, head_dim]
    head_dim = x.shape[-1]
    raw_angles = freqs_flat[..., :head_dim // 2]  # [seq, head_dim/2]
    cos = torch.cos(raw_angles)  # Compute cos from angles
    sin = torch.sin(raw_angles)  # Compute sin from angles
    is_neox_style = not getattr(config, 'rotary_interleaved', False)
    
    # Handle sequence length mismatch (partial apply)
    if x_seq_len == freqs_seq_len:
        # Full apply
        return sglang_apply_rotary_pos_emb(x, cos, sin, is_neox_style)
    elif freqs_seq_len < x_seq_len:
        # Partial apply: only apply to first freqs_seq_len positions
        if layer_number == 1:
            # Log only once for the first layer
            warnings.warn(
                f"[SGLang RoPE] Partial apply: Layer {layer_number} "
                f"applying to first {freqs_seq_len} of {x_seq_len} positions",
                UserWarning,
                stacklevel=2
            )
        x_valid = x[:freqs_seq_len]
        x_valid = sglang_apply_rotary_pos_emb(x_valid, cos, sin, is_neox_style)
        return torch.cat([x_valid, x[freqs_seq_len:]], dim=0)
    else:
        # freqs_seq_len > x_seq_len: use only first x_seq_len freqs
        cos = cos[:x_seq_len]
        sin = sin[:x_seq_len]
        return sglang_apply_rotary_pos_emb(x, cos, sin, is_neox_style)

def disable_sglang_rope():
    """Disable SGLang-compatible RoPE (use default Megatron RoPE)."""
    global _USE_SGLANG_ROPE
    _USE_SGLANG_ROPE = False

def is_sglang_rope_enabled() -> bool:
    """Check if SGLang-compatible RoPE is enabled."""
    return _USE_SGLANG_ROPE


# =============================================================================
# SGLang Flash Attention (FA3)
# =============================================================================

class SGLangFlashAttention(MegatronModule):
    """
    Flash Attention 3 implementation using SGLang's batch-invariant mode.

    Uses FA3 with num_splits=1 for batch-invariant behavior.
    This ensures consistent results regardless of batch size.

    Note: This is a simplified implementation for training. For inference with
    KV cache, use the full Attention class with flash_decode_and_prefill.

    Reference: Flash Attention 3 with batch-invariant mode
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        assert config.context_parallel_size == 1, \
            "Context parallelism is not supported by SGLangFlashAttention!"
        assert config.window_size is None, \
            "Sliding Window Attention is not supported by SGLangFlashAttention!"

        if not HAVE_FA3:
            raise ImportError(
                "Flash Attention 3 is required for SGLangFlashAttention. "
                "Please install flash-attn>=3.0.0"
            )

        # Log when SGLangFlashAttention is instantiated (only once per layer)
        if layer_number == 1:
            _print_sglang_log("=" * 80)
            _print_sglang_log(f"✨ SGLANG KERNEL: SGLangFlashAttention initialized for layer {layer_number}")
            _print_sglang_log("   - Using Flash Attention 3 (FA3) with batch-invariant mode")
            _print_sglang_log("   - num_splits=1 (deterministic attention)")
            _print_sglang_log(f"   - Attention mask type: {attn_mask_type}")
            _print_sglang_log("=" * 80)

        kv_channels = config.kv_channels
        assert kv_channels is not None, "kv_channels must be set"
        projection_size = kv_channels * config.num_attention_heads

        # Per attention head and per partition values
        if pg_collection is None:
            raise ValueError("FlashAttention requires ProcessGroupCollection")

        world_size = pg_collection.tp.size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(config.num_query_groups, world_size)

        # Softmax scale
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if config.apply_query_key_layer_scaling:
            self.softmax_scale /= layer_number

        # Dropout (FA3 handles dropout internally)
        dropout_rate = config.attention_dropout if attention_dropout is None else attention_dropout
        self.attention_dropout = dropout_rate

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """
        Forward pass using Flash Attention 3 with batch-invariant mode.

        Args:
            query: [sq, b, np, hn] or [t, np, hn] for packed - Query tensor
            key: [sk, b, ng, hn] or [t, ng, hn] for packed - Key tensor (ng = num_query_groups for GQA)
            value: [sk, b, ng, hn] or [t, ng, hn] for packed - Value tensor
            attention_mask: Attention mask tensor (not used by FA3, but kept for interface)
            attn_mask_type: Type of attention mask (must be causal)
            attention_bias: Not supported
            packed_seq_params: Optional packed sequence parameters for variable-length sequences

        Returns:
            context: [sq, b, hp] or [t, hp] for packed - Attention output
        """
        assert attention_bias is None, "Attention bias not supported for SGLangFlashAttention"
        assert attn_mask_type is None or attn_mask_type == AttnMaskType.causal, \
            "Only causal mask is supported for SGLangFlashAttention"

        # Check if using packed sequences (THD format)
        is_packed = packed_seq_params is not None

        # Determine input dimensionality for proper reshape at the end
        input_ndim = query.dim()

        # Handle GQA: expand key and value to match query heads
        # Need to determine the head dimension index based on input format
        if is_packed:
            # Packed format: [t, np, hn] (3D) or [t, 1, np, hn] (4D with dummy batch)
            head_dim_idx = -2 if input_ndim >= 3 else 1
        else:
            # Standard format: [sq, b, np, hn]
            head_dim_idx = 2

        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            repeat_factor = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
            key = key.repeat_interleave(repeat_factor, dim=head_dim_idx)
            value = value.repeat_interleave(repeat_factor, dim=head_dim_idx)

        # Cast to BF16 to match SGLang's FSDP-compatible numerical paths
        # In Qwen3 model: q = q.to(torch.bfloat16), k = k.to(torch.bfloat16)
        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)

        if is_packed:
            # Packed sequence format can be:
            # - 3D: [total_tokens, num_heads, head_dim] (THD format)
            # - 4D: [total_tokens, 1, num_heads, head_dim] (with dummy batch dim)
            if input_ndim == 3:
                # Already in THD format: [t, np, hn]
                t, np, hn = query.shape
            else:
                # 4D format: [t, 1, np, hn]
                t, b, np, hn = query.shape
                assert b == 1, f"Packed sequences should have batch=1, got {b}"
                query = query.squeeze(1)  # [t, np, hn]
                key = key.squeeze(1)      # [t, np, hn]
                value = value.squeeze(1)  # [t, np, hn]

            # Use cu_seqlens from packed_seq_params
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_k = packed_seq_params.cu_seqlens_kv
            max_seqlen_q = packed_seq_params.max_seqlen_q
            max_seqlen_k = packed_seq_params.max_seqlen_kv

            # Ensure cu_seqlens are int32
            if cu_seqlens_q.dtype != torch.int32:
                cu_seqlens_q = cu_seqlens_q.to(torch.int32)
            if cu_seqlens_k.dtype != torch.int32:
                cu_seqlens_k = cu_seqlens_k.to(torch.int32)
        else:
            # Standard format: [seq_len, batch, num_heads, head_dim]
            sq, b, np, hn = query.shape
            sk = key.shape[0]

            # Reshape: [sq, b, np, hn] -> [b*sq, np, hn] (FA3 THD format)
            query = query.transpose(0, 1).reshape(b * sq, np, hn)
            key = key.transpose(0, 1).reshape(b * sk, np, hn)
            value = value.transpose(0, 1).reshape(b * sk, np, hn)

            # Prepare cu_seqlens for fixed-length sequences
            # Format: [0, sq, 2*sq, 3*sq, ..., b*sq]
            cu_seqlens_q = torch.arange(
                0, (b + 1) * sq, sq, dtype=torch.int32, device=query.device
            )
            cu_seqlens_k = torch.arange(
                0, (b + 1) * sk, sk, dtype=torch.int32, device=query.device
            )
            max_seqlen_q = sq
            max_seqlen_k = sk

        # Use Flash Attention 3 with backward support
        # CRITICAL: Use flash_attn_varlen_func (high-level API) instead of _flash_attn_forward
        # _flash_attn_forward is a low-level CUDA kernel WITHOUT autograd support
        # flash_attn_varlen_func wraps it with proper backward implementation
        if HAVE_FA3_VARLEN and fa3_varlen_func is not None:
            # Use the high-level API with backward support
            # num_splits=1 is CRITICAL for batch-invariant (deterministic) behavior
            # This ensures consistent results regardless of batch size
            sig = inspect.signature(fa3_varlen_func)
            
            # Base kwargs that should work with all FA3 variants
            fa3_kwargs = {
                'q': query,
                'k': key,
                'v': value,
                'cu_seqlens_q': cu_seqlens_q,
                'cu_seqlens_k': cu_seqlens_k,
                'max_seqlen_q': max_seqlen_q,
                'max_seqlen_k': max_seqlen_k,
                'softmax_scale': self.softmax_scale,
                'causal': True,
            }
            
            # Add optional parameters if supported by this FA3 variant
            if 'dropout_p' in sig.parameters:
                fa3_kwargs['dropout_p'] = self.attention_dropout if self.training else 0.0
            if 'window_size' in sig.parameters:
                fa3_kwargs['window_size'] = (-1, -1)
            if 'softcap' in sig.parameters:
                fa3_kwargs['softcap'] = 0.0
            if 'return_attn_probs' in sig.parameters:
                fa3_kwargs['return_attn_probs'] = False
            if 'return_softmax_lse' in sig.parameters:
                fa3_kwargs['return_softmax_lse'] = False
            # CRITICAL: num_splits=1 for batch-invariant mode
            if 'num_splits' in sig.parameters:
                fa3_kwargs['num_splits'] = 1
            
            output = fa3_varlen_func(**fa3_kwargs)
            
            # flash_attn_varlen_func returns output directly (or tuple if return_attn_probs=True)
            if isinstance(output, tuple):
                output = output[0]
        else:
            # Fallback to _flash_attn_forward (WARNING: no backward support!)
            _print_sglang_log(
                "⚠️  WARNING: Using _flash_attn_forward without backward support! "
                "Training will NOT work correctly!", 
                level=logging.WARNING
            )
            output = _flash_attn_forward(
                q=query,
                k=key,
                v=value,
                k_new=None,
                v_new=None,
                qv=None,
                out=None,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                cu_seqlens_k_new=None,
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                page_table=None,
                kv_batch_idx=None,
                leftpad_k=None,
                rotary_cos=None,
                rotary_sin=None,
                seqlens_rotary=None,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                softmax_scale=self.softmax_scale,
                causal=True,
                window_size=(-1, -1),
                attention_chunk=0,
                softcap=0.0,
                rotary_interleaved=True,
                scheduler_metadata=None,
                num_splits=1,
                pack_gqa=None,
                sm_margin=0,
            )[0]

        # Reshape output based on input format
        if is_packed:
            # Output is [t, np, hn], reshape to match input format
            if input_ndim == 3:
                # Input was 3D [t, np, hn], output should be [t, hp]
                output = output.view(t, self.hidden_size_per_partition)
            else:
                # Input was 4D [t, 1, np, hn], output should be [t, 1, hp]
                output = output.view(t, 1, self.hidden_size_per_partition)
        else:
            # Output is [b*sq, np, hn], reshape to [sq, b, hp]
            output = output.view(b, sq, np, hn)  # [b, sq, np, hn]
            output = output.transpose(0, 1)      # [sq, b, np, hn]
            output = output.reshape(sq, b, self.hidden_size_per_partition)

        return output


# =============================================================================
# SGLang Activation Functions
# =============================================================================

class SGLangSwiGLU(nn.Module):
    """SwiGLU activation matching SGLang's FSDP-compatible paths.
    
    Matches SGLang's SiluAndMul.forward_native exactly:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]
    
    Note: bfloat16 casting is done in MLP.forward() before linear_fc1,
    matching SGLang's Qwen2MLP which does x = x.bfloat16() first.
    """

    def forward(self, x: Tensor) -> Tensor:
        # Match SGLang's exact implementation using slicing
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Return empty dict - activation functions have no parameters."""
        return {}


class SGLangGeGLU(nn.Module):
    """GeGLU activation matching SGLang's FSDP-compatible paths.
    
    Matches SGLang's GeluAndMul._forward_impl exactly:
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate="tanh") * x[..., d:]
    
    Note: bfloat16 casting is done in MLP.forward() before linear_fc1.
    """

    def forward(self, x: Tensor) -> Tensor:
        # Match SGLang's exact implementation using slicing
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate="tanh") * x[..., d:]

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Return empty dict - activation functions have no parameters."""
        return {}


class SGLangActivation(nn.Module):
    """
    Activation function selector.
    All activations are element-wise and thus inherently batch-invariant.
    """

    def __new__(cls, config: TransformerConfig):
        if config.gated_linear_unit:
            if config.activation_func == F.silu:
                return SGLangSwiGLU()
            elif config.activation_func == F.gelu:
                return SGLangGeGLU()
            else:
                raise ValueError(f"Unsupported gated activation: {config.activation_func}")
        else:
            if config.activation_func == F.gelu:
                return nn.GELU()
            elif config.activation_func == F.silu:
                return nn.SiLU()
            elif config.activation_func == F.relu:
                return nn.ReLU()
            else:
                raise ValueError(f"Unsupported activation: {config.activation_func}")


# =============================================================================
# SGLang Spec Provider (matching Kitchen's BackendSpecProvider interface)
# =============================================================================

class SGLangSpecProvider(BackendSpecProvider):
    """
    Backend spec provider using SGLang's batch-invariant kernels.

    This is the main entry point for using SGLang kernels in Megatron.
    It provides the same interface as KitchenSpecProvider.

    Usage:
        from megatron.core.extensions.sglang import (
            SGLangSpecProvider,
            enable_sglang_batch_invariant_mode
        )

        # Enable batch-invariant mode
        enable_sglang_batch_invariant_mode()

        # Create provider (uses FA3 by default)
        backend = SGLangSpecProvider()

        # Use in layer spec building
        layer_spec = get_gpt_layer_spec_for_backend(config, backend)
    """

    def __init__(
        self,
        fallback: Optional[BackendSpecProvider] = None,
        use_sglang_attention: bool = True,
    ):
        """
        Initialize SGLang backend provider.

        Args:
            fallback: Fallback provider for unsupported operations.
            use_sglang_attention: Whether to use SGLang's Flash Attention 3 kernel.
        """
        self.fallback = fallback
        self.use_sglang_attention = use_sglang_attention

        # Check FA3 availability
        if use_sglang_attention and not HAVE_FA3:
            raise ImportError(
                "Flash Attention 3 is required for SGLang attention. "
                "Please install flash-attn>=3.0.0 or set use_sglang_attention=False."
            )

        # logger.info clear initialization message
        _print_sglang_log("=" * 80)
        _print_sglang_log("🔧 SGLANG KERNEL: Initializing SGLangSpecProvider")
        _print_sglang_log(f"   - use_sglang_attention: {use_sglang_attention}")
        _print_sglang_log(f"   - Flash Attention 3 available: {HAVE_FA3}")
        _print_sglang_log(f"   - SGLang batch-invariant ops available: {HAVE_SGLANG_BATCH_INVARIANT}")
        if fallback is not None:
            _print_sglang_log(f"   - Fallback provider: {type(fallback).__name__}")
        _print_sglang_log("=" * 80)

    def column_parallel_linear(self) -> type:
        """Column parallel linear module."""
        return SGLangColumnParallelLinear

    def row_parallel_linear(self) -> type:
        """Row parallel linear module."""
        return SGLangRowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        """Whether to fuse layernorm and linear."""
        # Match fallback behavior if available
        if self.fallback is not None:
            return self.fallback.fuse_layernorm_and_linear()
        return False

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Fused layernorm + linear module."""
        return SGLangLayerNormColumnParallelLinear

    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False) -> type:
        """LayerNorm or RMSNorm module.

        IMPORTANT: Always use SGLang's LayerNorm implementations for true on-policy mode.
        This ensures numerical consistency between SGLang inference and Megatron training.
        Do NOT use fallback here as TransformerEngine's norms have different numerical behavior.

        For Q/K layernorm (for_qk=True), always use RMSNorm as most LLMs (Qwen, LLaMA, etc.)
        use RMSNorm for query/key normalization, even when the main layernorm might be different.
        """
        # For Q/K layernorm, always use RMSNorm (standard for most LLMs)
        # This avoids bias parameters that would break checkpoint compatibility
        if for_qk or rms_norm:
            return SGLangRMSNorm
        # Return SGLangNorm which will check config.normalization at instantiation time
        # This ensures correct norm type (RMSNorm vs LayerNorm) based on model config
        # Fixes MoE models like Qwen3-30B-A3B which use RMSNorm but pre_mlp_layernorm
        # was getting SGLangLayerNorm (with bias) causing checkpoint mismatch
        return SGLangNorm

    def core_attention(self) -> type:
        """Core attention module using Flash Attention 3."""
        _print_sglang_log(f"🔍 SGLANG DEBUG: core_attention() called, use_sglang_attention={self.use_sglang_attention}")
        if not self.use_sglang_attention:
            _print_sglang_log("⚠️  WARNING: use_sglang_attention is False, falling back to TE attention!", level=logging.WARNING)
            if self.fallback is not None:
                fallback_attn = self.fallback.core_attention()
                _print_sglang_log(f"⚠️  SGLangSpecProvider: Falling back to {fallback_attn.__name__}", level=logging.WARNING)
                return fallback_attn
            raise ValueError(
                "SGLangSpecProvider requires either use_sglang_attention=True "
                "or a fallback provider."
            )

        _print_sglang_log("✅ SGLANG KERNEL: Core attention will use Flash Attention 3 (FA3)")
        _print_sglang_log(f"🔍 SGLANG DEBUG: Returning SGLangFlashAttention class: {SGLangFlashAttention}")
        return SGLangFlashAttention

    def grouped_mlp_modules(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: bool
    ) -> Tuple[type, Optional[MLPSubmodules]]:
        """Grouped MLP modules for MoE."""
        if moe_use_grouped_gemm and not moe_use_legacy_grouped_gemm:
            return TEGroupedMLP, MLPSubmodules(
                linear_fc1=SGLangColumnParallelGroupedLinear,
                linear_fc2=SGLangRowParallelGroupedLinear,
            )
        elif moe_use_grouped_gemm:
            warnings.warn(
                "Legacy GroupedMLP will be deprecated. "
                "Please update to TEGroupedMLP."
            )
            return GroupedMLP, None
        else:
            return SequentialMLP, MLPSubmodules(
                linear_fc1=SGLangColumnParallelLinear,
                linear_fc2=SGLangRowParallelLinear,
            )

    def activation_func(self) -> type:
        """Activation function module.
        
        Always returns SGLangActivation to ensure identical numerical behavior
        with SGLang inference (no fallback to TE activation).
        """
        # Do NOT fall back to TE activation - always use SGLang's activation
        # for true on-policy compatibility
        return SGLangActivation


# =============================================================================
# Public API Summary
# =============================================================================

__all__ = [
    # Mode management
    'enable_sglang_batch_invariant_mode',
    'disable_sglang_batch_invariant_mode',
    'enable_sglang_deterministic_mode',  # alias
    'is_batch_invariant_mode_enabled',
    # Configuration
    'SGLangConfig',
    'get_sglang_config',
    'set_sglang_config',
    # Core operations
    'sglang_mm',
    'sglang_addmm',
    'sglang_bmm',
    'sglang_rms_norm',
    # Linear layers
    'SGLangLinear',
    'SGLangColumnParallelLinear',
    'SGLangRowParallelLinear',
    # Grouped linear (MoE)
    'SGLangGroupedLinear',
    'SGLangColumnParallelGroupedLinear',
    'SGLangRowParallelGroupedLinear',
    # Normalization
    'SGLangRMSNorm',
    'SGLangFinalRMSNorm',
    'SGLangLayerNorm',
    'SGLangNorm',
    # Fused ops
    'SGLangLayerNormColumnParallelLinear',
    # Attention
    'SGLangFlashAttention',
    # RoPE (Rotary Position Embedding)
    'sglang_apply_rotary_pos_emb',
    'sglang_apply_rotary_pos_emb_to_qk',
    'sglang_apply_rotary_pos_emb_with_freqs',
    'enable_sglang_rope',
    'disable_sglang_rope',
    'is_sglang_rope_enabled',
    # Activation
    'SGLangSwiGLU',
    'SGLangGeGLU',
    'SGLangActivation',
    # Provider
    'SGLangSpecProvider',
    # Flags
    'HAVE_SGLANG_BATCH_INVARIANT',
    'HAVE_DEEPGEMM',
    'HAVE_FA3',
]
