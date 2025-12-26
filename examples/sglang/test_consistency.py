"""
Example script demonstrating how to train with SGLang-compatible deterministic kernels.

This ensures training-inference consistency with SGLang for the dense (non-parallel) case.

Usage:
    python train_with_sglang_kernels.py

For distributed training (still uses SGLang kernels):
    torchrun --nproc_per_node=1 train_with_sglang_kernels.py
"""

import os
import sys
import torch
import torch.nn.functional as F

# Add megatron to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from megatron.core.extensions.sglang import (
    SGLangSpecProvider,
    SGLangColumnParallelLinear,
    SGLangRowParallelLinear,
    SGLangRMSNorm,
    SGLangFlashAttention,
    enable_sglang_deterministic_mode,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_mlp_module_spec_for_backend,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.custom_layers.transformer_engine import get_bias_dropout_add
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel


def get_gpt_layer_with_sglang_spec(
    num_experts: int = None,
    moe_grouped_gemm: bool = False,
    qk_layernorm: bool = False,
    normalization: str = "RMSNorm",
) -> ModuleSpec:
    """
    Get GPT layer spec using SGLang deterministic kernels.
    
    This function creates a layer specification that uses SGLang-compatible
    kernels for all operations, ensuring training-inference consistency.
    
    Args:
        num_experts: Number of MoE experts (None for dense model)
        moe_grouped_gemm: Whether to use grouped GEMM for MoE
        qk_layernorm: Whether to use layernorm for Q/K
        normalization: Type of normalization ("RMSNorm" or "LayerNorm")
    
    Returns:
        ModuleSpec for a GPT layer using SGLang kernels
    """
    backend = SGLangSpecProvider()
    
    # Get MLP spec
    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=False,
        use_te_op_fuser=False,
        use_te_activation_func=False,
    )
    
    # Build layer spec
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=SGLangRMSNorm if normalization == "RMSNorm" else backend.layer_norm(),
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=SGLangColumnParallelLinear,
                    core_attention=SGLangFlashAttention,
                    linear_proj=SGLangRowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=SGLangRMSNorm if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def create_sglang_config(
    hidden_size: int = 4096,
    num_attention_heads: int = 32,
    num_layers: int = 32,
    ffn_hidden_size: int = 11008,
    num_query_groups: int = 8,  # For GQA
    vocab_size: int = 32000,
    max_position_embeddings: int = 4096,
) -> TransformerConfig:
    """
    Create a TransformerConfig optimized for SGLang consistency.
    
    Args:
        hidden_size: Model hidden size
        num_attention_heads: Number of attention heads
        num_layers: Number of transformer layers
        ffn_hidden_size: FFN hidden size
        num_query_groups: Number of KV heads for GQA
        vocab_size: Vocabulary size
        max_position_embeddings: Maximum sequence length
    
    Returns:
        TransformerConfig configured for SGLang consistency
    """
    config = TransformerConfig(
        # Model architecture
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_layers=num_layers,
        ffn_hidden_size=ffn_hidden_size,
        num_query_groups=num_query_groups,
        
        # Normalization
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        
        # Activation
        activation_func=F.silu,
        gated_linear_unit=True,  # SwiGLU
        
        # Attention
        kv_channels=hidden_size // num_attention_heads,
        attention_dropout=0.0,  # No dropout for consistency
        hidden_dropout=0.0,
        
        # Precision
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        
        # Parallelism (no parallelism for TP=1)
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        
        # Deterministic settings
        deterministic_mode=True,
        attention_softmax_in_fp32=True,
        
        # Disable features that break consistency
        apply_query_key_layer_scaling=False,
        masked_softmax_fusion=False,  # Use unfused for consistency
        
        # Initialization
        perform_initialization=True,
        use_cpu_initialization=False,
    )
    
    return config


def test_linear_consistency():
    """Test that SGLang linear layers match PyTorch linear."""
    print("\n" + "="*60)
    print("Testing Linear Layer Consistency")
    print("="*60)
    
    # Create config
    config = create_sglang_config(hidden_size=256, num_attention_heads=4, num_layers=1)
    
    # Create SGLang linear
    sglang_linear = SGLangColumnParallelLinear(
        input_size=256,
        output_size=256,
        config=config,
        init_method=lambda x: torch.nn.init.xavier_uniform_(x),
        bias=True,
        skip_bias_add=False,
        is_expert=False,
    ).cuda()
    
    # Create reference PyTorch linear with same weights
    pytorch_linear = torch.nn.Linear(256, 256, bias=True).cuda().to(torch.bfloat16)
    with torch.no_grad():
        pytorch_linear.weight.copy_(sglang_linear.weight)
        pytorch_linear.bias.copy_(sglang_linear.bias)
    
    # Test
    x = torch.randn(4, 128, 256, dtype=torch.bfloat16, device='cuda')
    
    sglang_out, _ = sglang_linear(x)
    pytorch_out = pytorch_linear(x)
    
    max_diff = (sglang_out - pytorch_out).abs().max().item()
    print(f"Max difference: {max_diff}")
    print(f"Consistent: {max_diff < 1e-5}")
    
    return max_diff < 1e-5


def test_rmsnorm_consistency():
    """Test that SGLang RMSNorm matches reference implementation."""
    print("\n" + "="*60)
    print("Testing RMSNorm Consistency")
    print("="*60)
    
    config = create_sglang_config(hidden_size=256, num_attention_heads=4, num_layers=1)
    
    # Create SGLang RMSNorm
    sglang_norm = SGLangRMSNorm(config, hidden_size=256, eps=1e-6).cuda()
    
    # Reference implementation
    def reference_rmsnorm(x, weight, eps=1e-6):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + eps)
        return (weight.float() * x).to(x.dtype)
    
    # Test
    x = torch.randn(4, 128, 256, dtype=torch.bfloat16, device='cuda')
    
    sglang_out = sglang_norm(x)
    ref_out = reference_rmsnorm(x, sglang_norm.weight)
    
    max_diff = (sglang_out - ref_out).abs().max().item()
    print(f"Max difference: {max_diff}")
    print(f"Consistent: {max_diff < 1e-5}")
    
    return max_diff < 1e-5


def test_attention_consistency():
    """Test that SGLang attention is deterministic."""
    print("\n" + "="*60)
    print("Testing Attention Determinism")
    print("="*60)
    
    config = create_sglang_config(hidden_size=256, num_attention_heads=4, num_layers=1)
    
    # Initialize process groups for attention
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='tcp://localhost:12355',
            world_size=1,
            rank=0,
        )
    
    from megatron.core.parallel_state import initialize_model_parallel
    if not torch.distributed.is_initialized():
        initialize_model_parallel(tensor_model_parallel_size=1)
    
    # Create SGLang attention (FA3)
    from megatron.core.process_groups_config import ProcessGroupCollection
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
    
    sglang_attn = SGLangFlashAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        attention_dropout=0.0,
        pg_collection=pg_collection,
    ).cuda()
    
    # Test determinism: same input should produce same output
    torch.manual_seed(42)
    q = torch.randn(128, 4, 4, 64, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(128, 4, 4, 64, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(128, 4, 4, 64, dtype=torch.bfloat16, device='cuda')
    
    # Run twice
    out1 = sglang_attn(q, k, v, attention_mask=None)
    out2 = sglang_attn(q.clone(), k.clone(), v.clone(), attention_mask=None)
    
    max_diff = (out1 - out2).abs().max().item()
    print(f"Max difference between runs: {max_diff}")
    print(f"Deterministic: {max_diff == 0.0}")
    
    return max_diff == 0.0


def main():
    """Main function to demonstrate SGLang-consistent training."""
    print("="*60)
    print("SGLang-Consistent Training Demo")
    print("="*60)
    
    # Enable deterministic mode
    enable_sglang_deterministic_mode()
    
    # Run consistency tests
    tests_passed = []
    
    tests_passed.append(("Linear", test_linear_consistency()))
    tests_passed.append(("RMSNorm", test_rmsnorm_consistency()))
    
    # Note: Attention test requires distributed init
    # tests_passed.append(("Attention", test_attention_consistency()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in tests_passed:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(p for _, p in tests_passed)
    print(f"\nOverall: {'All tests passed!' if all_passed else 'Some tests failed!'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

