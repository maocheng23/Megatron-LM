# This module provides model-specific configurations for true on-policy training,
# ensuring Megatron produces bit-exact same results as SGLang inference.
#
# Each model may have different MoE configurations (routing method, topk logic, etc.)
# This config system allows easy extension for new models.

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict


class RouterType(Enum):
    """Router implementation type."""
    MEGATRON_DEFAULT = "megatron_default"
    SGLANG_SEPARATE = "sglang_separate"  # SGLang's separate gate + topk


class PermuteType(Enum):
    """Token permutation implementation type."""
    MEGATRON_DEFAULT = "megatron_default"
    SGLANG_ALIGN = "sglang_align"  # SGLang's moe_align_block_size


class ScoringFunc(Enum):
    """TopK scoring function."""
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"


@dataclass
class TrueOnPolicyConfig:
    """Configuration for true on-policy training.

    This config ensures Megatron produces the same intermediate results as SGLang
    for a specific model architecture.

    Attributes:
        model_name: Name of the model this config is for
        enabled: Whether true on-policy mode is enabled
        router_type: Which router implementation to use
        permute_type: Which permute implementation to use
        scoring_func: TopK scoring function (softmax, sigmoid)
        renormalize: Whether to renormalize topk weights
        moe_softcapping: Softcapping value for router logits (0.0 = disabled)
        use_grouped_topk: Whether to use grouped topk (DeepSeek style)
        topk_group: Number of expert groups for grouped topk
        num_expert_group: Number of groups for grouped topk
        block_size: Block size for moe_align_block_size
        correction_bias_enabled: Whether model uses correction bias
    """
    model_name: str = ""
    enabled: bool = False
    router_type: RouterType = RouterType.MEGATRON_DEFAULT
    permute_type: PermuteType = PermuteType.MEGATRON_DEFAULT
    scoring_func: ScoringFunc = ScoringFunc.SOFTMAX
    renormalize: bool = True
    moe_softcapping: float = 0.0
    use_grouped_topk: bool = False
    topk_group: Optional[int] = None
    num_expert_group: Optional[int] = None
    block_size: int = 128
    correction_bias_enabled: bool = False

    # Debug options
    debug_print: bool = False


# =============================================================================
# Model-specific presets (currently only Qwen3-MoE is implemented)
# =============================================================================

def get_qwen3_moe_config() -> TrueOnPolicyConfig:
    """Get true on-policy config for Qwen3 MoE models.

    SGLang's Qwen3MoE uses:
    - Separate gate (ReplicatedLinear) + topk (TopK class)
    - TopK: topk_softmax from sgl_kernel
    - renormalize=config.norm_topk_prob (usually True)
    - No softcapping (moe_softcapping=0)
    - No grouped topk
    - moe_align_block_size for token permutation
    """
    return TrueOnPolicyConfig(
        model_name="qwen3_moe",
        enabled=True,
        router_type=RouterType.SGLANG_SEPARATE,
        permute_type=PermuteType.SGLANG_ALIGN,
        scoring_func=ScoringFunc.SOFTMAX,
        renormalize=True,
        moe_softcapping=0.0,
        use_grouped_topk=False,
        block_size=128,
        correction_bias_enabled=False,
    )


# =============================================================================
# Config registry
# =============================================================================

_MODEL_CONFIG_REGISTRY: Dict[str, TrueOnPolicyConfig] = {
    "qwen3_moe": get_qwen3_moe_config(),
    "qwen3-moe": get_qwen3_moe_config(),
    "qwen3moe": get_qwen3_moe_config(),
    "qwen3_next": get_qwen3_moe_config(),
    "qwen3-next": get_qwen3_moe_config(),
}


def get_true_on_policy_config(model_name: str) -> TrueOnPolicyConfig:
    """Get true on-policy config for a model by name.

    Args:
        model_name: Model name (case-insensitive)

    Returns:
        TrueOnPolicyConfig for the model

    Raises:
        ValueError: If model is not supported
    """
    key = model_name.lower().replace("-", "_")
    if key in _MODEL_CONFIG_REGISTRY:
        return _MODEL_CONFIG_REGISTRY[key]

    # Try without underscores
    key_no_underscore = key.replace("_", "")
    for k, v in _MODEL_CONFIG_REGISTRY.items():
        if k.replace("_", "") == key_no_underscore:
            return v

    supported = list(set(k for k in _MODEL_CONFIG_REGISTRY.keys()))
    raise ValueError(
        f"Model '{model_name}' not supported. Supported: {supported}"
    )


def register_true_on_policy_config(model_name: str, config: TrueOnPolicyConfig):
    """Register a custom config for a model."""
    _MODEL_CONFIG_REGISTRY[model_name.lower()] = config


def list_supported_models() -> list:
    """List all supported models."""
    return list(set(k.replace("_", "-") for k in _MODEL_CONFIG_REGISTRY.keys()))
