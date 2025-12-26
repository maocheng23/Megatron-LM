# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Tuple

import torch

from megatron.core.jit import jit_fuser

# pylint: disable=missing-function-docstring


def _bias_dropout_add_func(x_with_bias, residual, prob, training, use_fp32_residual=False, output_dtype=None):
    # type: (Tuple[Tensor, Optional[Tensor]], Tensor, float, bool, bool, Optional[torch.dtype]) -> Tensor
    # NOTE: Previously, the argument `bias` used to be passed as
    # `bias.expand_as(residual)` when the `bias_dropout_func` is called from the
    # transformer layer but broadcasting should automatically take care of that.
    # Also, looking at broadcasting semantics, `expand_as` and broadcasting
    # seem to be identical performance-wise (both just change the view).
    #
    # If use_fp32_residual=True (for SGLang mode), perform residual sum in FP32,
    # then convert to output_dtype (bf16 for intermediate layers, fp32 for final layer).

    x, bias = x_with_bias  # unpack

    # Run in-place if in eval mode and inputs do not require gradients
    inplace = (
        not training
        and not x.requires_grad
        and not residual.requires_grad
        and (bias is None or not bias.requires_grad)
    )

    # For SGLang mode: perform residual sum in FP32 to match SGLang's behavior
    # SGLang converts tensors to FP32, performs sum, uses FP32 sum for RMSNorm,
    # then converts back to bf16 (for intermediate layers) or keeps fp32 (for final layer)
    if use_fp32_residual:
        # Store original dtype for output conversion
        orig_dtype = x.dtype if output_dtype is None else output_dtype
        
        # Convert to FP32 for residual sum (matching SGLang's forward_native behavior)
        x_fp32 = x.to(torch.float32)
        residual_fp32 = residual.to(torch.float32)
        
        # Perform operations in FP32
        if bias is not None:
            bias_fp32 = bias.to(torch.float32) if bias is not None else None
            if inplace:
                x_fp32.add_(bias_fp32)
            else:
                x_fp32 = x_fp32 + bias_fp32
        out_fp32 = torch.nn.functional.dropout(x_fp32, p=prob, training=training, inplace=inplace)
        if inplace:
            out_fp32.add_(residual_fp32)
        else:
            out_fp32 = residual_fp32 + out_fp32
        
        # Convert back to original dtype (bf16 for intermediate layers, fp32 for final layer)
        return out_fp32.to(orig_dtype)
    else:
        # Original behavior: cast residual to same dtype as x
        residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)

        # The Dropout operation, Residual Addition and the tensor returning can be
        # done generically outside the if statement, but that stops fusing of Bias
        # Addition-Dropout-Residual Addition operation. So doing it together inside
        # the conditional branch to improve performance
        if bias is not None:
            if inplace:
                x.add_(bias)
            else:
                x = x + bias
            out = torch.nn.functional.dropout(x, p=prob, training=training, inplace=inplace)
            if inplace:
                out.add_(residual)
            else:
                out = residual + out
            return out
        else:
            out = torch.nn.functional.dropout(x, p=prob, training=training, inplace=inplace)
            if inplace:
                out.add_(residual)
            else:
                out = residual + out
            return out


def bias_dropout_add_unfused(training, use_fp32_residual=False, output_dtype=None):
    def _bias_dropout_add(x_with_bias, residual, prob):
        return _bias_dropout_add_func(x_with_bias, residual, prob, training, use_fp32_residual, output_dtype)

    return _bias_dropout_add


def get_bias_dropout_add_sglang(training, fused, is_final_layer=False):
    """Get bias_dropout_add function for SGLang mode.
    
    Args:
        training: Whether in training mode
        fused: Whether to use fused kernel
        is_final_layer: If True, output dtype is fp32; if False, output dtype is bf16
    
    Returns:
        bias_dropout_add function configured for SGLang mode
    """
    output_dtype = torch.float32 if is_final_layer else torch.bfloat16
    
    if fused:
        # For fused kernels, we can't easily pass extra parameters, so we use unfused for SGLang mode
        # This is acceptable since SGLang mode prioritizes numerical consistency over performance
        return bias_dropout_add_unfused(training, use_fp32_residual=True, output_dtype=output_dtype)
    else:
        return bias_dropout_add_unfused(training, use_fp32_residual=True, output_dtype=output_dtype)


@jit_fuser
def bias_dropout_add_fused_train(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return _bias_dropout_add_func(x_with_bias, residual, prob, True, False, None)


@jit_fuser
def bias_dropout_add_fused_inference(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return _bias_dropout_add_func(x_with_bias, residual, prob, False, False, None)


def get_bias_dropout_add(training, fused, use_sglang=False, is_final_layer=False):
    """Get bias_dropout_add function.
    
    Args:
        training: Whether in training mode
        fused: Whether to use fused kernel
        use_sglang: If True, use FP32 for residual sum (matching SGLang behavior)
        is_final_layer: If True and use_sglang=True, output dtype is fp32; if False, output dtype is bf16
    
    Returns:
        bias_dropout_add function
    """
    if use_sglang:
        # For SGLang mode, use FP32 residual sum to match SGLang's behavior
        # SGLang converts tensors to FP32, performs sum, uses FP32 sum for RMSNorm,
        # then converts back to bf16 (for intermediate layers) or keeps fp32 (for final layer)
        output_dtype = torch.float32 if is_final_layer else torch.bfloat16
        # For SGLang mode, we use unfused to support fp32 residual
        # This is acceptable since SGLang mode prioritizes numerical consistency over performance
        return bias_dropout_add_unfused(training, use_fp32_residual=True, output_dtype=output_dtype)
    
    if fused:
        # jit scripting for a nn.module (with dropout) is not
        # triggering the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if training:
            return bias_dropout_add_fused_train
        else:
            return bias_dropout_add_fused_inference
    else:
        return bias_dropout_add_unfused(training)
