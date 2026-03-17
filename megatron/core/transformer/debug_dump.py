"""Debug layer dump utility for true-on-policy alignment debugging.

Dumps intermediate tensors during Megatron forward pass for comparison with SGLang.
Controlled by SLIME_DEBUG_LAYER_DUMP=1 environment variable.

Usage:
  SLIME_DEBUG_LAYER_DUMP=1 python examples/true_on_policy/run_moe_megatron.py

Dumps are saved to /tmp/megatron_debug/ as .pt files with naming convention:
  {name}_fwd{fwd_count}.pt

Compare with SGLang dumps using: python scripts/compare_moe_dumps.py
"""
import os

import torch

_DEBUG_DUMP = os.environ.get("SLIME_DEBUG_LAYER_DUMP", "0") == "1"
_DUMP_DIR = os.environ.get("SLIME_DEBUG_DUMP_DIR", "/tmp/megatron_debug")
_DUMP_MAX_FWD = int(os.environ.get("SLIME_DEBUG_DUMP_MAX_FWD", "4"))
_dump_fwd_count = [0]


def is_dump_enabled():
    return _DEBUG_DUMP and _dump_fwd_count[0] < _DUMP_MAX_FWD


def get_fwd_count():
    return _dump_fwd_count[0]


def increment_fwd_count():
    _dump_fwd_count[0] += 1


def dsave(name, tensor):
    """Save a tensor for debug comparison with SGLang dumps. Only rank 0 saves."""
    if not is_dump_enabled():
        return
    # Only save from TP rank 0 to avoid race condition
    try:
        import torch.distributed as _dist
        if _dist.is_initialized() and _dist.get_rank() != 0:
            return
    except Exception:
        pass
    os.makedirs(_DUMP_DIR, exist_ok=True)
    fwd = _dump_fwd_count[0]
    t = tensor.detach().cpu()
    path = os.path.join(_DUMP_DIR, f"{name}_fwd{fwd}.pt")
    torch.save(t, path)
    print(
        f"[DUMP] fwd{fwd} {name}: dtype={t.dtype} shape={list(t.shape)} "
        f"mean={t.float().mean():.8f} absmax={t.float().abs().max():.8f}",
        flush=True,
    )
