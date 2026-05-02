from typing import Any, Dict, List, Tuple

import torch
from torch import nn


def load_state_dict_shape_safe(
    model: nn.Module,
    state_dict: Dict[str, Any],
    strict: bool = False,
) -> Tuple[Any, List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]]:
    """
    Load weights that match by key and tensor shape.

    PyTorch raises on shape mismatch even when strict=False. This helper drops
    incompatible tensors (e.g. old head when finetuning adds QM features).
    """
    model_sd = model.state_dict()
    skipped: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    filtered: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if key not in model_sd:
            continue
        if not torch.is_tensor(value):
            continue
        if value.shape != model_sd[key].shape:
            skipped.append((key, tuple(value.shape), tuple(model_sd[key].shape)))
            continue
        filtered[key] = value
    load_msg = model.load_state_dict(filtered, strict=strict)
    return load_msg, skipped
