from __future__ import annotations

import logging
import os
from typing import Type

import torch.nn as nn

logger = logging.getLogger(__name__)

_NOISY_LINEAR_CLS: Type[nn.Linear] = nn.Linear
TORCHRL_AVAILABLE = False

if os.environ.get("BALATROBOT_ENABLE_TORCHRL", "").lower() in {"1", "true", "yes"}:
    try:
        import torchrl.modules as _torchrl_modules  # type: ignore

        _NOISY_LINEAR_CLS = _torchrl_modules.NoisyLinear  # type: ignore[attr-defined]
        TORCHRL_AVAILABLE = True
    except Exception as exc:  # noqa: BLE001 - import side effects
        logger.warning(
            "torchrl modules unavailable (%s). Falling back to nn.Linear without noise.",
            exc,
        )


def noisy_linear(
    in_features: int,
    out_features: int,
    *,
    std_init: float = 0.017,
    bias: bool = True,
) -> nn.Linear:
    if TORCHRL_AVAILABLE:
        return _NOISY_LINEAR_CLS(in_features, out_features, std_init=std_init, bias=bias)  # type: ignore[misc]
    return nn.Linear(in_features, out_features, bias=bias)


def noisy_linear_cls(noisy: bool) -> Type[nn.Linear]:
    if noisy and TORCHRL_AVAILABLE:
        return _NOISY_LINEAR_CLS
    return nn.Linear


__all__ = ["noisy_linear", "noisy_linear_cls", "TORCHRL_AVAILABLE"]
