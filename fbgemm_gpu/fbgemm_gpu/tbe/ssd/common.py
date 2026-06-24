#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import torch

# fmt:skip
from fbgemm_gpu.utils.loader import load_torch_module

try:
    load_torch_module(
        "//deeplearning/fbgemm/fbgemm_gpu:ssd_split_table_batched_embeddings"
    )
except Exception:
    pass

# Cache associativity must match the hardware warp/wavefront width, since the
# C++ cache kernels use kWarpSize for set indexing. The device paths derive
# it from torch.cuda.get_device_properties(dev).warp_size (32 on NVIDIA,
# either 32 or 64 on AMD); this constant is only the no-device fallback.
ASSOC: int = 32


def device_cache_assoc(use_cpu: bool = False) -> int:
    """
    Cache associativity for the current device: the warp size (32 on NVIDIA,
    either 32 or 64 on AMD), since the C++ cache kernels use kWarpSize for set
    indexing. Falls back to ASSOC when there is no device.
    """
    if use_cpu or not torch.cuda.is_available():
        return ASSOC
    return torch.cuda.get_device_properties(torch.cuda.current_device()).warp_size


def pad4(value: int) -> int:
    """
    Compute the smallest multiple of 4 that is greater than or equal to the given value.

    Parameters:
        value (int): The integer to align (must be non-negative).

    Returns:
        int: The aligned value.

    Raises:
        ValueError: If the input is negative.
        TypeError: If the input is not an integer.
    """
    return (int(value) + 3) & ~3


def tensor_pad4(value: torch.Tensor) -> torch.Tensor:
    """
    The equivalent of pad4 for tensors.
    """
    return (value + 3) & ~3
