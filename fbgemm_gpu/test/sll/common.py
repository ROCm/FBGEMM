# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import fbgemm_gpu
import fbgemm_gpu.sll
import torch

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if not open_source:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")

if open_source:
    # pyre-ignore[21]
    from test_utils import TEST_WITH_ROCM
else:
    from fbgemm_gpu.test.test_utils import TEST_WITH_ROCM

# Device strategy for the SLL tests that compare a GPU kernel against a
# reference on the same device.  The CPU backend (fbgemm_gpu/sll/cpu) is plain
# PyTorch with no device-specific code, so running the CPU leg on ROCm
# duplicates the CUDA CI job without adding coverage.  It is also the dominant
# cost there: the CPU backends loop over B in Python, which issues thousands of
# small tensor ops per Hypothesis example, and that is disproportionately slow
# on the many-vCPU ROCm runners.
device_types: list[str] = ["cuda"] if TEST_WITH_ROCM else ["cpu", "cuda"]


def clone_tensor(data: torch.Tensor) -> torch.Tensor:
    if data.requires_grad:
        return data.detach().clone().requires_grad_()
    return data.detach().clone()
