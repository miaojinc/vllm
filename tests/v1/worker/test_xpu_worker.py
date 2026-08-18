# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import torch

from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.worker.xpu_worker import (
    _XPU_MEMORY_INFO_FALLBACK_RESERVE,
    _repair_xpu_memory_snapshot,
)


def test_repair_xpu_zero_free_memory_snapshot() -> None:
    total_memory = 24 * 1024**3
    torch_memory = 2 * 1024**3
    snapshot = MemorySnapshot(
        free_memory=0,
        total_memory=total_memory,
        device=torch.device("xpu:0"),
        auto_measure=False,
    )

    with patch.object(torch.accelerator, "memory_reserved", return_value=torch_memory):
        _repair_xpu_memory_snapshot(snapshot)

    assert snapshot.free_memory == (
        total_memory - torch_memory - _XPU_MEMORY_INFO_FALLBACK_RESERVE
    )
    assert snapshot.cuda_memory == torch_memory + _XPU_MEMORY_INFO_FALLBACK_RESERVE
    assert snapshot.non_torch_memory == _XPU_MEMORY_INFO_FALLBACK_RESERVE


def test_repair_xpu_preserves_valid_memory_snapshot() -> None:
    snapshot = MemorySnapshot(
        free_memory=20 * 1024**3,
        total_memory=24 * 1024**3,
        device=torch.device("xpu:0"),
        auto_measure=False,
    )

    with patch.object(torch.accelerator, "memory_reserved") as memory_reserved:
        _repair_xpu_memory_snapshot(snapshot)

    memory_reserved.assert_not_called()
    assert snapshot.free_memory == 20 * 1024**3