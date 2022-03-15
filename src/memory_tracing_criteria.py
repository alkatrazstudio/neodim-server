# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

import torch
from transformers import StoppingCriteria

from gpu_info import GpuMemStats


class MemoryTracingCriteria(StoppingCriteria):
    def __init__(
        self,
        mem_stats_arrays: list[list[GpuMemStats]],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mem_stats_arrays = mem_stats_arrays

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stats_array = GpuMemStats.from_all_devices()
        self.mem_stats_arrays.append(stats_array)

        # The inference will be stopped when any criteria return true.
        # In this case we always return false.
        return False
