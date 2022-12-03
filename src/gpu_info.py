# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>
from dataclasses import dataclass

import torch


class GpuMemStats:
    def __init__(
        self,
        reserved: int,
        allocated: int,
        free: int
    ):
        self.reserved = reserved
        self.allocated = allocated
        self.free = free

    @staticmethod
    def from_device(gpu_index: int) -> "GpuMemStats":
        free, _ = torch.cuda.mem_get_info(gpu_index)
        reserved = torch.cuda.memory_reserved(gpu_index)
        allocated = torch.cuda.memory_allocated(gpu_index)

        stats = GpuMemStats(
            reserved=reserved,
            allocated=allocated,
            free=free
        )
        return stats

    @staticmethod
    def from_all_devices() -> list["GpuMemStats"]:
        stats_array = []
        for gpu_index in range(torch.cuda.device_count()):
            stats = GpuMemStats.from_device(gpu_index)
            stats_array.append(stats)
        return stats_array


@dataclass
class GpuInfo:
    name: str
    memory_total: int
    memory_reserved_start: int
    memory_allocated_start: int
    memory_free_start: int
    memory_reserved_end: int
    memory_allocated_end: int
    memory_free_end: int
    memory_reserved_min: int
    memory_allocated_min: int
    memory_free_min: int
    memory_reserved_max: int
    memory_allocated_max: int
    memory_free_max: int

    @staticmethod
    def from_device(gpu_index: int, mem_stats_array: list[GpuMemStats]) -> "GpuInfo":
        name = torch.cuda.get_device_properties(gpu_index).name
        _, memory_total = torch.cuda.mem_get_info(gpu_index)

        start_stats = mem_stats_array[0]
        end_stats = mem_stats_array[-1]

        reserved_min = min(m.reserved for m in mem_stats_array)
        allocated_min = min(m.allocated for m in mem_stats_array)
        free_min = min(m.free for m in mem_stats_array)
        reserved_max = max(m.reserved for m in mem_stats_array)
        allocated_max = max(m.allocated for m in mem_stats_array)
        free_max = max(m.free for m in mem_stats_array)

        gpu = GpuInfo(
            name=name,
            memory_total=memory_total,
            memory_reserved_start=start_stats.reserved,
            memory_allocated_start=start_stats.allocated,
            memory_free_start=start_stats.free,
            memory_reserved_end=end_stats.reserved,
            memory_allocated_end=end_stats.allocated,
            memory_free_end=end_stats.free,
            memory_reserved_min=reserved_min,
            memory_allocated_min=allocated_min,
            memory_free_min=free_min,
            memory_reserved_max=reserved_max,
            memory_allocated_max=allocated_max,
            memory_free_max=free_max
        )
        return gpu

    @staticmethod
    def from_all_devices(mem_stats_arrays: list[list[GpuMemStats]]) -> list["GpuInfo"]:
        gpus = []
        for gpu_index in range(torch.cuda.device_count()):
            mem_stats_array = [arr[gpu_index] for arr in mem_stats_arrays]
            gpu = GpuInfo.from_device(gpu_index, mem_stats_array)
            gpus.append(gpu)
        return gpus
