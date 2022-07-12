# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

from enum import Enum
from types import MethodType
from typing import Final, Optional

from transformers import LogitsProcessorList, LogitsWarper, PreTrainedModel
from transformers import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, TypicalLogitsWarper

from third_party.warpers import TailFreeLogitsWarper, TopALogitsWarper


class WarperId(Enum):
    TEMPERATURE = "temperature"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TYPICAL = "typical"
    TFS = "tfs"
    TOP_A = "top_a"
    ALL = "*"


DEFAULT_WARPER_TYPES: Final[dict[type(LogitsWarper), WarperId]] = {
    TemperatureLogitsWarper: WarperId.TEMPERATURE,
    TopKLogitsWarper:        WarperId.TOP_K,
    TopPLogitsWarper:        WarperId.TOP_P,
    TypicalLogitsWarper:     WarperId.TYPICAL,
    TailFreeLogitsWarper:    WarperId.TFS,
    TopALogitsWarper:        WarperId.TOP_A
}


def sort_warpers(warper: LogitsWarper, order: list[WarperId], all_index: int) -> int:
    warper_type = type(warper)
    if warper_type not in DEFAULT_WARPER_TYPES:
        # put all unknown warpers at the end
        return len(order) + 1  # len(order) can be occupied by all_index
    warper_type_id = DEFAULT_WARPER_TYPES[warper_type]

    try:
        warper_pos = order.index(warper_type_id)
    except ValueError:
        warper_pos = all_index

    return warper_pos


# Transformers do not provide any way to set custom logits warpers
def override_get_logits_warper(
    model: PreTrainedModel,
    tfs: Optional[float] = None,
    top_a: Optional[float] = None,
    order: Optional[list[WarperId]] = None
) -> None:
    def new_get_logits_warper(self, *args, **kwargs) -> LogitsProcessorList:
        warpers = self.original_get_logits_warper(*args, **kwargs)
        if tfs is not None:
            tfs_warper = TailFreeLogitsWarper(tfs=tfs)
            warpers.append(tfs_warper)
        if top_a is not None:
            top_a_warper = TopALogitsWarper(top_a=top_a)
            warpers.append(top_a_warper)

        final_order = [] if order is None else order
        try:
            all_index = final_order.index(WarperId.ALL)
        except ValueError:
            all_index = len(final_order)

        warpers.sort(key=lambda warper: sort_warpers(warper, final_order, all_index))
        return warpers

    model.original_get_logits_warper = model._get_logits_warper
    model._get_logits_warper = MethodType(new_get_logits_warper, model)


def restore_get_logits_warper(model: PreTrainedModel) -> None:
    model._get_logits_warper = model.original_get_logits_warper
    delattr(model, "original_get_logits_warper")
