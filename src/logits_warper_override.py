# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

from types import MethodType
from typing import Optional

from transformers import LogitsProcessorList, PreTrainedModel

from third_party.warpers import TailFreeLogitsWarper


# Transformers do not provide any way to set custom logits warpers
def override_get_logits_warper(model: PreTrainedModel, tfs: Optional[float]) -> None:
    def new_get_logits_warper(self, *args, **kwargs) -> LogitsProcessorList:
        processors = self.original_get_logits_warper(*args, **kwargs)
        if tfs is not None:
            tfs_processor = TailFreeLogitsWarper(tfs=tfs)
            processors.append(tfs_processor)
        return processors

    model.original_get_logits_warper = model._get_logits_warper
    model._get_logits_warper = MethodType(new_get_logits_warper, model)


def restore_get_logits_warper(model: PreTrainedModel) -> None:
    model._get_logits_warper = model.original_get_logits_warper
    delattr(model, "original_get_logits_warper")
