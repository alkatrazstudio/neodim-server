# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>


from enum import Enum
import gc
from typing import Optional

import torch
from transformers import GPT2PreTrainedModel, GPTJPreTrainedModel, GPTNeoPreTrainedModel
from transformers import PreTrainedModel, XGLMPreTrainedModel


class ModelType(Enum):
    GPT_NEO = "gpt_neo"
    GPT_J = "gptj"
    XGLM = "xglm"
    GPT2 = "gpt2"


def model_type(model: PreTrainedModel) -> ModelType:
    if isinstance(model, GPTNeoPreTrainedModel):
        return ModelType.GPT_NEO
    if isinstance(model, GPTJPreTrainedModel):
        return ModelType.GPT_J
    if isinstance(model, XGLMPreTrainedModel):
        return ModelType.XGLM
    if isinstance(model, GPT2PreTrainedModel):
        return ModelType.GPT2
    raise RuntimeError(f"Unsupported model class: {model.__class__.__name__}")


def num_layers(model: PreTrainedModel) -> int:
    if hasattr(model.config, "num_layers"):
        return model.config.num_layers
    return model.config.n_layer


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def normalize_str_list(strs: Optional[list[str]] = None) -> list[str]:
    if not strs:
        strs = []
    if isinstance(strs, str):
        strs = [strs]
    strs = sorted(strs, key=len, reverse=True)
    strs = [s for s in strs if s]
    return strs


def supports_layers_distribution(model: PreTrainedModel) -> bool:
    mtype = model_type(model)
    return mtype in [ModelType.GPT_NEO, ModelType.GPT_J, ModelType.XGLM]


def format_float(f: float, digits: int = 1) -> str:
    return ("{0:."+str(digits)+"f}").format(f).rstrip("0").rstrip(".")


def format_gb(b: float) -> str:
    gb = format_float(b / 1024 / 1024 / 1024)
    s = f"{gb}GiB"
    return s
