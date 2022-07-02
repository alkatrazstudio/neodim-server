# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

import gc
from enum import Enum
from typing import Optional, Union

import torch
from transformers import GPT2PreTrainedModel, GPTJPreTrainedModel, GPTNeoPreTrainedModel
from transformers import OPTPreTrainedModel, XGLMPreTrainedModel
from transformers import PretrainedConfig, PreTrainedModel


class ModelType(Enum):
    GPT_NEO = "gpt_neo"
    GPT_J = "gptj"
    XGLM = "xglm"
    GPT2 = "gpt2"
    OPT = "opt"


def model_type(model_or_config: Union[PreTrainedModel, PretrainedConfig]) -> ModelType:
    config = model_or_config if isinstance(model_or_config, PretrainedConfig) else model_or_config.config
    if isinstance(config, GPTNeoPreTrainedModel.config_class):
        return ModelType.GPT_NEO
    if isinstance(config, GPTJPreTrainedModel.config_class):
        return ModelType.GPT_J
    if isinstance(config, XGLMPreTrainedModel.config_class):
        return ModelType.XGLM
    if isinstance(config, GPT2PreTrainedModel.config_class):
        return ModelType.GPT2
    if isinstance(config, OPTPreTrainedModel.config_class):
        return ModelType.OPT
    raise RuntimeError(f"Unsupported model config class: {config.__class__.__name__}")


def num_layers(config: PretrainedConfig) -> int:
    if hasattr(config, "num_layers"):
        return config.num_layers
    if hasattr(config, "n_layer"):
        return config.n_layer
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    raise RuntimeError(f"Cannot determine the number of layers for {config.__class__.__name__}")


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


def format_float(f: float, digits: int = 1) -> str:
    return ("{0:."+str(digits)+"f}").format(f).rstrip("0").rstrip(".")


def format_gb(b: float) -> str:
    gb = format_float(b / 1024 / 1024 / 1024)
    s = f"{gb}GiB"
    return s
