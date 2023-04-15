# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

from enum import Enum
from typing import Final, Optional, TypedDict, Union

from tools import ModelType


class DeviceMapValue(Enum):
    FIRST_DEVICE = "FIRST_DEVICE"
    CPU_IF_USED = "CPU_IF_USED"


class DeviceMapInfo(TypedDict):
    layer_key_template: str
    device_map_template: dict[str, DeviceMapValue]


DeviceMap = dict[str, Union[str, int]]
DEVICE_CPU: Final[str] = "cpu"


DEVICE_MAP_TEMPLATES: Final[dict[ModelType, DeviceMapInfo]] = {
    ModelType.GPT_NEO: {
        "layer_key_template": "transformer.h.{layer}",
        "device_map_template": {
            "transformer.wte": DeviceMapValue.FIRST_DEVICE,
            "transformer.wpe": DeviceMapValue.FIRST_DEVICE,
            "transformer.ln_f": DeviceMapValue.CPU_IF_USED,
            "lm_head": DeviceMapValue.FIRST_DEVICE
        }
    },
    ModelType.GPT_J: {
        "layer_key_template": "transformer.h.{layer}",
        "device_map_template": {
            "transformer.wte": DeviceMapValue.FIRST_DEVICE,
            "transformer.ln_f": DeviceMapValue.CPU_IF_USED,
            "lm_head": DeviceMapValue.FIRST_DEVICE
        }
    },
    ModelType.GPT_NEOX: {
        "layer_key_template": "gpt_neox.layers.{layer}",
        "device_map_template": {
            "gpt_neox.embed_in": DeviceMapValue.FIRST_DEVICE,
            "gpt_neox.final_layer_norm": DeviceMapValue.CPU_IF_USED,
            "embed_out": DeviceMapValue.FIRST_DEVICE
        }
    },
    ModelType.GPT2: {
        "layer_key_template": "transformer.h.{layer}",
        "device_map_template": {
            "transformer.wte": DeviceMapValue.FIRST_DEVICE,
            "transformer.wpe": DeviceMapValue.FIRST_DEVICE,
            "transformer.ln_f": DeviceMapValue.CPU_IF_USED,
            "lm_head": DeviceMapValue.FIRST_DEVICE
        }
    },
    ModelType.XGLM: {
        "layer_key_template": "model.layers.{layer}",
        "device_map_template": {
            "model.embed_tokens": DeviceMapValue.FIRST_DEVICE,
            "model.layer_norm": DeviceMapValue.FIRST_DEVICE,
            "model.embed_positions": DeviceMapValue.CPU_IF_USED,
            "lm_head": DeviceMapValue.FIRST_DEVICE
        }
    },
    ModelType.OPT: {
        "layer_key_template": "model.decoder.layers.{layer}",
        "device_map_template": {
            "model.decoder.embed_tokens": DeviceMapValue.FIRST_DEVICE,
            "model.decoder.embed_positions": DeviceMapValue.FIRST_DEVICE,
            "model.decoder.final_layer_norm": DeviceMapValue.FIRST_DEVICE,
            "lm_head": DeviceMapValue.FIRST_DEVICE
        }
    },
    ModelType.CODEGEN: {
        "layer_key_template": "transformer.h.{layer}",
        "device_map_template": {
            "transformer.wte": DeviceMapValue.FIRST_DEVICE,
            "transformer.ln_f": DeviceMapValue.CPU_IF_USED,
            "lm_head": DeviceMapValue.FIRST_DEVICE
        }
    },
    ModelType.BLOOM: {
        "layer_key_template": "transformer.h.{layer}",
        "device_map_template": {
            "transformer.word_embeddings": DeviceMapValue.FIRST_DEVICE,
            "transformer.word_embeddings_layernorm": DeviceMapValue.FIRST_DEVICE,
            "transformer.ln_f": DeviceMapValue.FIRST_DEVICE,
            "lm_head": DeviceMapValue.FIRST_DEVICE
        }
    }
}


def build(model_type: ModelType, layers_count: int, gpu_layers: list[int]) -> Optional[DeviceMap]:
    gpu_layers_count = sum(gpu_layers)
    if not gpu_layers_count:
        return None

    layer_key_template = DEVICE_MAP_TEMPLATES[model_type]["layer_key_template"]
    device_map_template = DEVICE_MAP_TEMPLATES[model_type]["device_map_template"]

    if gpu_layers_count == 0:
        first_device = DEVICE_CPU
    else:
        first_device = next(i for i, layer in enumerate(gpu_layers) if layer > 0)

    cpu_layers_count = layers_count - gpu_layers_count
    cpu_if_used = DEVICE_CPU if cpu_layers_count > 0 else first_device

    device_map: DeviceMap = {}
    for (key, val) in device_map_template.items():
        match val:
            case DeviceMapValue.FIRST_DEVICE:
                val = first_device

            case DeviceMapValue.CPU_IF_USED:
                val = cpu_if_used

            case _:
                raise f"Unknown value type: {val}"

        device_map[key] = val

    layer_index = 0
    for device_index, layers_on_device in enumerate(gpu_layers):
        for _ in range(layers_on_device):
            layer_key = layer_key_template.format(layer=layer_index)
            device_map[layer_key] = device_index
            layer_index += 1

    for _ in range(cpu_layers_count):
        layer_key = layer_key_template.format(layer=layer_index)
        device_map[layer_key] = DEVICE_CPU
        layer_index += 1

    return device_map
