# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

import bitsandbytes as bnb
from torch import nn
from transformers.utils import bitsandbytes


# This is a modified version of replace_8bit_linear in transformers/utils/bitsandbytes.py
# The following changes were made:
# 1. modules_to_not_convert can contain full module paths instead of just immediate names
# 2. the default value for modules_to_not_convert is effectively a list instead of a string
# 3. "model" is renamed to "parent_module" to not confuse it with the actual model
# 4. removed redundant check for len(modules)
def replace_8bit_linear(parent_module, threshold=6.0, modules_to_not_convert=None, parent_layer_path=""):
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert

    parent_layer_prefix = "" if parent_layer_path == "" else parent_layer_path + "."
    for name, module in parent_module.named_children():
        layer_path = parent_layer_prefix + name

        if layer_path in modules_to_not_convert:
            continue

        replace_8bit_linear(module, threshold, modules_to_not_convert, layer_path)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            with bitsandbytes.init_empty_weights():
                parent_module._modules[name] = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    has_fp16_weights=False,
                    threshold=threshold,
                )

    return parent_module


def override_bnb():
    bitsandbytes.replace_8bit_linear = replace_8bit_linear
