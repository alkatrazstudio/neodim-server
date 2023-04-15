# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

import argparse
import time
from argparse import Namespace
from typing import Final, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

import ai
import dev_map
import server
import tools
from ai import GeneratedOutput, ModelPrecision
from server import Callback, ServerRequestData


AVAILABLE_LAYERS_CHAR: Final[str] = "a"


def get_request_callback(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, gpu_device: Optional[int]) -> Callback:
    def request_callback(request: ServerRequestData) -> GeneratedOutput:
        tools.cleanup()

        t_start = time.time()

        gen_out = ai.generate(
            model=model,
            tokenizer=tokenizer,
            gpu_device=gpu_device,
            request=request
        )

        t_elapsed = round(time.time() - t_start)

        free_mems = [
            f"GPU_{gpu_index}={tools.format_gb(gpu.memory_free_end)}"
            for gpu_index, gpu in enumerate(gen_out.gpus)
        ]
        free_mems_str = ", ".join(free_mems)

        print(f"Generated {gen_out.generated_tokens_count} tokens in {t_elapsed}s " +
              f"with {gen_out.used_input_tokens_count} input tokens. Free VRAM: [{free_mems_str}]")

        tools.cleanup()
        return gen_out

    return request_callback


def print_gpu_info() -> None:
    title = server.name_and_version()
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    print()

    print(f"Using CUDA v{torch.version.cuda}")
    print()

    gpu_devices_count = torch.cuda.device_count()
    if not gpu_devices_count:
        print("No CUDA GPUs found")
    else:
        print("Found GPUs:")
        for gpu_index in range(gpu_devices_count):
            mem_free, mem_total = torch.cuda.mem_get_info(gpu_index)
            gpu_name = torch.cuda.get_device_properties(gpu_index).name
            mem_total_str = tools.format_gb(mem_total)
            mem_free_str = tools.format_gb(mem_free)
            print(f"{gpu_index}. {gpu_name} ({mem_total_str} total, {mem_free_str} free)")
    print()


def parse_layers(layer_strs: Optional[list[str]], layers_count: int) -> list[int]:
    if layer_strs is None:
        return [1]

    explicit_count = 0
    for layer_str in layer_strs:
        if layer_str != AVAILABLE_LAYERS_CHAR:
            explicit_count += int(layer_str)
    available_count = layers_count - explicit_count

    layers = [
        int(layer_str)
        if layer_str != AVAILABLE_LAYERS_CHAR
        else available_count
        for layer_str in layer_strs
    ]
    return layers


def load_model(
    path: str,
    revision: Optional[str] = None,
    precision: ModelPrecision = ModelPrecision.FLOAT16,
    cache_dir: Optional[str] = None,
    layer_strs: Optional[list[str]] = None
) -> tuple[PreTrainedModel, PreTrainedTokenizer, Optional[int]]:
    if not revision:
        print(f"Loading the model: {path}")
    else:
        print(f"Loading the model: {path} ({revision})")

    t_start = time.time()

    config = ai.load_config(path, revision, cache_dir)
    model_type = tools.model_type(config)

    layers_count = tools.num_layers(config)
    layers = parse_layers(layer_strs, layers_count)
    layers_str = ", ".join(f"GPU_{device_index}={layer}" for device_index, layer in enumerate(layers) if layer > 0)
    gpu_layers_count = sum(layers)
    cpu_layers_count = layers_count - gpu_layers_count
    if cpu_layers_count > 0:
        layers_str = layers_str + f", CPU={cpu_layers_count}"
    print(f"Total layers: {layers_count} ({layers_str})")
    gpus_count = len([x for x in layers if x])

    device_map = dev_map.build(model_type, layers_count, layers)

    model, tokenizer = ai.load_model(
        path=path,
        revision=revision,
        cache_dir=cache_dir,
        device_map=device_map,
        precision=precision
    )

    t_elapsed = round(time.time() - t_start)
    print(f"Model {model.__class__.__name__} ({model_type.name}) loaded in {t_elapsed}s")
    if device_map is not None:
        gpu_device = next(i for i, layer in enumerate(layers) if layer > 0)
        print(f"Distributed {gpu_layers_count} layer(s) to {gpus_count} GPU(s), and {cpu_layers_count} layers to CPU")
    else:
        print("Moving the entire model to CPU")
        model = ai.move_to_cpu(model)
        gpu_device = None

    print()

    return model, tokenizer, gpu_device


def print_free_ram():
    gpu_devices_count = torch.cuda.device_count()
    free_mems = []
    for gpu_index in range(gpu_devices_count):
        memory_free, _ = torch.cuda.mem_get_info(gpu_index)
        free_mems.append(memory_free)
    free_mem_strs = [tools.format_gb(m) for m in free_mems]
    free_mem_str = ", ".join(free_mem_strs)
    print(f"Free VRAM: {free_mem_str}")
    print()


def run_ai_server(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    gpu_device: Optional[int] = None,
    ip: str = "127.0.0.1",
    port: int = 8787
) -> None:
    callback = get_request_callback(model, tokenizer, gpu_device)
    print("Server is running")
    print(f"Endpoint for text generation: http://{ip}:{port}{server.ENDPOINT_PATH}")
    print("Press CTRL+C to stop")
    print()
    print("------")
    print()
    server.run(ip, port, callback)

    print()
    print("------")
    print()
    print("Neodim Server is terminated.")


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog="start.sh",
        description=f"{server.name_and_version()} - natural language model AI via HTTP",
        epilog="License: AGPLv3\n"
               "More info and help: https://github.com/alkatrazstudio/neodim-server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False
    )
    parser.add_argument("--model",
                        help="Path to the model or its name on Hugging Face "
                             "(paths must either be absolute or start with a dot)")
    parser.add_argument("--model-revision",
                        help="Model revision/branch on Hugging Face")
    parser.add_argument("--cache-dir",
                        help="Cache directory for downloading Hugging Face models")
    parser.add_argument("--listen-address",
                        help="Listen on this address (pass 0.0.0.0 to listen on all addresses). "
                             "Default: 127.0.0.1",
                        default="127.0.0.1")
    parser.add_argument("--listen-port",
                        help="Listen on this port. "
                             "Default: 8787",
                        default="8787")
    parser.add_argument("--layers",
                        help="Distribute model's layers between GPUs (pass 0 to not use GPUs). "
                             "Default: 1",
                        default="1")
    parser.add_argument("--precision",
                        help=f"Load the model in this precision ({', '.join([x.value for x in ModelPrecision])}). "
                             f"Default: {ModelPrecision.FLOAT16.value}",
                        default=ModelPrecision.FLOAT16)
    parser.add_argument("--version",
                        help="Show the version of this Neodim Server",
                        action="store_true")
    args = parser.parse_args()

    args.listen_port = int(args.listen_port)
    args.layer_strs = args.layers.split(",")
    args.precision = ModelPrecision(args.precision)
    return args


def main() -> None:
    args = parse_args()

    if args.version:
        print(server.SERVER_VERSION)
        return

    if not args.model:
        raise RuntimeError("--model is missing")

    print_gpu_info()
    model, tokenizer, gpu_device = load_model(
        path=args.model,
        revision=args.model_revision,
        precision=args.precision,
        cache_dir=args.cache_dir,
        layer_strs=args.layer_strs
    )
    run_ai_server(model, tokenizer, gpu_device, args.listen_address, args.listen_port)


if __name__ == "__main__":
    main()
