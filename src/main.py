# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

import argparse
import time
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

import ai
from ai import GeneratedOutput
import server
from server import RequestData, Callback
import tools


def get_request_callback(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, gpu_device: Optional[int]) -> Callback:
    def request_callback(req: RequestData) -> GeneratedOutput:
        tools.cleanup()

        t_start = time.time()

        gen_out = ai.generate(
            model=model,
            tokenizer=tokenizer,
            prompt=req.prompt,
            preamble=req.preamble,
            gen_tokens_len=req.generated_tokens_count,
            truncate_prompt_until=req.truncate_prompt_until,
            max_total_tokens=req.max_total_tokens,
            stop_strings=req.stop_strings,
            repetition_penalty=req.repetition_penalty,
            repetition_penalty_range=req.repetition_penalty_range,
            repetition_penalty_slope=req.repetition_penalty_slope,
            repetition_penalty_include_preamble=req.repetition_penalty_include_preamble,
            repetition_penalty_include_generated=req.repetition_penalty_include_generated,
            repetition_penalty_truncate_to_input=req.repetition_penalty_truncate_to_input,
            repetition_penalty_prompt=req.repetition_penalty_prompt,
            top_p=req.top_p,
            top_k=req.top_k,
            tfs=req.tfs,
            temperature=req.temperature,
            sequences_count=req.sequences_count,
            gpu_device=gpu_device
        )

        t_elapsed = round(time.time() - t_start)

        free_mems = [tools.format_gb(gpu.memory_free_end) for gpu in gen_out.gpus]
        free_mems_str = ", ".join(free_mems)

        print(f"Generated {gen_out.generated_tokens_count} tokens in {t_elapsed}s " +
              f"with {gen_out.used_input_tokens_count} input tokens. Free VRAM: [{free_mems_str}]")

        tools.cleanup()
        return gen_out

    return request_callback


def run_ai_server(
    model_path: str,
    model_revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    ip: str = "127.0.0.1",
    port: int = 8787,
    layers: Optional[list[int]] = None
) -> None:
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

    t_start = time.time()
    if not model_revision:
        print(f"Loading the model: {model_path}")
    else:
        print(f"Loading the model: {model_path} ({model_revision})")
    model, tokenizer = ai.load(model_path, model_revision, cache_dir)
    mtype = tools.model_type(model)
    t_elapsed = round(time.time() - t_start)
    print(f"Model {model.__class__.__name__} ({mtype.name}) loaded in {t_elapsed}s")
    print()

    if layers is None:
        layers = [1]
    n_layers = tools.num_layers(model)
    t_start = time.time()
    print(f"Distributing {n_layers} model layers: {layers}")
    if sum(layers) == 0:
        print("Moving the entire model to CPU")
        model = ai.move_to_cpu(model)
        gpu_device = None
    else:
        gpu_device = next((x for x in layers if x == n_layers), -1)
        if gpu_device >= 0:
            print(f"Moving the entire model to GPU {gpu_device}")
            model = ai.move_to_gpu(model, gpu_device)
        else:
            gpu_layers = sum(layers)
            cpu_layers = n_layers - gpu_layers
            gpu_count = len([x for x in layers if x])
            print(f"Distributing {gpu_layers} layer(s) to {gpu_count} GPU(s), and {cpu_layers} layers to CPU")
            ai.distribute_layers(model, layers)
            gpu_device = next(i for i, layer in enumerate(layers) if layer > 0)
    t_elapsed = round(time.time() - t_start)
    print(f"Layers distributed in {t_elapsed}s")

    free_mems = []
    for gpu_index in range(gpu_devices_count):
        memory_free, _ = torch.cuda.mem_get_info(gpu_index)
        free_mems.append(memory_free)
    free_mem_strs = [tools.format_gb(m) for m in free_mems]
    free_mem_str = ", ".join(free_mem_strs)
    print(f"Free VRAM: {free_mem_str}")
    print()

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to the model or its name on Hugging Face")
    parser.add_argument("--model-revision", help="Model revision/branch on Hugging Face")
    parser.add_argument("--cache-dir", help="Cache directory for Hugging Face models")
    parser.add_argument("--listen-address", help="Listen on this address", default="127.0.0.1")
    parser.add_argument("--listen-port", help="Listen on this port", default="8787")
    parser.add_argument("--layers", help="Distribute model's layers between GPUs", default="1")
    parser.add_argument("--version", help="The version of this Neodim Server", action="store_true")
    args = parser.parse_args()

    if args.version:
        print(server.SERVER_VERSION)
        return

    if not args.model:
        raise RuntimeError("--model is missing")

    listen_port = int(args.listen_port)
    layers = [int(x) for x in args.layers.split(",")]
    run_ai_server(
        model_path=args.model,
        model_revision=args.model_revision,
        cache_dir=args.cache_dir,
        ip=args.listen_address,
        port=listen_port,
        layers=layers
    )


if __name__ == "__main__":
    main()
