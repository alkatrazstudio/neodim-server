# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, PretrainedConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteriaList

import logits_warper_override as lwo
import tokenizer as tok
import tools
from dev_map import DeviceMap
from gpu_info import GpuInfo, GpuMemStats
from logits_warper_override import WarperId
from memory_tracing_criteria import MemoryTracingCriteria
from rep_pen_processor import AdvancedRepetitionPenaltyLogitsProcessor, RepPenGenerated
from stop_tokens_criteria import StopStringsType, StopTokensCriteria


@dataclass
class GeneratedSequence:
    generated_text: str
    stop_string: str
    stop_string_match: str
    trimmed_tail: str
    repetition_penalty_text_at_end: str

    def to_dict(self) -> dict:
        return {
            "generated_text": self.generated_text,
            "stop_string": self.stop_string,
            "stop_string_match": self.stop_string_match,
            "trimmed_tail": self.trimmed_tail,
            "repetition_penalty_text_at_end": self.repetition_penalty_text_at_end
        }


@dataclass
class GeneratedOutput:
    original_input_tokens_count: int
    used_input_tokens_count: int
    preamble_tokens_count: int
    used_prompt: str
    original_prompt_tokens_count: int
    used_prompt_tokens_count: int
    repetition_penalty_text_at_start: str
    used_repetition_penalty_tokens_count_at_start: int
    used_repetition_penalty_range_at_start: int
    used_repetition_penalty_tokens_count_at_end: int
    used_repetition_penalty_range_at_end: int
    generated_tokens_count: int
    output_tokens_count: int
    sequences: list[GeneratedSequence]
    gpus: list[GpuInfo]

    def to_dict(self) -> dict:
        return {
            "original_input_tokens_count": self.original_input_tokens_count,
            "used_input_tokens_count": self.used_input_tokens_count,
            "preamble_tokens_count": self.preamble_tokens_count,
            "used_prompt": self.used_prompt,
            "original_prompt_tokens_count": self.original_prompt_tokens_count,
            "used_prompt_tokens_count": self.used_prompt_tokens_count,
            "repetition_penalty_text_at_start": self.repetition_penalty_text_at_start,
            "used_repetition_penalty_tokens_count_at_start": self.used_repetition_penalty_tokens_count_at_start,
            "used_repetition_penalty_range_at_start": self.used_repetition_penalty_range_at_start,
            "used_repetition_penalty_tokens_count_at_end": self.used_repetition_penalty_tokens_count_at_end,
            "used_repetition_penalty_range_at_end": self.used_repetition_penalty_range_at_end,
            "generated_tokens_count": self.generated_tokens_count,
            "output_tokens_count": self.output_tokens_count,
            "sequences": [s.to_dict() for s in self.sequences],
            "gpus": [g.to_dict() for g in self.gpus]
        }


class ModelPrecision(Enum):
    ORIGINAL = "original"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8 = "int8"


def load_config(
    model_path: str,
    model_revision: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> PretrainedConfig:
    cfg = AutoConfig.from_pretrained(
        model_path,
        revision=model_revision,
        cache_dir=cache_dir,
        gradient_checkpointing=None
    )
    return cfg


def load_model(
    path: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    device_map: Optional[DeviceMap] = None,
    precision: ModelPrecision = ModelPrecision.FLOAT16
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    args = {}
    match precision:
        case ModelPrecision.ORIGINAL:
            pass

        case ModelPrecision.FLOAT32:
            args["torch_dtype"] = torch.float32

        case ModelPrecision.FLOAT16:
            args["torch_dtype"] = torch.float16

        case ModelPrecision.INT8:
            args["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(
        path,
        revision=revision,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        device_map=device_map,
        **args
    )

    tokenizer = AutoTokenizer.from_pretrained(path)

    # These defaults are used when you pass None to a corresponding parameter in model.generate().
    # Set them all to None, so that passing None to generate() will actually mean None.
    model.config.temperature = None
    model.config.top_k = None
    model.config.top_p = None
    model.config.typical_p = None
    model.config.repetition_penalty = None
    model.config.diversity_penalty = None
    model.config.no_repeat_ngram_size = None
    model.config.encoder_no_repeat_ngram_size = None

    # silence the Transformers warning "Setting `pad_token_id` to `eos_token_id`"
    if model.config.pad_token_id is None and model.config.eos_token_id is not None:
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def move_to_cpu(model: PreTrainedModel) -> PreTrainedModel:
    model = model.to("cpu").float()
    return model


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    gen_tokens_len: int,
    max_total_tokens: int,
    preamble: str = "",
    stop_strings: Optional[list[str]] = None,
    stop_strings_type: Optional[StopStringsType] = StopStringsType.STRING,
    stop_strings_required_matches_count: int = 1,
    truncate_prompt_until: Optional[list[str]] = None,
    gpu_device: Optional[int] = 0,
    repetition_penalty: Optional[float] = None,
    repetition_penalty_range: Optional[int] = None,
    repetition_penalty_slope: Optional[float] = None,
    repetition_penalty_include_preamble: bool = False,
    repetition_penalty_include_generated: RepPenGenerated = RepPenGenerated.SLIDE,
    repetition_penalty_truncate_to_input: bool = False,
    repetition_penalty_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[float] = None,
    tfs: Optional[float] = None,
    typical: Optional[float] = None,
    top_a: Optional[float] = None,
    warpers_order: Optional[list[WarperId]] = None,
    sequences_count: int = 1
) -> GeneratedOutput:
    if not preamble and not prompt:
        raise RuntimeError("Specify preamble and/or prompt")

    mem_stats_arrays = [GpuMemStats.from_all_devices()]

    max_tokens_for_gen = max_total_tokens - gen_tokens_len
    tok_res = tok.tokenize_input(prompt, preamble, truncate_prompt_until, model, tokenizer, max_tokens_for_gen)
    input_tokens_len = len(tok_res.input_tokens)
    sequences = []

    if gen_tokens_len or not input_tokens_len:
        stop_strings = tools.normalize_str_list(stop_strings)
        stop_list = StoppingCriteriaList()
        if len(stop_strings):
            stop_criteria = StopTokensCriteria(
                input_tokens_len=input_tokens_len,
                stop_strings=stop_strings,
                stop_strings_type=stop_strings_type,
                required_matches_count=stop_strings_required_matches_count,
                model=model,
                tokenizer=tokenizer,
                sequences_count=sequences_count
            )
            stop_list.append(stop_criteria)
        else:
            stop_criteria = None
        tracing_criteria = MemoryTracingCriteria(mem_stats_arrays)
        stop_list.append(tracing_criteria)

        logit_processors = LogitsProcessorList()
        if repetition_penalty is not None and repetition_penalty_range is not None:
            if repetition_penalty_prompt:
                rep_pen_custom_ids = tok.str_to_tokens(repetition_penalty_prompt, model, tokenizer)
                rep_pen_custom_ids = [rep_pen_custom_ids] * sequences_count
                rep_pen_custom_ids = torch.LongTensor(rep_pen_custom_ids)
                if gpu_device is None:
                    rep_pen_custom_ids = rep_pen_custom_ids.to("cpu")
                else:
                    rep_pen_custom_ids = rep_pen_custom_ids.to(gpu_device)

                if not repetition_penalty_range:
                    if repetition_penalty_include_preamble:
                        repetition_penalty_range = rep_pen_custom_ids.shape[-1] + tok_res.preamble_tokens_count
                    else:
                        repetition_penalty_range = rep_pen_custom_ids.shape[-1]
            else:
                rep_pen_custom_ids = None

                if not repetition_penalty_range:
                    if repetition_penalty_include_preamble:
                        repetition_penalty_range = len(tok_res.input_tokens)
                    else:
                        repetition_penalty_range = tok_res.trimmed_prompt_tokens_count

            penalty_processor = AdvancedRepetitionPenaltyLogitsProcessor(
                penalty=repetition_penalty,
                input_len=input_tokens_len,
                penalty_range=repetition_penalty_range,
                penalty_slope=repetition_penalty_slope,
                include_generated=repetition_penalty_include_generated,
                preamble_tokens_count=tok_res.preamble_tokens_count,
                include_preamble=repetition_penalty_include_preamble,
                truncate_to_input=repetition_penalty_truncate_to_input,
                prompt_tokens=rep_pen_custom_ids
            )
            logit_processors.append(penalty_processor)
        else:
            penalty_processor = None

        in_tensor = torch.tensor(tok_res.input_tokens, dtype=torch.long)
        in_tensor = in_tensor[None]
        if gpu_device is None:
            in_tensor = in_tensor.to("cpu")
        else:
            in_tensor = in_tensor.to(gpu_device)

        lwo.override_get_logits_warper(
            model,
            tfs=tfs,
            top_a=top_a,
            order=warpers_order
        )
        try:
            out_tensor = model.generate(
                in_tensor,
                do_sample=True,
                min_length=input_tokens_len + gen_tokens_len,
                max_length=input_tokens_len + gen_tokens_len,
                use_cache=True,  # insanely slow without the cache (but uses much less VRAM)
                num_return_sequences=sequences_count,
                num_beams=1,
                num_beam_groups=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                typical_p=typical,
                stopping_criteria=stop_list,
                logits_processor=logit_processors,
                return_dict_in_generate=False
            )
            lwo.restore_get_logits_warper(model)
        except Exception as e:
            lwo.restore_get_logits_warper(model)
            raise e

        out_tokens_len = out_tensor.shape[-1]

        if penalty_processor is None or penalty_processor.first_tokens is None:
            repetition_penalty_text_at_start = ""
            used_repetition_penalty_range_at_start = 0
            used_repetition_penalty_tokens_count_at_start = 0
            used_repetition_penalty_range_at_end = 0
            used_repetition_penalty_tokens_count_at_end = 0
        else:
            used_repetition_penalty_range_at_start = penalty_processor.first_tokens.shape[-1]
            repetition_penalty_text_at_start = tok.tokens_to_str(penalty_processor.first_tokens[0], model, tokenizer)
            used_repetition_penalty_tokens_count_at_start = penalty_processor.first_range
            used_repetition_penalty_range_at_end = penalty_processor.last_tokens.shape[-1]
            used_repetition_penalty_tokens_count_at_end = penalty_processor.last_range

        input_text_len = len(preamble) + len(tok_res.trimmed_prompt)

        for seq_idx, out_tokens in enumerate(out_tensor):
            if penalty_processor is None or penalty_processor.last_tokens is None:
                repetition_penalty_text_at_end = ""
            else:
                repetition_penalty_text_at_end = tok.tokens_to_str(
                    penalty_processor.last_tokens[seq_idx], model, tokenizer)

            out_txt = tok.tokens_to_str(out_tokens, model, tokenizer)

            gen_txt = out_txt[input_text_len:]

            if stop_criteria and stop_criteria.matches[seq_idx]:
                match = stop_criteria.matches[seq_idx]
                stop_string = match.stop_string
                stop_string_match = match.match
                gen_txt_trimmed = gen_txt[0:match.start_index]
                trimmed_tail = gen_txt[match.start_index + len(match.match):]
            else:
                gen_txt_trimmed = gen_txt
                stop_string = ""
                stop_string_match = ""
                trimmed_tail = ""

            seq = GeneratedSequence(
                generated_text=gen_txt_trimmed,
                stop_string=stop_string,
                stop_string_match=stop_string_match,
                trimmed_tail=trimmed_tail,
                repetition_penalty_text_at_end=repetition_penalty_text_at_end
            )
            sequences.append(seq)
    else:
        repetition_penalty_text_at_start = ""
        used_repetition_penalty_range_at_start = 0
        used_repetition_penalty_tokens_count_at_start = 0
        used_repetition_penalty_range_at_end = 0
        used_repetition_penalty_tokens_count_at_end = 0
        out_tokens_len = len(tok_res.input_tokens)

        for _ in range(sequences_count):
            seq = GeneratedSequence(
                generated_text="",
                stop_string="",
                stop_string_match="",
                trimmed_tail="",
                repetition_penalty_text_at_end=""
            )
            sequences.append(seq)

    mem_stats_arrays.append(GpuMemStats.from_all_devices())
    gpus = GpuInfo.from_all_devices(mem_stats_arrays)

    gen_out = GeneratedOutput(
        original_input_tokens_count=tok_res.original_input_tokens_count,
        used_input_tokens_count=input_tokens_len,
        preamble_tokens_count=tok_res.preamble_tokens_count,
        used_prompt=tok_res.trimmed_prompt,
        original_prompt_tokens_count=tok_res.original_prompt_tokens_count,
        used_prompt_tokens_count=tok_res.trimmed_prompt_tokens_count,
        repetition_penalty_text_at_start=repetition_penalty_text_at_start,
        used_repetition_penalty_range_at_start=used_repetition_penalty_range_at_start,
        used_repetition_penalty_tokens_count_at_start=used_repetition_penalty_tokens_count_at_start,
        used_repetition_penalty_range_at_end=used_repetition_penalty_range_at_end,
        used_repetition_penalty_tokens_count_at_end=used_repetition_penalty_tokens_count_at_end,
        output_tokens_count=out_tokens_len,
        generated_tokens_count=out_tokens_len - len(tok_res.input_tokens),
        sequences=sequences,
        gpus=gpus
    )

    return gen_out
