# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer

import logits_warper_override as lwo
import tokenizer as tok
import tools
from dev_map import DeviceMap
from gpu_info import GpuInfo, GpuMemStats
from logits_warper_override import WarperId
from memory_tracing_criteria import MemoryTracingCriteria
from rep_pen_warper import RepetitionPenaltyLogitsWarper, RepPenGenerated
from stop_tokens_criteria import StopStringsType, StopTokensCriteria
from tokenizer import TokenizerResult


@dataclass
class RequestData:
    prompt: str
    generated_tokens_count: int
    max_total_tokens: int
    preamble: str = ""
    stop_strings: Optional[list[str]] = None
    stop_strings_type: Optional[StopStringsType] = StopStringsType.STRING
    stop_strings_required_matches_count: int = 1
    truncate_prompt_until: Optional[list[str]] = None
    gpu_device: Optional[int] = 0
    repetition_penalty: Optional[float] = None
    repetition_penalty_range: Optional[int] = None
    repetition_penalty_slope: Optional[float] = None
    repetition_penalty_include_preamble: bool = False
    repetition_penalty_include_generated: RepPenGenerated = RepPenGenerated.SLIDE
    repetition_penalty_truncate_to_input: bool = False
    repetition_penalty_prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tfs: Optional[float] = None
    typical: Optional[float] = None
    top_a: Optional[float] = None
    penalty_alpha: Optional[float] = None
    warpers_order: Optional[list[WarperId]] = None
    sequences_count: int = 1
    words_whitelist: Optional[list[str]] = None
    words_blacklist: Optional[list[str]] = None
    no_repeat_ngram_size: Optional[int] = None

    def __post_init__(self):
        if self.top_p == 0:
            self.top_p = None
        if self.top_k == 0:
            self.top_k = None
        if self.tfs == 0:
            self.tfs = None
        if self.typical == 0:
            self.typical = None
        if self.top_a == 0:
            self.top_a = None
        if self.penalty_alpha == 0:
            self.penalty_alpha = None
        if self.temperature == 0:
            self.temperature = None
        if self.repetition_penalty == 0:
            self.repetition_penalty = None
        if self.repetition_penalty_range is None:
            self.repetition_penalty_range = 0
        if self.repetition_penalty_slope == 0:
            self.repetition_penalty_slope = None
        if self.repetition_penalty_prompt == "":
            self.repetition_penalty_prompt = None
        if self.no_repeat_ngram_size == 0:
            self.no_repeat_ngram_size = None


@dataclass
class GeneratedSequence:
    output_text: str
    generated_text: str
    stop_string: str
    stop_string_match: str
    trimmed_tail: str
    repetition_penalty_text_at_end: str


@dataclass
class GeneratedOutput:
    original_input_tokens_count: int
    used_input_tokens_count: int
    preamble_tokens_count: int
    preamble: str
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
            args["torch_dtype"] = torch.float16
            args["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_skip_modules=["lm_head"]
            )

    model = AutoModelForCausalLM.from_pretrained(
        path,
        revision=revision,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        device_map=device_map,
        **args
    )

    tokenizer = AutoTokenizer.from_pretrained(path)

    return model, tokenizer


def default_generation_config_by_model(model: PreTrainedModel) -> GenerationConfig:
    # These defaults are used when you pass None to a corresponding parameter in model.generate().
    # Set them all to None, so that passing None to generate() will actually mean None.
    config = GenerationConfig(
        temperature=None,
        top_k=None,
        top_p=None,
        typical_p=None,
        repetition_penalty=None,
        diversity_penalty=None,
        no_repeat_ngram_size=None,
        encoder_no_repeat_ngram_size=None
    )

    # silence the Transformers warning "Setting `pad_token_id` to `eos_token_id`"
    if model.config.pad_token_id is None and model.config.eos_token_id is not None:
        config.update(
            pad_token_id=model.config.eos_token_id
        )

    return config


def move_to_cpu(model: PreTrainedModel) -> PreTrainedModel:
    model = model.to("cpu").float()
    return model


def create_repetition_penalty_warper(
    tokenizer: PreTrainedTokenizer,
    request: RequestData,
    tok_res: TokenizerResult,
    input_tokens_len: int
) -> Optional[RepetitionPenaltyLogitsWarper]:
    r = request
    if r.repetition_penalty is not None and r.repetition_penalty_range is not None:
        repetition_penalty_range = r.repetition_penalty_range
        if r.repetition_penalty_prompt:
            rep_pen_custom_ids = tok.str_to_tokens(r.repetition_penalty_prompt, tokenizer)
            rep_pen_custom_ids = [rep_pen_custom_ids] * r.sequences_count
            rep_pen_custom_ids = torch.LongTensor(rep_pen_custom_ids)
            if r.gpu_device is None:
                rep_pen_custom_ids = rep_pen_custom_ids.to("cpu")
            else:
                rep_pen_custom_ids = rep_pen_custom_ids.to(r.gpu_device)

            if not r.repetition_penalty_range:
                if r.repetition_penalty_include_preamble:
                    repetition_penalty_range = rep_pen_custom_ids.shape[-1] + tok_res.preamble_tokens_count
                else:
                    repetition_penalty_range = rep_pen_custom_ids.shape[-1]
        else:
            rep_pen_custom_ids = None

            if not r.repetition_penalty_range:
                if r.repetition_penalty_include_preamble:
                    repetition_penalty_range = len(tok_res.input_tokens)
                else:
                    repetition_penalty_range = tok_res.trimmed_prompt_tokens_count

        penalty_warper = RepetitionPenaltyLogitsWarper(
            penalty=r.repetition_penalty,
            input_len=input_tokens_len,
            penalty_range=repetition_penalty_range,
            penalty_slope=r.repetition_penalty_slope,
            include_generated=r.repetition_penalty_include_generated,
            preamble_tokens_count=tok_res.preamble_tokens_count,
            include_preamble=r.repetition_penalty_include_preamble,
            truncate_to_input=r.repetition_penalty_truncate_to_input,
            prompt_tokens=rep_pen_custom_ids
        )
    else:
        penalty_warper = None

    return penalty_warper


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    request: RequestData,
    gpu_device: Optional[int] = 0
) -> GeneratedOutput:
    r = request

    if not r.preamble and not r.prompt:
        raise RuntimeError("Specify preamble and/or prompt")

    mem_stats_arrays = [GpuMemStats.from_all_devices()]

    max_tokens_for_gen = r.max_total_tokens - r.generated_tokens_count
    tok_res = tok.tokenize_input(r.prompt, r.preamble, r.truncate_prompt_until, tokenizer, max_tokens_for_gen)
    input_tokens_len = len(tok_res.input_tokens)
    sequences = []

    if r.generated_tokens_count or not input_tokens_len:
        stop_strings = tools.normalize_str_list(r.stop_strings)
        stop_list = StoppingCriteriaList()
        if len(stop_strings):
            stop_criteria = StopTokensCriteria(
                input_tokens_len=input_tokens_len,
                stop_strings=stop_strings,
                stop_strings_type=r.stop_strings_type,
                required_matches_count=r.stop_strings_required_matches_count,
                tokenizer=tokenizer,
                sequences_count=r.sequences_count
            )
            stop_list.append(stop_criteria)
        else:
            stop_criteria = None
        tracing_criteria = MemoryTracingCriteria(mem_stats_arrays)
        stop_list.append(tracing_criteria)

        penalty_warper = create_repetition_penalty_warper(
            tokenizer=tokenizer,
            request=r,
            tok_res=tok_res,
            input_tokens_len=input_tokens_len
        )

        in_tensor = torch.tensor(tok_res.input_tokens, dtype=torch.long)
        in_tensor = in_tensor[None]
        if gpu_device is None:
            in_tensor = in_tensor.to("cpu")
        else:
            in_tensor = in_tensor.to(gpu_device)

        lwo.override_get_logits_warper(
            model,
            tfs=r.tfs,
            top_a=r.top_a,
            order=r.warpers_order,
            repetition_penalty_warper=penalty_warper
        )
        try:
            generation_config = default_generation_config_by_model(model)

            do_sample = r.penalty_alpha is None

            warpers_params = {
                "temperature": r.temperature,
                "top_p": r.top_p,
                "top_k": r.top_k,
                "typical_p": r.typical,
            }

            if do_sample:
                logits_processor = LogitsProcessorList()
            else:
                # Penalty Alpha disables all warpers, so we need to move all of them to the processors,
                # so they can be passed to the generate().
                # The only way to get the warpers beforehand is via the "protected" method _get_logits_warper().
                cfg = copy.deepcopy(generation_config)
                cfg.update(**warpers_params)
                logits_processor = model._get_logits_warper(generation_config=cfg)

            if r.words_whitelist is not None:
                bad_words_ids = tok.bad_words_by_whitelist(r.words_whitelist, tokenizer)
            else:
                bad_words_ids = None

            if r.words_blacklist:
                blacklisted_word_ids = tok.bad_words_by_blacklist(r.words_blacklist, tokenizer)
                if bad_words_ids is None:
                    bad_words_ids = []
                bad_words_ids += blacklisted_word_ids

            if model.config.eos_token_id is not None:
                begin_suppress_tokens = [model.config.eos_token_id]
            else:
                begin_suppress_tokens = None

            out_tensor = model.generate(
                in_tensor,
                generation_config=generation_config,

                do_sample=do_sample,
                min_length=input_tokens_len + r.generated_tokens_count,
                max_length=input_tokens_len + r.generated_tokens_count,
                use_cache=True,  # insanely slow without the cache (but uses much less VRAM)
                num_return_sequences=r.sequences_count,
                num_beams=1,
                num_beam_groups=1,
                penalty_alpha=r.penalty_alpha,
                logits_processor=logits_processor,
                stopping_criteria=stop_list,
                return_dict_in_generate=False,
                bad_words_ids=bad_words_ids,
                begin_suppress_tokens=begin_suppress_tokens,
                no_repeat_ngram_size=r.no_repeat_ngram_size,

                **warpers_params
            )
            lwo.restore_get_logits_warper(model)
        except Exception as e:
            lwo.restore_get_logits_warper(model)
            raise e

        out_tokens_len = out_tensor.shape[-1]

        if penalty_warper is None or penalty_warper.first_tokens is None:
            repetition_penalty_text_at_start = ""
            used_repetition_penalty_range_at_start = 0
            used_repetition_penalty_tokens_count_at_start = 0
            used_repetition_penalty_range_at_end = 0
            used_repetition_penalty_tokens_count_at_end = 0
        else:
            used_repetition_penalty_range_at_start = penalty_warper.first_tokens.shape[-1]
            repetition_penalty_text_at_start = tok.tokens_to_str(penalty_warper.first_tokens[0], tokenizer)
            used_repetition_penalty_tokens_count_at_start = penalty_warper.first_range
            used_repetition_penalty_range_at_end = penalty_warper.last_tokens.shape[-1]
            used_repetition_penalty_tokens_count_at_end = penalty_warper.last_range

        input_text_len = len(r.preamble) + len(tok_res.trimmed_prompt)

        for seq_idx, out_tokens in enumerate(out_tensor):
            if penalty_warper is None or penalty_warper.last_tokens is None:
                repetition_penalty_text_at_end = ""
            else:
                repetition_penalty_text_at_end = tok.tokens_to_str(
                    penalty_warper.last_tokens[seq_idx], tokenizer)

            out_txt = tok.tokens_to_str(out_tokens, tokenizer)

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
                output_text=out_txt,
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

        for _ in range(r.sequences_count):
            seq = GeneratedSequence(
                output_text="",
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
        preamble=tok_res.preamble,
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
