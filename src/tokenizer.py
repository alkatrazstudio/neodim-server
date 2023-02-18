# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

import sys
from typing import Final, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

import tools
from tools import ModelType


S_NEWLINE: Final[str] = "</s>"


class TokenizerResult:
    preamble_tokens_count: int
    original_prompt_tokens_count: int
    original_input_tokens_count: int
    trimmed_prompt: str
    trimmed_prompt_tokens_count: int
    input_tokens: list[int]


def is_s_newline(model: PreTrainedModel) -> bool:
    return tools.model_type(model) in [ModelType.XGLM, ModelType.OPT]


def has_extra_nl_space(tokenizer: PreTrainedTokenizer) -> bool:
    if hasattr(tokenizer, "__has_extra_nl_space"):
        return tokenizer.__has_extra_nl_space

    # For a XGLM tokenizer the "\na" will be transformed into "\n\n a".
    # We detect this case here, but "\n a" will also transform into "\n\n a",
    # so this method is not foolproof.
    s = tokenizer.decode(tokenizer.encode(S_NEWLINE + "a"))
    tokenizer.__has_extra_nl_space = s == S_NEWLINE + S_NEWLINE + " a"
    return tokenizer.__has_extra_nl_space


def has_extra_nl(tokenizer: PreTrainedTokenizer) -> bool:
    if hasattr(tokenizer, "__has_extra_nl"):
        return tokenizer.__has_extra_nl

    # For an OPT tokenizer the "\na" will be transformed into "\n\na".
    s = tokenizer.decode(tokenizer.encode(S_NEWLINE + "a"))
    tokenizer.__has_extra_nl = s == S_NEWLINE + S_NEWLINE + "a"
    return tokenizer.__has_extra_nl


def str_to_tokens(text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> list[int]:
    if not text:
        return []

    s_nl = is_s_newline(model)
    if s_nl:
        text = text.replace("\n", S_NEWLINE)

    # Using a fake max_length to silence the warning about "the specified maximum sequence length"
    tokens = tokenizer.encode(text, max_length=sys.maxsize, truncation=True)
    if s_nl and (has_extra_nl_space(tokenizer) or has_extra_nl(tokenizer)) and len(tokens):
        tokens = tokens[1:]
    return tokens


def tokens_to_str(
    tokens: Union[list[int], torch.Tensor],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer
) -> str:
    if not len(tokens):
        return ""

    text = tokenizer.decode(tokens)
    if is_s_newline(model):
        if has_extra_nl_space(tokenizer):
            text = text.replace(S_NEWLINE + " ", "\n")
        text = text.replace(S_NEWLINE, "\n")
    return text


def tokenize_input(
    prompt: str,
    preamble: str,
    truncate_untils: Optional[list[str]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_tokens_len: int
) -> TokenizerResult:
    truncate_untils = tools.normalize_str_list(truncate_untils)
    err = RuntimeError(f"Cannot fit the prompt in {max_tokens_len} tokens while using specified stop_strings")

    res = TokenizerResult()
    preamble_tokens = str_to_tokens(preamble, model, tokenizer)
    res.preamble_tokens_count = len(preamble_tokens)
    res.trimmed_prompt = prompt
    prompt_tokens = str_to_tokens(prompt, model, tokenizer)
    res.original_prompt_tokens_count = len(prompt_tokens)
    res.trimmed_prompt_tokens_count = res.original_prompt_tokens_count
    res.input_tokens = preamble_tokens + prompt_tokens
    res.original_input_tokens_count = len(res.input_tokens)

    if not prompt and not preamble:
        return res

    max_allowed_prompt_tokens = max_tokens_len - res.preamble_tokens_count
    if max_allowed_prompt_tokens < 0:
        # seems like even preamble does not fit inside max_tokens_len
        raise err

    if max_allowed_prompt_tokens >= res.original_prompt_tokens_count:
        # the prompt is short enough, no need to trim anything
        return res

    trimmed_prompt_tokens = prompt_tokens[-max_allowed_prompt_tokens:]
    res.trimmed_prompt = tokens_to_str(trimmed_prompt_tokens, model, tokenizer)
    res.trimmed_prompt_tokens_count = len(trimmed_prompt_tokens)

    if not truncate_untils:
        # no specific truncate_until strings, do not trim any further
        res.input_tokens = preamble_tokens + trimmed_prompt_tokens
        return res

    remove_pos = -1
    reduce_string = ""
    for truncate_until in truncate_untils:
        pos = res.trimmed_prompt.find(truncate_until)
        if pos >= 0 and (remove_pos == -1 or pos < remove_pos):
            remove_pos = pos
            reduce_string = truncate_until

    if remove_pos == -1:
        # no truncate_until strings found
        if not preamble:
            # we now need to empty the whole prompt, but if there's no preamble we can't do that
            raise err

        res.trimmed_prompt = ""
        res.trimmed_prompt_tokens_count = 0
        res.input_tokens = preamble_tokens
        return res

    # trim from the end of the closest truncate_until string
    res.trimmed_prompt = res.trimmed_prompt[remove_pos + len(reduce_string):]
    if not res.trimmed_prompt and not preamble:
        # we eliminated the whole prompt, but there's no preamble either
        raise err

    trimmed_prompt_tokens = str_to_tokens(res.trimmed_prompt, model, tokenizer)
    res.trimmed_prompt_tokens_count = len(trimmed_prompt_tokens)
    res.input_tokens = preamble_tokens + trimmed_prompt_tokens
    return res
