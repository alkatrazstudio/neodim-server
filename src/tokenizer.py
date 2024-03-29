# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

import itertools
import sys
from typing import Final

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

import tools


S_NEWLINE: Final[str] = "</s>"
BREAK_TOKEN: Final[str] = "<|neodimbreak|>"


class TokenizerResult:
    preamble_tokens_count: int
    original_prompt_tokens_count: int
    original_input_tokens_count: int
    preamble: str
    trimmed_prompt: str
    trimmed_prompt_tokens_count: int
    input_tokens: list[int]


def is_s_newline(tokenizer: PreTrainedTokenizer) -> bool:
    if hasattr(tokenizer, "__is_s_newline"):
        return tokenizer.__is_s_newline

    result = tokenizer.decode(tokenizer.encode("\n"))
    tokenizer.__is_s_newline = S_NEWLINE in result
    return tokenizer.__is_s_newline


def has_extra_nl_space(tokenizer: PreTrainedTokenizer) -> bool:
    if hasattr(tokenizer, "__has_extra_nl_space"):
        return tokenizer.__has_extra_nl_space

    # For a XGLM tokenizer the "\na" will be transformed into "\n\n a".
    # We detect this case here, but "\n a" will also transform into "\n\n a",
    # so this method is not foolproof.
    s = tokenizer.decode(tokenizer.encode(S_NEWLINE + "a"))
    tokenizer.__has_extra_nl_space = s == S_NEWLINE + S_NEWLINE + " a"
    return tokenizer.__has_extra_nl_space


def get_ignored_prefix(tokenizer: PreTrainedTokenizer) -> str:
    if hasattr(tokenizer, "__ignored_prefix"):
        return tokenizer.__ignored_prefix

    test_str = "Test"
    out_str = tokenizer.decode(str_to_tokens(test_str, tokenizer))
    dec_pos = out_str.index(test_str)
    tokenizer.__ignored_prefix = out_str[:dec_pos]
    return tokenizer.__ignored_prefix


def has_extra_nl(tokenizer: PreTrainedTokenizer) -> bool:
    if hasattr(tokenizer, "__has_extra_nl"):
        return tokenizer.__has_extra_nl

    # For an OPT tokenizer the "\na" will be transformed into "\n\na".
    s = tokenizer.decode(tokenizer.encode(S_NEWLINE + "a"))
    tokenizer.__has_extra_nl = s == S_NEWLINE + S_NEWLINE + "a"
    return tokenizer.__has_extra_nl


def get_ignored_tokens(tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    if hasattr(tokenizer, "__ignored_tokens"):
        return tokenizer.__ignored_tokens

    special_ids = tokenizer.all_special_ids
    if is_s_newline(tokenizer):
        vocab = tokenizer.get_vocab()
        if S_NEWLINE in vocab:
            s_id = vocab[S_NEWLINE]
            special_ids = list(set(special_ids) - {s_id})

    tokenizer.__ignored_tokens = torch.tensor(special_ids, dtype=torch.int64)
    return tokenizer.__ignored_tokens


def str_to_tokens(text: str, tokenizer: PreTrainedTokenizer) -> list[int]:
    if not text:
        return []

    s_nl = is_s_newline(tokenizer)
    if s_nl:
        text = text.replace("\n", S_NEWLINE)

    # Using a fake max_length to silence the warning about "the specified maximum sequence length"
    tokens = tokenizer.encode(text, max_length=sys.maxsize, truncation=True)
    if s_nl and (has_extra_nl_space(tokenizer) or has_extra_nl(tokenizer)) and len(tokens):
        tokens = tokens[1:]
    return tokens


def remove_ignored_tokens(
    tokens: list[int] | torch.Tensor,
    tokenizer: PreTrainedTokenizer
) -> list[int] | torch.Tensor:
    ignored_tokens = get_ignored_tokens(tokenizer)
    if isinstance(tokens, torch.Tensor):
        ignored_tokens = ignored_tokens.to(tokens.device)
        tokens = tokens[torch.logical_not(torch.isin(tokens, ignored_tokens))]
    else:
        ignored_tokens = ignored_tokens.tolist()
        tokens = [token for token in tokens if token not in ignored_tokens]
    return tokens


def add_break_token(tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> None:
    tokenizer.add_tokens([BREAK_TOKEN])
    model.resize_token_embeddings(len(tokenizer))


def tokens_to_str(
    tokens: list[int] | torch.Tensor,
    tokenizer: PreTrainedTokenizer
) -> str:
    # Ignore special tokens to avoid adding things like <|endoftext|> to the output.
    # Can't use skip_special_tokens=True because some special tokens (e.g. </s>) must not be skipped.
    tokens = remove_ignored_tokens(tokens, tokenizer)

    if not len(tokens):
        return ""

    text = tokenizer.decode(tokens)
    if is_s_newline(tokenizer):
        if has_extra_nl_space(tokenizer):
            text = text.replace(S_NEWLINE + " ", "\n")
        text = text.replace(S_NEWLINE, "\n")

    prefix = get_ignored_prefix(tokenizer)
    if text.startswith(prefix):
        text = text[len(prefix):]
    return text


def normalize_str(text: str, tokenizer: PreTrainedTokenizer) -> tuple[str, list[int]]:
    special_tokens = tokenizer.all_special_tokens

    while True:
        new_text = text
        for token in special_tokens:
            new_text = new_text.replace(token, "")
        is_same = new_text == text
        text = new_text
        if is_same:
            break

    tokens = str_to_tokens(text, tokenizer)
    normalized_text = tokens_to_str(tokens, tokenizer)
    return normalized_text, tokens


def tokenize_input(
    prompt: str,
    preamble: str,
    truncate_untils: list[str] | None,
    tokenizer: PreTrainedTokenizer,
    max_tokens_len: int
) -> TokenizerResult:
    truncate_untils = tools.normalize_str_list(truncate_untils)
    err = RuntimeError(f"Cannot fit the prompt in {max_tokens_len} tokens while using specified stop_strings")

    res = TokenizerResult()
    preamble, preamble_tokens = normalize_str(preamble, tokenizer)
    res.preamble_tokens_count = len(preamble_tokens)

    # never encode/decode the prompt without the preamble,
    # because doing so may add extra "concatenation" tokens/symbols at the start of the prompt
    full_input, full_input_tokens = normalize_str(preamble + prompt, tokenizer)
    prompt = full_input[len(preamble):]
    prompt_tokens = full_input_tokens[res.preamble_tokens_count:]

    res.preamble = preamble
    res.trimmed_prompt = prompt
    res.original_prompt_tokens_count = len(prompt_tokens)
    res.trimmed_prompt_tokens_count = res.original_prompt_tokens_count
    res.input_tokens = full_input_tokens
    res.original_input_tokens_count = len(res.input_tokens)

    if not prompt and not preamble:
        return res

    max_allowed_prompt_tokens = max_tokens_len - res.preamble_tokens_count
    if res.preamble_tokens_count > max_tokens_len:
        # even the preamble does not fit inside the maximum allowed input tokens
        raise err

    if res.preamble_tokens_count == max_tokens_len:
        # only preamble fits the maximum allowed input tokens
        res.trimmed_prompt = ""
        res.trimmed_prompt_tokens_count = 0
        res.input_tokens = preamble_tokens
        return res

    if max_tokens_len >= res.original_input_tokens_count:
        # both preamble and prompt fit inside the maximum allowed input tokens, no need to trim anything
        return res

    trimmed_prompt_tokens = prompt_tokens[-max_allowed_prompt_tokens:]
    full_input_tokens = preamble_tokens + trimmed_prompt_tokens
    full_input = tokens_to_str(full_input_tokens, tokenizer)
    res.trimmed_prompt = full_input[len(preamble):]
    res.trimmed_prompt_tokens_count = len(trimmed_prompt_tokens)

    if not truncate_untils:
        # no specific truncate_until strings, do not trim any further
        res.input_tokens = full_input_tokens
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

    res.input_tokens = str_to_tokens(preamble + res.trimmed_prompt, tokenizer)
    res.trimmed_prompt_tokens_count = len(res.input_tokens) - res.preamble_tokens_count
    return res


def word_list_to_tokens_lists(words: list[str], tokenizer: PreTrainedTokenizer) -> list[list[int]]:
    token_lists = []
    for word in words:
        if word.startswith(" "):
            spaced_word = word
            non_spaced_word = word.lstrip()
        else:
            spaced_word = " " + word
            non_spaced_word = word

        non_spaced_ids = remove_ignored_tokens(str_to_tokens(non_spaced_word, tokenizer), tokenizer)

        non_concat_word = BREAK_TOKEN + non_spaced_word
        non_concat_ids = remove_ignored_tokens(str_to_tokens(non_concat_word, tokenizer), tokenizer)
        non_concat_ids = non_concat_ids[1:]

        token_lists.append(non_spaced_ids)
        if non_concat_ids != non_spaced_ids:
            token_lists.append(non_concat_ids)
        else:
            spaced_ids = remove_ignored_tokens(str_to_tokens(spaced_word, tokenizer), tokenizer)
            token_lists.append(spaced_ids)
    return token_lists


def bad_words_by_whitelist(whitelist_words: list[str], tokenizer: PreTrainedTokenizer) -> list[list[int]]:
    whitelist_words_ids = word_list_to_tokens_lists(whitelist_words, tokenizer)
    whitelist_ids = list(itertools.chain.from_iterable(whitelist_words_ids))
    bad_ids = [[token_id] for token_id in tokenizer.get_vocab().values() if token_id not in whitelist_ids]
    return bad_ids


def bad_words_by_blacklist(blacklist_words: list[str], tokenizer: PreTrainedTokenizer) -> list[list[int]]:
    bad_ids = word_list_to_tokens_lists(blacklist_words, tokenizer)
    return bad_ids
