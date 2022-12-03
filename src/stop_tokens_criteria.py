# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

import re
from enum import Enum
from re import Pattern
from typing import Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteria

import tokenizer as tok


class StopStringsType(Enum):
    STRING = "string"
    REGEX = "regex"


class StopTokensCriteria(StoppingCriteria):
    stop_strings: list[Union[str, Pattern[str]]]

    def __init__(
        self,
        input_tokens_len: int,
        stop_strings: list[str],
        stop_strings_type: StopStringsType,
        required_matches_count: int,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sequences_count: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.input_tokens_len = input_tokens_len
        if stop_strings_type == StopStringsType.REGEX:
            self.stop_strings = [re.compile(s) for s in stop_strings]
        else:
            self.stop_strings = stop_strings

        self.stop_strings_type = stop_strings_type
        self.tokenizer = tokenizer
        self.model = model

        self.is_stopped = [False for _ in range(sequences_count)]
        self.required_matches_count = required_matches_count
        self.matches_left = [
            [required_matches_count for _ in range(len(stop_strings))]
            for _ in range(sequences_count)
        ]
        if stop_strings_type == StopStringsType.STRING:
            self.max_stop_string_len = max(len(s) for s in stop_strings)
        else:
            self.max_stop_string_len = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Tokens can represent any text, so we can't just compare input_ids to encoded stop_strings.
        # E.g. if we compare to " " token directly, the inference won't stop on "! " token.
        for seq_index, input_tokens in enumerate(input_ids):
            if self.is_stopped[seq_index]:
                continue

            total_len = input_tokens.shape[0]
            generated_tokens_count = total_len - self.input_tokens_len
            if generated_tokens_count <= 0:
                continue

            if self.max_stop_string_len is None:
                analyzed_tokens_count = generated_tokens_count
            else:
                # There's no point in decoding a whole sequence when using simple strings.
                # We just decode the last N tokens from the end.
                # N = <number of characters in a stop string> + 1
                # (extra token to include possible concatenation symbols)
                analyzed_tokens_count = min(self.max_stop_string_len, generated_tokens_count) + 1
            analyzed_tokens = input_tokens[-analyzed_tokens_count:]
            analyzed_text = tok.tokens_to_str(analyzed_tokens, self.model, self.tokenizer)
            if generated_tokens_count == 1 and self.max_stop_string_len is not None:
                # special case: exclude the end of the input text
                prev_token_str = tok.tokens_to_str(analyzed_tokens[0:1], self.model, self.tokenizer)
                analyzed_text = analyzed_text[len(prev_token_str):]

            for str_index, stop_string in enumerate(self.stop_strings):
                if self.stop_strings_type == StopStringsType.REGEX:
                    match = stop_string.search(analyzed_text)
                    has_match = match is not None
                else:
                    has_match = analyzed_text.find(stop_string) >= 0
                if has_match:
                    self.matches_left[seq_index][str_index] = self.matches_left[seq_index][str_index] - 1
                    if self.matches_left[seq_index][str_index] <= 0:
                        self.is_stopped[seq_index] = True
                else:
                    self.matches_left[seq_index][str_index] = self.required_matches_count
                if self.is_stopped[seq_index]:
                    break

        all_stopped = all(self.is_stopped)
        return all_stopped
