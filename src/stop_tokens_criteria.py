# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from regex import regex
from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteria

import tokenizer as tok


class StopStringsType(Enum):
    STRING = "string"
    REGEX = "regex"


@dataclass
class StopStringMatch:
    stop_string: str
    start_index: int
    match: str


class StopTokensCriteria(StoppingCriteria):
    matches: list[Optional[StopStringMatch]]

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
        self.stop_strings = stop_strings
        if stop_strings_type == StopStringsType.REGEX:
            self.stop_regexes = [regex.compile(s) for s in stop_strings]

        self.stop_strings_type = stop_strings_type
        self.tokenizer = tokenizer
        self.model = model

        self.matches = [None for _ in range(sequences_count)]
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
            if self.matches[seq_index]:
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
                    rx_match = self.stop_regexes[str_index].search(analyzed_text)
                    match = StopStringMatch(
                        stop_string=stop_string,
                        start_index=rx_match.start(),
                        match=analyzed_text[rx_match.start():rx_match.end()]
                    ) if rx_match else None
                else:
                    start_index = analyzed_text.find(stop_string)
                    if start_index >= 0:
                        # now we actually need the full text to determine the match position
                        if generated_tokens_count != 1 and analyzed_tokens_count != generated_tokens_count:
                            analyzed_tokens = input_tokens[-generated_tokens_count:]
                            analyzed_text = tok.tokens_to_str(analyzed_tokens, self.model, self.tokenizer)
                            start_index = analyzed_text.find(stop_string)
                        match = StopStringMatch(
                            stop_string=stop_string,
                            start_index=start_index,
                            match=stop_string
                        ) if start_index >= 0 else None
                    else:
                        match = None
                if match:
                    self.matches_left[seq_index][str_index] = self.matches_left[seq_index][str_index] - 1
                    if self.matches_left[seq_index][str_index] <= 0:
                        self.matches[seq_index] = match
                else:
                    self.matches_left[seq_index][str_index] = self.required_matches_count
                if self.matches[seq_index]:
                    break

        all_stopped = all(self.matches)
        return all_stopped
