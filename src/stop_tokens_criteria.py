# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, StoppingCriteria

import tokenizer as tok


class StopTokensCriteria(StoppingCriteria):
    def __init__(
        self,
        input_tokens_len: int,
        stop_strings: list[str],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sequences_count: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.input_tokens_len = input_tokens_len
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.model = model

        self.is_stopped = [False for _ in range(sequences_count)]
        self.max_stop_string_len = max(len(s) for s in stop_strings)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Tokens can represent any text, so we can't just compare input_ids to encoded stop_strings.
        # E.g. if we compare to " " token directly, the inference won't stop on "! " token.
        for i, input_tokens in enumerate(input_ids):
            if self.is_stopped[i]:
                continue

            total_len = input_tokens.shape[0]
            generated_tokens_count = total_len - self.input_tokens_len
            if generated_tokens_count <= 0:
                continue

            # There's no point in decoding a whole sequence.
            # We just decode the last N tokens from the end.
            # N = <number of characters in a stop string> + 1 (extra token to include possible concatenation symbols)
            analyzed_tokens_count = min(self.max_stop_string_len, generated_tokens_count) + 1
            analyzed_tokens = input_tokens[-analyzed_tokens_count:]
            analyzed_text = tok.tokens_to_str(analyzed_tokens, self.model, self.tokenizer)
            if generated_tokens_count == 1:
                # special case: exclude the end of the input text
                prev_token_str = tok.tokens_to_str(analyzed_tokens[0:1], self.model, self.tokenizer)
                analyzed_text = analyzed_text[len(prev_token_str):]

            for stop_string in self.stop_strings:
                self.is_stopped[i] = analyzed_text.find(stop_string) >= 0
                if self.is_stopped[i]:
                    break

        all_stopped = all(self.is_stopped)
        return all_stopped
