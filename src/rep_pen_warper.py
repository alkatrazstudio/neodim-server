# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

from enum import Enum
from typing import Optional

import torch

from third_party.warpers import AdvancedRepetitionPenaltyLogitsProcessor as _RepPenWarper


# how to include the generated text in the penalty range
class RepPenGenerated(Enum):
    IGNORE = "ignore"  # do not include
    EXPAND = "expand"  # include all generated text
    SLIDE = "slide"  # include all newly generated text but keep the original range length


# warpers.py is "external" file,
# so to make is easier to keep it in sync with upstream there should not be any modifications to it.
# Instead, all modifications should happen outside, in child classes.
class RepetitionPenaltyLogitsWarper(_RepPenWarper):
    first_tokens: Optional[torch.LongTensor] = None
    last_tokens: Optional[torch.LongTensor] = None
    first_range: Optional[int] = None
    last_range: Optional[int] = None

    def __init__(
        self,
        penalty: float,
        input_len: int,
        penalty_range: int,
        preamble_tokens_count: int = 0,
        include_preamble: bool = False,
        penalty_slope: Optional[float] = None,
        include_generated: RepPenGenerated = RepPenGenerated.SLIDE,
        truncate_to_input: bool = False,
        prompt_tokens: Optional[torch.LongTensor] = None
    ):
        super().__init__()
        if penalty_slope is None:
            penalty_slope = 0

        self.include_generated = include_generated
        self.penalty_range = penalty_range
        self.original_penalty_range = penalty_range
        self.preamble_tokens_count = preamble_tokens_count
        self.include_preamble = include_preamble
        self.original_penalty = penalty
        self.penalty = penalty
        self.original_penalty_slope = penalty_slope
        self.penalty_slope = penalty_slope
        self.input_len = input_len
        self.truncate_to_input = truncate_to_input
        self.prompt_tokens = prompt_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # self.penalty is overwritten in AdvancedRepetitionPenaltyLogitsProcessor.__call__,
        # it causes a crash on second invocation,
        # therefore we need to restore self.penalty before each call
        self.penalty = self.original_penalty

        if self.preamble_tokens_count and not self.include_preamble:
            input_ids = input_ids[..., self.preamble_tokens_count:]
            preamble_tokens_left = 0
        else:
            preamble_tokens_left = self.preamble_tokens_count

        if self.prompt_tokens is not None:
            # replace the initial prompt tokens with the custom ones,
            # but leave the preamble and the generated text intact
            preamble_tokens = input_ids[..., :preamble_tokens_left]
            generated_tokens = input_ids[..., self.input_len:]
            input_ids = torch.cat((preamble_tokens, self.prompt_tokens, generated_tokens), -1)
            input_len = preamble_tokens.shape[-1] + self.prompt_tokens.shape[-1]
        else:
            input_len = self.input_len

        if self.include_generated == RepPenGenerated.IGNORE:
            input_ids = input_ids[..., :input_len]
        elif self.include_generated == RepPenGenerated.EXPAND:
            tokens_len = input_ids.shape[-1]
            generated_len = tokens_len - input_len
            self.penalty_range = self.original_penalty_range + generated_len
        else:
            input_ids = input_ids[..., -self.original_penalty_range:]

        if self.first_tokens is None:
            self.first_tokens = input_ids
        self.last_tokens = input_ids

        if self.truncate_to_input:
            self.penalty_range = min(input_ids.shape[-1], self.penalty_range)

        if self.first_range is None:
            self.first_range = self.penalty_range
        self.last_range = self.penalty_range

        if not self.penalty_range or input_ids.shape[-1] == 0:
            return scores

        # applying penalty_slope requires penalty_range > 1
        # (see the original AdvancedRepetitionPenaltyLogitsProcessor in warpers.py)
        if self.penalty_range == 1:
            self.penalty_slope = 0
        else:
            self.penalty_slope = self.original_penalty_slope

        return super().__call__(input_ids, scores)
