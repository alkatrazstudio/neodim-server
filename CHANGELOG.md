# Neodim Server - CHANGELOG


## v0.11 (April 23, 2023)

- Added: [no_repeat_ngram_size](README.md#no_repeat_ngram_size-int-optional) parameter
- Added: [words_blacklist](README.md#words_blacklist-string-optional) parameter
- Added: [words_whitelist](README.md#words_whitelist-string-optional) parameter
- Added: support for [LLaMA](https://huggingface.co/models?other=llama) models
- Added: support for safetensors
- Improved: 8-bit models can now be loaded directly
- Fixed: the playground misses some input parameters
- Fixed: multiple encoding/decoding problems
- Changed: the server now tries to generate at least one token


## v0.10 (March 4, 2023)

- Fixed: specifying `top_k` gives a type error
- Added: contrastive search
  (see [penalty_alpha](README.md#penalty_alpha-float-optional) parameter)


## v0.9 (February 19, 2023)

- Fixed: wrong truncation of the inference result
- Changed: special tokens (e.g. `<|endoftext|>` or `</s>`) are now always removed
  from the input and the output
- Added: server version validation
  ([required_server_version](README.md#required_server_version-string-optional) request parameter)
- Added: full output from the model is returned in the
  [output_text](README.md#sequencesoutput_text-string) response field
- Added: actually used preamble is returned in the
  [preamble](README.md#preamble-string) response field
- Improved: silenced messages: "the specified maximum sequence length" and "Welcome to bitsandbytes"


## v0.8 (December 18, 2022)

- Changed: using CUDA 11.7
- Changed: `stop_strings` and `truncate_prompt_until` are not sorted anymore
- Added: support for
  [GPT-NeoX](https://huggingface.co/models?other=gpt_neox),
  [CodeGen](https://huggingface.co/models?other=codegen) and
  [BLOOM](https://huggingface.co/models?other=bloom) models
- Added: ability to specify regular expressions as stop strings (see
  [stop_strings_type](README.md#stop_strings_type-enumstringregex-optionaldefaultstring),
  [stop_strings_required_matches_count](README.md#stop_strings_required_matches_count-int-optional-default1),
  request parameters and
  [stop_string_match](README.md#sequencesstop_string_match-string)
  response parameter)
- Improved: layers distribution is now supported for 8-bit precision
  (i.e. `layers` can be set to any supported value when `precision=int8`)
- Improved: more readable display of free VRAM and layers distribution
- Improved: repetition penalty can now be specified in
  [warpers_order](README.md#warpers_order-string-optional)


## v0.7 (September 18, 2022)

- Added: ability to load the model in 32-bit and 8-bit precisions
  ([precision](README.md#precision-originalfloat32float16int8-optional-defaultfloat16) CLI param)
- Fixed: fast tokenizer is not working for OPT models


## v0.6 (July 16, 2022)

- Added: typical sampling (`typical` request param)
- Added: top-a sampling (`top-a` request param)
- Added: the order of filters/sampling/warpers can now be set (`warpers_order` request param)
- Added: the playground settings can now be reset to their defaults


## v0.5 (July 2, 2022)

- Changed: using CUDA 11.6 (to support modern GPUs)
- Improved: loading models is now 4 times faster
- Improved: GPT2 models now support layers distribution
- Added: support for OPT models
- Added: use `a` symbol to put all unspecified layers on a GPU


## v0.4 (June 26, 2022)

- Fixed: wrong device detection when moving the entire model to GPU


## v0.3 (April 10, 2022)

- Added: `--version` flag to show the current version


## v0.2 (April 2, 2022)

- Fixed: avoid the crash with penalty slope when penalty range = 1


## v0.1 (March 20, 2022)

- Initial release
