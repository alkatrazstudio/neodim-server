# Neodim Server - CHANGELOG


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
