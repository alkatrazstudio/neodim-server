# Neodim Server - CHANGELOG


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
