# Neodim Server

Neodim Server is an HTTP-server
that provides natural language text generation
using a selected language model.

---

Main website: https://github.com/alkatrazstudio/neodim-server

Current stable version: **v0.12** (June 11, 2023) • [CHANGELOG](CHANGELOG.md)

Repository branches:
* [master](https://github.com/alkatrazstudio/neodim-server/tree/master) - current stable version
* [dev](https://github.com/alkatrazstudio/neodim-server/tree/dev) - ongoing development

---

**NOTE:** before reading any further you must have at least a general grasp of the following topics:
machine learning, text generation, language model, HTTP server, command line interface.
Also, running this on Linux is recommended.
If you have any other OS you need to install Python, Git, Bash, sed and md5sum.


## Contents

- [How it works](#how-it-works)
- [Security and production use](#security-and-production-use)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Supported models](#supported-models)
  - [Notable models](#notable-models)
- [Downloading the language models](#downloading-the-language-models)
  - [Auto-download](#auto-download)
  - [Download via Git](#download-via-git)
  - [Manual download](#manual-download)
- [CLI](#cli)
- [Prompt and preamble](#prompt-and-preamble)
  - [Recommendations](#recommendations)
  - [Example](#example)
- [API request](#api-request)
- [API response](#api-response)
- [Squeezing the VRAM](#squeezing-the-vram)
  - [Running in console mode](#running-in-console-mode)
  - [Distributing model layers on CPU](#distributing-model-layers-on-cpu)
  - [Lowering the precision](#lowering-the-precision)
- [Third-party libraries](#third-party-libraries)
- [License](#license)
- [Client applications](#client-applications)


## How it works

Neodim Server uses [Transformers](https://huggingface.co/docs/transformers/index)
to provide the actual inference (text generation).
It can use some [language models from Hugging Face](https://huggingface.co/models?library=pytorch&pipeline_tag=text-generation&sort=downloads)
(it will download them automatically if needed)
or locally available models (pre-downloaded from Hugging Face or from anywhere else).
See below for a list of supported models.

After the model is loaded, the web server is started.
You can pass the text generation parameters to it,
and the server will use the model to generate some text and then the server will return this text (or an error).

Neodim Server is a stateless server.
It will not save anything between requests.
All required information will be passed with the request.


## Security and production use

Neodim Server as-is should only be used locally or internally.
It's not designed to be exposed over the Internet or used by multiple clients simultaneously.
For example, it uses the built-in Python HTTP web-server
(which is usually not tested for any security vulnerabilities),
can only handle one request at a time
and can easily be crashed or hanged (e.g. if you pass incorrect `Content-Length` header).

Neodim Server also lacks a lot of sanity checks and is not foolproof to use.


## Requirements

* CUDA-enabled GPU (e.g. any NVIDIA GTX/RTX), supporting at least CUDA v11.8.
* Appropriate amount of free VRAM (e.g., at least 5GB of free VRAM for `GPT-J-6B` model)
* Appropriate amount or free RAM (e.g., at least 15GB of free RAM for `GPT-J-6B` model)
* Python 3.9+ (check with `python3 --version`)
* Preferably Linux (Windows may work, but the compatibility is not tested and not maintained)

You may also need to update your GPU drivers.
On Ubuntu you can do it with `sudo ubuntu-drivers install`.
To download the repo and (optionally) models, you will need Git.
On Ubuntu you can install it with `sudo apt install git`.
To run the server the recommended way (via `start.sh`, see below)
you will need Python Virtual Environment tools.
On Ubuntu you can install them with `sudo apt install python3-venv`.


## Quick Start

```sh
# WARNING: this may download up to 2GB of data
git clone https://github.com/alkatrazstudio/neodim-server
neodim-server/start.sh --model=EleutherAI/gpt-neo-125M
```

This will start a server with the smallest language model:
[GPT-Neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125M).
This model is not suitable for any adequate text generation,
but it will allow you to test if Neodim Server works at all on your machine.

Open `playground.html` in your browser and play with parameters, etc.
All CLI options and parameters are explained below.

Here is another example:
```sh
# WARNING: this may download up to 13GB of data
neodim-server/start.sh --model=EleutherAI/gpt-j-6B --model-revision=float16 --listen-address=0.0.0.0 --layers=14 --precision=float4
```
It will use the official [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6B) model
by [EleutherAI](https://www.eleuther.ai)
from the [`float16` branch](https://huggingface.co/EleutherAI/gpt-j-6B/tree/float16).
Then it will be dynamically converted to 4-bit float to reduce VRAM usage.
The server will also listen on all network interfaces (`--listen-address=0.0.0.0`)
which allows any external client to connect to the server.
It will also only load half of the model to the GPU, and the rest will be on the CPU.

If you have `curl` installed you can try to manually send an HTTP request:
```sh
curl -Ss --data '{
  "prompt": "I walk across the dead train yard, remembering who we are.",
  "generated_tokens_count": 32,
  "max_total_tokens": 128,
  "temperature": 0.6,
  "tfs": 0.95,
  "truncate_prompt_until": [" "],
  "repetition_penalty": 3
}' http://127.0.0.1:8787/generate | jq -r '.sequences[].generated_text'
```
(`jq` is used here just to display the result in a human-readable form,
but you can remove it to view the raw JSON response)


## Supported models

* [GPT-Neo](https://huggingface.co/models?other=gpt_neo),
  [GPT-J](https://huggingface.co/models?other=gptj),
  [GPT-NeoX](https://huggingface.co/models?other=gpt_neox),
  [GPT 2](https://huggingface.co/models?other=gpt2),
  [OPT](https://huggingface.co/models?other=opt),
  [CodeGen](https://huggingface.co/models?other=codegen),
  [BLOOM](https://huggingface.co/models?other=bloom),
  [LLaMA](https://huggingface.co/models?other=llama)
  \- fully supported, except some corner cases.
* [XGLM](https://huggingface.co/models?other=xglm) - only some `fairseq` models are supported.
  You may try your luck with
  [KoboldAI fairseq-dense-* models](https://huggingface.co/KoboldAI).

However, each model may have some quirks which are out of the scope of this documentation.
For example, XGLM and OPT models have trouble with whitespace and newlines.


### Notable models

* [EleutherAI models](https://huggingface.co/EleutherAI) -
  original `GPT-Neo`, `GPT-J` and `GPT-NeoX` models,
  not specifically trained for any task
  (jack of all trades, master of none)

* [KoboldAI models](https://huggingface.co/KoboldAI) -
  models that are used in [KoboldAI](https://github.com/KoboldAI/KoboldAI-Client),
  trained for various tasks (mostly adventure games and novel writing)

* [sberbank-ai/rugpt3large_based_on_gpt2](https://huggingface.co/sberbank-ai/rugpt3large_based_on_gpt2) -
  at the time of writing this document,
  this was the best Transformers-compatible model that can generate Russian text


## Downloading the language models

There are at least three ways to get a language model on your PC.


### Auto-download

Let Neodim Server download the model from HuggingFace.
In this case you need to pass the model's name (e.g. `EleutherAI/gpt-neo-2.7B`)
to the [--model](#model-string-required) parameter.


### Download via Git

Specifying a model's name may be convenient, but it has some drawbacks:
* You need an internet connection even if the model was previously loaded
* It's hard to move the downloaded model to a different place or make a backup.

So, you may want to download the model manually first.
You can do it with Git, for example:
```sh
# first enable LFS if it's not enabled already
git lfs install

# then clone the model's repository
git clone https://huggingface.co/EleutherAI/gpt-neo-2.7B

# or if you want to clone a specific branch
git clone -b float16 https://huggingface.co/EleutherAI/gpt-j-6B
```

After that you'll be able to specify the cloned directory path in the `model` parameter.

**NOTES:**

* The whole Git repository will take twice the space that is really needed to run the model.
  The `.git` subfolder will contain the entire copy of the language model.
  If you don't plan on frequently updating the model you can remove the `.git` subfolder to save some disk space.

* The repository may contain some other files that is not needed for Neodim Server.
  For example, a language model for Rust or something like that.
  The list of required files depends on a model,
  but the only big file(s) that is used by Neodim Server are `pytorch_model*.bin` or `model-*.safetensors`.
  If there are some other multi-gigabyte files (e.g. `rust_model.ot` or `flax_model.msgpack`),
  you can safely remove them.


### Manual download

Downloading models with Git is easy, but as discussed in the previous section,
you may need to do some cleanup afterwards.

It's also possible to manually download all needed files from HuggingFace.

1. Go to the model's page, e.g. https://huggingface.co/EleutherAI/gpt-j-6B.
2. Go to "Files and versions" tab.
3. Choose the required branch from the dropdown, e.g. `float16`.
4. Download all needed files by clicking the "down arrow" icon.
5. Put all these files in the same folder.

Like it was said in the previous section, the only multi-gigabyte file(s) you need are `pytorch_model*.bin` or `model-*.safetensors`.
All other huge files are not needed.
However, all other small files (e.g. `*.json` and `*.txt`) are usually needed to run a language model.


## CLI

To start the server run the `start.sh` script.
You can pass some parameters to it in the following forms: `--<param-name>=val` or `--<param-name> val`.
For example, `--model=EleutherAI/gpt-neo-1.3B` or `--model EleutherAI/gpt-neo-1.3B`.

Pass `--help` or `-h` to show a short help.
Pass `--version` to show the version of your Neodim Server.

Below is the list of all supported parameters.


### `model`: string (required)

You can specify either a path to the directory of your model,
or its name on Hugging Face (e.g. `EleutherAI/gpt-neo-2.7B`).

The directory path must either be absolute (e.g. `/path/to/model`)
or start with a dot (e.g. `./some/path` or `../some/path`).

Read more on how to [download the language models](#downloading-the-language-models).

### `model-revision`: string (optional, default=\<default branch for the repo\>)

If you specified a Hugging Face model name in the `model` parameter
then `model-revision` will let you specify a specific branch.

By default, it will download the main branch.

### `cache-dir`: string (optional, default=\<transformer's default\>)

A directory for storing downloaded models.

By default, it's chosen by `transformers` framework.

### `listen-address`: string (optional, default="127.0.0.1")

Listen on this IP address.

### `listen-port`: int (optional, default=8787)

Listen on this port.

### `layers`: (int|"a")[] (optional, default="a")

Distribute model's layers on GPUs.

It's a comma-separated list of integers.
Each integer represents how many layers should be put on a GPU with that index.
All unspecified layers will be put on CPU.

You can also use a letter `a` (stands for "**a**vailable")
for one GPU to put all unspecified layers on it.
For example, `--layers=a` will put all model's layers on the first GPU.
`--layers=2,a,3` will put 2 layers on the first GPU, 3 layers on the third GPU
and the rest of the layers on the second GPU.

GPUs and their indexes are shown at server startup.
The number of layers is shown after the model is loaded,
but you also can find this number in `config.json` of the model
(see `n_layer`, `num_layers` or `num_hidden_layers` fields).

For example:
```sh
./start.sh --model=KoboldAI/GPT-J-6B-Skein --layers=0,2
```
This will put no layers on the first GPU, 2 layers on the second GPU and the rest of the layers on the CPU.

The more layers you put on GPUs the faster the model will work
and the more VRAM it will use.
You can also put all layers on CPU (`--layers=0`)
or put all layers on a single GPU (e.g. `--layers=a`, `--layers=0,a` etc.).

### `precision`: original|float32|float16|float4|int8|gptq2|gptq4|gptq8 (optional, default=float16)

Neodim Server can load 16-bit and 32-bit models,
but using this parameter the model can be load in memory with a specified precision.
This allows to seek balance between VRAM consumption and the quality of the generated text.

* `float16` - 16-bit floating precision, a.k.a. "half-precision".
  This is the default.
* `float32` - 32-bit floating precision, a.k.a. "full-precision".
  Compared to `float16`, loading the model in `float32` may improve the quality of the generated text,
  but the model will consume nearly twice as much VRAM.
* `int8` - 8-bit integer precision.
  Compared to `float16`, loading the model in `int8` may lower the quality of the generated text,
  but the model will consume around 1.5 times less VRAM.
* `float4` - 4-bit float precision.
  Compared to `int8`, loading the model in `float4` may lower the quality of the generated text,
  but the model will consume around 1.5 times less VRAM.
* `gptq2`, `gptq4`, `gptq8` - [read below](#gptq)
* `original` - use the original precision of the model.

It only makes sense to downgrade the model's precision.
For example, if you load 16-bit model in 32-bit precision mode
then it will consume twice as much VRAM, but it won't improve the quality of the generated text.

#### **INT8**

1. You need to install CUDA Toolkit.
   There are several ways to do it:
   - On Ubuntu: `sudo apt install nvidia-cuda-toolkit`
   - The official NVIDIA page: https://developer.nvidia.com/cuda-downloads
   - [Conda](https://docs.conda.io/): `conda install -c nvidia/label/cuda-<cuda version> cuda`
2. CPU-layers won't be converted to 8-bit.
3. You can load 8-bit models directly, but you still need to specify `--precision=int8`.
   Also, in this case all layers must be on GPU, e.g. `--layers=a`.
4. Some models may fail to load completely.

#### **FLOAT4**

The above notes apply to `float4` precision as well.

Also, when using `--precision=float4` the following additional CLI parameters are available:
[--quantization-type](#quantization-type-fp4nf4-optional-defaultnf4)
and [--double-quantization](#double-quantization-truefalse-optional-defaulttrue).

#### **GPTQ**

Neodim Server supports 2-bit, 4-bit and 8-bit models quantized with [GPTQ](https://arxiv.org/abs/2210.17323).
4-bit models consume roughly 1.5-2 times less VRAM than 8-bit models,
and only have slightly lesser quality (the bigger the model - the less the quality loss).
On HuggingFace Hub such models will have either "GPTQ" and/or "4bit" and "128g/64g/32g" in their name, but not "GGML" (that's a different quantization technique, not compatible with Neodim Server).

Example 4-bit GPTQ model: https://huggingface.co/TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g.

To load GPTQ models you need to specify additional parameters to CLI:
[--model-basename](#model-basename-string-optional),
[--group-size](#group-size-int-optional),
[--true-sequential](#true-sequential-truefalse-optional-defaulttrue)
and [--safetensors](#safetensors-truefalse-optional-defaulttrue).
The last three already have popular defaults, so most of the time you only need to specify `--model-basename`.
There are also some additional parameters that can influence the model behavior:
[--fused-attention](#fused-attention-truefalse-optional-defaultfalse)
and [--fused-mlp](#fused-mlp-truefalse-optional-defaultfalse).

Current limitations and gotchas:

* Some models may not load, it depends on how they were quantized.
* `--precision=gptq8` is different from `--precision=int8`,
  and `--precision=gptq4` is different from `--precision=float4` - GPTQ uses a different quantization algorithms, so it's not possible to load a GPTQ model with `int8` and `float4` precisions, and vice versa.
* `--precision=int8` and `--precision=float4` can quantize the model on fly, i.e. you can load float16 model in INT8 precision. GPTQ models, on the other hand, need to be quantized beforehand.
* You will need `tokenizer.json` for GPTQ models, otherwise they will be very slow.

Example that shows how to load the abovementioned model:
```sh
./start.sh --model=/opt/models/vicuna-13B-1.1-GPTQ-4bit-128g --precision=gptq4 --model-basename=vicuna-13B-1.1-GPTQ-4bit-128g.latest
```

You can also load GPTQ models directly from the HuggingFace Hub:
```sh
./start.sh --model Neko-Institute-of-Science/LLaMA-7B-4bit-128g --precision=gptq4 --model-basename=llama-7b-4bit-128g
```

### `model-basename`: string (optional)

Specify a model basename, i.e. the filename without directory and extenstion.
This setting only applied for [GPTQ](#gptq) models, i.e. if `--precision=gptq*`.

For example, to load the model `/some/dir/wizardLM-7B-GPTQ.safetensors`,
use this: `--model=/some/dir --model-basename=wizardLM-7B-GPTQ --safetensors=true`.

### `group-size`: int (optional)

The group size parameter for a [GPTQ](#gptq) model.
This value is usually mentioned in the description of the model.
Sometimes it's in the filename - a number with a `g` suffix,
e.g. `koala-13B-GPTQ-4bit-128g` has group size of `128`.

### `true-sequential`: true|false (optional, default=true)

The "true sequential" flag for a [GPTQ](#gptq) model.
This value is usually mentioned in the description of the model.

### `safetensors`: true|false (optional, default=true)

Whether to load [GPTQ](#gptq) model in `safetensors` format.
If `false`, then the `*.bin` or `*.pt` file will be loaded.

### `fused-attention`: true|false (optional, default=false)

Inject fused attention.
Only applicable to [GPTQ](#gptq) models, but may not be incompatible with some of them.
May increase the inference speed at the cost of higher VRAM usage.

### `fused-mlp`: true|false (optional, default=false)

Inject fused MLP.
Only applicable to [GPTQ](#gptq) models, but may not be incompatible with some of them.
May increase the inference speed at the cost of higher VRAM usage.

### `quantization-type`: fp4|nf4 (optional, default=nf4)

Use [4-bit floating point](https://huggingface.co/blog/4bit-transformers-bitsandbytes#fp4-precision-in-a-few-words) (`fp4`)
or [4-bit NormalFloat](https://arxiv.org/abs/2305.14314) (`nf4`)
when performing the quantization.
This mayt affect the quality of the generated text.
Only applicable to `float4` precision.

### `double-quantization`: true|false (optional, default=true)

Perform double quantization.
Disabling it will reduce the VRAM consumption by approximately 10%,
but may lower the quality of the generated text.


## Prompt and preamble

With each request you can pass two chunks of text to the model.
The first chunk is "preamble" and the second chunk is "prompt".

The final input text will be the result of concatenation of preamble and prompt
(without any spaces).
However, prompt may be truncated from its beginning
so that the final input text can fit inside the total allowed number of tokens.
The preamble is never truncated.

Either preamble or prompt can be empty, but not both at the same time.

Both preamble and prompt may be modified before they are passed to the language model.
It can happen if they contain special tokens that are handled differently by the model.
See the response parameters
[used_prompt](#used_prompt-string) and [preamble](#preamble-string)
to see what was actually passed to the model.


### Recommendations

A lot of models give incorrect results if the input text has spaces in particular places.
Below are the list of recommendations to reduce the number of inference errors.

* Strip all leading and trailing spaces from the input text (`prompt + preamble`).
  It's safe to leave leading and trailing newlines, though.
* Squish multiple consequent spaces into one.
* Do not put any spaces right after a newline.
* For newlines only use `\n` and not `\r`.

Check each model output independently to see what happens when you put spaces in various places.


### Example

Let's assume we have this preamble:
```
This is a story about Alice.


```
(notice the two line breaks)

And the prompt:
> In another moment down went Alice after it, never once considering how in the world she was to get out again. The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.

Now let's assume that the whole text (preamble + prompt) does not fit within maximum allowed tokens.
Therefore, the prompt will be truncated from the beginning,
and the actual input text that will be passed to the model may look like this:

```
This is a story about Alice.

like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.
```


## API request

The request body must be a JSON that represents an object with the fields described below.

**Example:**
```json
{
  "preamble": "The Butcher of Ark\n\n",
  "prompt": "It was a dull, cold and wet morning that would change my life forever. Yes... Somehow it almost seemed as if that day Mother Nature, as a response to the festivities of the preceding night, had decided to recover herself with a dreamy, nondescript day.",
  "generated_tokens_count": 40,
  "max_total_tokens": 80,
  "temperature": 0.6,
  "tfs": 0.95,
  "repetition_penalty_range": 20,
  "repetition_penalty_slope": 1.25,
  "repetition_penalty": 2.5,
  "repetition_penalty_include_generated": "slide",
  "truncate_prompt_until": [" "],
  "stop_strings": [".", "!", "?"],
  "sequences_count": 2
}
```

### `prompt`: string (optional, default="")

Specify your prompt here.

### `preamble`: string (optional, default="")

Specify your preamble here.
It will be prepended to the prompt.
Note that the preamble will be prepended as is,
without trimming any leading or trailing spaces.

### `sequences_count`: int (optional, default=1)

Number of sequences to generate.
A sequence is a text that was generated based on preamble and prompt.

By default, only one sequence is generated.
When using GPU, additional sequences do not slow down the inference, but require more VRAM.

### `generated_tokens_count`: int (required)

The amount of tokens to generate for each sequence.
A token can represent any amount of text, but usually it's around 4-6 letters and/or a punctuation sign(s).

It's not guaranteed that the generated text will contain all `generated_tokens_count` tokens,
but the generated text will never exceed this amount of tokens.

See also: [can_stop_early](#can_stop_early-bool-optional-defaultfalse).

### `max_total_tokens`: int (required)

Combined length of `preamble`, `prompt` and the resulting generated text must fit within this amount of tokens.
The `prompt` will be truncated from the beginning if needed.
The maximum allowed length of `prompt` will be
`max_total_tokens - <length of preamble in tokens> - generated_tokens_count`.

### `temperature`: float (optional)

Changes the appearance probability of all generated tokens.
In a nutshell, it will make more probable tokens even more probable,
and less probable tokens even less probable.

Recommended values: `0.4 - 0.7`.
Higher values will make the generated text nonsensical.
Lower values will remove almost all variety (the text will resemble the model training data),
and text will more likely to start to repeat itself in a loop.
However, YMMV.

By default, the temperature is not applied, which is the same as setting it to `1` (not recommended).

### `top_k`: int (optional)

Only consider this amount of top-probability results.
For example, if `top_k = 2` the algorithm will only choose between two top results for a next token.

By default, this filter is off.
It's not really recommended using this filter alone by itself.
However, it can be used in combination with other filters,
and it's required when using [penalty_alpha](#penalty_alpha-float-optional).
The value of `40` is considered to be adequate.

### `top_p`: float (optional)

Top-p sampling (sometimes called "nucleus sampling")
chooses from the smallest possible set of tokens whose cumulative probability exceeds the given probability.
Higher values will include more tokens, which will make the text more random.

Allowed range: `0 < x < 1`.
Recommended values: `0.6 - 0.95`.

By default, this filter is off.

### `tfs`: float (optional)

Tail-free sampling will try to cut off the tokens that have much lower probability (the "tail")
than tokens with higher probability.
Higher values will include more of the "tail", which will make the text more random.
Lower values will cut off more tokens, which will make text less random.

Allowed range: `0 < x < 1`.
Recommended values: `0.9 - 0.95`.

By default, this filter is off.

### `typical`: float (optional)

Typical sampling will pick samples based not on appearance probability,
but on expected amount of information produced.
It can make the resulting text more interesting by excluding the most likely words
if there's a big pool of words to choose from for the next token.

Lower values will make the generated text to maintain the same "rhythm" as the prompt.
Higher values will include more choices for the next token,
and the text will be more random.

Allowed range: `0 < x < 1`.
Recommended values: `0.2` is assumed to be a good choice for a generic story-writing;
however, higher values may also produce good results, depending on the expected result.

By default, this filter is off.

### `top_a`: float (optional)

Top-a sampling will filter out any tokens that are less likely to appear than the top-rated token.

Lower values will keep more tokens.
The value of `1` will only allow the top-rated token to be chosen.

Allowed range: `0 < x <= 1`.
Recommended values: `0.8 - 0.95`.

By default, this filter is off.

### `penalty_alpha`: float (optional)

If not zero, enables [contrastive search](https://huggingface.co/blog/introducing-csearch).

This works somewhat similar to the repetition penalty.
Bigger values will make the generated text more random and less repetitive,
but after some threshold it becomes more and more incoherent.
For more details read the link above.

This parameter will switch the inference to a completely different mode,
so there are some caveats:

* Since the contrastive search only applies to `top_k` results,
  `top_k` parameter is mandatory, and should be greater than 1.
* All other warpers (`top_p`, `tfs`, `typical`, `top_a`, `temperature` and the repetition penalty)
  may work differently, so you may need to adjust or disable them.
* The memory consumption depends linearly on `top_k`.
* The output is deterministic, i.e. you will get the same output each time
  for the same input parameters and prompt/preamble.

Allowed range: `0 < x <= 1`.
Recommended values: `penalty_alpha=0.6`, `top_k=4` (taken from the example in the link above).

By default, this parameter is zero and therefore the contrastive search is disabled.

### `warpers_order`: string[] (optional)

Sets the order in which all of the above filters/sampling methods (called "warpers") are applied.

Can be an array containing the names of filters:
`"repetition_penalty"`, `"temperature"`, `"top_k"`, `"top_p"`, `"typical"`, `"tfs"` or `"top_a"`.
By default, these warpers are applied in this order.
However, the default order may change in future versions,
so it's better not to rely on it.

You can also specify a special value `"*"` that will imply all unspecified warpers.
This value is automatically added at the end of `warpers_order` if not specified explicitly.

**Examples:**

- `["tfs", "*", "temperature"]` - apply the tail-free sampling first,
  then apply everything else except the temperature, then apply the temperature.
- `["typical"]` - apply the typical sampling first, then apply everything else.

### `repetition_penalty`: float (optional)

Change the probability of the tokens that are already included in the input text (`preamble` and/or `prompt`).

Values higher than `1` will decrease the probability that tokens in the input text will appear in the generated text.
Higher values will apply more penalty,
which means that new generated tokens will more likely be different from input tokens.

The value of zero `0` will disable this penalty.
Recommended values: `1 - 3`.

By default, the repetition penalty is off (i.e. the same as specifying `1`).

### `repetition_penalty_include_preamble`: bool (optional, default=false)

If `false` then repetition penalty will only be applied to the tokens in the `prompt`.

### `repetition_penalty_range`: int (optional, default=0)

Limit repetition penalty text to this number of tokens.

The initial penalty range will be located at the end of the input text.
However, the range may change as new tokens are generated,
depending on `repetition_penalty_include_generated`.

The value of `0` (default) will expand the initial penalty range to all input tokens
(`preamble` + `prompt` if `repetition_penalty_include_preamble` is set, or just `prompt` otherwise).

### `repetition_penalty_slope`: float (optional)

Apply more penalty to the tokens that are closer to the end of the chosen penalty text.
The value of `1` will lower the penalty linearly from end to the start of the penalty range.
Higher values will increase the difference in penalty between the first half of the penalty range and its second half.

By default, the slope is not applied.
The value of `0` will also disable this feature.

You can extend the slope curve beyond the beginning of the input text
by specifying `repetition_penalty_range` that is bigger than the current input tokens count
(however, `repetition_penalty_truncate_to_input` must be set to `false` in this case).

### `repetition_penalty_include_generated`: enum("ignore","allow","expand","slide") (optional,default="slide")

How to include newly generated tokens into the repetition penalty range.

* `"ignore"` - do not include newly generated text into the repetition penalty range

* `"expand"` - include all generated text
  (the range window will be including all newly generated tokens, the range start will remain at place)

* `"slide"` - include all new generated text but keep the original range length
  (the range window will be "sliding" forward as the new tokens are generated)

### `repetition_penalty_truncate_to_input`: bool (optional, default=false)

If `true` then the repetition penalty range will never exceed the length of the input text.

### `repetition_penalty_prompt`: string (optional)

If specified, this text will be treated as if it was passed to the `prompt`
but only for the purpose of calculating the repetition penalty.
The text can be any non-zero length.

This allows you to fully customize what words will receive the penalty.
For example, you may want to pass a filtered `prompt` here
(e.g. removing all punctuation and newlines - otherwise they will receive the penalty too),
or you may want to pass some extra text that does not fit inside `max_total_tokens`.

### `required_server_version`: string (optional)

This will validate the server version before executing the request.
This parameter must hold a valid version specifier, as described in the
[PEP 440 – Version Identification and Dependency Specification](https://peps.python.org/pep-0440/#version-specifiers).
If the server version does not match the specifier,
then the request will fail immediately.

**Examples of possible version specifiers:**

* `==1.2` - will only match v1.2
* `~=1.2` - will match v1.2 and all newer v1.*
* `>=1, <4, !=1.2, !=2.*` - will match all v1.* except v1.2, and all v3.*

### `stop_strings`: string[] (optional)

Stop the inference when any of these strings are encountered in generated text.
After that the sequences will be truncated up to that string (excluding the string itself).

For example, if the currently generated text is `I saw a ` and the `stop_strings` is `[","]`,
then if the next token is `cat, but` the final sequence will be `I saw a cat`.
However, the API also will return the rest of the string (`, but`) separately.

If multiple sequences are requested, the inference will stop when all sequences have any of these strings
(or the requested number of tokens are generated).
All sequences will be truncated accordingly as well.

You may want to use different stop strings for different scenarios.
For example, if you want to generate no more than one sentence then you may set `stop_strings = [".", "!", "?"]`
(see [stop_strings_type](#stop_strings_type-enumstringregex-optionaldefaultstring) for more complex example).
If you want to stop at the end of the line (e.g. generating a reply in a chat) then use `["\n"]`.
If you use AI as a story-writer for an interactive adventure
where your actions start with a new paragraph and a `>` prompt,
you may want to use something like this: `["\n>", "\n\n"]`.

### `stop_strings_type`: enum("string","regex") (optional,default="string")

* **string** - treats `stop_strings` as simple strings and searches for occurrences of these strings.

* **regex** - treats `stop_strings` as regular expressions.
  The syntax from the [regex package](https://pypi.org/project/regex/)
  will be used (no flags are passed to `regex.compile`).
  An example of stopping at the end of sentence:
  `(?i)(?<!\W(dr|jr|mr|mrs|ms|prof|sr|st))[\.\!\?](?=\s)`.
  When passing regular expressions via JSON you need to escape the ``\`` symbol,
  for example to use `[\.\!\?]` as the regular expression,
  you need to pass this JSON string: `[\\.\\!\\?]`.
  Note that using regular expressions may result in slower inference.

### `stop_strings_required_matches_count`: int (optional, default=1)

For a `stop_string` to stop the inference, it needs to match this many times in a row.
Currently, it only really matters for regular expressions with look-aheads.
For example, you want to stop the inference when a string `Brad` is found,
but not when it is followed by `bury`.
The `stop_string` will be `Brad(?!bury)` with `stop_strings_type=regex`.
Let's assume that the AI generated the string `we saw that Brad` at some point.
It matches the regular expression, because it is not followed by `bury`,
but you can't possibly know that the next token won't be `bury`.
However, if you specify `stop_strings_required_matches_count=2`
the AI will be required to generate one more token.
If it's not `bury`, then it means that there were `2` matches in a row
and the inference needs to be stopped.
But if the next token is `bury` then the inference continues
and the "match counter" resets back to the value of `stop_strings_required_matches_count`.

### `truncate_prompt_until`: string[] (optional)

If the initial `prompt` does not fit into the required number of tokens it needs to be truncated.
The `prompt` will be truncated from the beginning until it fits the desired number of tokens,
and then it will be truncated until one of the strings in `truncate_prompt_until` is encountered.

For example, let's assume the input text is `We had a nice chat. And then she left.`,
but the whole prompt does not fit inside the desired number of tokens
(`max_total_tokens - <length of preamble in tokens> - generated_tokens_count`).
Then if `truncate_prompt_until = ["."]` then it will be truncated like this:
<code>&nbsp;And then she left.</code> (notice the space at the beginning).
If `truncate_prompt_until = [" "]` it may be truncated like this:
`nice chat. And then she left.`.
The truncation will stop as early as possible.

**NOTE:** If `truncate_prompt_until` is not set
the truncation will be token-by-token,
which is probably not what you want,
e.g. it may leave half of the word in the beginning, like this:
`ce chat. And then she left.`

If the `prompt` does not need to be truncated then `truncate_prompt_until` is ignored.

### `words_whitelist`: string[] (optional)

If not `null`, the output sequence will only contain these specified words.

**Example:** `["feed", "seed"]`

**Notes:**

* "Words" don't have to be actual words.
  You can put an entire sentence as a string.
  For example: `["cookies and cream !"]`.
  However, see the next note.

* Currently, this whitelist will also allow any permutations of tokens that define the specified words.
  For example, if `words_whitelist = ["privacy", "intimidate"]`
  and the tokens that form these words are `"priv", "acy", "intim", "id", "ate"`,
  then the server may generate something like `"private intimacy"`.

* Empty list means a whitelist with zero words, i.e. all words are banned.
  To disable the whitelist pass `null` or don't pass anything.

### `words_blacklist`: string[] (optional)

The output sequence will not contain the specified words or word sequences.

**Example:** `["word", "Many words?"]`

**Notes:**

* Only the whole sequence will be banned.
  For example, if `words_blacklist = ["Hello world"]`,
  the server will still able to generate separate words "Hello" and "world",
  but not the exact phrase "Hello world".
  Because of that, this blacklist is not mutually exclusive with the
  [whitelist](#words_whitelist-string-optional).

* The word sequences are banned using tokens.
  It means that only the exact token sequence is banned.
  But if the word sequence can be represented by a different set of tokens,
  it may still appear in the output.
  For example, if `words_blacklist = ["Mathematical"]`
  then the following sequence of tokens may be banned `"Mathe", "matical"`.
  However, if there are tokens `"Math", "ematic", "al"` then this word can still appear in the output

* As a consequence of the previous gotcha, the blacklist is case-sensitive.
  For example, if you ban the word "go", then the word "GO" may still appear as it considered a different word.

### `no_repeat_ngram_size`: int (optional)

N-gram is a sequence of a consecutive tokens.
"N-gram size" is a length of that sequence.
If `no_repeat_ngram_size = M` then the output sequence will not contain
any N-grams with size of `M` tokens more than once (including the input text).

For example, let's take the input sequence that consists of these tokens:
`With boots and guns` (for simplicity, let's assume that each word is a single token).
Now, if `no_repeat_ngram_size = 2` then the output won't be able to contain
`With boots`, `boots and` or `and guns`.
This parameter also take into account any currently generated text, not just the original input.
So if the next two generated tokens are `shouting loud`,
then this N-gram will also not appear in any further output (as well as `guns shouting`, etc).

If `no_repeat_ngram_size = 1` then it means that the output text will not contain any tokens from the input
and will not contain any duplicate tokens.

By default, the value of this parameter is zero, which means that no N-gram restrictions apply.

### `can_stop_early`: bool (optional, default=false)

If `true` then the model may generate less than [generated_tokens_count](#generated_tokens_count-int-required).
For example, when the model decides that the story it generated is ended.
If `false` the model will be instructed to not stop generating text
until all `generated_tokens_count` tokens are generated
or until any of the [stop strings](#stop_strings-string-optional) are encountered.


## API response

The API response will be in JSON and can be one of two things: error or data.

The error will look like this:

```json
{
  "error": "Error description"
}
```

If the request is completed successfully then the result will be an object with the fields described below.

**Example:**
```json
{
  "original_input_tokens_count": 60,
  "used_input_tokens_count": 40,
  "preamble_tokens_count": 5,
  "preamble": "The Butcher of Ark\n\n",
  "used_prompt": "almost seemed as if that day Mother Nature, as a response to the festivities of the preceding night, had decided to recover herself with a dreamy, nondescript day.",
  "original_prompt_tokens_count": 55,
  "used_prompt_tokens_count": 35,
  "repetition_penalty_text_at_start": " had decided to recover herself with a dreamy, nondescript day.",
  "used_repetition_penalty_tokens_count_at_start": 20,
  "used_repetition_penalty_range_at_start": 15,
  "used_repetition_penalty_tokens_count_at_end": 20,
  "used_repetition_penalty_range_at_end": 15,
  "generated_tokens_count": 29,
  "output_tokens_count": 69,
  "sequences": [
    {
      "generated_text": " It was the kind of morning that could have been described as perfect, with a soft breeze blowing through the trees and flowers that lined her street",
      "output_text": "The Butcher of Ark\n\nIt was a dull, cold and wet morning that would change my life forever. Yes... Somehow it almost seemed as if that day Mother Nature, as a response to the festivities of the preceding night, had decided to recover herself with a dreamy, nondescript day. It was the kind of morning that could have been described as perfect, with a soft breeze blowing through the trees and flowers that lined her street",
      "stop_string": ".",
      "stop_string_match": ".",
      "trimmed_tail": "",
      "repetition_penalty_text_at_end": ", with a soft breeze blowing through the trees and flowers that lined her street"
    },
    {
      "generated_text": " The sky was cloudless and the air had that crisp, clean feeling of being just out side in a forest",
      "output_text": "The Butcher of Ark\n\nIt was a dull, cold and wet morning that would change my life forever. Yes... Somehow it almost seemed as if that day Mother Nature, as a response to the festivities of the preceding night, had decided to recover herself with a dreamy, nondescript day. The sky was cloudless and the air had that crisp, clean feeling of being just out side in a forest",
      "stop_string": ".",
      "stop_string_match": ".",
      "trimmed_tail": "\nThe sun was warm on",
      "repetition_penalty_text_at_end": " feeling of being just out side in a forest.\nThe sun was warm"
    }
  ],
  "gpus": [
    {
      "name": "NVIDIA GeForce GTX 1060 6GB",
      "memory_total": 6370361344,
      "memory_reserved_start": 2162163712,
      "memory_allocated_start": 2151981568,
      "memory_free_start": 1599209472,
      "memory_reserved_end": 2376073216,
      "memory_allocated_end": 2151986688,
      "memory_free_end": 1385299968,
      "memory_reserved_min": 2162163712,
      "memory_allocated_min": 2151981568,
      "memory_free_min": 1385299968,
      "memory_reserved_max": 2376073216,
      "memory_allocated_max": 2349271040,
      "memory_free_max": 1599209472
    }
  ]
}
```

### `original_input_tokens_count`: int

Number of tokens that was in the `preamble` + `prompt`.

### `used_input_tokens_count`: int

Number of tokens that were actually passed to the language model.

### `preamble`: string

The preamble that was used by the language model.
Due to some preprocessing it may differ from what was passed in the request.

### `preamble_tokens_count`: int

Number of tokens in the `preamble`.

### `used_prompt`: string

The part of the `prompt` that was passed to the language model.

### `original_prompt_tokens_count`: int

The number of tokens in the original `prompt`.

### `used_prompt_tokens_count`: int

The number of tokens in `used_prompt`.

### `repetition_penalty_text_at_start`: string

The text that was used for repetition penalty at the start of the inference.

### `used_repetition_penalty_tokens_count_at_start`: int

The number of tokens in `repetition_penalty_text_at_start`.

### `used_repetition_penalty_range_at_start`: int

The actual repetition penalty range that was used at the start of the inference.

Note that it may be larger than `used_repetition_penalty_tokens_count_at_start`,
i.e. the requested range is bigger than the current repetition penalty text.
The tokens in `used_repetition_penalty_tokens_count_at_start` are used to get the tokens to be penalized,
while the penalty range is used for calculating the penalty curve using `repetition_penalty_slope`.

### `used_repetition_penalty_tokens_count_at_end`: int

The number of tokens in `sequences[].repetition_penalty_text_at_end`.

### `used_repetition_penalty_range_at_end`: int

The actual repetition penalty range that was used at the end of the inference.

Also, see the note in `used_repetition_penalty_range_at_start`.

### `generated_tokens_count`: int

The number of tokens that were generated in each sequence.
This will also include all discarded/trimmed tokens
(`sequences[].stop_string` and `sequences[].trimmed_tail`).

### `output_tokens_count`: int

The total number of tokens for the whole output of the language model per sequence.
This includes all input tokens + all generated tokens within one sequence
(all sequences will have the same amount of tokens).

### `sequences`: array

An array of objects with the information about the generated sequences.

### `sequences[].generated_text`: string

The text that was generated by the model.
If the inference was stopped by any of the `stop_strings`
the `generated_text` won't include this stop string and the rest of text
(these can be found in other fields, see below).

### `sequences[].output_text`: string

All the text that was fetched as a result of the inference.
It contains everything that the language model returned preamble, prompt and generated text
(after some preprocessing from Neodim Server, just like all other output).

### `sequences[].stop_string`: string

Encountered stop string (from `stop_strings`) if any.

### `sequences[].stop_string_match`: string

The text that matched the `stop_string`.
If `stop_strings_type=string` then it will be the same as the `stop_string`.
For `stop_strings_type=regex` it will be the text that matched the regular expression.

### `sequences[].trimmed_tail`: string

The generated text after a stop string if any.

Note that when multiple sequences are requested the inference for each sequence
will only stop when all sequences either meet a stop string or `generated_tokens_count` are generated.
Because of that, the `trimmed_tail` may contain a lot of discarded text for some sequences that were stopped earlier,
and it's not an error.
The whole `generated_text + stop_string + trimmed_tail` combo will give you all generated text,
so you can decide what to do with it
(e.g. you may store it for later and use it for the next inference to save computing time).

### `sequences[].repetition_penalty_text_at_end`: string

The text that was used for repetition penalty at the end of the inference for this particular sequence.

### `gpus`: array

An array of objects with the information about system's GPUs.
All CUDA-enabled GPUs will be shown even if they were not used.

### `gpus[].name`: string

The name of the GPU.

### `gpus[].memory_total`: int

Total memory of the GPU.

### `gpus[].memory_reserved_start`: int
### `gpus[].memory_allocated_start`: int
### `gpus[].memory_free_start`: int

Reserved/allocated/free VRAM at the start of the inference, in bytes.

### `gpus[].memory_reserved_end`: int
### `gpus[].memory_allocated_end`: int
### `gpus[].memory_free_end`: int

Reserved/allocated/free VRAM at the end of the inference, in bytes.

### `gpus[].memory_reserved_min`: int
### `gpus[].memory_allocated_min`: int
### `gpus[].memory_free_min`: int

Minimum values of reserved/allocated/free VRAM during the inference, in bytes.

### `gpus[].memory_reserved_max`: int
### `gpus[].memory_allocated_max`: int
### `gpus[].memory_free_max`: int

Maximum values of reserved/allocated/free VRAM during the inference, in bytes.


## Squeezing the VRAM

Decent language models take up a lot of precious VRAM.
There are a couple of ways to optimize VRAM usage.


### Running in console mode

On Linux you can run Neodim Server in `runlevel 3` (multi-user with networking, but without a display manager).
It will allow you to use all VRAM that your GPU has, without spending it on your DE/WM.

You can do it the following way:

1. Log out of your account (but not shutdown the computer) and get to the login screen.
2. Switch to another console with `CTRL+ALT+F2` (try using any other `F-keys` if it does not work).
3. Login to the console.
4. Run `sudo init 3`.
5. You may be redirected back to the original console. Just press `CTRL+ALT+F2` again.
6. Run Neodim Server as usual (`.../neodim-server/start.sh ...`).


### Distributing model layers on CPU

By default, Neodim Server will distribute only the first model layer on your GPU.
The rest will go to the CPU.
This will prioritize VRAM savings over performance.
In terms of speed, there is a huge difference between no layers on GPU and 1 layer on GPU,
but there's very little difference between 1 layer on GPU and 2 layers on GPU.
Play around with `--layers` parameter to find a suitable combination.


### Lowering the precision

By default, the model is loaded in 16-bit floating precision.
However, it's possible to try to load the model in 8-bit or 4-bit precision.
This may lower the quality of the generated text, but the model will consume only half/quarter as much VRAM.
There are limitations when loading the model in 8-bit or 4-bit precision.
See more details in
[precision](#precision-originalfloat32float16float4int8gptq2gptq4gptq8-optional-defaultfloat16)
parameter description.


### Using GPTQ models

Using `4-bit` [GPTQ](#gptq) precision will reduce VRAM consumption by a factor of 3 compared to regular `float16` models.


## Third-party libraries

There are some embedded third-party libraries.
Read more [here](src/third_party/THIRD_PARTY.md).

Other direct dependencies:
* [transformers](https://pypi.org/project/transformers/)
* [torch](https://pypi.org/project/torch/)
* [bitsandbytes](https://pypi.org/project/bitsandbytes/)
* [jsons](https://pypi.org/project/jsons/)
* [regex](https://pypi.org/project/regex/)
* [packaging](https://pypi.org/project/packaging/)
* [auto-gptq](https://pypi.org/project/auto-gptq/)


## License

[AGPL v3](LICENSE)


## Client applications

Here's a list of applications that use Neodim Server:

* [Neodim Chat](https://github.com/alkatrazstudio/neodim-chat) -
  Android application for chatting with a bot
