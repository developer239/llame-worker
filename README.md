# Llame Worker 🦙🦙

A small C++ library for **one-off multimodal prompts** against local GGUF
models, built on [llama.cpp](https://github.com/ggerganov/llama.cpp)'s
multimodal (mtmd) interface.  Designed as the local-inference core for CLI tools, MCP servers. 

## Requirements

You need a vision-capable GGUF model **and** its matching multimodal
projector (`mmproj-*.gguf`) - for example `gemma-3-4b-it` plus
`mmproj-model-f16.gguf` from the ggml-org repositories on Hugging Face.
For video, `ffmpeg` (ideally with `ffprobe`) must be installed and on
PATH; it is invoked as a subprocess, never linked.

## Installation

```bash
git submodule add https://github.com/developer239/llame-worker externals/llame-worker
git submodule update --init --recursive
```

```cmake
add_subdirectory(externals/llame-worker)
target_link_libraries(<your_target> PRIVATE LlamaVision)
```

## API reference

### `VisionModelParams` - load once

| Field | Default | Description |
|---|---|---|
| `modelPath` | - | Main `.gguf` model. Required. |
| `projectorPath` | - | `mmproj` `.gguf`. Required. |
| `gpuLayerCount` | `999` | Offload everything that fits; `0` = CPU only. |
| `projectorOnGpu` | `true` | Run image encoding on the GPU. |
| `contextSize` | `4096` | Raise for multi-image or video prompts. |
| `batchSize` | `2048` | Also the micro-batch. Must exceed the per-image token count for models with non-causal image attention. |
| `threadCount` | `0` | `0` = hardware concurrency. |
| `systemPrompt` | factual default | Applied to every call unless overridden. |
| `verbose` | `false` | `true` re-enables llama.cpp logging (process-global). |

### `GenerateParams` - per call

| Field | Default | Description |
|---|---|---|
| `prompt` | - | Plain text; may contain media markers. |
| `imagePaths` | `{}` | Absolute paths. JPEG/PNG/BMP/TGA/GIF (stb_image); **WebP and JXL are not supported**. |
| `maxTokens` | `512` | Generation cap. |
| `temperature` | `0.2` | `<= 0` switches to greedy sampling. |
| `topK` / `topP` / `minP` | `40` / `0.95` / `0.05` | Standard nucleus sampling stack. |
| `repeatPenalty` | `1.0` | `1.0` = disabled (window: last 64 tokens). |
| `seed` | random | Fix for reproducible output. |
| `systemPromptOverride` | `""` | Empty = use the load-time system prompt. |

`Generate` returns `GenerateResult { ok, text, error, promptTokenCount,
generatedTokenCount, truncated }`. `truncated` means generation stopped at
`maxTokens` or the context edge rather than a natural stop token.

## License

See the LICENSE file for details.
