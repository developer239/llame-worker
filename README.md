# LlamaVision 🦙👁

A small C++ library for **one-off multimodal prompts** against local GGUF
models, built on [llama.cpp](https://github.com/ggerganov/llama.cpp)'s
multimodal (mtmd) interface. Load a vision model once, then hand it an
image path (or several, or sampled video frames) plus a prompt, and stream
back the answer. No conversation state, no server, no OpenCV.

Designed as the local-inference core for CLI tools, MCP servers, and Node
bindings: "one-off" describes the call, not the process - keep the instance
alive and the multi-second model load is paid exactly once.

**Supported systems:** macOS, Linux, Windows.

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

## Usage

### Describe an image

```cpp
#include <iostream>

#include "llama-vision.h"

int main() {
  LlamaVision llama;

  VisionModelParams params;
  params.modelPath = "models/gemma-3-4b-it-f16.gguf";
  params.projectorPath = "models/mmproj-model-f16.gguf";
  if (!llama.Load(params)) {
    std::cerr << llama.LoadError() << std::endl;
    return 1;
  }

  auto result = llama.DescribeImage("/absolute/path/to/screenshot.png");
  if (!result.ok) {
    std::cerr << result.error << std::endl;
    return 1;
  }
  std::cout << result.text << std::endl;
  return 0;
}
```

### Streaming, multiple images, custom prompts

```cpp
GenerateParams request;
request.prompt = "What changed between these two screenshots?";
request.imagePaths = {"/tmp/before.png", "/tmp/after.png"};
request.maxTokens = 300;

auto result = llama.Generate(request, [](const std::string& piece) {
  std::cout << piece << std::flush;
});
```

Text-only prompts work too - leave `imagePaths` empty and the library is a
plain local LLM.

### Summarize a video (frame sampling)

```cpp
#include "video-frames.h"

auto frames = ExtractVideoFrames("/absolute/path/to/clip.mp4");
if (frames.ok) {
  GenerateParams request;
  request.prompt =
      "These images are frames sampled from one video, in order. "
      "Summarize what happens.";
  request.imagePaths = frames.framePaths;

  auto result = llama.Generate(request);
  CleanupVideoFrames(frames);  // after generation - frames are read inside it

  std::cout << result.text << std::endl;
}
```

This gives keyframe-level understanding (screen recordings, clip gist),
not true motion reasoning. Each frame costs a few hundred prompt tokens
once projected; `GenerateResult.promptTokenCount` shows what a request
actually cost, and `VideoFrameParams.maxFrames` plus
`VisionModelParams.contextSize` are the knobs that trade coverage for
context.

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

### Media markers

By default images are placed before your text (what most vision models
expect). To position them yourself, write `LlamaVision::MediaMarker()`
(literally `<__media__>`) into the prompt - one marker per image, in order.

## Design notes

Every `Generate` starts from a cleared KV cache: no history, no cross-call
state, no cache-position bookkeeping. This is deliberate - multi-turn chat
can be layered on top later without touching the engine.

One call at a time per instance: llama.cpp's multimodal evaluation is not
thread-safe. Run generation on a worker thread if the caller must stay
responsive (this is exactly what a Node binding or MCP server should do),
but never call `Generate` concurrently on the same instance.

## Troubleshooting

"failed to load the multimodal projector" usually means the `mmproj` does
not match the model (embedding-size mismatch). "failed to load or decode
image" on a file that exists is usually WebP or JXL. A prompt-too-large
error reports the exact token count - raise `contextSize` or send fewer or
smaller images. A crash mentioning `n_ubatch >= n_tokens` means
`batchSize` is smaller than one image's token count on a model with
non-causal image attention - raise it.

## License

See the LICENSE file for details.
