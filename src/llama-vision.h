#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Parameters for loading a model. Loading is the expensive step (seconds to
// tens of seconds); do it once per process and reuse the instance.
struct VisionModelParams {
  std::string modelPath;      // path to the main .gguf model (required)
  std::string projectorPath;  // path to the mmproj .gguf (required)

  int gpuLayerCount = 999;    // 999 = offload every layer that fits; 0 = CPU
  bool projectorOnGpu = true;

  // Context window in tokens. Each image typically costs a few hundred to
  // ~1500 prompt tokens after projection; raise this for multi-image or
  // video-frame prompts.
  int contextSize = 4096;

  // Prompt-processing batch size. Also used as the micro-batch: models with
  // non-causal attention over image tokens (e.g. gemma-4-class) require an
  // entire image chunk to fit in one micro-batch, so keep this comfortably
  // above the per-image token count. Lower it to save memory on small
  // machines if your model's images are small.
  int batchSize = 2048;

  int threadCount = 0;  // 0 = use hardware concurrency

  // Applied to every Generate() unless overridden per call. May be empty.
  std::string systemPrompt =
      "You are a precise visual assistant. Describe exactly what is shown, "
      "concisely and factually.";

  bool verbose = false;  // false = suppress llama.cpp/ggml logging
                         // (note: the log handler is process-global)
};

// Parameters for a single, independent generation.
struct GenerateParams {
  std::string prompt;
  std::vector<std::string> imagePaths;  // absolute paths; may be empty

  int maxTokens = 512;
  float temperature = 0.2f;  // low by default: factual, near-deterministic;
                             // <= 0 switches to greedy sampling
  int topK = 40;             // <= 0 disables
  float topP = 0.95f;        // >= 1 disables
  float minP = 0.05f;        // <= 0 disables
  float repeatPenalty = 1.0f;  // 1.0 disables (over the last 64 tokens)
  uint32_t seed = 0xFFFFFFFF;  // LLAMA_DEFAULT_SEED = random each call

  std::string systemPromptOverride;  // empty = use the load-time prompt
};

struct GenerateResult {
  bool ok = false;
  std::string text;   // the response; on error, whatever was generated
                      // before the failure
  std::string error;  // set when !ok

  int32_t promptTokenCount = 0;     // text + projected image tokens
  int32_t generatedTokenCount = 0;
  bool truncated = false;  // stopped at maxTokens or the context edge
                           // instead of a natural end-of-generation token
};

using TokenCallback = std::function<void(const std::string& piece)>;

// A stateless single-shot engine over llama.cpp's multimodal (mtmd) API.
// Load once, then call Generate() many times; every call starts from an
// empty KV cache and is completely independent of previous calls.
//
// Thread-safety: one call at a time per instance. Run it on a worker
// thread if you need a responsive caller (this is what the Node binding
// will do), but never call Generate() concurrently on the same instance.
// A moved-from instance may only be destroyed or assigned to.
class LlamaVision {
 public:
  LlamaVision();
  ~LlamaVision();

  LlamaVision(const LlamaVision&) = delete;
  LlamaVision& operator=(const LlamaVision&) = delete;
  LlamaVision(LlamaVision&&) noexcept;
  LlamaVision& operator=(LlamaVision&&) noexcept;

  // Loads the model and multimodal projector. Returns false on failure;
  // the reason is available via LoadError(). Calling Load() again replaces
  // the previously loaded model.
  bool Load(const VisionModelParams& params);
  bool IsLoaded() const;
  void Unload();
  const std::string& LoadError() const;

  // One-off generation. Text-only prompts (no image paths) are fine.
  // Supported image formats are what stb_image decodes: JPEG, PNG, BMP,
  // TGA, GIF. Notably NOT WebP or JXL.
  GenerateResult Generate(
      const GenerateParams& params, const TokenCallback& onToken = nullptr
  );

  // Convenience for the most common case: one image in, description out.
  GenerateResult DescribeImage(
      const std::string& imagePath,
      const std::string& prompt = "Describe this image.",
      const TokenCallback& onToken = nullptr
  );

  // The literal marker that stands for "an image goes here" inside a
  // prompt. If the prompt contains no markers, one per image is prepended
  // automatically (images first, then your text - the layout most vision
  // models expect). If it contains markers, their count must equal the
  // number of image paths.
  static const char* MediaMarker();

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};
