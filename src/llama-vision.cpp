#include "llama-vision.h"

#include <algorithm>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

namespace {

void SilentLog(ggml_log_level, const char*, void*) {}

void InitBackendsOnce() {
  static std::once_flag once;
  std::call_once(once, []() {
    ggml_backend_load_all();
    llama_backend_init();
  });
}

int ResolveThreadCount(int requested) {
  if (requested > 0) return requested;
  const unsigned int hardware = std::thread::hardware_concurrency();
  return hardware > 0 ? static_cast<int>(hardware) : 4;
}

size_t CountOccurrences(const std::string& text, const std::string& needle) {
  if (needle.empty()) return 0;
  size_t count = 0;
  for (size_t pos = text.find(needle); pos != std::string::npos;
       pos = text.find(needle, pos + needle.size())) {
    ++count;
  }
  return count;
}

struct LlamaModelDeleter {
  void operator()(llama_model* model) const { llama_model_free(model); }
};
struct LlamaContextDeleter {
  void operator()(llama_context* context) const { llama_free(context); }
};
struct MtmdContextDeleter {
  void operator()(mtmd_context* context) const { mtmd_free(context); }
};
struct SamplerDeleter {
  void operator()(llama_sampler* sampler) const {
    llama_sampler_free(sampler);
  }
};
struct ChunksDeleter {
  void operator()(mtmd_input_chunks* chunks) const {
    mtmd_input_chunks_free(chunks);
  }
};
struct BitmapDeleter {
  void operator()(mtmd_bitmap* bitmap) const { mtmd_bitmap_free(bitmap); }
};

using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;
using ChunksPtr = std::unique_ptr<mtmd_input_chunks, ChunksDeleter>;
using BitmapPtr = std::unique_ptr<mtmd_bitmap, BitmapDeleter>;

// Ensures the prompt carries exactly one media marker per image. If the
// caller wrote no markers, images are placed before the text, which is the
// layout most vision models expect.
std::string InsertMediaMarkers(
    const std::string& prompt, size_t imageCount, std::string& error
) {
  const std::string marker = mtmd_default_marker();
  const size_t present = CountOccurrences(prompt, marker);
  if (present == imageCount) return prompt;

  if (present == 0) {
    std::string result;
    result.reserve(imageCount * (marker.size() + 1) + prompt.size());
    for (size_t i = 0; i < imageCount; ++i) {
      result += marker;
      result += '\n';
    }
    result += prompt;
    return result;
  }

  error = "prompt contains " + std::to_string(present) +
          " media marker(s) but " + std::to_string(imageCount) +
          " image(s) were provided";
  return {};
}

llama_sampler* BuildSamplerChain(const PromptParams& params) {
  llama_sampler_chain_params chainParams =
      llama_sampler_chain_default_params();
  chainParams.no_perf = true;
  llama_sampler* chain = llama_sampler_chain_init(chainParams);

  if (params.repeatPenalty != 1.0f) {
    llama_sampler_chain_add(
        chain,
        llama_sampler_init_penalties(64, params.repeatPenalty, 0.0f, 0.0f)
    );
  }

  if (params.temperature <= 0.0f) {
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    return chain;
  }

  if (params.topK > 0) {
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(params.topK));
  }
  if (params.topP < 1.0f) {
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(params.topP, 1));
  }
  if (params.minP > 0.0f) {
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(params.minP, 1));
  }
  llama_sampler_chain_add(chain, llama_sampler_init_temp(params.temperature));
  llama_sampler_chain_add(chain, llama_sampler_init_dist(params.seed));
  return chain;
}

std::string TokenToPiece(const llama_vocab* vocabulary, llama_token token) {
  char buffer[256];
  int32_t length = llama_token_to_piece(
      vocabulary, token, buffer, sizeof(buffer), 0, true);
  if (length >= 0) return std::string(buffer, static_cast<size_t>(length));

  std::vector<char> largeBuffer(static_cast<size_t>(-length));
  length = llama_token_to_piece(
      vocabulary, token, largeBuffer.data(), largeBuffer.size(), 0, true);
  if (length < 0) return {};
  return std::string(largeBuffer.data(), static_cast<size_t>(length));
}

}  // namespace

class LlamaVision::Impl {
 public:
  bool Load(const VisionModelParams& params) {
    Unload();
    loadError.clear();

    if (params.modelPath.empty()) {
      loadError = "modelPath must not be empty";
      return false;
    }
    if (params.projectorPath.empty()) {
      loadError =
          "projectorPath (mmproj) is required; this library only supports "
          "multimodal models";
      return false;
    }

    // The log handlers are process-global in llama.cpp; the last Load()
    // wins. Silence both the llama/ggml core and the mtmd projector unless
    // verbose logging was requested.
    llama_log_set(params.verbose ? nullptr : SilentLog, nullptr);
    mtmd_log_set(params.verbose ? nullptr : SilentLog, nullptr);
    InitBackendsOnce();

    const int threads = ResolveThreadCount(params.threadCount);

    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = params.gpuLayerCount;

    model.reset(
        llama_model_load_from_file(params.modelPath.c_str(), modelParams));
    if (!model) {
      loadError = "failed to load model from " + params.modelPath;
      return false;
    }

    llama_context_params contextParams = llama_context_default_params();
    contextParams.n_ctx = static_cast<uint32_t>(params.contextSize);
    contextParams.n_batch = static_cast<uint32_t>(params.batchSize);
    // Models with non-causal attention over image tokens require a whole
    // image chunk to fit in one micro-batch; keeping n_ubatch == n_batch
    // avoids mid-chunk splits (and the GGML_ASSERT they trigger).
    contextParams.n_ubatch = static_cast<uint32_t>(params.batchSize);
    contextParams.n_threads = threads;
    contextParams.n_threads_batch = threads;

    context.reset(llama_init_from_model(model.get(), contextParams));
    if (!context) {
      loadError = "failed to create the llama context";
      Unload();
      return false;
    }

    mtmd_context_params mtmdParams = mtmd_context_params_default();
    mtmdParams.use_gpu = params.projectorOnGpu;
    mtmdParams.print_timings = false;
    mtmdParams.n_threads = threads;

    mtmdContext.reset(mtmd_init_from_file(
        params.projectorPath.c_str(), model.get(), mtmdParams));
    if (!mtmdContext) {
      loadError = "failed to load the multimodal projector from " +
                  params.projectorPath +
                  " (check that the mmproj file matches the model)";
      Unload();
      return false;
    }

    if (!mtmd_support_vision(mtmdContext.get())) {
      loadError = "the provided multimodal projector does not support vision";
      Unload();
      return false;
    }

    loadedParams = params;
    return true;
  }

  bool IsLoaded() const {
    return model != nullptr && context != nullptr && mtmdContext != nullptr;
  }

  void Unload() {
    mtmdContext.reset();  // references the model, so free it first
    context.reset();
    model.reset();
  }

  const std::string& LoadError() const { return loadError; }

  PromptResult Prompt(
      const PromptParams& params, const TokenCallback& onToken
  ) {
    PromptResult result;
    if (!IsLoaded()) {
      result.error = "no model is loaded; call Load() first";
      return result;
    }
    if (params.prompt.empty() && params.imagePaths.empty()) {
      result.error = "nothing to do: prompt and imagePaths are both empty";
      return result;
    }

    // Every call is independent: start from an empty KV cache.
    llama_memory_clear(llama_get_memory(context.get()), true);

    // 1) Decode the images (stb_image: JPEG/PNG/BMP/TGA/GIF; not WebP).
    std::vector<BitmapPtr> bitmaps;
    std::vector<const mtmd_bitmap*> bitmapPtrs;
    bitmaps.reserve(params.imagePaths.size());
    for (const std::string& path : params.imagePaths) {
      BitmapPtr bitmap(
          mtmd_helper_bitmap_init_from_file(mtmdContext.get(), path.c_str()));
      if (!bitmap) {
        result.error = "failed to load or decode image: " + path +
                       " (WebP and JXL are not supported)";
        return result;
      }
      bitmapPtrs.push_back(bitmap.get());
      bitmaps.push_back(std::move(bitmap));
    }

    // 2) Build the single-turn templated prompt.
    std::string userText =
        InsertMediaMarkers(params.prompt, bitmaps.size(), result.error);
    if (!result.error.empty()) return result;

    const std::string& systemPrompt = params.systemPromptOverride.empty()
                                          ? loadedParams.systemPrompt
                                          : params.systemPromptOverride;

    std::string formatted =
        ApplyChatTemplate(systemPrompt, userText, result.error);
    if (formatted.empty()) {
      if (result.error.empty()) result.error = "failed to build the prompt";
      return result;
    }

    // 3) Tokenize text + images into chunks.
    mtmd_input_text textInput{};
    textInput.text = formatted.c_str();
    textInput.add_special = true;   // fresh sequence: tokenizer adds BOS
    textInput.parse_special = true; // template output has special tokens

    ChunksPtr chunks(mtmd_input_chunks_init());
    if (!chunks) {
      result.error = "failed to allocate input chunks";
      return result;
    }

    const int32_t tokenizeResult = mtmd_tokenize(
        mtmdContext.get(),
        chunks.get(),
        &textInput,
        bitmapPtrs.data(),
        bitmapPtrs.size()
    );
    if (tokenizeResult == 1) {
      result.error =
          "media marker count does not match the number of images";
      return result;
    }
    if (tokenizeResult != 0) {
      result.error = "failed to tokenize or preprocess the input "
                     "(mtmd error " + std::to_string(tokenizeResult) + ")";
      return result;
    }

    // 4) Budget check, then evaluate the whole prompt in one pass.
    const auto contextTokens =
        static_cast<int32_t>(llama_n_ctx(context.get()));
    result.promptTokenCount =
        static_cast<int32_t>(mtmd_helper_get_n_tokens(chunks.get()));
    if (result.promptTokenCount >= contextTokens) {
      result.error =
          "prompt needs " + std::to_string(result.promptTokenCount) +
          " tokens but the context holds " + std::to_string(contextTokens) +
          "; raise contextSize or send fewer/smaller images";
      return result;
    }

    llama_pos pastTokenCount = 0;
    const int32_t evalResult = mtmd_helper_eval_chunks(
        mtmdContext.get(),
        context.get(),
        chunks.get(),
        /*n_past=*/0,
        /*seq_id=*/0,
        static_cast<int32_t>(loadedParams.batchSize),
        /*logits_last=*/true,
        &pastTokenCount
    );
    if (evalResult != 0) {
      result.error = "failed to evaluate the prompt (error " +
                     std::to_string(evalResult) + ")";
      return result;
    }

    // 5) Sample the response.
    SamplerPtr sampler(BuildSamplerChain(params));
    const llama_vocab* vocabulary = llama_model_get_vocab(model.get());

    for (int produced = 0;; ++produced) {
      if (produced >= params.maxTokens || pastTokenCount + 1 > contextTokens) {
        result.truncated = true;
        break;
      }

      llama_token token =
          llama_sampler_sample(sampler.get(), context.get(), -1);
      if (llama_vocab_is_eog(vocabulary, token)) break;

      std::string piece = TokenToPiece(vocabulary, token);
      if (!piece.empty()) {
        if (onToken) onToken(piece);
        result.text += piece;
      }
      ++result.generatedTokenCount;

      llama_batch batch = llama_batch_get_one(&token, 1);
      if (llama_decode(context.get(), batch) != 0) {
        result.error = "llama_decode failed mid-generation";
        return result;  // result.text keeps the partial output
      }
      ++pastTokenCount;
    }

    result.ok = true;
    return result;
  }

 private:
  std::string ApplyChatTemplate(
      const std::string& systemPrompt, const std::string& userText,
      std::string& error
  ) const {
    const char* chatTemplate = llama_model_chat_template(model.get(), nullptr);
    if (chatTemplate == nullptr) {
      chatTemplate = "chatml";  // built-in fallback template
    }

    std::vector<llama_chat_message> messages;
    if (!systemPrompt.empty()) {
      messages.push_back({"system", systemPrompt.c_str()});
    }
    messages.push_back({"user", userText.c_str()});

    const int32_t requiredSize = llama_chat_apply_template(
        chatTemplate, messages.data(), messages.size(), true, nullptr, 0);
    if (requiredSize <= 0) {
      error = "chat template application failed";
      return {};
    }

    std::string formatted(static_cast<size_t>(requiredSize), '\0');
    const int32_t writtenSize = llama_chat_apply_template(
        chatTemplate,
        messages.data(),
        messages.size(),
        true,
        formatted.data(),
        requiredSize
    );
    if (writtenSize <= 0) {
      error = "chat template application failed";
      return {};
    }
    formatted.resize(static_cast<size_t>(writtenSize));
    return formatted;
  }

  std::string loadError;
  VisionModelParams loadedParams;
  std::unique_ptr<llama_model, LlamaModelDeleter> model;
  std::unique_ptr<llama_context, LlamaContextDeleter> context;
  std::unique_ptr<mtmd_context, MtmdContextDeleter> mtmdContext;
};

// ---- Public class: pure forwarding ----

LlamaVision::LlamaVision() : pimpl(std::make_unique<Impl>()) {}
LlamaVision::~LlamaVision() = default;
LlamaVision::LlamaVision(LlamaVision&&) noexcept = default;
LlamaVision& LlamaVision::operator=(LlamaVision&&) noexcept = default;

bool LlamaVision::Load(const VisionModelParams& params) {
  return pimpl->Load(params);
}

bool LlamaVision::IsLoaded() const { return pimpl->IsLoaded(); }

void LlamaVision::Unload() { pimpl->Unload(); }

const std::string& LlamaVision::LoadError() const {
  return pimpl->LoadError();
}

PromptResult LlamaVision::Prompt(
    const PromptParams& params, const TokenCallback& onToken
) {
  return pimpl->Prompt(params, onToken);
}

PromptResult LlamaVision::DescribeImage(
    const std::string& imagePath, const std::string& prompt,
    const TokenCallback& onToken
) {
  PromptParams params;
  params.prompt = prompt;
  params.imagePaths.push_back(imagePath);
  return pimpl->Prompt(params, onToken);
}

PromptResult LlamaVision::DescribeVideo(
    const std::string& videoPath, const std::string& prompt,
    const VideoFrameParams& frameParams, const TokenCallback& onToken
) {
  VideoFrameResult frames = ExtractVideoFrames(videoPath, frameParams);
  if (!frames.ok) {
    PromptResult result;
    result.error = frames.error;
    return result;
  }

  PromptParams params;
  params.prompt = prompt;
  params.imagePaths = frames.framePaths;

  PromptResult result = pimpl->Prompt(params, onToken);
  CleanupVideoFrames(frames);
  return result;
}

const char* LlamaVision::MediaMarker() { return mtmd_default_marker(); }
