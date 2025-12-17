#include "llama-chat.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

class LlamaChat::Impl {
 public:
  Impl() {
    // Load all available backends (CPU, CUDA, Metal, etc.)
    ggml_backend_load_all();
  }

  ~Impl() {
    if (mtmdCtx) {
      mtmd_free(mtmdCtx);
      mtmdCtx = nullptr;
    }
  }

  bool InitializeModel(
      const std::string& model_path, const ModelParams& params
  ) {
    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = params.nGpuLayers;
    modelParams.vocab_only = params.vocabularyOnly;
    modelParams.use_mmap = params.useMemoryMapping;
    modelParams.use_mlock = params.useModelLock;

    model.reset(llama_model_load_from_file(model_path.c_str(), modelParams));
    if (!model) {
      std::cerr << "Failed to load model from " << model_path << std::endl;
      return false;
    }

    // Get vocabulary from model
    vocab = llama_model_get_vocab(model.get());
    if (!vocab) {
      std::cerr << "Failed to get vocabulary from model" << std::endl;
      return false;
    }

    // Store params for later use
    storedModelParams = params;

    return true;
  }

  bool InitializeContext(const ContextParams& params) {
    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = params.nContext;
    ctxParams.n_threads = params.nThreads;
    ctxParams.n_batch = params.nBatch;

    ctx.reset(llama_init_from_model(model.get(), ctxParams));
    if (!ctx) {
      std::cerr << "Failed to create the llama_context" << std::endl;
      return false;
    }

    // Store context params
    storedContextParams = params;

    // Initialize the sampler chain
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;  // Disable performance metrics for cleaner output
    sampler.reset(llama_sampler_chain_init(sparams));
    if (!sampler) {
      std::cerr << "Failed to create sampler chain" << std::endl;
      return false;
    }

    // Try to find the EOT token for Llama 3 style models
    auto eot_tokens = Encode("<|eot_id|>", false, true);
    if (eot_tokens.size() == 1) {
      eotToken = eot_tokens[0].tokenId;
    } else {
      // Fallback: use EOS token
      eotToken = llama_vocab_eos(vocab);
    }

    // Initialize multimodal context if mmproj path was provided
    if (!storedModelParams.multiModalPath.empty()) {
      InitializeMultiModal();
    }

    return true;
  }

  bool InitializeMultiModal() {
    if (storedModelParams.multiModalPath.empty()) {
      return false;
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = storedModelParams.offloadMultiModalToGPU;
    mparams.print_timings = false;
    mparams.n_threads = storedContextParams.nThreads;

    mtmdCtx = mtmd_init_from_file(
        storedModelParams.multiModalPath.c_str(),
        model.get(),
        mparams
    );

    if (!mtmdCtx) {
      std::cerr << "Failed to load multimodal projector from "
                << storedModelParams.multiModalPath << std::endl;
      return false;
    }

    hasVisionSupport = mtmd_support_vision(mtmdCtx);
    return true;
  }

  [[nodiscard]] std::vector<LlamaToken> Encode(
      const std::string& text, bool addBos, bool parseSpecial = false
  ) const {
    // First call with NULL to get the number of tokens needed
    int nTokens = llama_tokenize(
        vocab,
        text.c_str(),
        text.length(),
        nullptr,
        0,
        addBos,
        parseSpecial
    );

    // nTokens is negative, representing the required buffer size
    if (nTokens == 0) {
      return {};
    }

    int requiredSize = nTokens < 0 ? -nTokens : nTokens;
    std::vector<llama_token> llamaTokens(requiredSize);

    nTokens = llama_tokenize(
        vocab,
        text.c_str(),
        text.length(),
        llamaTokens.data(),
        llamaTokens.size(),
        addBos,
        parseSpecial
    );

    if (nTokens < 0) {
      std::cerr << "Tokenization failed with error code: " << nTokens
                << std::endl;
      return {};
    }

    llamaTokens.resize(nTokens);

    std::vector<LlamaToken> tokens;
    tokens.reserve(nTokens);
    for (auto token : llamaTokens) {
      tokens.emplace_back(token);
    }

    return tokens;
  }

  void Prompt(
      const std::string& userMessage,
      const std::function<void(const std::string&)>& callback
  ) {
    AddUserMessage(userMessage);
    RunQueryStream([this, &callback](const std::string& piece) {
      callback(piece);
    });
  }

  void PromptWithImage(
      const std::string& userMessage, const ImageInput& image,
      const std::function<void(const std::string&)>& callback
  ) {
    if (!mtmdCtx || !hasVisionSupport) {
      std::cerr << "Warning: Multimodal support not available. "
                << "Make sure to provide multiModalPath in ModelParams."
                << std::endl;
      // Fall back to text-only prompt
      Prompt(userMessage, callback);
      return;
    }

    // Load the image
    mtmd_bitmap* bitmap = mtmd_helper_bitmap_init_from_file(
        mtmdCtx, image.path.c_str()
    );
    if (!bitmap) {
      std::cerr << "Failed to load image from " << image.path << std::endl;
      Prompt(userMessage, callback);
      return;
    }

    // Build prompt with image marker
    std::string messageWithMarker = userMessage;
    const char* marker = mtmd_default_marker();

    // If the message doesn't contain an image marker, prepend one
    if (messageWithMarker.find(marker) == std::string::npos) {
      messageWithMarker = std::string(marker) + "\n" + userMessage;
    }

    AddUserMessage(messageWithMarker);

    // Run multimodal query
    RunMultiModalQueryStream(bitmap, callback);

    // Clean up bitmap
    mtmd_bitmap_free(bitmap);
  }

  void SetSystemPrompt(const std::string& systemPrompt) {
    conversationHistory.clear();
    conversationHistory.push_back({"system", systemPrompt});
  }

  void ResetConversation() {
    conversationHistory.clear();
    nPast = 0;
    llama_memory_clear(llama_get_memory(ctx.get()), true);
  }

  [[nodiscard]] bool HasVisionSupport() const {
    return hasVisionSupport;
  }

 private:
  struct LlamaModelDeleter {
    void operator()(llama_model* m) const { llama_model_free(m); }
  };

  struct LlamaContextDeleter {
    void operator()(llama_context* c) const { llama_free(c); }
  };

  struct LlamaSamplerDeleter {
    void operator()(llama_sampler* s) const { llama_sampler_free(s); }
  };

  struct Message {
    std::string role;
    std::string content;
  };

  std::vector<Message> conversationHistory;
  std::unique_ptr<llama_model, LlamaModelDeleter> model = nullptr;
  std::unique_ptr<llama_context, LlamaContextDeleter> ctx = nullptr;
  std::unique_ptr<llama_sampler, LlamaSamplerDeleter> sampler = nullptr;
  const llama_vocab* vocab = nullptr;
  llama_token eotToken;

  // Multimodal support
  mtmd_context* mtmdCtx = nullptr;
  bool hasVisionSupport = false;
  ModelParams storedModelParams;
  ContextParams storedContextParams;
  llama_pos nPast = 0;

  void BuildPrompt(std::string& prompt) const {
    std::ostringstream oss;
    oss << "<|begin_of_text|>";

    // Add system prompt first
    for (const auto& msg : conversationHistory) {
      if (msg.role == "system") {
        oss << "<|start_header_id|>" << msg.role << "<|end_header_id|>"
            << msg.content << "<|eot_id|>";
        break;  // Assume there's only one system message
      }
    }

    size_t totalTokens = 0;
    const size_t maxTokens =
        1024;  // Adjust this based on your model's context size
    for (auto it = conversationHistory.begin(); it != conversationHistory.end();
         ++it) {
      if (it->role != "system") {
        std::string messageContent = "<|start_header_id|>" + it->role +
                                     "<|end_header_id|>" + it->content +
                                     "<|eot_id|>";
        auto tokens = Encode(messageContent, false);
        if (totalTokens + tokens.size() > maxTokens) {
          break;
        }
        oss << messageContent;
        totalTokens += tokens.size();
      }
    }

    oss << "<|start_header_id|>assistant<|end_header_id|>";

    prompt = oss.str();
  }

  [[nodiscard]] std::string TokenToString(llama_token token) const {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) {
      // Buffer too small, try with larger buffer
      std::vector<char> largeBuf(-n);
      n = llama_token_to_piece(
          vocab,
          token,
          largeBuf.data(),
          largeBuf.size(),
          0,
          true
      );
      if (n < 0) {
        return "";
      }
      return std::string(largeBuf.data(), n);
    }
    return std::string(buf, n);
  }

  void SetupSamplerChain(const SamplingParams& params) {
    // Clear existing samplers
    sampler.reset();

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    sampler.reset(llama_sampler_chain_init(sparams));

    // Add samplers in order
    // Add penalties sampler for repeat penalty
    if (params.repeatPenalty != 1.0f || params.frequencyPenalty != 0.0f ||
        params.presencePenalty != 0.0f) {
      llama_sampler_chain_add(
          sampler.get(),
          llama_sampler_init_penalties(
              64,  // penalty_last_n
              params.repeatPenalty,     // penalty_repeat (1.0 = disabled)
              params.frequencyPenalty,  // penalty_freq (0.0 = disabled)
              params.presencePenalty    // penalty_present (0.0 = disabled)
          )
      );
    }

    // Add top-k sampler
    if (params.topK > 0) {
      llama_sampler_chain_add(
          sampler.get(),
          llama_sampler_init_top_k(params.topK)
      );
    }

    // Add top-p sampler
    if (params.topP < 1.0f) {
      llama_sampler_chain_add(
          sampler.get(),
          llama_sampler_init_top_p(params.topP, 1)
      );
    }

    // Add temperature sampler
    llama_sampler_chain_add(
        sampler.get(),
        llama_sampler_init_temp(params.temperature)
    );

    // Add distribution sampler (required to actually sample a token)
    llama_sampler_chain_add(
        sampler.get(),
        llama_sampler_init_dist(LLAMA_DEFAULT_SEED)
    );
  }

  void AddUserMessage(const std::string& message) {
    // TODO: make configurable
    const size_t maxHistorySize = 10;

    if (conversationHistory.size() >= maxHistorySize) {
      conversationHistory.erase(conversationHistory.begin());
    }

    conversationHistory.push_back({"user", message});
  }

  void RunQueryStream(const std::function<void(const std::string&)>& callback) {
    std::string prompt;
    BuildPrompt(prompt);

    SamplingParams params;
    SetupSamplerChain(params);

    auto tokens = Encode(prompt, false, true);
    if (tokens.empty()) {
      std::cerr << "Failed to tokenize prompt" << std::endl;
      return;
    }

    // Convert to llama_token vector
    std::vector<llama_token> tokenIds;
    tokenIds.reserve(tokens.size());
    for (const auto& t : tokens) {
      tokenIds.push_back(t.tokenId);
    }

    // Create batch for the prompt
    llama_batch batch = llama_batch_get_one(tokenIds.data(), tokenIds.size());

    // Decode the prompt
    if (llama_decode(ctx.get(), batch) != 0) {
      throw std::runtime_error("llama_decode() failed for prompt");
    }

    size_t nCur = tokenIds.size();
    std::string assistantResponse;

    while (nCur < params.maxTokens + tokenIds.size()) {
      // Sample next token
      llama_token new_token =
          llama_sampler_sample(sampler.get(), ctx.get(), -1);

      // Check for end of generation
      if (llama_vocab_is_eog(vocab, new_token) || new_token == eotToken) {
        break;
      }

      // Convert token to string
      std::string piece = TokenToString(new_token);

      callback(piece);
      assistantResponse += piece;

      // Create batch for the new token
      batch = llama_batch_get_one(&new_token, 1);

      if (llama_decode(ctx.get(), batch) != 0) {
        throw std::runtime_error("Failed to evaluate");
      }

      nCur += 1;
    }

    conversationHistory.push_back({"assistant", assistantResponse});
  }

  void RunMultiModalQueryStream(
      mtmd_bitmap* bitmap,
      const std::function<void(const std::string&)>& callback
  ) {
    std::string prompt;
    BuildPrompt(prompt);

    SamplingParams params;
    SetupSamplerChain(params);

    // Prepare text input for mtmd_tokenize
    mtmd_input_text text;
    text.text = prompt.c_str();
    text.add_special = (nPast == 0);  // Add BOS if at start
    text.parse_special = true;

    // Create input chunks
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    if (!chunks) {
      std::cerr << "Failed to create input chunks" << std::endl;
      return;
    }

    // Tokenize the prompt with image
    const mtmd_bitmap* bitmaps[] = { bitmap };
    int32_t res = mtmd_tokenize(
        mtmdCtx,
        chunks,
        &text,
        bitmaps,
        1  // Number of bitmaps
    );

    if (res != 0) {
      std::cerr << "Failed to tokenize multimodal prompt, error: " << res << std::endl;
      mtmd_input_chunks_free(chunks);
      return;
    }

    // Evaluate the chunks (this handles both text tokens and image embeddings)
    llama_pos new_n_past;
    int eval_result = mtmd_helper_eval_chunks(
        mtmdCtx,
        ctx.get(),
        chunks,
        nPast,
        0,  // seq_id
        storedContextParams.nBatch,
        true,  // logits_last
        &new_n_past
    );

    mtmd_input_chunks_free(chunks);

    if (eval_result != 0) {
      std::cerr << "Failed to evaluate multimodal chunks" << std::endl;
      return;
    }

    nPast = new_n_past;

    // Generate response tokens
    std::string assistantResponse;
    llama_batch batch = llama_batch_init(1, 0, 1);

    for (size_t i = 0; i < params.maxTokens; ++i) {
      // Sample next token
      llama_token new_token = llama_sampler_sample(sampler.get(), ctx.get(), -1);

      // Check for end of generation
      if (llama_vocab_is_eog(vocab, new_token) || new_token == eotToken) {
        break;
      }

      // Convert token to string
      std::string piece = TokenToString(new_token);
      callback(piece);
      assistantResponse += piece;

      // Prepare batch for next token using common helpers
      common_batch_clear(batch);
      common_batch_add(batch, new_token, nPast++, {0}, true);

      if (llama_decode(ctx.get(), batch) != 0) {
        std::cerr << "Failed to decode token" << std::endl;
        break;
      }
    }

    llama_batch_free(batch);
    conversationHistory.push_back({"assistant", assistantResponse});
  }
};

ImageInput ImageInput::FromPath(const std::string& path) {
  ImageInput input;
  input.path = path;
  return input;
}

LlamaChat::LlamaChat() : pimpl(std::make_unique<Impl>()) {}
LlamaChat::~LlamaChat() = default;

bool LlamaChat::InitializeModel(
    const std::string& modelPath, const ModelParams& params
) {
  try {
    return pimpl->InitializeModel(modelPath, params);
  } catch (const std::exception& e) {
    std::cerr << "InitializeModel exception: " << e.what() << std::endl;
    return false;
  }
}

bool LlamaChat::InitializeContext(const ContextParams& params) {
  try {
    return pimpl->InitializeContext(params);
  } catch (const std::exception& e) {
    std::cerr << "InitializeContext exception: " << e.what() << std::endl;
    return false;
  }
}

void LlamaChat::SetSystemPrompt(const std::string& systemPrompt) {
  pimpl->SetSystemPrompt(systemPrompt);
}

void LlamaChat::ResetConversation() { pimpl->ResetConversation(); }

void LlamaChat::Prompt(
    const std::string& userMessage,
    const std::function<void(const std::string&)>& callback
) {
  return pimpl->Prompt(userMessage, callback);
}

void LlamaChat::PromptWithImage(
    const std::string& userMessage, const ImageInput& image,
    const std::function<void(const std::string&)>& callback
) {
  return pimpl->PromptWithImage(userMessage, image, callback);
}

std::vector<LlamaToken> LlamaChat::Encode(
    const std::string& text, bool addBos
) const {
  return pimpl->Encode(text, addBos);
}

bool LlamaChat::HasVisionSupport() const {
  return pimpl->HasVisionSupport();
}