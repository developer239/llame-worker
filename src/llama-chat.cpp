#include "llama-chat.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "ResponseStreamer.h"
#include "common.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

namespace {
bool verboseLoggingEnabled = false;

void LogCallback(ggml_log_level level, const char* text, void* _userData) {
  if (verboseLoggingEnabled) {
    switch (level) {
      case GGML_LOG_LEVEL_ERROR:
        std::cerr << text;
        break;
      case GGML_LOG_LEVEL_WARN:
      case GGML_LOG_LEVEL_INFO:
      case GGML_LOG_LEVEL_DEBUG:
      default:
        std::cout << text;
        break;
    }
  }
}
}  // namespace

struct LlamaModelDeleter {
  void operator()(llama_model* model) const {
    if (model) {
      llama_model_free(model);
    }
  }
};

struct LlamaContextDeleter {
  void operator()(llama_context* context) const {
    if (context) {
      llama_free(context);
    }
  }
};

struct MtmdContextDeleter {
  void operator()(mtmd_context* context) const {
    if (context) {
      mtmd_free(context);
    }
  }
};

class LlamaChat::Impl {
 public:
  Impl() {
    llama_log_set(LogCallback, nullptr);
    ggml_backend_load_all();
  }

  ~Impl() noexcept = default;

  bool InitializeModel(
      const std::string& modelPath, const ModelParams& params
  ) {
    verboseLoggingEnabled = params.verboseLogging;
    llama_log_set(LogCallback, nullptr);

    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = params.gpuLayerCount;
    modelParams.vocab_only = params.vocabularyOnly;
    modelParams.use_mmap = params.useMemoryMapping;
    modelParams.use_mlock = params.useModelLock;

    model.reset(llama_model_load_from_file(modelPath.c_str(), modelParams));
    if (!model) {
      std::cerr << "Failed to load model from " << modelPath << std::endl;
      return false;
    }

    storedModelParams = params;
    return true;
  }

  bool InitializeContext(const ContextParams& params) {
    if (storedModelParams.multiModalProjectorPath.empty()) {
      throw std::runtime_error(
          "multiModalProjectorPath is required. "
          "This library only supports multimodal models."
      );
    }

    llama_context_params contextParams = llama_context_default_params();
    contextParams.n_ctx = params.contextSize;
    contextParams.n_threads = params.threadCount;
    contextParams.n_batch = params.batchSize;

    context.reset(llama_init_from_model(model.get(), contextParams));
    if (!context) {
      std::cerr << "Failed to create the llama_context" << std::endl;
      return false;
    }

    storedContextParams = params;

    const llama_vocab* vocabulary = llama_model_get_vocab(model.get());
    endOfTurnToken = llama_vocab_eot(vocabulary);
    if (endOfTurnToken == LLAMA_TOKEN_NULL) {
      endOfTurnToken = llama_vocab_eos(vocabulary);
    }

    InitializeMultiModal();

    responseStreamer = std::make_unique<ResponseStreamer>(
        context.get(),
        model.get(),
        endOfTurnToken
    );

    return true;
  }

  void Prompt(
      const std::string& userMessage,
      const std::function<void(const std::string&)>& callback,
      const std::optional<ImageInput>& image
  ) {
    std::string messageContent = userMessage;

    mtmd_bitmap* bitmap = nullptr;
    if (image.has_value()) {
      bitmap = mtmd_helper_bitmap_init_from_file(
          multiModalContext.get(),
          image->path.c_str()
      );
      if (!bitmap) {
        throw std::runtime_error("Failed to load image from " + image->path);
      }

      const char* marker = mtmd_default_marker();
      if (messageContent.find(marker) == std::string::npos) {
        messageContent = std::string(marker) + "\n" + userMessage;
      }
    }

    AddUserMessage(messageContent);
    RunQuery(bitmap, callback);

    if (bitmap) {
      mtmd_bitmap_free(bitmap);
    }
  }

  void SetSystemPrompt(const std::string& systemPrompt) {
    conversationHistory.clear();
    conversationHistory.push_back({"system", systemPrompt});
  }

  void ResetConversation() {
    conversationHistory.clear();
    pastTokenCount = 0;
    llama_memory_clear(llama_get_memory(context.get()), true);
  }

  std::vector<LlamaToken> Encode(
      const std::string& text, bool addBos, bool parseSpecial = false
  ) const {
    const llama_vocab* vocabulary = llama_model_get_vocab(model.get());

    int tokenCount = llama_tokenize(
        vocabulary,
        text.c_str(),
        text.length(),
        nullptr,
        0,
        addBos,
        parseSpecial
    );

    if (tokenCount == 0) {
      return {};
    }

    int requiredSize = tokenCount < 0 ? -tokenCount : tokenCount;
    std::vector<llama_token> llamaTokens(requiredSize);

    tokenCount = llama_tokenize(
        vocabulary,
        text.c_str(),
        text.length(),
        llamaTokens.data(),
        llamaTokens.size(),
        addBos,
        parseSpecial
    );

    if (tokenCount < 0) {
      std::cerr << "Tokenization failed with error code: " << tokenCount
                << std::endl;
      return {};
    }

    llamaTokens.resize(tokenCount);

    std::vector<LlamaToken> tokens;
    tokens.reserve(tokenCount);
    for (auto token : llamaTokens) {
      tokens.emplace_back(token);
    }

    return tokens;
  }

 private:
  struct Message {
    std::string role;
    std::string content;
  };

  std::vector<Message> conversationHistory;
  std::unique_ptr<llama_model, LlamaModelDeleter> model;
  std::unique_ptr<llama_context, LlamaContextDeleter> context;
  std::unique_ptr<mtmd_context, MtmdContextDeleter> multiModalContext;
  std::unique_ptr<ResponseStreamer> responseStreamer;

  llama_token endOfTurnToken;
  ModelParams storedModelParams;
  ContextParams storedContextParams;
  llama_pos pastTokenCount = 0;

  void InitializeMultiModal() {
    mtmd_context_params multiModalParams = mtmd_context_params_default();
    multiModalParams.use_gpu = storedModelParams.offloadMultiModalToGPU;
    multiModalParams.print_timings = false;
    multiModalParams.n_threads = storedContextParams.threadCount;

    multiModalContext.reset(mtmd_init_from_file(
        storedModelParams.multiModalProjectorPath.c_str(),
        model.get(),
        multiModalParams
    ));

    if (!multiModalContext) {
      throw std::runtime_error(
          "Failed to load multimodal projector from " +
          storedModelParams.multiModalProjectorPath
      );
    }

    if (!mtmd_support_vision(multiModalContext.get())) {
      throw std::runtime_error(
          "The provided multimodal projector does not support vision. "
          "This library only supports vision-capable multimodal models."
      );
    }
  }

  std::string BuildFormattedPrompt(bool addBos) const {
    const char* chatTemplate = llama_model_chat_template(model.get(), nullptr);

    std::vector<llama_chat_message> allMessages;
    for (const auto& message : conversationHistory) {
      allMessages.push_back({message.role.c_str(), message.content.c_str()});
    }

    std::string formattedPrompt;

    if (addBos) {
      int32_t size = llama_chat_apply_template(
          chatTemplate,
          allMessages.data(),
          allMessages.size(),
          true,
          nullptr,
          0
      );
      if (size > 0) {
        formattedPrompt.resize(size + 1);
        llama_chat_apply_template(
            chatTemplate,
            allMessages.data(),
            allMessages.size(),
            true,
            formattedPrompt.data(),
            formattedPrompt.size()
        );
        formattedPrompt.resize(size);
      }
    } else {
      int32_t fullSize = llama_chat_apply_template(
          chatTemplate,
          allMessages.data(),
          allMessages.size(),
          true,
          nullptr,
          0
      );

      int32_t previousSize = 0;
      if (allMessages.size() > 1) {
        previousSize = llama_chat_apply_template(
            chatTemplate,
            allMessages.data(),
            allMessages.size() - 1,
            true,
            nullptr,
            0
        );
      }

      if (fullSize > 0) {
        std::string fullPrompt;
        fullPrompt.resize(fullSize + 1);
        llama_chat_apply_template(
            chatTemplate,
            allMessages.data(),
            allMessages.size(),
            true,
            fullPrompt.data(),
            fullPrompt.size()
        );
        fullPrompt.resize(fullSize);

        if (previousSize > 0 && previousSize < fullSize) {
          formattedPrompt = fullPrompt.substr(previousSize);
        } else {
          formattedPrompt = fullPrompt;
        }
      }
    }

    return formattedPrompt;
  }

  void AddUserMessage(const std::string& message) {
    if (conversationHistory.size() >= LlamaChatConfig::MAX_HISTORY_SIZE) {
      conversationHistory.erase(conversationHistory.begin());
    }

    conversationHistory.push_back({"user", message});
  }

  void RunQuery(
      mtmd_bitmap* bitmap,
      const std::function<void(const std::string&)>& callback
  ) {
    SamplingParams samplingParams;

    bool addBos = (pastTokenCount == 0);
    std::string formattedPrompt = BuildFormattedPrompt(addBos);

    if (formattedPrompt.empty()) {
      throw std::runtime_error("Failed to format chat prompt");
    }

    mtmd_input_text textInput;
    textInput.text = formattedPrompt.c_str();
    textInput.add_special = addBos;
    textInput.parse_special = true;

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    if (!chunks) {
      throw std::runtime_error("Failed to create input chunks");
    }

    const mtmd_bitmap* bitmaps[] = {bitmap};
    int bitmapCount = bitmap ? 1 : 0;

    int32_t tokenizeResult = mtmd_tokenize(
        multiModalContext.get(),
        chunks,
        &textInput,
        bitmaps,
        bitmapCount
    );

    if (tokenizeResult != 0) {
      mtmd_input_chunks_free(chunks);
      throw std::runtime_error(
          "Failed to tokenize multimodal prompt, error: " +
          std::to_string(tokenizeResult)
      );
    }

    llama_pos newPastTokenCount;
    int evaluateResult = mtmd_helper_eval_chunks(
        multiModalContext.get(),
        context.get(),
        chunks,
        pastTokenCount,
        0,
        storedContextParams.batchSize,
        true,
        &newPastTokenCount
    );

    mtmd_input_chunks_free(chunks);

    if (evaluateResult != 0) {
      throw std::runtime_error("Failed to evaluate multimodal chunks");
    }

    pastTokenCount = newPastTokenCount;

    std::string assistantResponse =
        responseStreamer->Stream(pastTokenCount, samplingParams, callback);

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
) const {
  try {
    return pimpl->InitializeModel(modelPath, params);
  } catch (const std::exception& exception) {
    std::cerr << "InitializeModel exception: " << exception.what() << std::endl;
    return false;
  }
}

bool LlamaChat::InitializeContext(const ContextParams& params) const {
  try {
    return pimpl->InitializeContext(params);
  } catch (const std::exception& exception) {
    std::cerr << "InitializeContext exception: " << exception.what()
              << std::endl;
    return false;
  }
}

void LlamaChat::SetSystemPrompt(const std::string& systemPrompt) const {
  pimpl->SetSystemPrompt(systemPrompt);
}

void LlamaChat::ResetConversation() const { pimpl->ResetConversation(); }

void LlamaChat::Prompt(
    const std::string& userMessage,
    const std::function<void(const std::string&)>& callback,
    const std::optional<ImageInput>& image
) const {
  pimpl->Prompt(userMessage, callback, image);
}

std::vector<LlamaToken> LlamaChat::Encode(
    const std::string& text, const bool addBos
) const {
  return pimpl->Encode(text, addBos);
}
