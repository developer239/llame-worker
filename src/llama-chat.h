#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

typedef int llama_token;

struct LlamaToken {
  llama_token tokenId;
  explicit LlamaToken(llama_token id = 0) : tokenId(id) {}
};

struct ModelParams {
  int nGpuLayers = 0;
  bool vocabularyOnly = false;
  bool useMemoryMapping = true;
  bool useModelLock = false;

  // Path to mmproj (multimodal projector) file
  std::string multiModalPath;
  bool offloadMultiModalToGPU = true;
};

struct ContextParams {
  size_t nContext = 4096;
  int nThreads = 6;
  int nBatch = 512;
};

struct SamplingParams {
  size_t maxTokens = 1000;
  float temperature = 1.0f;
  int32_t topK = 45;
  float topP = 0.95f;
  float repeatPenalty = 1.0f;
  float frequencyPenalty = 1.0f;
  float presencePenalty = 0.0f;
  std::vector<LlamaToken> repeatPenaltyTokens;
};

struct ImageInput {
  std::string path;

  static ImageInput FromPath(const std::string& path);
};

class LlamaChat {
 public:
  LlamaChat();
  ~LlamaChat();

  LlamaChat(const LlamaChat&) = delete;
  LlamaChat& operator=(const LlamaChat&) = delete;

  LlamaChat(LlamaChat&&) noexcept = default;
  LlamaChat& operator=(LlamaChat&&) noexcept = default;

  bool InitializeModel(const std::string& modelPath, const ModelParams& params);
  bool InitializeContext(const ContextParams& params);
  void SetSystemPrompt(const std::string& systemPrompt);
  void ResetConversation();

  void Prompt(
      const std::string& userMessage,
      const std::function<void(const std::string&)>& callback
  );

  void PromptWithImage(
      const std::string& userMessage, const ImageInput& image,
      const std::function<void(const std::string&)>& callback
  );

  [[nodiscard]] std::vector<LlamaToken> Encode(
      const std::string& text, bool addBos = true
  ) const;

  [[nodiscard]] bool HasVisionSupport() const;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};