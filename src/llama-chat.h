#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "LlamaChatConfig.h"

typedef int llama_token;

struct LlamaToken {
  llama_token tokenId;
  explicit LlamaToken(llama_token id = 0) : tokenId(id) {}
};

struct ModelParams {
  int gpuLayerCount = 0;
  bool vocabularyOnly = false;
  bool useMemoryMapping = true;
  bool useModelLock = false;

  std::string multiModalProjectorPath;
  bool offloadMultiModalToGPU = true;
};

struct ContextParams {
  size_t contextSize = 4096;
  int threadCount = 6;
  int batchSize = 512;
};

struct SamplingParams {
  size_t maxTokens = 1000;
  float temperature = 1.0f;
  int32_t topK = 45;
  float topP = 0.95f;
  float repeatPenalty = 1.0f;
  float frequencyPenalty = 1.0f;
  float presencePenalty = 0.0f;
  int penaltyLastN = LlamaChatConfig::DEFAULT_PENALTY_LAST_N;
  unsigned int seed = LlamaChatConfig::DEFAULT_SEED;
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

  bool InitializeModel(const std::string& modelPath, const ModelParams& params
  ) const;
  bool InitializeContext(const ContextParams& params) const;

  void SetSystemPrompt(const std::string& systemPrompt) const;
  void ResetConversation() const;

  void Prompt(
      const std::string& userMessage,
      const std::function<void(const std::string&)>& callback,
      const std::optional<ImageInput>& image = std::nullopt
  ) const;

  std::vector<LlamaToken> Encode(
      const std::string& text, bool addBos = true
  ) const;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};
