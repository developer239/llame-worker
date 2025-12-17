#pragma once

#include <functional>
#include <memory>
#include <string>

#include "llama.h"

struct SamplingParams;

struct LlamaSamplerDeleter {
  void operator()(llama_sampler* sampler) const {
    if (sampler) {
      llama_sampler_free(sampler);
    }
  }
};

class ResponseStreamer {
 public:
  ResponseStreamer(
      llama_context* context, llama_model* model, llama_token endOfTurnToken
  );

  ~ResponseStreamer() noexcept;

  ResponseStreamer(const ResponseStreamer&) = delete;
  ResponseStreamer& operator=(const ResponseStreamer&) = delete;
  ResponseStreamer(ResponseStreamer&&) = delete;
  ResponseStreamer& operator=(ResponseStreamer&&) = delete;

  std::string Stream(
      llama_pos& pastTokenCount, const SamplingParams& params,
      const std::function<void(const std::string&)>& callback
  );

 private:
  llama_context* context;
  llama_model* model;
  llama_token endOfTurnToken;
  std::unique_ptr<llama_sampler, LlamaSamplerDeleter> sampler;

  void SetupSamplerChain(const SamplingParams& params);
  std::string TokenToString(llama_token token) const;
};
