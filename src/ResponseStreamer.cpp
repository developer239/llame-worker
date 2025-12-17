#include "ResponseStreamer.h"

#include <stdexcept>
#include <vector>

#include "common.h"
#include "llama-chat.h"

ResponseStreamer::ResponseStreamer(
    llama_context* context, llama_model* model, llama_token endOfTurnToken
)
    : context(context),
      model(model),
      endOfTurnToken(endOfTurnToken),
      sampler(nullptr) {}

ResponseStreamer::~ResponseStreamer() noexcept = default;

void ResponseStreamer::SetupSamplerChain(const SamplingParams& params) {
  sampler.reset();

  auto samplerChainParams = llama_sampler_chain_default_params();
  samplerChainParams.no_perf = true;
  sampler.reset(llama_sampler_chain_init(samplerChainParams));

  if (params.repeatPenalty != 1.0f || params.frequencyPenalty != 0.0f ||
      params.presencePenalty != 0.0f) {
    llama_sampler_chain_add(
        sampler.get(),
        llama_sampler_init_penalties(
            params.penaltyLastN,
            params.repeatPenalty,
            params.frequencyPenalty,
            params.presencePenalty
        )
    );
  }

  if (params.topK > 0) {
    llama_sampler_chain_add(
        sampler.get(),
        llama_sampler_init_top_k(params.topK)
    );
  }

  if (params.topP < 1.0f) {
    llama_sampler_chain_add(
        sampler.get(),
        llama_sampler_init_top_p(params.topP, 1)
    );
  }

  llama_sampler_chain_add(
      sampler.get(),
      llama_sampler_init_temp(params.temperature)
  );

  llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(params.seed));
}

std::string ResponseStreamer::TokenToString(const llama_token token) const {
  const llama_vocab* vocabulary = llama_model_get_vocab(model);
  char buffer[256];
  int length =
      llama_token_to_piece(vocabulary, token, buffer, sizeof(buffer), 0, true);

  if (length < 0) {
    std::vector<char> largeBuffer(-length);
    length = llama_token_to_piece(
        vocabulary,
        token,
        largeBuffer.data(),
        largeBuffer.size(),
        0,
        true
    );

    if (length < 0) {
      return "";
    }

    return std::string(largeBuffer.data(), length);
  }

  return std::string(buffer, length);
}

std::string ResponseStreamer::Stream(
    llama_pos& pastTokenCount, const SamplingParams& params,
    const std::function<void(const std::string&)>& callback
) {
  SetupSamplerChain(params);

  const llama_vocab* vocabulary = llama_model_get_vocab(model);
  std::string assistantResponse;
  llama_batch batch = llama_batch_init(1, 0, 1);

  for (size_t tokenIndex = 0; tokenIndex < params.maxTokens; ++tokenIndex) {
    llama_token newToken = llama_sampler_sample(sampler.get(), context, -1);

    if (llama_vocab_is_eog(vocabulary, newToken) ||
        newToken == endOfTurnToken) {
      break;
    }

    std::string piece = TokenToString(newToken);
    callback(piece);
    assistantResponse += piece;

    common_batch_clear(batch);
    common_batch_add(batch, newToken, pastTokenCount++, {0}, true);

    if (llama_decode(context, batch) != 0) {
      llama_batch_free(batch);
      throw std::runtime_error("Failed to decode token");
    }
  }

  llama_batch_free(batch);

  return assistantResponse;
}
