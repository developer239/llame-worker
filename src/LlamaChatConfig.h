#pragma once

struct LlamaChatConfig {
  static constexpr size_t MAX_HISTORY_SIZE = 10;
  static constexpr int DEFAULT_PENALTY_LAST_N = 64;
  static constexpr unsigned int DEFAULT_SEED = 0xFFFFFFFF;
};
