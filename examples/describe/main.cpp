// Minimal CLI over the library. Build with -DLLAMEWORKER_BUILD_EXAMPLES=ON.
//
// Usage:
//   llameworker-describe <model.gguf> <mmproj.gguf> [image ...]
//                         [--video <clip>] [-p <prompt>] [--verbose]

#include <cstdio>
#include <print>
#include <string>
#include <string_view>
#include <vector>

#include "llameworker.h"
#include "video-frames.h"

namespace lw = llameworker;

namespace {

void printUsage(const char* program) {
  std::println(stderr, "Usage:");
  std::println(stderr, "  {} <model.gguf> <mmproj.gguf> [image ...]", program);
  std::println(stderr, "        [--video <clip>] [-p <prompt>] [--verbose]");
}

void printPiece(std::string_view piece) { std::print("{}", piece); }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    printUsage(argv[0]);
    return 1;
  }

  lw::VisionModelParams modelParams;
  modelParams.modelPath = argv[1];
  modelParams.projectorPath = argv[2];

  std::string prompt;
  std::string videoPath;
  std::vector<std::string> imagePaths;

  for (int i = 3; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "-p" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (argument == "--video" && i + 1 < argc) {
      videoPath = argv[++i];
    } else if (argument == "--verbose") {
      modelParams.verbose = true;
    } else {
      imagePaths.push_back(argument);
    }
  }

  lw::LlameWorker llameworker;
  if (!llameworker.load(modelParams)) {
    std::println(stderr, "Load failed: {}", llameworker.loadError());
    return 1;
  }

  if (prompt.empty()) {
    if (imagePaths.empty() && videoPath.empty()) {
      prompt = "Say hello.";
    } else if (!videoPath.empty()) {
      prompt =
          "These images are frames sampled from one video, in order. "
          "Describe what happens.";
    } else {
      prompt = "Describe this image.";
    }
  }

  // When callers mix video frames with explicit image paths, the frames must
  // stay on disk until prompt() finishes.
  lw::VideoFrameResult frames;
  if (!videoPath.empty() && !imagePaths.empty()) {
    frames = lw::extractVideoFrames(videoPath);
    if (!frames.ok) {
      std::println(stderr, "Frame extraction failed: {}", frames.error);
      return 1;
    }
    std::println(stderr, "[extracted {} frame(s)]", frames.framePaths.size());
  }

  lw::PromptResult result;
  if (!videoPath.empty() && imagePaths.empty()) {
    result = llameworker.describeVideo(videoPath, prompt, {}, printPiece);
  } else {
    imagePaths.insert(
        imagePaths.end(), frames.framePaths.begin(), frames.framePaths.end()
    );
    lw::PromptParams promptParams{
        .prompt = prompt,
        .imagePaths = imagePaths,
    };

    result = llameworker.prompt(promptParams, printPiece);
  }
  std::print("\n");

  lw::cleanupVideoFrames(frames);

  if (!result.ok) {
    std::println(stderr, "Prompt failed: {}", result.error);
    return 1;
  }
  std::println(
      stderr, "[prompt tokens: {}, generated: {}{}]", result.promptTokenCount,
      result.generatedTokenCount, result.truncated ? ", truncated" : "");
  return 0;
}
