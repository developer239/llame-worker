// Minimal CLI over the library. Build with -DLLAMAVISION_BUILD_EXAMPLES=ON.
//
// Usage:
//   llama-vision-describe <model.gguf> <mmproj.gguf> [image ...]
//                         [--video <clip>] [-p <prompt>] [--verbose]

#include <iostream>
#include <string>
#include <vector>

#include "llama-vision.h"
#include "video-frames.h"

namespace {

void PrintUsage(const char* program) {
  std::cerr << "Usage:\n"
            << "  " << program << " <model.gguf> <mmproj.gguf> [image ...]\n"
            << "        [--video <clip>] [-p <prompt>] [--verbose]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    PrintUsage(argv[0]);
    return 1;
  }

  VisionModelParams modelParams;
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

  LlamaVision llama;
  if (!llama.Load(modelParams)) {
    std::cerr << "Load failed: " << llama.LoadError() << std::endl;
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
  // stay on disk until Prompt() finishes.
  VideoFrameResult frames;
  if (!videoPath.empty() && !imagePaths.empty()) {
    frames = ExtractVideoFrames(videoPath);
    if (!frames.ok) {
      std::cerr << "Frame extraction failed: " << frames.error << std::endl;
      return 1;
    }
    std::cerr << "[extracted " << frames.framePaths.size() << " frame(s)]"
              << std::endl;
  }

  PromptResult result;
  if (!videoPath.empty() && imagePaths.empty()) {
    result = llama.DescribeVideo(
        videoPath,
        prompt,
        {},
        [](const std::string& piece) { std::cout << piece << std::flush; }
    );
  } else {
    imagePaths.insert(
        imagePaths.end(), frames.framePaths.begin(), frames.framePaths.end()
    );
    PromptParams promptParams;
    promptParams.prompt = prompt;
    promptParams.imagePaths = imagePaths;

    result = llama.Prompt(
        promptParams,
        [](const std::string& piece) { std::cout << piece << std::flush; }
    );
  }
  std::cout << std::endl;

  CleanupVideoFrames(frames);

  if (!result.ok) {
    std::cerr << "Prompt failed: " << result.error << std::endl;
    return 1;
  }
  std::cerr << "[prompt tokens: " << result.promptTokenCount
            << ", generated: " << result.generatedTokenCount
            << (result.truncated ? ", truncated]" : "]") << std::endl;
  return 0;
}
