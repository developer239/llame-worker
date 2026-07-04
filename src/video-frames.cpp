#include "video-frames.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Quotes one argument for the shell used by std::system/popen.
// POSIX: single-quote wrapping with '\'' escapes. Windows: plain double
// quotes (sufficient for typical paths; paths containing '"' are not
// supported there).
std::string Quote(const std::string& text) {
#ifdef _WIN32
  return "\"" + text + "\"";
#else
  std::string quoted = "'";
  for (char character : text) {
    if (character == '\'') {
      quoted += "'\\''";
    } else {
      quoted += character;
    }
  }
  quoted += "'";
  return quoted;
#endif
}

bool RunAndCapture(const std::string& command, std::string& output) {
#ifdef _WIN32
  FILE* pipe = _popen(command.c_str(), "r");
#else
  FILE* pipe = popen(command.c_str(), "r");
#endif
  if (!pipe) return false;

  char buffer[256];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }

#ifdef _WIN32
  const int status = _pclose(pipe);
#else
  const int status = pclose(pipe);
#endif
  return status == 0;
}

// Returns the duration in seconds, or 0.0 when it cannot be determined
// (ffprobe missing, unreadable file, ...).
double ProbeDurationSeconds(
    const VideoFrameParams& params, const std::string& videoPath
) {
  const std::string command =
      Quote(params.ffprobePath) +
      " -v error -show_entries format=duration"
      " -of default=noprint_wrappers=1:nokey=1 " +
      Quote(videoPath);
  std::string output;
  if (!RunAndCapture(command, output)) return 0.0;
  try {
    return std::stod(output);
  } catch (...) {
    return 0.0;
  }
}

std::string MakeTempDirectory(std::string& error) {
  std::random_device randomDevice;
  for (int attempt = 0; attempt < 8; ++attempt) {
    std::ostringstream name;
    name << "llameworker-frames-" << std::hex << randomDevice()
         << randomDevice();
    const fs::path directory = fs::temp_directory_path() / name.str();
    std::error_code errorCode;
    if (fs::create_directory(directory, errorCode)) {
      return directory.string();
    }
  }
  error = "failed to create a temporary directory for frames";
  return {};
}

}  // namespace

VideoFrameResult ExtractVideoFrames(
    const std::string& videoPath, const VideoFrameParams& params
) {
  VideoFrameResult result;

  std::error_code errorCode;
  if (!fs::exists(videoPath, errorCode)) {
    result.error = "video not found: " + videoPath;
    return result;
  }
  if (params.maxFrames <= 0) {
    result.error = "maxFrames must be positive";
    return result;
  }

  result.directory = MakeTempDirectory(result.error);
  if (result.directory.empty()) return result;

  // Spread frames across the whole video when the duration is known;
  // otherwise fall back to sampling from the start. The fps clamp keeps
  // short clips from producing near-duplicate frames.
  const double duration = ProbeDurationSeconds(params, videoPath);
  double framesPerSecond =
      duration > 0.0 ? static_cast<double>(params.maxFrames) / duration : 1.0;
  framesPerSecond = std::min(framesPerSecond, params.maxSampleFps);

  // min(...) in the scale expression prevents upscaling; the inner quotes
  // stop the filtergraph parser from treating its commas as separators.
  std::ostringstream filter;
  filter << "fps=" << std::fixed << std::setprecision(6) << framesPerSecond
         << ",scale='min(" << params.maxEdgePixels << ",iw)':'min("
         << params.maxEdgePixels << ",ih)':force_original_aspect_ratio="
         << "decrease";

  const fs::path outputPattern = fs::path(result.directory) / "frame-%04d.jpg";
  const std::string command =
      Quote(params.ffmpegPath) + " -v error -y -i " + Quote(videoPath) +
      " -vf " + Quote(filter.str()) + " -frames:v " +
      std::to_string(params.maxFrames) + " " + Quote(outputPattern.string());

  if (std::system(command.c_str()) != 0) {
    result.error =
        "ffmpeg failed; is it installed and on PATH (or set ffmpegPath)?";
    CleanupVideoFrames(result);
    result.directory.clear();
    return result;
  }

  std::vector<std::string> frames;
  for (const fs::directory_entry& entry :
       fs::directory_iterator(result.directory, errorCode)) {
    frames.push_back(entry.path().string());
  }
  std::sort(frames.begin(), frames.end());

  if (frames.empty()) {
    result.error = "ffmpeg produced no frames from " + videoPath;
    CleanupVideoFrames(result);
    result.directory.clear();
    return result;
  }

  result.framePaths = std::move(frames);
  result.ok = true;
  return result;
}

void CleanupVideoFrames(const VideoFrameResult& result) {
  if (result.directory.empty()) return;
  std::error_code errorCode;
  fs::remove_all(result.directory, errorCode);
}
