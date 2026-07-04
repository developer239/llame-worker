#include "video-frames.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <random>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace llameworker {

namespace {

constexpr int kMaxTempDirAttempts = 8;

// Quotes one argument for the shell used by std::system/popen.
// POSIX: single-quote wrapping with '\'' escapes. Windows: plain double
// quotes (sufficient for typical paths; paths containing '"' are not
// supported there).
std::string quoteForShell(const std::string& text) {
#ifdef _WIN32
  return "\"" + text + "\"";
#else
  std::string quoted = "'";
  for (const char character : text) {
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

bool runAndCapture(const std::string& command, std::string& output) {
#ifdef _WIN32
  FILE* pipe = _popen(command.c_str(), "r");
#else
  FILE* pipe = popen(command.c_str(), "r");
#endif
  if (!pipe) return false;

  std::array<char, 256> buffer{};
  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    output += buffer.data();
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
double probeDurationSeconds(
    const VideoFrameParams& params, const std::string& videoPath
) {
  const std::string command =
      quoteForShell(params.ffprobePath) +
      " -v error -show_entries format=duration"
      " -of default=noprint_wrappers=1:nokey=1 " +
      quoteForShell(videoPath);
  std::string output;
  if (!runAndCapture(command, output)) return 0.0;

  const char* begin = output.data();
  const char* const end = output.data() + output.size();
  while (begin < end && (*begin == ' ' || *begin == '\n' || *begin == '\r' ||
                         *begin == '\t')) {
    ++begin;
  }
  double duration = 0.0;
  const auto [ptr, errorCode] = std::from_chars(begin, end, duration);
  if (errorCode != std::errc{}) return 0.0;
  return duration;
}

std::string makeTempDirectory(std::string& error) {
  std::random_device randomDevice;
  std::error_code lastError;
  for (int attempt = 0; attempt < kMaxTempDirAttempts; ++attempt) {
    const std::string name =
        std::format("llameworker-frames-{:x}{:x}", randomDevice(),
                    randomDevice());
    const fs::path directory = fs::temp_directory_path() / name;
    if (fs::create_directory(directory, lastError)) {
      return directory.string();
    }
  }
  error = std::format(
      "failed to create a temporary directory for frames: {}",
      lastError.message());
  return {};
}

}  // namespace

VideoFrameResult extractVideoFrames(
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

  result.directory = makeTempDirectory(result.error);
  if (result.directory.empty()) return result;

  // Spread frames across the whole video when the duration is known;
  // otherwise fall back to sampling from the start. The fps clamp keeps
  // short clips from producing near-duplicate frames.
  const double duration = probeDurationSeconds(params, videoPath);
  double framesPerSecond =
      duration > 0.0 ? static_cast<double>(params.maxFrames) / duration : 1.0;
  framesPerSecond = std::min(framesPerSecond, params.maxSampleFps);

  // min(...) in the scale expression prevents upscaling; the inner quotes
  // stop the filtergraph parser from treating its commas as separators.
  const std::string filter = std::format(
      "fps={:.6f},scale='min({},iw)':'min({},ih)'"
      ":force_original_aspect_ratio=decrease",
      framesPerSecond, params.maxEdgePixels, params.maxEdgePixels);

  const fs::path outputPattern = fs::path(result.directory) / "frame-%04d.jpg";
  // Capture ffmpeg's own diagnostics (it writes them to stderr) so a
  // failure surfaces the real reason instead of a generic guess.
  const std::string command =
      quoteForShell(params.ffmpegPath) + " -v error -y -i " +
      quoteForShell(videoPath) + " -vf " + quoteForShell(filter) +
      " -frames:v " + std::to_string(params.maxFrames) + " " +
      quoteForShell(outputPattern.string()) + " 2>&1";

  std::string ffmpegOutput;
  if (!runAndCapture(command, ffmpegOutput)) {
    result.error = ffmpegOutput.empty()
                       ? "ffmpeg failed; is it installed and on PATH "
                         "(or set ffmpegPath)?"
                       : "ffmpeg failed: " + ffmpegOutput;
    cleanupVideoFrames(result);
    result.directory.clear();
    return result;
  }

  std::vector<std::string> frames;
  for (const auto& entry :
       fs::directory_iterator(result.directory, errorCode)) {
    frames.push_back(entry.path().string());
  }
  std::ranges::sort(frames);

  if (frames.empty()) {
    result.error = "ffmpeg produced no frames from " + videoPath;
    cleanupVideoFrames(result);
    result.directory.clear();
    return result;
  }

  result.framePaths = std::move(frames);
  result.ok = true;
  return result;
}

void cleanupVideoFrames(const VideoFrameResult& result) {
  if (result.directory.empty()) return;
  std::error_code errorCode;
  fs::remove_all(result.directory, errorCode);
}

}  // namespace llameworker
