#pragma once

#include <string>
#include <vector>

// Frame extraction for video files. This module knows nothing about the
// engine: it produces image paths, which you then pass to
// LlamaVision::Prompt() like any other images.
//
// Runtime requirement: ffmpeg (and ideally ffprobe) installed and on PATH,
// or pointed to via the params below.
struct VideoFrameParams {
  int maxFrames = 8;          // hard cap on extracted frames
  double maxSampleFps = 2.0;  // never sample faster than this (short clips
                              // would otherwise yield near-duplicate frames)
  int maxEdgePixels = 720;    // downscale so the longest edge fits this;
                              // never upscales
  std::string ffmpegPath = "ffmpeg";
  std::string ffprobePath = "ffprobe";
};

struct VideoFrameResult {
  bool ok = false;
  std::vector<std::string> framePaths;  // ordered, earliest first
  std::string directory;  // temp directory holding the frames
  std::string error;      // set when !ok
};

// Extracts up to maxFrames JPEG frames. When ffprobe can report the video
// duration, frames are spread evenly across the whole video; otherwise the
// fallback samples 1 frame per second from the start.
// The caller owns the temp directory: keep it until generation is done,
// then remove it with CleanupVideoFrames().
VideoFrameResult ExtractVideoFrames(
    const std::string& videoPath, const VideoFrameParams& params = {}
);

// Removes the temp directory created by ExtractVideoFrames. Safe to call
// on a failed or empty result.
void CleanupVideoFrames(const VideoFrameResult& result);
