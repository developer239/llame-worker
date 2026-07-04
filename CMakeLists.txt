cmake_minimum_required(VERSION 3.24)
project(llameworker VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LLAMEWORKER_BUILD_EXAMPLES "Build the example programs" OFF)

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR
            "The llama.cpp submodule is missing. "
            "Run 'git submodule update --init --recursive'")
endif ()

# mtmd (the multimodal library) lives under tools/ and needs common to
# configure, so LLAMA_BUILD_TOOLS stays ON. Tests, examples, and the server
# are trimmed for build speed.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

# EXCLUDE_FROM_ALL keeps llama.cpp's own tool executables out of the default
# build: we only need the 'llama' and 'mtmd' library targets, which are still
# built because llameworker links them explicitly. This also sidesteps an
# upstream quirk at the current pin where 'llama-cli' fails to configure with
# the server disabled (it links a server-only target). Downstream projects
# that add_subdirectory(llame-worker) inherit this same behaviour.
add_subdirectory(externals/llama.cpp EXCLUDE_FROM_ALL)

add_library(llameworker STATIC
        src/llama-vision.cpp
        src/llama-vision.h
        src/video-frames.cpp
        src/video-frames.h
)

target_compile_features(llameworker PUBLIC cxx_std_17)

# 'llama' transitively exports the ggml headers; 'mtmd' is the multimodal
# library. 'common' is no longer linked: v2 uses no common_* symbols.
target_link_libraries(llameworker PUBLIC llama mtmd)

target_include_directories(llameworker
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        PRIVATE
        # Belt and braces: some submodule revisions do not export the mtmd
        # header directory on the target.
        ${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp/tools/mtmd
)

# std::filesystem needs an explicit link on GCC < 9.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(llameworker PRIVATE stdc++fs)
endif ()

if (LLAMEWORKER_BUILD_EXAMPLES)
    add_executable(llameworker-describe examples/describe/main.cpp)
    target_link_libraries(llameworker-describe PRIVATE llameworker)
endif ()

install(TARGETS llameworker DESTINATION lib)
install(FILES src/llama-vision.h src/video-frames.h DESTINATION include)
