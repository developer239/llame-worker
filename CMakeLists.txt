cmake_minimum_required(VERSION 3.24)
project(LlamaChat VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LIB_NAME LlamaChat)

option(LLAMACHAT_DISABLE_LOGS "Disable llama.cpp logging at compile time" ON)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR "The llama.cpp submodule is missing. Please run 'git submodule update --init --recursive'")
endif()

# Enable building common and tools (for mtmd) when using llama.cpp as subdirectory
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp)

set(SOURCES
        src/llama-chat.cpp
        src/llama-chat.h
        src/ResponseStreamer.cpp
        src/ResponseStreamer.h
        src/LlamaChatConfig.h
)

add_library(${LIB_NAME} STATIC ${SOURCES})

if(LLAMACHAT_DISABLE_LOGS)
    target_compile_definitions(${LIB_NAME} PUBLIC LOG_DISABLE_LOGS=1)
    target_compile_definitions(llama PUBLIC LOG_DISABLE_LOGS=1)
    target_compile_definitions(common PUBLIC LOG_DISABLE_LOGS=1)
    target_compile_definitions(mtmd PUBLIC LOG_DISABLE_LOGS=1)
endif()

# Link libraries - mtmd is the multimodal library
target_link_libraries(${LIB_NAME} PUBLIC
        llama
        common
        mtmd
        ${CMAKE_THREAD_LIBS_INIT}
)

target_include_directories(${LIB_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp/src
        ${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp/ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp/common
        ${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp/tools/mtmd  # ADD THIS LINE
)

install(TARGETS ${LIB_NAME} DESTINATION lib)
install(FILES src/llama-chat.h DESTINATION include)