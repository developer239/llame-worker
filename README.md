# LlamaChat 🦙🦙🦙

LlamaChat is a C++ library designed for running **multimodal** language models using the [llama.cpp](https://github.com/ggerganov/llama.cpp) framework. It provides an easy-to-use interface for loading models, querying them with text and images, and streaming responses in C++ applications.

> **Note:** This library exclusively supports multimodal (vision-capable) models. A multimodal projector file is required for initialization.

**Supported Systems:**

- MacOS
- Windows
- Linux

## Installation

### Add LlamaChat as a Submodule

First, add this library as a submodule in your project:

```bash
$ git submodule add https://github.com/developer239/llama-chat externals/llama-chat
```

Load the module's dependencies:

```bash
$ git submodule update --init --recursive
```

### Update Your CMake

In your project's `CMakeLists.txt`, add the following lines to include and link the LlamaChat library:

```cmake
add_subdirectory(externals/llama-chat)
target_link_libraries(<your_target> PRIVATE LlamaChat)
```

## Usage

### Basic Text-Only Usage

```cpp
#include "llama-chat.h"
#include <iostream>

int main() {
    LlamaChat llama;
    
    ModelParams modelParams;
    modelParams.gpuLayerCount = 32;
    modelParams.multiModalProjectorPath = "path/to/mmproj.gguf";
    
    if (!llama.InitializeModel("path/to/model.gguf", modelParams)) {
        std::cerr << "Failed to initialize the model." << std::endl;
        return 1;
    }
    
    ContextParams contextParams;
    contextParams.contextSize = 4096;
    
    if (!llama.InitializeContext(contextParams)) {
        std::cerr << "Failed to initialize the context." << std::endl;
        return 1;
    }

    llama.SetSystemPrompt("You are a helpful AI assistant.");

    llama.Prompt("How do I write hello world in C++?", [](const std::string& piece) {
        std::cout << piece << std::flush;
    });

    return 0;
}
```

### Multimodal Usage (Text + Image)

```cpp
#include "llama-chat.h"
#include <iostream>

int main() {
    LlamaChat llama;
    
    ModelParams modelParams;
    modelParams.gpuLayerCount = 32;
    modelParams.multiModalProjectorPath = "path/to/mmproj.gguf";
    
    if (!llama.InitializeModel("path/to/model.gguf", modelParams)) {
        std::cerr << "Failed to initialize the model." << std::endl;
        return 1;
    }
    
    ContextParams contextParams;
    contextParams.contextSize = 4096;
    
    if (!llama.InitializeContext(contextParams)) {
        std::cerr << "Failed to initialize the context." << std::endl;
        return 1;
    }

    llama.SetSystemPrompt("You are a helpful AI assistant that can analyze images.");

    // Prompt with an image
    ImageInput image = ImageInput::FromPath("path/to/image.jpg");
    
    llama.Prompt(
        "What do you see in this image?",
        [](const std::string& piece) {
            std::cout << piece << std::flush;
        },
        image  // Optional image parameter
    );

    return 0;
}
```

### Streaming Responses

The `Prompt` method implements streaming responses by providing a callback function. This is useful for long outputs where you want to display text as it's generated.

## API Reference

### LlamaChat Class

The `LlamaChat` class provides methods to interact with multimodal language models loaded through llama.cpp.

#### Public Methods

- `LlamaChat()`: Constructor. Initializes the LlamaChat object.
- `~LlamaChat()`: Destructor. Cleans up resources.
- `bool InitializeModel(const std::string& modelPath, const ModelParams& params)`: Initializes the model with the specified path and parameters.
- `bool InitializeContext(const ContextParams& params)`: Initializes the context with the specified parameters. Throws if multimodal projector is not configured or doesn't support vision.
- `void SetSystemPrompt(const std::string& systemPrompt)`: Sets the system prompt for the conversation.
- `void ResetConversation()`: Resets the conversation history and context.
- `void Prompt(const std::string& userMessage, const std::function<void(const std::string&)>& callback, const std::optional<ImageInput>& image = std::nullopt)`: Processes the user message (with optional image) and streams the response.

#### Structs

##### LlamaToken

Represents a token in the model's vocabulary.

| Field | Type | Description |
|-------|------|-------------|
| `tokenId` | `int` | The unique identifier of the token |

##### ModelParams

Parameters for model initialization.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpuLayerCount` | `int` | `0` | Number of layers to offload to GPU. Set to 0 for CPU-only |
| `vocabularyOnly` | `bool` | `false` | Only load the vocabulary, no weights |
| `useMemoryMapping` | `bool` | `true` | Use memory mapping for faster loading |
| `useModelLock` | `bool` | `false` | Force system to keep model in RAM |
| `multiModalProjectorPath` | `std::string` | `""` | **Required.** Path to the multimodal projector (mmproj) file |
| `offloadMultiModalToGPU` | `bool` | `true` | Whether to offload multimodal processing to GPU |

##### ContextParams

Parameters for context initialization.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `contextSize` | `size_t` | `4096` | Size of the context window (in tokens) |
| `threadCount` | `int` | `6` | Number of threads to use for computation |
| `batchSize` | `int` | `512` | Number of tokens to process in parallel |

##### SamplingParams

Parameters for text generation sampling.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `maxTokens` | `size_t` | `1000` | Maximum number of tokens to generate |
| `temperature` | `float` | `1.0` | Controls randomness in generation |
| `topK` | `int32_t` | `45` | Limits sampling to the k most likely tokens |
| `topP` | `float` | `0.95` | Limits sampling to a cumulative probability |
| `repeatPenalty` | `float` | `1.0` | Penalty for repeating tokens (1.0 = disabled) |
| `frequencyPenalty` | `float` | `1.0` | Penalty based on token frequency in generated text |
| `presencePenalty` | `float` | `0.0` | Penalty for tokens already present in generated text |
| `penaltyLastN` | `int` | `64` | Number of previous tokens to consider for penalties |
| `seed` | `unsigned int` | `0xFFFFFFFF` | Random seed for sampling (default = random) |
| `repeatPenaltyTokens` | `std::vector<LlamaToken>` | `{}` | Specific tokens to consider for repeat penalty |

##### ImageInput

Represents an image input for multimodal prompts.

| Field | Type | Description |
|-------|------|-------------|
| `path` | `std::string` | Path to the image file |

**Static Methods:**
- `static ImageInput FromPath(const std::string& path)`: Creates an ImageInput from a file path.

## Supported Models

This library requires multimodal models with vision support. Compatible models include:

- LLaVA models
- Other vision-language models supported by llama.cpp's mtmd (multimodal) interface

You need both the main model file (`.gguf`) and the corresponding multimodal projector file (`mmproj-*.gguf`).

## License

See the LICENSE file for details.
