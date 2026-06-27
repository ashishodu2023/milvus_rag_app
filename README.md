# Milvus RAG App — Local C++ RAG Pipeline on Apple Silicon

A fully local, production-grade Retrieval-Augmented Generation (RAG) system built in C++, running entirely on Apple Silicon (M1/M2/M3) with Metal GPU acceleration. No cloud dependencies — embeddings, vector search, and LLM inference all run on-device.

---

## Architecture

```
User Query
    │
    ▼
LocalInferenceEngine (all-MiniLM-L6-v2)
    │  Embed query → float vector
    ▼
MilvusRAGClient
    │  Vector similarity search (cosine / IP)
    ▼
Retrieved Context
    │
    ▼
LocalInferenceEngine (Llama 3 8B Instruct)
    │  Grounded generation
    ▼
Final Answer
```

**Components:**
- **Embedding Model** — `all-MiniLM-L6-v2` (GGUF, 384-dim, ~90MB)
- **Vector Database** — Milvus standalone (Docker) with HNSW index
- **LLM** — Llama 3 8B Instruct (GGUF Q4_K_M, ~4.6GB)
- **Inference Runtime** — llama.cpp via Homebrew, Metal-accelerated
- **Build System** — CMake 3.20+, Apple Clang

---

## Prerequisites

| Tool | Install |
|------|---------|
| Homebrew | `https://brew.sh` |
| CMake | `brew install cmake` |
| llama.cpp | `brew install llama.cpp` |
| Docker Desktop | `https://docker.com/products/docker-desktop` |
| Milvus C++ SDK | Built from source (see below) |

---

## Project Structure

```
milvus_rag_app/
├── src/
│   └── main.cpp                  # Entry point
├── include/
│   ├── LocalInferenceEngine.hpp  # llama.cpp wrapper (embed + generate)
│   └── MilvusClientWrapper.hpp   # Milvus SDK wrapper (insert + search)
├── models/
│   ├── all-MiniLM-L6-v2.gguf    # Embedding model
│   └── llama-3-8b-instruct.gguf # LLM
├── build/                        # CMake build output
└── CMakeLists.txt
```

---

## Setup

### 1. Clone and install dependencies

```bash
# Install llama.cpp and build tools
brew install cmake llama.cpp

# Build Milvus C++ SDK
git clone https://github.com/milvus-io/milvus-sdk-cpp.git ~/milvus-sdk-cpp
cd ~/milvus-sdk-cpp

# Install Conan via pipx (required for Milvus deps)
brew install pipx
pipx install conan
pipx ensurepath
source ~/.zshrc

# Build and install SDK to ~/milvus
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/milvus
make -j$(sysctl -n hw.logicalcpu)
make install
```

### 2. Download models

```bash
mkdir -p ~/Documents/milvus_rag_app/models
cd ~/Documents/milvus_rag_app/models

# Embedding model (~90MB)
curl -L -o all-MiniLM-L6-v2.gguf \
  "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q5_K_M.gguf"

# LLM (~4.6GB)
curl -L -o llama-3-8b-instruct.gguf \
  "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
```

### 3. Start Milvus

```bash
# Download official standalone compose file
curl -L https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml \
  -o docker-compose.yml

# Remap ports if 9000/9001 are in use locally
sed -i '' 's/9000:9000/9003:9000/' docker-compose.yml
sed -i '' 's/9001:9001/9002:9001/' docker-compose.yml

# Start
docker compose up -d

# Verify
curl http://localhost:9091/healthz
```

### 4. Build the app

```bash
cd ~/Documents/milvus_rag_app
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.logicalcpu)
```

### 5. Run

```bash
cd ~/Documents/milvus_rag_app
./build/milvus_rag_app 2>/dev/null
```

---

## CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)

set(CMAKE_C_COMPILER   "/usr/bin/clang"   CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE STRING "" FORCE)

project(MilvusMacRAG VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3 -mcpu=apple-m1)
add_compile_definitions(GGML_USE_ACCELERATE GGML_USE_METAL)

find_package(Threads REQUIRED)

set(MILVUS_INCLUDE_DIR "$ENV{HOME}/milvus/include")
set(MILVUS_LIB_DIR     "$ENV{HOME}/milvus/lib")
find_library(MILVUS_SDK_LIB milvus_sdk PATHS ${MILVUS_LIB_DIR} REQUIRED)

set(HOMEBREW_PREFIX "/opt/homebrew")
set(LLAMA_INCLUDE_DIR "${HOMEBREW_PREFIX}/include")
set(LLAMA_LIB_DIR     "${HOMEBREW_PREFIX}/lib")

find_library(LLAMA_LIB        llama        PATHS ${LLAMA_LIB_DIR} REQUIRED)
find_library(LLAMA_COMMON_LIB llama-common PATHS ${LLAMA_LIB_DIR} REQUIRED)
find_library(GGML_LIB         ggml         PATHS ${LLAMA_LIB_DIR} REQUIRED)
find_library(GGML_BASE_LIB    ggml-base    PATHS ${LLAMA_LIB_DIR} REQUIRED)

add_executable(milvus_rag_app src/main.cpp)

target_include_directories(milvus_rag_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LLAMA_INCLUDE_DIR}
    ${MILVUS_INCLUDE_DIR}
)

target_link_libraries(milvus_rag_app PRIVATE
    Threads::Threads
    ${MILVUS_SDK_LIB}
    ${LLAMA_LIB}
    ${LLAMA_COMMON_LIB}
    ${GGML_LIB}
    ${GGML_BASE_LIB}
    "-framework Accelerate"
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
    "-framework MetalPerformanceShaders"
)
```

> **Important:** Must use Apple Clang (`/usr/bin/clang++`), not GCC. The Milvus SDK and nlohmann/json headers use Apple's libc++ ABI which is incompatible with GCC's libstdc++.

---

## Sample Output

```
[System Init] Bootstrapping...
[Ingestion] Embedding document...
[Ingestion Success]

[Query]: What is the mandatory rotation window interval for API tokens?
[Context (score=9.51507)]: Security Directive: API access tokens for production servers
must be rotated every 48 hours. Vault backend server clusters are isolated on subnet 10.240.4.0/24.
[Running inference...]

[Answer]:
48 hours. API access tokens for production servers must be rotated every 48 hours.
```

---

## Key Implementation Notes

### LocalInferenceEngine

- Uses `llama_model_load_from_file` / `llama_init_from_model` (modern llama.cpp API)
- Embedding: `llama_get_embeddings_seq(ctx, 0)` — seq-level pooling, not context-level
- Generation: `llama_get_logits_ith(ctx, -1)` — last token logits only
- Cache reset: `llama_memory_clear(llama_get_memory(ctx), true)`
- All 32+ layers offloaded to Metal GPU via `n_gpu_layers = 99`

### MilvusRAGClient

- Collection schema: `id` (INT64 PK), `embedding` (FLOAT_VECTOR 384-dim), `text` (VARCHAR 2048)
- Index: HNSW with Inner Product metric (`M=16`, `efConstruction=64`)
- Search results accessed via `rawResults.Results()` → `SingleResult::Scores()` + `OutputField<VarCharFieldData>("text")`

### Suppress Verbose Output

```cpp
static void llama_null_log(ggml_log_level, const char*, void*) {}
llama_log_set(llama_null_log, nullptr);  // Call before loading any model
```

Run with `2>/dev/null` to suppress Metal/gRPC stderr output.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `g++` uses Apple Clang instead of GCC | Run `ls /opt/homebrew/bin/g++*` and alias to versioned binary |
| Milvus port conflict (9000/9001) | Remap in docker-compose.yml with `sed` |
| `llama_kv_cache_clear` not found | Use `llama_memory_clear(llama_get_memory(ctx), true)` |
| Namespace errors with GCC | Switch to Apple Clang — Milvus SDK requires it |
| Model file not found | Run app from project root, not from `build/` directory |
| Milvus connection refused | Start Docker and run `docker compose up -d` |

---

## Dependencies Summary

| Library | Version | Source |
|---------|---------|--------|
| llama.cpp | 0.9820 | Homebrew |
| ggml | 0.15.3 | Homebrew (via llama.cpp) |
| Milvus C++ SDK | latest | Built from source |
| nlohmann/json | bundled | Milvus SDK |
| Apple Metal | system | macOS frameworks |
| CMake | 3.20+ | Homebrew |

---

## License

MIT