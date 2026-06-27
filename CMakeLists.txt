cmake_minimum_required(VERSION 3.20)
project(MilvusMacRAG VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_COMPILER   "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
# 1. Apple Silicon M1 Optimization
message(STATUS "Configuring for Apple Silicon M1 (ARM64)")
add_compile_options(-O3 -mcpu=apple-m1)
add_compile_definitions(GGML_USE_ACCELERATE GGML_USE_METAL)

# 2. System Threads
find_package(Threads REQUIRED)

# 3. Milvus SDK
set(MILVUS_INCLUDE_DIR "/Users/ashishverma/milvus/include")
set(MILVUS_LIB_DIR     "/Users/ashishverma/milvus/lib")
find_library(MILVUS_SDK_LIB milvus_sdk PATHS ${MILVUS_LIB_DIR} REQUIRED)

# 4. llama.cpp + ggml (Homebrew)
set(HOMEBREW_PREFIX "/opt/homebrew")
set(LLAMA_INCLUDE_DIR "${HOMEBREW_PREFIX}/include")
set(LLAMA_LIB_DIR     "${HOMEBREW_PREFIX}/lib")

find_library(LLAMA_LIB      llama       PATHS ${LLAMA_LIB_DIR} REQUIRED)
find_library(GGML_LIB       ggml        PATHS ${LLAMA_LIB_DIR} REQUIRED)
find_library(GGML_BASE_LIB  ggml-base   PATHS ${LLAMA_LIB_DIR})
find_library(GGML_METAL_LIB ggml-metal  PATHS ${LLAMA_LIB_DIR})
find_library(GGML_BLAS_LIB  ggml-blas   PATHS ${LLAMA_LIB_DIR})
find_library(GGML_CPU_LIB   ggml-cpu    PATHS ${LLAMA_LIB_DIR})

# 5. Executable
add_executable(milvus_rag_app src/main.cpp)

# 6. Include Dirs
target_include_directories(milvus_rag_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LLAMA_INCLUDE_DIR}
    ${MILVUS_INCLUDE_DIR}
)

find_library(LLAMA_LIB        llama        PATHS ${LLAMA_LIB_DIR} REQUIRED)
find_library(LLAMA_COMMON_LIB llama-common PATHS ${LLAMA_LIB_DIR} REQUIRED)
find_library(GGML_LIB         ggml         PATHS ${LLAMA_LIB_DIR} REQUIRED)
find_library(GGML_BASE_LIB    ggml-base    PATHS ${LLAMA_LIB_DIR} REQUIRED)

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