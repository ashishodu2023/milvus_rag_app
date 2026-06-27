#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <llama.h>

class LocalInferenceEngine {
private:
    struct llama_model* model = nullptr;
    struct llama_context* ctx = nullptr;
    const struct llama_vocab* vocab = nullptr;

public:
    LocalInferenceEngine(const std::string& modelPath, bool isEmbeddingModel) {
        llama_backend_init();

        auto m_params = llama_model_default_params();
        m_params.n_gpu_layers = 99;

        model = llama_model_load_from_file(modelPath.c_str(), m_params);
        if (!model) throw std::runtime_error("Failed to load model: " + modelPath);

        vocab = llama_model_get_vocab(model);
        if (!vocab) throw std::runtime_error("Failed to extract vocab.");

        auto c_params = llama_context_default_params();
        c_params.embeddings = isEmbeddingModel;
        c_params.n_ctx = 2048;

        ctx = llama_init_from_model(model, c_params);
        if (!ctx) throw std::runtime_error("Failed to initialize context.");
    }

    ~LocalInferenceEngine() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
    }

    std::vector<float> getEmbedding(const std::string& text) {
    std::vector<llama_token> tokens(text.length() + 2);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.length(),
                                  tokens.data(), tokens.size(), true, false);
    if (n_tokens < 0) throw std::runtime_error("Tokenization failed.");
    tokens.resize(n_tokens);

    llama_memory_clear(llama_get_memory(ctx), true);  // FIXED
    llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));

    const float* rawEmb = llama_get_embeddings_seq(ctx, 0);
    if (!rawEmb) throw std::runtime_error("Failed to get embeddings.");

    int n_embd = llama_model_n_embd(model);
    return std::vector<float>(rawEmb, rawEmb + n_embd);
}

    std::string generateResponse(const std::string& prompt) {
    std::vector<llama_token> tokens(prompt.length() + 2);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                                  tokens.data(), tokens.size(), true, false);
    if (n_tokens < 0) throw std::runtime_error("Tokenization failed.");
    tokens.resize(n_tokens);

    llama_memory_clear(llama_get_memory(ctx), true);  // FIXED
    llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));

    std::string output;
    int n_vocab = llama_vocab_n_tokens(vocab);

    for (int i = 0; i < 200; ++i) {
        float* logits = llama_get_logits_ith(ctx, -1);
        llama_token nextToken = (llama_token)std::distance(
            logits, std::max_element(logits, logits + n_vocab));

        if (llama_vocab_is_eog(vocab, nextToken)) break;

        char buf[128];
        int n = llama_token_to_piece(vocab, nextToken, buf, sizeof(buf), 0, false);
        if (n > 0) output.append(buf, n);

        llama_decode(ctx, llama_batch_get_one(&nextToken, 1));
    }
    return output;
}
};