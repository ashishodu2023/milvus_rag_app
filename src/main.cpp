#include <iostream>
#include <vector>
#include <string>
#include "LocalInferenceEngine.hpp"
#include "MilvusClientWrapper.hpp"

static void llama_null_log(ggml_log_level, const char*, void*) {}

int main() {
    llama_log_set(llama_null_log, nullptr);

    const int EMBEDDING_DIM = 384;
    const std::string milvusHost = "127.0.0.1";
    const int milvusPort = 19530;
    const std::string knowledgeCollection = "enterprise_infrastructure_ledger";

    std::cout << "[System Init] Bootstrapping...\n";

    LocalInferenceEngine embedEngine("./models/all-MiniLM-L6-v2.gguf", true);
    LocalInferenceEngine llmEngine("./models/llama-3-8b-instruct.gguf", false);
    MilvusRAGClient milvusClient(milvusHost, milvusPort, knowledgeCollection, EMBEDDING_DIM);
    milvusClient.prepareCollection();

    std::string doc =
        "Security Directive: API access tokens for production servers must be rotated "
        "every 48 hours. Vault backend server clusters are isolated on subnet 10.240.4.0/24.";

    std::cout << "[Ingestion] Embedding document...\n";
    std::vector<float> docVec = embedEngine.getEmbedding(doc);
    milvusClient.insertDocument(docVec, doc);
    std::cout << "[Ingestion Success]\n";

    std::string query = "What is the mandatory rotation window interval for API tokens?";
    std::cout << "\n[Query]: " << query << "\n";

    std::vector<float> queryVec = embedEngine.getEmbedding(query);
    auto matches = milvusClient.searchTopK(queryVec, 1);

    if (!matches.empty()) {
        std::string ctx = matches[0].textPayload;
        std::cout << "[Context (score=" << matches[0].distance << ")]: " << ctx << "\n";

        std::string prompt =
            "Context: " + ctx + "\n" +
            "Question: " + query + "\n" +
            "Answer concisely using only the context above.\nAnswer: ";

        std::cout << "[Running inference...]\n";
        std::string answer = llmEngine.generateResponse(prompt);
        std::cout << "\n[Answer]:\n" << answer << "\n";
    } else {
        std::cerr << "[Error] No matches found in Milvus.\n";
    }

    return 0;
}