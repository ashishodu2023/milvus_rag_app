#include <iostream>
#include <vector>
#include <string>
#include "LocalInferenceEngine.hpp"
#include "MilvusClientWrapper.hpp"

static void llama_null_log(ggml_log_level, const char*, void*) {}

void printBanner() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════╗\n";
    std::cout << "║       Milvus RAG — Local AI Assistant        ║\n";
    std::cout << "║   llama.cpp + Milvus | Apple Metal (M1)      ║\n";
    std::cout << "╚══════════════════════════════════════════════╝\n\n";
}

void ingestPhase(LocalInferenceEngine& embedEngine, MilvusRAGClient& milvusClient) {
    std::cout << "── INGESTION MODE ──────────────────────────────\n";
    std::cout << "Enter documents to add to the knowledge base.\n";
    std::cout << "Type 'done' on a new line when finished.\n\n";

    int docCount = 0;
    while (true) {
        std::cout << "Document " << (docCount + 1) << " > ";
        std::string line, doc;

        while (std::getline(std::cin, line)) {
            if (line == "done") break;
            if (!doc.empty()) doc += " ";
            doc += line;
        }

        if (doc.empty()) break;

        std::cout << "  [Embedding...] ";
        std::vector<float> vec = embedEngine.getEmbedding(doc);
        milvusClient.insertDocument(vec, doc);
        docCount++;
        std::cout << "stored. (" << docCount << " total)\n\n";

        std::cout << "Add another document? (yes/done): ";
        std::string choice;
        std::getline(std::cin, choice);
        if (choice != "yes" && choice != "y") break;
        std::cout << "\n";
    }

    std::cout << "\n✓ Knowledge base ready with " << docCount << " document(s).\n";
}

void chatPhase(LocalInferenceEngine& embedEngine,
               LocalInferenceEngine& llmEngine,
               MilvusRAGClient& milvusClient) {
    std::cout << "\n── CHAT MODE ───────────────────────────────────\n";
    std::cout << "Ask questions about your documents.\n";
    std::cout << "Type 'exit' or 'quit' to stop.\n\n";

    while (true) {
        std::cout << "You > ";
        std::string query;
        std::getline(std::cin, query);

        if (query.empty()) continue;
        if (query == "exit" || query == "quit") {
            std::cout << "\nGoodbye!\n";
            break;
        }

        // Search top-k matching documents
        std::vector<float> queryVec = embedEngine.getEmbedding(query);
        auto matches = milvusClient.searchTopK(queryVec, 3);

        if (matches.empty()) {
            std::cout << "AI  > No relevant context found in knowledge base.\n\n";
            continue;
        }

        // Build context from top-k results
        std::string context;
        for (size_t i = 0; i < matches.size(); ++i) {
            context += "[" + std::to_string(i + 1) + "] " + matches[i].textPayload + "\n";
        }

        std::cout << "\n  [Context score: " << matches[0].distance << "]\n";

        // Llama 3 native chat template — prevents filler/separator output
        std::string prompt =
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful assistant. Answer using only the context provided. "
            "Be concise. Do not add separators or formatting.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "Context:\n" + context +
            "\nQuestion: " + query +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";

        std::cout << "AI  > ";
        std::string answer = llmEngine.generateResponse(prompt);
        std::cout << answer << "\n\n";
    }
}

int main() {
    llama_log_set(llama_null_log, nullptr);

    printBanner();

    const int EMBEDDING_DIM = 384;
    const std::string milvusHost = "127.0.0.1";
    const int milvusPort = 19530;
    const std::string collection = "rag_knowledge_base";

    std::cout << "[Loading embedding model...]\n";
    LocalInferenceEngine embedEngine("./models/all-MiniLM-L6-v2.gguf", true);

    std::cout << "[Loading LLM...]\n";
    LocalInferenceEngine llmEngine("./models/llama-3-8b-instruct.gguf", false);

    std::cout << "[Connecting to Milvus...]\n";
    MilvusRAGClient milvusClient(milvusHost, milvusPort, collection, EMBEDDING_DIM);
    milvusClient.prepareCollection();

    std::cout << "✓ All systems ready.\n\n";

    ingestPhase(embedEngine, milvusClient);
    chatPhase(embedEngine, llmEngine, milvusClient);

    return 0;
}