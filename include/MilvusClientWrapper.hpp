#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <milvus/MilvusClient.h>
#include <milvus/Status.h>
#include <milvus/types/CollectionSchema.h>
#include <milvus/types/SearchArguments.h>
#include <milvus/types/SearchResults.h>
#include <milvus/types/IndexDesc.h>

struct MilvusQueryResult {
    float distance;
    std::string textPayload;
};

class MilvusRAGClient {
private:
    std::shared_ptr<milvus::MilvusClient> client;
    std::string collectionName;
    int dimension;

public:
    MilvusRAGClient(const std::string& host, int port,
                    const std::string& collName, int dim)
        : collectionName(collName), dimension(dim) {
        milvus::ConnectParam param(host, port);
        client = milvus::MilvusClient::Create();
        milvus::Status status = client->Connect(param);
        if (!status.IsOk())
            throw std::runtime_error("Milvus connect failed: " + status.Message());
    }

    void prepareCollection() {
        bool hasCollection = false;
        client->HasCollection(collectionName, hasCollection);

        if (hasCollection) {
            client->LoadCollection(collectionName);
            return;
        }

        milvus::CollectionSchema schema(collectionName);
        schema.AddField(milvus::FieldSchema("id", milvus::DataType::INT64,
                                             "Primary Key", true, true));
        schema.AddField(milvus::FieldSchema("embedding",
                                             milvus::DataType::FLOAT_VECTOR, "Vector")
                            .WithDimension(dimension));
        schema.AddField(milvus::FieldSchema("text", milvus::DataType::VARCHAR, "Text")
                            .WithMaxLength(2048));

        client->CreateCollection(schema);

        // FIXED: correct 3-arg constructor matching your SDK
        milvus::IndexDesc indexDesc("embedding", "embedding_hnsw_index",
                                     milvus::IndexType::HNSW,
                                     milvus::MetricType::IP);
        indexDesc.AddExtraParam("M", "16");
        indexDesc.AddExtraParam("efConstruction", "64");

        // FIXED: CreateIndex requires collection_name as first arg
        client->CreateIndex(collectionName, indexDesc);
        client->LoadCollection(collectionName);
    }

    void insertDocument(const std::vector<float>& embedding, const std::string& text) {
        std::vector<milvus::FieldDataPtr> fields;

        std::vector<std::vector<float>> vecBatch = { embedding };
        fields.push_back(std::make_shared<milvus::FloatVecFieldData>(
            "embedding", std::move(vecBatch)));

        std::vector<std::string> textBatch = { text };
        fields.push_back(std::make_shared<milvus::VarCharFieldData>(
            "text", std::move(textBatch)));

        milvus::DmlResults results;
        client->Insert(collectionName, "", fields, results);
    }

    std::vector<MilvusQueryResult> searchTopK(const std::vector<float>& queryEmb, int topK = 1) {
    milvus::SearchArguments searchArgs;
    searchArgs.SetCollectionName(collectionName);
    searchArgs.AddTargetVector("embedding", queryEmb);
    searchArgs.AddOutputField("text");
    searchArgs.SetTopK(topK);
    searchArgs.SetMetricType(milvus::MetricType::IP);
    searchArgs.AddExtraParam("ef", "64");

    milvus::SearchResults rawResults;
    milvus::Status status = client->Search(searchArgs, rawResults);

    std::vector<MilvusQueryResult> mapped;
    if (!status.IsOk()) return mapped;

    // FIXED: Results() method, SingleResult uses Scores() and OutputField()
    for (const auto& result : rawResults.Results()) {
        const auto& distances = result.Scores();
        auto textField = result.OutputField<milvus::VarCharFieldData>("text");

        for (size_t i = 0; i < distances.size(); ++i) {
            std::string txt;
            if (textField && i < textField->Data().size()) {
                txt = textField->Data()[i];
            }
            mapped.push_back({ distances[i], txt });
        }
    }
    return mapped;
}
};