// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "llamaembproxy.h"
#include "llama.h"
#include "common/common.h"

#include <memory>
#include <iostream>
#include <cstring>
#include <any>

GLOBAL_USE_NAMESPACE

// Helper function to add sequence to batch
static void batch_add_seq(llama_batch &batch, const std::vector<int32_t> &tokens, int seq_id) {
    for (size_t i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i], i, {seq_id}, i == tokens.size() - 1);
    }
}

// Helper function to decode batch and get embeddings
static void batch_decode(llama_context *ctx, llama_batch &batch, float *output, int n_seq, int n_embd) {
    // Clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);

    // Run model
    std::cerr << __func__ << ": n_tokens = " << batch.n_tokens 
              << ", n_seq = " << n_seq << std::endl;
              
    if (llama_decode(ctx, batch) < 0) {
        std::cerr << __func__ << ": failed to decode" << std::endl;
        return;
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        // Try to get sequence embeddings - supported only when pooling_type is not NONE
        const float *embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == nullptr) {
            embd = llama_get_embeddings_ith(ctx, i);
            if (embd == nullptr) {
                std::cerr << __func__ << ": failed to get embeddings for token " << i << std::endl;
                continue;
            }
        }

        float *out = output + batch.seq_id[i][0] * n_embd;
        common_embd_normalize(embd, out, n_embd, 2);
    }
}

LlamaEmbProxy::LlamaEmbProxy(const std::string &name)
    : EmbeddingProxy()
    , LlamaModelWrapper()
    , modelName(name) {}

std::string LlamaEmbProxy::name() const {
    return modelName;
}

std::list<std::vector<int32_t>> LlamaEmbProxy::tokenize(const std::list<std::string> &prompt,
                                                       const std::map<std::string, std::string> &params) {
    std::list<std::vector<int32_t>> ret;
    // Embedding token do not parse special
    for (const std::string &pmpt : prompt) {
        ret.push_back(common_tokenize(gCtx, pmpt, true, false));
    }
    return ret;
}

std::list<std::vector<float>> LlamaEmbProxy::embedding(const std::list<std::vector<int32_t>> &tokens,
                                                      const std::map<std::string, std::string> &params) {
    std::unique_lock<std::mutex> lk(gMtx);

    if (llama_pooling_type(gCtx) == LLAMA_POOLING_TYPE_NONE) {
        std::cerr << "do not support LLAMA_POOLING_TYPE_NONE" << std::endl;
        return {};
    }

    std::list<std::vector<float>> ret;
    const uint64_t n_batch = gParams->n_batch;
    const uint32_t nCtx = llama_n_ctx(gCtx);

    llama_kv_cache_clear(gCtx);

    std::list<std::vector<int32_t>> inputs = tokens;
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
        if (it->size() > n_batch) {
            std::cerr << "error: number of tokens in input line " << it->size()
                     << " exceeds batch size " << n_batch << std::endl;
            return ret;
        }

        if (it->size() > nCtx) {
            std::cerr << "error: number of tokens in input line " << it->size()
                     << " exceeds model context size " << nCtx << std::endl;
            return ret;
        }

        if (it->empty() || it->back() != llama_vocab_sep(gVocab)) {
            it->push_back(llama_vocab_sep(gVocab));
        }
    }

    const int n_prompts = inputs.size();
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // Allocate output
    const int n_embd = llama_model_n_embd(gModel);
    std::vector<float> embeddings(n_prompts * n_embd, 0);
    float *emb = embeddings.data();
    
    int k = 0;
    for (const auto &input : inputs) {
        common_batch_clear(batch);
        batch_add_seq(batch, input, 0);
        float *out = emb + k * n_embd;
        batch_decode(gCtx, batch, out, 1, n_embd);
        k++;
    }

    const int stride = n_embd * sizeof(float);
    for (int i = 0; i < n_prompts; ++i) {
        std::vector<float> tmp(n_embd);
        std::memcpy(tmp.data(), (char *)emb + i * stride, stride);
        ret.push_back(std::move(tmp));
    }

    llama_batch_free(batch);
    return ret;
}

bool LlamaEmbProxy::initialize(const std::string &bin,
                             const std::unordered_map<std::string, std::string> &params) {
    auto pcp = params;
    pcp["--embedding"] = std::string("");
    
    const int nctx = std::stoi(params.at("--ctx-size"));
    if (pcp.find("--batch-size") == pcp.end() && nctx > 0) {
        pcp["--batch-size"] = std::to_string(nctx);
    }

    if (pcp.find("--ubatch-size") == pcp.end() && nctx > 0) {
        pcp["--ubatch-size"] = std::to_string(nctx);
    }

    return LlamaModelWrapper::initialize(bin, pcp);
}
