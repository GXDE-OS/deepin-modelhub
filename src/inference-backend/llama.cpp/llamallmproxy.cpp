// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "llamallmproxy.h"
#include "llama.h"
#include "common/common.h"
#include "sampling.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <unistd.h>

GLOBAL_USE_NAMESPACE

LlamaLLMProxy::LlamaLLMProxy(const std::string &name)
    : LLMProxy()
    , LlamaServer(name) {}

std::string LlamaLLMProxy::name() const {
    return modelName;
}

std::vector<int32_t> LlamaLLMProxy::tokenize(const std::string &prompt,
                                            const std::map<std::string, std::string> &params) {
    return common_tokenize(gCtx, prompt, true, true);
}

std::string LlamaLLMProxy::detokenize(const std::vector<int32_t> &tokens,
                                     const std::map<std::string, std::string> &params) {
    std::string ret;
    for (size_t i = 0; i < tokens.size(); ++i) {
        ret += common_token_to_piece(gCtx, tokens[i], params.find("special") != params.end());
    }
    return ret;
}

std::vector<int32_t> LlamaLLMProxy::generate(const std::vector<int32_t> &tokens, const std::map<std::string, std::string> &params, generateStream stream, void *user)
{
    if (enabled())
        return generateParallel(tokens, params, stream, user);

    return generateSafe(tokens, params, stream, user);
}

std::vector<int32_t> LlamaLLMProxy::generateSafe(const std::vector<int32_t> &input,
                                            const std::map<std::string, std::string> &params,
                                            generateStream stream,
                                            void *user) {
    std::unique_lock<std::mutex> lk(gMtx);

    std::vector<int32_t> output;
    if (input.empty())
        return output;

    const int n_ctx = llama_n_ctx(gCtx);
    int predict = INT32_MAX;
    {
        auto pv = params.find("predict");
        if (pv != params.end()) {
            int value = std::stoi(pv->second);
            if (value > 0)
                predict = value;
        }
    }

    bool streamToken = params.find("stream_token") != params.end();

    if (input.size() > n_ctx - 4) {
        std::cerr << "prompt is too long (" << input.size() << " tokens, max " << (n_ctx - 4) << ")" << std::endl;
        return output;
    }

    int n_consumed = 0;
    common_sampler *ctx_sampling = common_sampler_init(gModel, gParams->sampling);

    std::vector<llama_token> embd;
    while (input.size() > n_consumed) {
        embd.push_back(input[n_consumed]);
        common_sampler_accept(ctx_sampling, input[n_consumed], false);
        ++n_consumed;
    }

    const int n_batch = llama_n_batch(gCtx);
    const int defaultSeqID = 0;

    // only a single seq_id per token is needed
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    int n_past = 0;
    LLMGenerateContext slot;

    while (true) {
        int n_eval = embd.size();
        bool decodeError = false;
        for (int32_t i = 0; i < n_eval; i += n_batch) {
            common_batch_clear(batch);

            const int32_t n_tokens = std::min(n_batch, n_eval - i);
            int j = 0;
            for (; j < n_tokens; ++j) {
                common_batch_add(batch, embd[i + j], n_past, {defaultSeqID}, false);
                n_past++;
            }

            if (i + j == n_eval)
                batch.logits[batch.n_tokens - 1] = true;

            if (llama_decode(gCtx, batch)) {
                std::cerr << "llama_decode: failed to eval" << std::endl;
                decodeError = true;
                break;
            }
        }

        if (decodeError)
            break;

        embd.clear();

        const llama_token id = common_sampler_sample(ctx_sampling, gCtx, -1);
        common_sampler_accept(ctx_sampling, id, true);

        output.push_back(id);
        embd.push_back(id);

        if (stream) {
            if (!streamToken) {
                auto send = processToken(id, slot);
                if (!send.empty()) {
                    if (!stream(send, user))
                        break;
                }
            } else {
                if (!stream(std::to_string(id), user))
                    break;
            }
        }

        if (embd.back() == llama_vocab_eos(gVocab))
           break;

        if (n_past >= n_ctx - 4)
            break;

        if (output.size() >= predict)
            break;
    }

    if (params.find("timings") != params.end()) {
        const llama_perf_context_data timings = llama_perf_context(gCtx);
        std::cerr << std::endl;
        //fprintf(stderr, "load time = %.2f ms\n", timings.t_load_ms);
        fprintf(stderr, "prompt eval time = %.2f ms / %d tokens (%.2f ms per token, %.2f tokens per second)\n",
                timings.t_p_eval_ms, timings.n_p_eval, timings.t_p_eval_ms / timings.n_p_eval, 1e3 / timings.t_p_eval_ms * timings.n_p_eval);
        fprintf(stderr, "eval time = %.2f ms / %d runs   (%.2f ms per token, %.2f tokens per second)\n",
                timings.t_eval_ms, timings.n_eval, timings.t_eval_ms / timings.n_eval, 1e3 / timings.t_eval_ms * timings.n_eval);
        //fprintf(stderr, "total time = %.2f ms / %d tokens\n", (timings.t_end_ms - timings.t_start_ms), (timings.n_p_eval + timings.n_eval));
    }

    common_sampler_free(ctx_sampling);
    llama_kv_cache_clear(gCtx);
    llama_batch_free(batch);

    return output;
}

std::vector<int32_t> LlamaLLMProxy::generateParallel(const std::vector<int32_t> &tokens, const std::map<std::string, std::string> &params,
                                             generateStream stream, void *user)
{
    std::vector<int32_t> output;
    if (tokens.empty())
        return output;

    int predict = INT32_MAX;
    {
        auto pv = params.find("predict");
        if (pv != params.end()) {
            int value = std::stoi(pv->second);
            if (value > 0)
                predict = value;
        }
    }

    bool streamToken = params.find("stream_token") != params.end();
    if (tokens.size() > ctxSize) {
        std::cerr << "prompt is too long (" << tokens.size() << " tokens, max " << ctxSize << ")" << std::endl;
        return output;
    }

    std::shared_ptr<SeverTask> task(new SeverTask);
    task->id = ggml_time_us();
    task->prompt_tokens = tokens;
    task->n_predict = predict;

    LLMGenerateContext ctx;
    auto pushFunc = [this, stream, streamToken, user, &ctx, &output](int id) {
        output.push_back(id);
        if (stream) {
            if (!streamToken) {
                auto send = processToken(id, ctx);
                if (!stream(send, user))
                    return false;
            } else {
                if (!stream(std::to_string(id), user))
                    return false;
            }
        }
        return true;
    };

    post(task);

    while (true) {
        if (task->state == Waiting) {
            if (stream) {
                if (!stream("", user)) {
                    task->state = Cancel;
#ifndef NDEBUG
                    std::cerr << "cancle task " << task->id << std::endl;
#endif
                    break;
                }
            }
            usleep(100 * 1000);
        } else if (task->state == Generating) {
            std::unique_lock<std::mutex> lock(task->pushMtx);
            task->conditionPush.wait(lock);

            std::vector<int32_t> tmp(task->predicted_tokens.begin() + output.size(), task->predicted_tokens.end());
            lock.unlock();

            bool cancel = false;
            if (tmp.empty()) {
                if (stream)
                    cancel = !stream("", user);
            } else {
                for (auto id : tmp) {
                    if (!pushFunc(id)) {
                        cancel = true;
                        break;
                    }
                }
            }

            if (cancel)
                task->state = Cancel;
        } else {
            break;
        }
    }

    if (task->state == Completed && task->predicted_tokens.size() > output.size()) {
        std::vector<int32_t> tmp(task->predicted_tokens.begin() + output.size(), task->predicted_tokens.end());
        for (auto id : tmp) {
            if (!pushFunc(id)) {
                break;
            }
        }
    }

    return task->predicted_tokens;
}

std::string LlamaLLMProxy::processToken(int32_t token, LLMGenerateContext &slot) const {
    std::string push;

    const std::string token_str = common_token_to_piece(gCtx, token, false);
    slot.generatedText += token_str;

    // check if there is incomplete UTF-8 character at the end
    bool incomplete = false;
    for (unsigned i = 1; i < 5 && i <= slot.generatedText.size(); ++i) {
        unsigned char c = slot.generatedText[slot.generatedText.size() - i];
        if ((c & 0xC0) == 0x80) {
            // continuation byte: 10xxxxxx
            continue;
        }
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character: 110xxxxx ...
            incomplete = i < 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character: 1110xxxx ...
            incomplete = i < 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character: 11110xxx ...
            incomplete = i < 4;
        }
        // else 1-byte character or invalid byte
        break;
    }

    if (!incomplete) {
        size_t pos = std::min(slot.pushedPos, slot.generatedText.size());
        push = slot.generatedText.substr(pos);
        slot.pushedPos += push.size();
    }

    return push;
}

