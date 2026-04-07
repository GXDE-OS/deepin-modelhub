// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "llamaserver.h"

#include "llama.h"
#include "common/common.h"

#include <iostream>
#include <algorithm>
#include <filesystem>

#include <unistd.h>
#include <assert.h>

namespace fs = std::filesystem;
GLOBAL_USE_NAMESPACE

#define CACHE_MINIMUM_COUNT 20

LlamaServer::LlamaServer(const std::string &name)
    : LlamaModelWrapper()
    , modelName(name)
{

}

LlamaServer::~LlamaServer()
{
    running = false;
    {
        std::unique_lock<std::mutex> lock(mutexTasks);
        conditionTasks.notify_all();
    }

    if (workThread.joinable())
        workThread.join();

    if (enabled())
        llama_batch_free(srvBatch);
}

bool LlamaServer::initialize(const std::string &bin, const std::unordered_map<std::string, std::string> &params)
{
    auto tparams = params;
    std::string tempDir;
    {
        auto it = tparams.find("--temp-cache-dir");
        if (it != tparams.end()) {
            tempDir= it->second;
            tparams.erase(it);
        }
    }

    if (!LlamaModelWrapper::initialize(bin, tparams))
        return false;

    if (!enabled())
        return true;

    std::cerr << "server cache dir:" << tempDir << std::endl;
    if (!tempDir.empty() && !fs::exists(tempDir)) {
        if (!fs::create_directories(tempDir)) {
            std::cerr << "can not create cache dir " << tempDir << std::endl;
            tempDir.clear();
        }
    }

    ctxSize = llama_n_ctx(gCtx) - gParams->n_parallel * 4;
    for (int i = 0; i < gParams->n_parallel; i++) {
        std::shared_ptr<SeverSlot> slot(new SeverSlot(i));
        srvSlots.insert(std::make_pair(i, slot));
        if (!tempDir.empty())
            srvCaches.push_back(std::shared_ptr<SeverCache>(new SeverCache(i, tempDir)));
    }
    srvBatch = llama_batch_init(llama_n_batch(gCtx), 0, 1);
    workThread = std::thread(&LlamaServer::run, this);
    return true;
}

void LlamaServer::post(std::shared_ptr<SeverTask> task, bool front)
{
    std::unique_lock<std::mutex> lock(mutexTasks);
    if (front) {
        queueTasks.push_front(std::move(task));
    } else {
        queueTasks.push_back(std::move(task));
    }
    conditionTasks.notify_one();
}

void LlamaServer::resetSlot(std::shared_ptr<LlamaServer::SeverSlot> slot)
{
    saveCache(slot);

    slot->task.reset();
    slot->suspend = false;
    slot->n_past = 0;
    slot->sampled = 0;
    slot->prompt_processed = false;
    slot->act_time = 0;

    llama_kv_cache_seq_rm(gCtx, slot->id, -1, -1);

    common_sampler_free(slot->smpl);
    slot->smpl = nullptr;
}

void LlamaServer::swapOutSlot(std::shared_ptr<LlamaServer::SeverSlot> slot)
{
    saveCache(slot);

    slot->suspend = true;
    slot->n_past = 0;
    slot->sampled = 0;
    slot->prompt_processed = false;

    llama_kv_cache_seq_rm(gCtx, slot->id, -1, -1);

    common_sampler_free(slot->smpl);
    slot->smpl = nullptr;
}

bool LlamaServer::enabled()
{
    return gParams->n_parallel > 1;
}

std::vector<int32_t> LlamaServer::cachedTokens(std::shared_ptr<LlamaServer::SeverSlot> slot)
{
    auto cached = slot->processed_tokens();
    if (cached.empty())
        return {};

    //! check this when update llama.cpp
    //! the slot may in batch and do not deocde.
    //! so it's processed tokens is fake.
    int cache_num = llama_kv_cache_seq_pos_max(gCtx, slot->id);
    cache_num = cache_num == 0 ? 0 : cache_num + 1;
    if (cache_num < cached.size()) {
#ifndef NDEBUG
       std::cerr << "slot " << slot->id << " processed tokens: " << cached.size() << " cached: " << cache_num << std::endl;
#endif
       cached.resize(cache_num);
    }

    return cached;
}

bool LlamaServer::exportCache(std::shared_ptr<LlamaServer::SeverSlot> slot, std::shared_ptr<SeverCache> cache)
{
    auto tokens = cachedTokens(slot);
    if (tokens.empty())
        return false;

#ifndef NDEBUG
    int64_t begin = ggml_time_ms();
#endif

    cache->cached_tokens = tokens;
    auto cacheSize = llama_state_seq_save_file(gCtx, cache->cacheFile.c_str(), slot->id, tokens.data(), tokens.size());

    if (cacheSize < 1) {
        fprintf(stderr, "error: slot %d fail to save cache to %d %s\n", slot->id, cache->id, cache->cacheFile);
        cache->clear();
        return false;
    }

    cache->update_time = ggml_time_ms();
#ifndef NDEBUG
    fprintf(stderr, "slot %d export cache %.2f MB to %d, tokens:%d, time: %dms\n", slot->id, cacheSize / float(1024 * 1024),
            cache->id, tokens.size(), ggml_time_ms() - begin);
#endif

    return true;
}

bool LlamaServer::restoreCache(std::shared_ptr<LlamaServer::SeverSlot> slot, std::shared_ptr<SeverCache> cache)
{
    if (cache->cached_tokens.empty())
        return false;

    std::vector<int32_t> tokens(cache->cached_tokens.size());
    size_t read_count = 0;
#ifndef NDEBUG
    int64_t begin = ggml_time_ms();
#endif
    auto size = llama_state_seq_load_file(gCtx, cache->cacheFile.c_str(), slot->id, tokens.data(), tokens.size(), &read_count);
    bool ok = cache->cached_tokens.size() == read_count && size > 0;
    if (!ok) {
        fprintf(stderr, "error: slot %d fail to restore cache from %d, need token %d, read token %d\n",
                slot->id, cache->id, cache->cached_tokens.size(), read_count);
        llama_kv_cache_seq_rm(gCtx, slot->id, -1, -1);
    } else {
#ifndef NDEBUG
        fprintf(stderr, "slot %d restore cache %.2f MB from %d, tokens:%d, time: %dms\n", slot->id, size / float(1024 * 1024), cache->id,
                read_count, ggml_time_ms() - begin);
#endif
    }

    cache->clear();
    return ok;
}

std::shared_ptr<LlamaServer::SeverSlot> LlamaServer::findSlotCache(const std::vector<int32_t> &prompt, int *len)
{
    std::shared_ptr<LlamaServer::SeverSlot> s;
    int last = 0;
    for (auto it = srvSlots.begin(); it != srvSlots.end(); ++it) {
        SeverSlot *slot = it->second.get();
        if (slot->task.get() == nullptr || slot->suspend)
            continue;

        auto cached = cachedTokens(it->second);
        int match = SeverCache::tokenLcp(prompt, cached);

        if (match > last) {
            last = match;
            s = it->second;
        }
    }

    if (len)
        *len = last;

    return s;
}

std::shared_ptr<LlamaServer::SeverCache> LlamaServer::findExtCache(const std::vector<int32_t> &prompt, int *len)
{
    std::shared_ptr<LlamaServer::SeverCache> s;
    int last = 0;
    float score = 0;
    for (auto it = srvCaches.begin(); it != srvCaches.end(); ++it) {
        SeverCache *cache = it->get();
        if (cache->cached_tokens.empty())
            continue;

        int match = SeverCache::tokenLcp(prompt, cache->cached_tokens);
        float ts = 0;
        if (SeverCache::scoreCache(prompt.size(), cache->cached_tokens.size(), match, ts)) {
            if (ts > score) {
                last = match;
                s = *it;
                score = ts;
            }
        }
    }

    if (len)
        *len = last;

    return s;
}

void LlamaServer::applyCache(std::shared_ptr<LlamaServer::SeverSlot> slot, const std::vector<int32_t> &prompt)
{
    // find cache.
    int len_ext = 0;
    auto cacheExt = findExtCache(prompt, &len_ext);
    if (len_ext > 1) {
        bool ok = restoreCache(slot, cacheExt);
        if (ok) {
            if (prompt.size() == len_ext)
                len_ext = prompt.size() - 1;

            slot->n_past = len_ext;
            llama_kv_cache_seq_rm(gCtx, slot->id, len_ext, -1);
        }
    } else {
        int len_slot = 0;
        auto cacheSlot = findSlotCache(prompt, &len_slot);
        if (cacheSlot.get() == nullptr || len_slot < 1)
            return;

        if (prompt.size() == len_slot)
            len_slot = prompt.size() - 1;

        slot->n_past = len_slot;
        llama_kv_cache_seq_cp(gCtx, cacheSlot->id, slot->id, 0, len_slot);
#ifndef NDEBUG
        std::cerr << "slot " << slot->id << " copy cache from slot " << cacheSlot->id << " past:"
                  << len_slot  << " sucessed: " << llama_kv_cache_seq_pos_max(gCtx, slot->id) << "(+1)"<< std::endl;
#endif

    }
}

void LlamaServer::saveCache(std::shared_ptr<LlamaServer::SeverSlot> slot)
{
    if (slot->n_past < CACHE_MINIMUM_COUNT)
        return;

    std::shared_ptr<LlamaServer::SeverCache> ext;
    for (auto it = srvCaches.begin(); it != srvCaches.end(); ++it) {
        SeverCache *p = it->get();
        if (p->cached_tokens.empty()) {
            exportCache(slot, *it);
            return;
        }

        if (ext.get() == nullptr)
            ext = *it;
        else {
            if (ext->update_time > p->update_time)
                ext = *it;
        }
    }

    if (ext)
        exportCache(slot, ext);
}

void LlamaServer::run()
{
    running = true;
    while (running) {
     {
         std::unique_lock<std::mutex> lock(mutexTasks);
         if (queueTasks.empty()) {
             llama_kv_cache_clear(gCtx);
             conditionTasks.wait(lock);

             if (!running)
                 return;
         }

         // remove canceled and completed task
         for (auto it = queueTasks.begin(); it != queueTasks.end();) {
            SeverTask *task = it->get();
            if (task->state == Cancel || task->state == Completed) {
                auto itslot = srvSlots.find(task->slot);
                if (itslot != srvSlots.end()) {
                    resetSlot(itslot->second);
                }

                task->conditionPush.notify_one();
#ifndef NDEBUG
                std::cerr << "remove task " << task->id << " reason " << task->state << std::endl;
#endif
                it = queueTasks.erase(it);
            } else {
                ++it;
            }
         }

         // find slot to task
         for (auto it = queueTasks.begin(); it != queueTasks.end(); ++it) {
            SeverTask *task = it->get();
            if (task->state == Waiting) {
                for (auto its = srvSlots.begin(); its != srvSlots.end(); ++its) {
                    SeverSlot *slot = its->second.get();
                    if (slot->task.get() == nullptr) {
                        auto old = task->state.exchange(Generating);
                        if (old == Cancel) {
                            task->state = Cancel;
#ifndef NDEBUG
                            std::cerr << "task is canceled on geting slot." << task->id << std::endl;
#endif
                            continue;
                        }

                        task->slot = slot->id;
                        slot->task = *it;
                        slot->act_time = ggml_time_ms();
#ifndef NDEBUG
                        std::cerr << "put task " << task->id << " to slot " << task->slot << std::endl;
#endif
                        break;
                    }
                }

                // no idle slot;
                if (task->slot < 0)
                    break;
            }
         }
     }

        updateSlot();
    }
}

void LlamaServer::updateSlot()
{
    // check ctx size
    std::vector<std::shared_ptr<SeverSlot>> actSlots;
    {
        int total_used = 0;
        for (auto it = srvSlots.begin(); it != srvSlots.end(); ++it) {
            SeverSlot *slot = it->second.get();
            if (slot->task.get() == nullptr)
                continue;

            actSlots.push_back(it->second);
            total_used += slot->n_past;
        }

        std::stable_sort(actSlots.begin(), actSlots.end(), [](const std::shared_ptr<SeverSlot> &t1,
                                    const std::shared_ptr<SeverSlot> &t2) {
            return t1->act_time < t2->act_time;
        });
#ifndef NDEBUG
        std::cerr << "all slots used:" << total_used << ", max:" << ctxSize << std::endl;
#endif

        int stop = -1;
        for (int i = 0; i < actSlots.size(); ++i) {
            if (i == stop)
                break;

            SeverSlot *slot = actSlots[i].get();
            int need = slot->task->predicted_tokens.size() + slot->task->prompt_tokens.size() - slot->n_past;
            if (total_used + need > ctxSize) {
                int canfree = 0;
                for (int j = actSlots.size() - 1; j > i; --j) {
                    SeverSlot *last = actSlots[j].get();
                    canfree += last->n_past;
                }

                if (total_used + need - canfree > ctxSize) {
                    if (i == 0) { // never get in.
                        std::cerr << "error: the first slot has no enough space, that is a bug!" << std::endl;
                        abort();
                    }
#ifndef NDEBUG
                    std::cerr << "slot id " << slot->id  << " num:" << i
                              << " no enough space need:" << need  << " total_used:"<< total_used
                              << " suspend: " << slot->id;
#endif
                    for (int j = actSlots.size() - 1; j > i; --j) {
                        SeverSlot *last = actSlots[j].get();
                        last->suspend = true;
#ifndef NDEBUG
                        std::cerr << " " << last->id;
#endif
                    }
#ifndef NDEBUG
                    std::cerr << std::endl;
#endif
                    slot->suspend = true;
                    break;
                } else {
                    for (int j = actSlots.size() - 1; j > i; --j) {
                        SeverSlot *last = actSlots[j].get();
                        int released = last->n_past;
#ifndef NDEBUG
                        std::cerr << "swap out slot:" << last->id << " free:" << released << std::endl;
#endif
                        swapOutSlot(actSlots[j]);

                        stop = j;
                        total_used -= released;

                        if (total_used + need <= ctxSize)
                            break;
                    }
                }
            }

            total_used += need;
            slot->suspend = false;
        }
    }

    common_batch_clear(srvBatch);
    int32_t n_batch  = llama_n_batch(gCtx);

    for (auto it = actSlots.begin(); it != actSlots.end(); ++it) {
        SeverSlot *slot = it->get();
        if (slot->suspend || !slot->prompt_processed)
            continue;

        slot->i_batch = srvBatch.n_tokens;
        common_batch_add(srvBatch, slot->sampled, slot->n_past, { slot->id }, false);

        slot->n_past += 1;
        srvBatch.logits[slot->i_batch] = true;
#ifndef NDEBUG
        std::cerr << "slot " << slot->id << " doing task " << slot->task->id
                  << " past:" << slot->n_past << " batch index:" << slot->i_batch << std::endl;
#endif
    }

    for (auto it = actSlots.begin(); it != actSlots.end(); ++it) {
        SeverSlot *slot = it->get();
        if (slot->prompt_processed || slot->suspend)
            continue;

        std::vector<int32_t> need_process = slot->task->prompt_tokens;
        need_process.insert(need_process.end(), slot->task->predicted_tokens.begin(), slot->task->predicted_tokens.end());

        if (slot->n_past < 1) {
            applyCache(*it, need_process);
        }
#ifndef NDEBUG
        std::cerr << "slot " << slot->id << " prepering task " << slot->task->id
                  << " prompt_tokens:" << slot->task->prompt_tokens.size()
                  << " predicted_tokens:" << slot->task->predicted_tokens.size()
                  << " past:" << slot->n_past << std::endl;
#endif

        while (slot->n_past < need_process.size() && srvBatch.n_tokens < n_batch) {
            slot->i_batch = srvBatch.n_tokens;
            common_batch_add(srvBatch, need_process[slot->n_past], slot->n_past, { slot->id }, false);
            slot->n_past += 1;
        }

        if (slot->n_past == need_process.size()) {
            slot->smpl = common_sampler_init(gModel, gParams->sampling);
            for (int i = 0; i < need_process.size(); ++i)
                common_sampler_accept(slot->smpl, need_process[i], false);

            srvBatch.logits[slot->i_batch] = true;
            slot->prompt_processed = true;
#ifndef NDEBUG
            std::cerr << "prompt loaded " << slot->n_past << " batch index:" << slot->i_batch << std::endl;
#endif
        } else {
#ifndef NDEBUG
            std::cerr << "no enough batch to load prompt,past:" << slot->n_past << " total:" << need_process.size() << std::endl;
#endif
        }
    }

    if (srvBatch.n_tokens < 1)
        return;

    int ret = llama_decode(gCtx, srvBatch);
    if (ret != 0) {
        if (ret == 1) {
            llama_kv_cache_defrag(gCtx);
            ret = llama_decode(gCtx, srvBatch);
            std::cerr << "decode error: can not find slot, try to defrag kv defrag and decode again:" << ret << std::endl;
        }

        if (ret != 0) {
            std::cerr << "decode error: " << ret << ", cache_token_count:" << llama_get_kv_cache_token_count(gCtx)
                      << " cache_used_cells:" << llama_get_kv_cache_used_cells(gCtx)
                      << " release all slot." << std::endl;
            for (auto it = actSlots.begin(); it != actSlots.end(); ++it)
                it->get()->task->state = Completed;
            return;
        }
    }

    for (auto it = actSlots.begin(); it != actSlots.end(); ++it) {
        SeverSlot *slot = it->get();
        if (slot->suspend) { // for cancel
            slot->task->conditionPush.notify_one();
            continue;
        }

        if (slot->prompt_processed) {
            const llama_token id = common_sampler_sample(slot->smpl, gCtx, slot->i_batch);
            common_sampler_accept(slot->smpl, id, true);
            slot->sampled = id;

            {
                std::unique_lock<std::mutex> lock(slot->task->pushMtx);
                slot->task->predicted_tokens.push_back(id);
            }

            if (id == llama_token_eos(gVocab)
                    || slot->n_past >= ctxSize
                    || slot->task->predicted_tokens.size() >= slot->task->n_predict) {
                slot->task->state = Completed;
            }
            slot->task->conditionPush.notify_one();
        }
    }
}

std::vector<int32_t> LlamaServer::SeverSlot::processed_tokens() const
{
    std::vector<int32_t> ret;
    if (n_past < 1 || task.get() == nullptr)
        return ret;

    if (n_past <= task->prompt_tokens.size()) {
        ret =  std::vector<int32_t>(task->prompt_tokens.begin(),
                                    task->prompt_tokens.begin() + n_past);
    } else {
        ret = task->prompt_tokens;
        int next = n_past - ret.size();
        ret.insert(ret.end(), task->predicted_tokens.begin(), task->predicted_tokens.begin() + next);
    }

    return ret;
}

void LlamaServer::SeverCache::clear()
{
    cached_tokens.clear();
    update_time = 0;
    fs::remove(cacheFile);
}

std::string LlamaServer::SeverCache::makeCacheFile(const std::string &dir, int id)
{
    return dir + "/" + std::to_string(id);
}

int LlamaServer::SeverCache::tokenLcp(const std::vector<int32_t> &input, const std::vector<int32_t> &cacheTokens)
{
    int i;
    for (i = 0; i < cacheTokens.size() && i < input.size() && cacheTokens[i] == input[i]; i++) {}

    return i;
}

bool LlamaServer::SeverCache::scoreCache(int input, int cached, int matched, float &score)
{
    score = 0;
    bool accept = false;
    if (input == 0 || cached == 0 || matched == 0)
        return accept;

    static float wMatchCount = 0.6;
    static float wMatchRate = 0.3;
    static float wReuse = 0.1;
    static float wRetry = 10;
    static int wDemarcation = 300;

    float ihr = matched / float(input);
    float chr = matched / float(cached);
    float amn = matched;
    float cur = amn / float(cached + input - amn);

    float retry_bonus = (ihr >= 0.9 && amn >= input * 0.9) ? wRetry : 0;

    score = (wMatchCount * amn) +
                 (wMatchRate * ( chr >= 0.3 ? 100 : chr * 100)) +
                 (wReuse * cur * 100) + retry_bonus;

    if (amn >= wDemarcation || chr >= wMatchRate)
        accept = true;

    return accept;
}
