// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef LLAMASERVER_H
#define LLAMASERVER_H

#include "llamamodelwrapper.h"

#include "llama.h"
#include "sampling.h"

#include <map>
#include <memory>
#include <deque>
#include <condition_variable>
#include <atomic>
#include <thread>

GLOBAL_BEGIN_NAMESPACE

class LlamaServer : public LlamaModelWrapper
{
public:
    enum TaskState{Waiting = 0, Cancel, Generating, Completed};
    struct SeverTask
    {
        int64_t id = -1;
        int slot = -1;
        std::atomic_int state = Waiting;
        int32_t n_predict = -1;

        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> predicted_tokens;
        std::mutex pushMtx;
        std::condition_variable conditionPush;
    };

    struct SeverCache {
        const int id;
        const std::string cacheFile;
        int64_t update_time = 0;
        std::vector<int32_t> cached_tokens;

        SeverCache(int idx, const std::string &dir) : id(idx), cacheFile(makeCacheFile(dir, idx)) {}
        void clear();
        static std::string makeCacheFile(const std::string &dir, int id);
        static int tokenLcp(const std::vector<int32_t> &input, const std::vector<int32_t> &cacheTokens);
        static bool scoreCache(int input, int cached, int matched, float &score);
    };

    struct SeverSlot
    {
        const int id;
        bool suspend = false;
        std::shared_ptr<SeverTask> task;
        int32_t n_past = 0;
        int32_t sampled = 0;
        int i_batch = 0;

        bool prompt_processed = false;

        int64_t act_time = 0;

        common_sampler *smpl = nullptr;

        SeverSlot(int idx) : id(idx) {}
        std::vector<int32_t> processed_tokens() const;
    };

public:
    explicit LlamaServer(const std::string &name);
    ~LlamaServer();
    bool initialize(const std::string &bin, const std::unordered_map<std::string, std::string> &params);
    void post(std::shared_ptr<SeverTask> task, bool front = false);
    void resetSlot(std::shared_ptr<SeverSlot> slot);
    void swapOutSlot(std::shared_ptr<SeverSlot> slot);
    bool enabled();
protected:
    std::vector<int32_t> cachedTokens(std::shared_ptr<SeverSlot> slot);
    bool exportCache(std::shared_ptr<SeverSlot> slot, std::shared_ptr<SeverCache> cache);
    bool restoreCache(std::shared_ptr<SeverSlot> slot, std::shared_ptr<SeverCache> cache);
    std::shared_ptr<SeverSlot> findSlotCache(const std::vector<int32_t> &prompt, int *len = nullptr);
    std::shared_ptr<SeverCache> findExtCache(const std::vector<int32_t> &prompt, int *len = nullptr);
    void applyCache(std::shared_ptr<SeverSlot> slot, const std::vector<int32_t> &prompt) ;
    void saveCache(std::shared_ptr<SeverSlot> slot);
    virtual void run();
    virtual void updateSlot();
protected:
    std::string modelName;
    std::map<int, std::shared_ptr<SeverSlot>> srvSlots;
    std::deque<std::shared_ptr<SeverTask>> queueTasks;
    std::vector<std::shared_ptr<SeverCache>> srvCaches;
    std::mutex mutexTasks;
    std::condition_variable conditionTasks;
    llama_batch srvBatch;
    int ctxSize;
    volatile bool running = false;
    std::thread workThread;
};

GLOBAL_END_NAMESPACE

#endif // LLAMASERVER_H
