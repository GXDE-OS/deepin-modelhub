// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MODELRUNNER_H
#define MODELRUNNER_H

#include "global_header.h"

#include "modelinfo.h"
#include "modelproxy.h"
#include "backendmetaobject.h"

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <string>
#include <map>
#include <set>

#define VERSION_CHECK(major, minor, patch) ((major<<16)|(minor<<8)|(patch))

GLOBAL_BEGIN_NAMESPACE

class ModelTask
{
public:
    ModelTask(class ModelRunner *);
    virtual ~ModelTask();
    virtual void doTask() = 0;
    virtual void cancel() {}
protected:
    class ModelRunner *runner = nullptr;
};

class ModelRunner
{
public:
    explicit ModelRunner();
    ~ModelRunner();
    static ModelRunner* instance();
    void postTask(std::shared_ptr<ModelTask> task);
    bool recvTask(std::shared_ptr<ModelTask> task);
    void terminate();
    int configVersion();
    std::string chatTemplate(std::string type = "");

    void start();
    void join();

    void clear();
public:
    ModelInfoPointer modelInfo;
    BackendMetaObjectPointer backendmo;
    std::shared_ptr<InferencePlugin> backendIns;
    std::shared_ptr<ModelProxy> modelProxy;
    std::string modelFormat;
    int parallel = 1;
    std::string cacheDir;
private:
    void run();

private:
    std::queue<std::shared_ptr<ModelTask>> taskQueue;
    std::mutex taskMtx;
    std::condition_variable taskCondition;
    bool running = false;
    
    std::mutex resultMtx;
    std::condition_variable resultCondition;
    std::set<std::shared_ptr<ModelTask>> workingList;
    std::vector<std::shared_ptr<ModelTask>> resultList;

    std::mutex tplMtx;
    std::map<std::string, std::string> chatTpl;
    int cfgVer = -1;

    std::map<std::thread::id, std::shared_ptr<std::thread>> threads;
};

GLOBAL_END_NAMESPACE

#endif // MODELRUNNER_H
