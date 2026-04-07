// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "modelrunner.h"

#include <iostream>
#include <sstream>
#include <filesystem>

GLOBAL_USE_NAMESPACE

static ModelRunner* kInstance = nullptr;

ModelRunner::ModelRunner()
{
    if (kInstance)
        std::abort();

    kInstance = this;
}

ModelRunner::~ModelRunner()
{
    kInstance = nullptr;
    terminate();
    join();
}

ModelRunner* ModelRunner::instance()
{
    return kInstance;
}

void ModelRunner::postTask(std::shared_ptr<ModelTask> task)
{
    if (!task)
        return;

    {
        std::lock_guard<std::mutex> lock(taskMtx);
        taskQueue.push(task);
    }

    taskCondition.notify_one();
}

bool ModelRunner::recvTask(std::shared_ptr<ModelTask> task)
{
    while (running) {
        std::unique_lock<std::mutex> lock(resultMtx);
        for (size_t i = 0; i < resultList.size(); i++) {
            if (resultList[i].get() == task.get()) {
                resultList.erase(resultList.begin() + i);
                return true;
            }
        }
        resultCondition.wait(lock);
    }

    return false;
}

void ModelRunner::terminate()
{
    {
        std::lock_guard<std::mutex> lock(taskMtx);
        for (auto it = workingList.begin(); it != workingList.end(); ++it)
            it->get()->cancel();

        if (!running)
            return;

        running = false;
    }

    taskCondition.notify_all();
}

int ModelRunner::configVersion()
{
    std::lock_guard<std::mutex> lock(tplMtx);

    int minVersion = VERSION_CHECK(0, 1, 0);
    if (cfgVer < 0) {
        auto strVer = modelInfo->version(modelFormat);
        std::cerr << "config version: " << strVer << std::endl;
        std::vector<std::string> tokens;
           std::istringstream iss(strVer);
           std::string token;
           while (std::getline(iss, token, '.')) {
               if (!token.empty())
                   tokens.push_back(token);
           }

           try {
               if (tokens.size() == 1) {
                   cfgVer = VERSION_CHECK(std::stoi(tokens[0]), 0, 0);
               } else if (tokens.size() == 2) {
                   cfgVer = VERSION_CHECK(std::stoi(tokens[0]), std::stoi(tokens[1]), 0);

               } else if (tokens.size() == 3){
                   cfgVer = VERSION_CHECK(std::stoi(tokens[0]), std::stoi(tokens[1]), std::stoi(tokens[2]));
               }
           } catch (...) {
               std::cerr << "invaild version string" << strVer << std::endl;
           }

           if (cfgVer < minVersion)
               cfgVer = minVersion;
    }

    return cfgVer;
}

std::string ModelRunner::chatTemplate(std::string type)
{
    std::lock_guard<std::mutex> lock(tplMtx);
    if (type.empty())
        type = "default";

    auto it = chatTpl.find(type);
    if (it != chatTpl.end())
        return it->second;

    auto tpl = modelInfo->chatTemplate(modelFormat, type);
    if (!tpl.empty())
        chatTpl[type] = tpl;

    return tpl;
}

void ModelRunner::start()
{
    running = true;
    for (int i = 0; i < parallel; ++i) {
        auto thread = std::make_shared<std::thread>(&ModelRunner::run, this);
        threads.insert(std::make_pair(thread->get_id(), thread));
        std::cerr << "start model task thread " << thread->get_id() << std::endl;
    }
}

void ModelRunner::join()
{
    for (auto it = threads.begin(); it != threads.end(); ++it) {
        auto th = it->second;
        if (th->joinable())
            th->join();
        std::cerr << "model task thread exited:" << th->get_id() << std::endl;
    }

    threads.clear();
}

void ModelRunner::clear()
{
    if (!cacheDir.empty()) {
#ifndef NDEBUG
        std::cerr << "rm cache dir:" << cacheDir << std::endl;
#endif
        std::filesystem::remove_all(cacheDir);
    }
}

void ModelRunner::run()
{
    while (running) {
        std::shared_ptr<ModelTask> task;
        {
            std::unique_lock<std::mutex> lock(taskMtx);
            if (task)
                workingList.erase(workingList.find(task));

            taskCondition.wait(lock, [this] { 
                return !taskQueue.empty() || !running; 
            });
            
            if (!running) break;
#ifndef NDEBUG
            std::cerr << "do task in work thread:" << std::this_thread::get_id() << std::endl;
#endif
            task = taskQueue.front();
            taskQueue.pop();

            workingList.insert(task);
        }

        if (task) {
            task->doTask();
            {
                std::lock_guard<std::mutex> lock(resultMtx);
                resultList.push_back(task);
            }
            resultCondition.notify_all();
        }
    }

    std::cerr << "ending model task loop" << std::endl;
    resultCondition.notify_all();
}

ModelTask::ModelTask(ModelRunner* r) : runner(r)
{
}

ModelTask::~ModelTask()
{
}
