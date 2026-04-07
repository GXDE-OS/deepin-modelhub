// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MODELSERVER_H
#define MODELSERVER_H

#include "global_header.h"
#include <string>
#include <memory>
#include <functional>
#include <chrono>
#include <thread>         // for std::thread
#include <mutex>         // for std::mutex
#include <condition_variable>  // for std::condition_variable
#include <nlohmann/json.hpp>  // for nlohmann::json

GLOBAL_BEGIN_NAMESPACE

class ModelProxy;
class RuntimeState;
class HttpServer;
class HttpContext;
class ModelRunner;
class LLMProxy;
class ChatCompletionsTask;
class ChatStreamTask;
class ASRTask;

// Simple timer implementation
class Timer {
public:
    using Callback = std::function<void()>;
    Timer();
    ~Timer();
    void setInterval(int milliseconds);
    void setSingleShot(bool single);
    void start();
    void stop();
    void setCallback(Callback cb);
    bool isActive() const;

private:
    void timerThread();
    std::unique_ptr<std::thread> thread;
    std::mutex mutex;
    std::condition_variable cv;
    bool running = false;
    bool singleShot = false;
    int interval = 0;
    Callback callback;
};

class ModelServer {
public:
    explicit ModelServer();
    ~ModelServer();

    void setHost(const std::string& host) {
        uHost = host;
    }
    
    void setPort(int port) {
        uPort = port;
    }

    void run(ModelRunner* mr);
    void stop();
    void setIdle(int s);
    void resetIdle();
    void stopIdle();
    static int instance(const std::string& model);

    void onRequest(void* ptr);
    void onIdle();
    static void exitServer();
protected:
    void chatCompletions(std::shared_ptr<ChatCompletionsTask> task, HttpContext *ctx);
    void chatStream(const nlohmann::json &root, LLMProxy *llm, HttpContext *ctx);
    void chatCheckStream(const nlohmann::json &root, LLMProxy *llm, HttpContext *ctx);
    bool checkUseTool(const nlohmann::json &root) const;
    void asr(std::shared_ptr<ASRTask> task, HttpContext *ctx);
private:
    static void deepseekr1(ChatStreamTask* cst, std::string &buff, std::string &think);
protected:
    ModelRunner* runner = nullptr;
    std::unique_ptr<std::fstream> stateFile;
    std::unique_ptr<RuntimeState> state;
    std::unique_ptr<HttpServer> http;
    std::string uHost;
    int uPort = -1;
    std::unique_ptr<Timer> idleTimer;
};

GLOBAL_END_NAMESPACE

#endif // MODELSERVER_H
