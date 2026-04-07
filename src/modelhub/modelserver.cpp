// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "modelserver.h"
#include "httpserver.h"
#include "runtimestate.h"
#include "modelproxy.h"
#include "embeddingproxy.h"
#include "llmproxy.h"
#include "modeltasks.h"
#include "asrproxy.h"

#include <nlohmann/json.hpp>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ctime>
#include <unistd.h>

using json = nlohmann::json;
GLOBAL_USE_NAMESPACE

// Timer implementation
Timer::Timer() {}

Timer::~Timer() {
    stop();
}

void Timer::setInterval(int milliseconds) {
    interval = milliseconds;
}

void Timer::setSingleShot(bool single) {
    singleShot = single;
}

void Timer::start() {
    std::lock_guard<std::mutex> lock(mutex);
    if (running) return;
    
    running = true;
    thread = std::make_unique<std::thread>(&Timer::timerThread, this);
}

void Timer::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        running = false;
    }
    cv.notify_one();
    if (thread && thread->joinable()) {
        thread->join();
    }
}

void Timer::setCallback(Callback cb) {
    callback = std::move(cb);
}

bool Timer::isActive() const {
    return running;
}

void Timer::timerThread() {
    while (running) {
        std::unique_lock<std::mutex> lock(mutex);
        if (cv.wait_for(lock, std::chrono::milliseconds(interval), 
            [this] { return !running; })) {
            break;
        }
        
        if (callback) callback();
        
        if (singleShot) {
            running = false;
            break;
        }
    }
}

static ModelServer *gInstance = nullptr;

void ModelServer::exitServer()
{
    if (gInstance && gInstance->runner->parallel > 1 && !gInstance->runner->cacheDir.empty()) {
        std::cerr << "safely exit server...." << std::endl;
        gInstance->runner->terminate();
        gInstance->stop();
    } else {
        std::cerr << "force exit server!" << std::endl;
        exit(0);
    }
}

// ModelServer implementation
ModelServer::ModelServer() {
    gInstance = this;
}

ModelServer::~ModelServer() {
    gInstance = nullptr;
    stop();
}

void ModelServer::run(ModelRunner* mr) {
    if (!mr || stateFile)
        return;

    runner = mr;
    RuntimeState::mkpath();
    state = std::make_unique<RuntimeState>(mr->modelProxy->name());
    
    stateFile = std::make_unique<std::fstream>(state->stateFile(), 
        std::ios::out | std::ios::trunc);
    if (!stateFile->is_open()) {
        std::cerr << "Cannot create state file " << state->stateFile() << std::endl;
        state.reset();
        stateFile.reset();
        return ;
    }

    http = std::make_unique<HttpServer>();

    const std::string host = uHost.empty() ? "127.0.0.1" : uHost;
    const int port = uPort > 0 ? uPort : HttpServer::randomPort();

    if (!http->initialize(host, port))
        return;

    http->setRequestCallback([this](void* ctx) {
        this->onRequest(ctx);
    });

    {
        std::map<std::string, std::string> st;
        st["model"] = mr->modelProxy->name();
        st["pid"] = std::to_string(getpid());
        st["host"] = host;
        st["port"] = std::to_string(port);
        
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        st["starttime"] = ss.str();
        
        state->writeState(*stateFile, st);
    }

    http->registerAPI(HttpServer::Post, "/embeddings");
    http->registerAPI(HttpServer::Post, "/chat/completions");
    http->registerAPI(HttpServer::Post, "/v1/chat/completions");
    http->registerAPI(HttpServer::Post, "/dsl");
    http->registerAPI(HttpServer::Get, "/stop");
    http->registerAPI(HttpServer::Post, "/chat/completions_check");
    http->registerAPI(HttpServer::Post, "/audio/transcriptions");

    resetIdle();

    runner->start();
    http->run();

    // remove cache file.
    runner->clear();
    return ;
}

void ModelServer::setIdle(int s) {
    if (s < 10 && idleTimer) {
        std::cerr << "idle time is invalid:" << s << std::endl;
        idleTimer.reset();
        return;
    }

    if (!idleTimer) {
        idleTimer = std::make_unique<Timer>();
        idleTimer->setSingleShot(true);
        idleTimer->setCallback([this]() { this->onIdle(); });
    }
    
    idleTimer->setInterval(s * 1000);
}

void ModelServer::resetIdle() {
    if (http)
        http->setLastWorkTime(
            std::chrono::system_clock::to_time_t(
                std::chrono::system_clock::now()));

    if (idleTimer)
        idleTimer->start();
}

void ModelServer::stopIdle() {
    if (http)
        http->setLastWorkTime(-1);

    if (idleTimer)
        idleTimer->stop();
}

void ModelServer::stop() {
    if (stateFile) {
        stateFile->close();
        std::remove(state->stateFile().c_str());
        stateFile.reset();
    }

    state.reset();
    http.reset();
}

int ModelServer::instance(const std::string& model) {
    if (model.empty())
        return ::getpid();

    RuntimeState st(model);
    return st.pid();
}

void ModelServer::onRequest(void* ptr) {
    // 停止空闲计时
    stopIdle();

    HttpContext* ctx = static_cast<HttpContext*>(ptr);
    const std::string api = HttpServer::getPath(ctx);
    const std::string reqBody = HttpServer::getBody(ctx);

    if (api == "/stop") {
        // Perform cleanup and graceful shutdown
        exitServer(); // Clean up resources
    } else if (api == "/embeddings") {
        if (auto emb = dynamic_cast<EmbeddingProxy*>(runner->modelProxy.get())) {
            std::shared_ptr<ModelTask> task(new EmbTask(runner, reqBody, emb, ctx));
            runner->postTask(task);
            runner->recvTask(task);
        } else {
            HttpServer::setStatus(ctx, 403);
            nlohmann::json obj;
            obj["invalid_request_error"] = "The API is not supported by model " + runner->modelProxy->name();
            HttpServer::setContent(ctx, obj.dump());
        }
    } else if (api == "/chat/completions" || api == "/v1/chat/completions") {
        if (auto llm = dynamic_cast<LLMProxy*>(runner->modelProxy.get())) {
            try {
                auto root = nlohmann::json::parse(reqBody);
                if (checkUseTool(root)) {
                    std::shared_ptr<ChatCompletionsTask> task(new ChatToolslTask(runner, root, llm));
                    chatCompletions(task, ctx);
                    return;
                } else {
                    if (root.value("stream", false)) {
                        chatStream(root, llm, ctx);
                        return;
                    } else {
                        std::shared_ptr<ChatCompletionsTask> task(new ChatCompletionsTask(runner, root, llm));
                        chatCompletions(task, ctx);
                        return;
                    }
                }
            } catch (const nlohmann::json::exception& e) {
                HttpServer::setStatus(ctx, 403);
                nlohmann::json obj;
                obj["invalid_request_error"] = "Invalid input content";
                HttpServer::setContent(ctx, obj.dump());
            }
        } else {
            HttpServer::setStatus(ctx, 500);
            nlohmann::json obj;
            obj["error"] = "API /chat/completions not supported by model " + runner->modelProxy->name();
            HttpServer::setContent(ctx, obj.dump());
        }
    } else if (api == "/dsl") {
        bool supported = false;
        if (auto llm = dynamic_cast<LLMProxy*>(runner->modelProxy.get())) {
            if (!runner->chatTemplate("dsl").empty()) {
                supported = true;
                try {
                    auto root = nlohmann::json::parse(reqBody);
                    std::shared_ptr<ChatCompletionsTask> task(new DslTask(runner, root, llm));
                    chatCompletions(task, ctx);
                    return;
                } catch (const nlohmann::json::exception& e) {
                    HttpServer::setStatus(ctx, 403);
                    nlohmann::json obj;
                    obj["invalid_request_error"] = "Invalid input content";
                    HttpServer::setContent(ctx, obj.dump());
                }
            }
        }

        if (!supported) {
            HttpServer::setStatus(ctx, 500);
            nlohmann::json obj;
            obj["error"] = "API /dsl not supported by model " + runner->modelProxy->name();
            HttpServer::setContent(ctx, obj.dump());
        }
    } else if (api == "/chat/completions_check") {
        if (auto llm = dynamic_cast<LLMProxy *>(runner->modelProxy.get())) {
            try {
                auto root = nlohmann::json::parse(reqBody);
                if (root.value("stream", false)) {
                    chatCheckStream(root, llm, ctx);
                    // must be return
                    return;
                } else {
                    std::shared_ptr<ChatCompletionsTask> task(new CompletionsCheckTask(runner, root, llm));
                    chatCompletions(task, ctx);
                    return;
                }
            } catch (const nlohmann::json::exception& e) {
                HttpServer::setStatus(ctx, 200);
                json obj;
                obj["content"] = "";
                obj["finish_reason"] = "stop";
                obj["status"] = "failed";
                obj["reason"] = "Invalid input content";
                HttpServer::setContent(ctx, obj.dump());
            }
        } else {
            HttpServer::setStatus(ctx, 200);
            json obj;
            obj["content"], "";
            obj["finish_reason"] ="stop";
            obj["status"] = "failed";
            obj["reason"] = "API /chat/completions_check not supported by model " + runner->modelProxy->name();
            HttpServer::setContent(ctx, obj.dump());
        }
    } else if (api == "/audio/transcriptions") {
        if (auto asrModel = dynamic_cast<ASRProxy *>(runner->modelProxy.get())) {
            try {
                std::string fileData = HttpServer::getFileContent(ctx, "file");
                std::string fileType = HttpServer::getFileType(ctx, "file");
                std::shared_ptr<ASRTask> task(new ASRTask(runner, fileData, fileType, asrModel));
                asr(task, ctx);
                return;
            } catch (const nlohmann::json::exception& e) {
                HttpServer::setStatus(ctx, 403);
                nlohmann::json obj;
                obj["invalid_request_error"] = "Invalid input content";
                HttpServer::setContent(ctx, obj.dump());
            }
        } else {
            HttpServer::setStatus(ctx, 500);
            nlohmann::json obj;
            obj["error"] = "API /audio/transcriptions not supported by model " + runner->modelProxy->name();
            HttpServer::setContent(ctx, obj.dump());
        }
    } else {
        HttpServer::setStatus(ctx, 403);
        nlohmann::json obj;
        obj["invalid_request_error"] = "The API is not open";
        HttpServer::setContent(ctx, obj.dump());
    }

    resetIdle();
}

void ModelServer::onIdle() {
    std::cerr << "server exit by idle" << std::endl;
    std::exit(0);
}

void ModelServer::chatCompletions(std::shared_ptr<ChatCompletionsTask> task, HttpContext *ctx)
{
    {
        auto init = task->initialize();
        if (init.first != 0) {
            HttpServer::setStatus(ctx, init.first);
            HttpServer::setContent(ctx, init.second);
            this->resetIdle();
            return;
        }
    }

    runner->postTask(task);
    auto chunk = [task](std::string& out, bool& stop) {
        ChatCompletionsTask *cst = task.get();
        std::unique_lock<std::mutex> lk(cst->genMtx);
        if (!cst->stop)
            cst->con.wait(lk);

        out = cst->text;
        cst->text.clear();
        stop = cst->stop;
        lk.unlock();
        return true;
    };

    auto onComplete = [this, task](bool x) {
        ChatCompletionsTask* cst = task.get();
        {
            std::unique_lock<std::mutex> lk(cst->genMtx);
            cst->stop = true;
        }

        runner->recvTask(task);
        this->resetIdle();
    };

    HttpServer::setChunckProvider(ctx, chunk, onComplete);
}

void ModelServer::chatStream(const nlohmann::json& root, LLMProxy* llm, HttpContext* ctx)
{
    std::shared_ptr<ChatStreamTask> task(new ChatStreamTask(runner, root, llm));
    {
        auto init = task->initialize();
        if (init.first != 0) {
            HttpServer::setStatus(ctx, init.first);
            HttpServer::setContent(ctx, init.second);
            this->resetIdle();
            return;
        }
    }

    runner->postTask(task);
    auto chunk = [task](std::string& out, bool& stop) {
        ChatStreamTask* cst = task.get();
        std::unique_lock<std::mutex> lk(cst->genMtx);
        if (!cst->stop)
            cst->con.wait(lk);

        std::string buff = cst->text;
        cst->text.clear();
        stop = cst->stop;

        // deepseek think
        std::string think;
        if (!buff.empty()) {
            ModelServer::deepseekr1(cst, buff, think);
            cst->first = false;
        }

        lk.unlock();

        if (buff.empty() && think.empty() && !stop)
            return true;

        nlohmann::json response;
        response["finish_reason"] = stop ? "stop" : "";

        nlohmann::json choice;
        choice["index"] = 0;
        choice["delta"]["role"] = "assistant";
        choice["delta"]["content"] = buff;

        if (!think.empty())
            choice["delta"]["reasoning_content"] = think;

        response["choices"] = nlohmann::json::array({choice});
        out = "data:" + response.dump() + "\n";
        return true;
    };

    auto onComplete = [this, task](bool) {
        ChatStreamTask* cst = task.get();
        {
            std::unique_lock<std::mutex> lk(cst->genMtx);
            cst->stop = true;
        }

        runner->recvTask(task);
        this->resetIdle();
    };

    HttpServer::setChunckProvider(ctx, chunk, onComplete);
}

void ModelServer::chatCheckStream(const nlohmann::json &root, LLMProxy *llm, HttpContext *ctx)
{
    std::shared_ptr<StreamCompletionsCheckTask> task(new StreamCompletionsCheckTask(runner, root, llm));
    task->initialize();
    runner->postTask(task);

    auto chunk = [task](std::string &out, bool &stop) {
        StreamCompletionsCheckTask *cst = task.get();

        std::unique_lock<std::mutex> lk(cst->genMtx);
        if (!cst->stop)
            cst->con.wait(lk);

        std::string buff = cst->text;
        cst->text.clear();
        stop = cst->stop;
        lk.unlock();

        if (buff.empty() && !stop)
            return true;

        nlohmann::json response;
        response["content"] = "";
        response["status"] = "success";
        response["reason"] = "success";
        {
            json choice;
            choice["delta"] = buff;
            choice["finish_reason"] = stop ? "stop" : "";

            response["choices"] = json::array({choice});
        }

        out = response.dump() + "\n";
        return true;
    };

    auto onComplete = [this, task](bool) {
        StreamCompletionsCheckTask *cst = task.get();
        {
            std::unique_lock<std::mutex> lk(cst->genMtx);
            cst->stop = true;
        }

        //relese task
        runner->recvTask(task);
        this->resetIdle();
    };

    HttpServer::setChunckProvider(ctx, chunk, onComplete);
}

bool ModelServer::checkUseTool(const nlohmann::json &root) const
{
    if (!root.contains("tools") || root["tools"].empty())
        return false;

    if (!root.contains("messages") || root["messages"].empty())
        return false;

    const auto& lastMessage = root["messages"].back();
    if (!lastMessage.contains("role"))
        return false;

    std::string role = lastMessage["role"];
    std::transform(role.begin(), role.end(), role.begin(), ::tolower);
    return role != "tool";
}

void ModelServer::asr(std::shared_ptr<ASRTask> task, HttpContext *ctx)
{
    runner->postTask(task);
    auto chunk = [task](std::string& out, bool& stop) {
        ASRTask *cst = task.get();
        std::unique_lock<std::mutex> lk(cst->genMtx);
        if (!cst->stop)
            cst->con.wait(lk);

        out = cst->text;
        cst->text.clear();
        stop = cst->stop;
        lk.unlock();
        return true;
    };

    auto onComplete = [this, task](bool x) {
        ASRTask *cst = task.get();
        {
            std::unique_lock<std::mutex> lk(cst->genMtx);
            cst->stop = true;
        }

        runner->recvTask(task);
        this->resetIdle();
    };

    HttpServer::setChunckProvider(ctx, chunk, onComplete);
}

void ModelServer::deepseekr1(ChatStreamTask *cst, std::string &buff, std::string &think)
{
    const std::string begin_think = "<think>";
    const std::string end_think = "</think>";
    if (cst->thinking && !cst->first) {
        int end = buff.find(end_think);
        if (end != std::string::npos) {
            think = buff.substr(0, end);
            buff = buff.substr(end + end_think.size());
            cst->thinking = false;
        } else {
            think = buff;
            buff.clear();
        }
    } else if (cst->first) {
        if (buff.find(begin_think) != std::string::npos) {
            cst->thinking = true;
            int end = buff.find(end_think, begin_think.size());
            if (end == std::string::npos) {
                think = buff.substr(begin_think.size());
                buff.clear();
            } else {
                cst->thinking = false;
                think = buff.substr(begin_think.size(), end - begin_think.size());
                buff = buff.substr(end + end_think.size());
            }
        }
    }
}
