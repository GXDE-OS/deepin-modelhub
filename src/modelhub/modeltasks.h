// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MODELTASKS_H
#define MODELTASKS_H

#include "modelserver.h"
#include "modelrunner.h"
#include <map>
#include <vector>
#include <list>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <string>
#include <nlohmann/json.hpp>

GLOBAL_BEGIN_NAMESPACE

class HttpTask : public ModelTask 
{
public:
    explicit HttpTask(ModelRunner *r, const std::string &req, ModelProxy *m, HttpContext *c);
protected:
    std::string reqBody;
    ModelProxy *model = nullptr;
    HttpContext *ctx = nullptr;
};

class EmbTask : public HttpTask
{
public:
    using HttpTask::HttpTask;
    void doTask() override;
};

class ChatCompletionsTask : public ModelTask
{
public:
    struct ChatHistory {
        std::optional<std::string> system = std::nullopt;
        std::optional<std::string> prompt = std::nullopt;
        std::optional<std::string> response = std::nullopt;
        std::optional<std::string> tool = std::nullopt;
    };
public:
    explicit ChatCompletionsTask(ModelRunner *r, const nlohmann::json &json_data, ModelProxy *m);
    virtual std::pair<int, std::string> initialize();
    void doTask() override;
    void cancel() override;
    virtual std::string formatPrompt();
    virtual std::string formatPromptV2();
    static bool splitThink(const std::string &input, std::string &content, std::string &think);
protected:
    virtual std::list<ChatHistory> parseMessage(const nlohmann::json &varList);
    virtual nlohmann::json parseMessageV2(const nlohmann::json &varList);
public:
    std::mutex genMtx;
    std::string text;
    std::condition_variable con;
    bool stop = false;
protected:
    nlohmann::json root;
    ModelProxy *model = nullptr;
    std::string curPrompt;
};

class ChatToolslTask : public ChatCompletionsTask
{
public:
    using ChatCompletionsTask::ChatCompletionsTask;
    void doTask() override;
    std::string formatPrompt() override;
    std::string formatPromptV2() override;
protected:
    std::string functionCalls();
    std::string toolCall(const std::string &in);
};

class ChatStreamTask : public ChatCompletionsTask
{
public:
    using ChatCompletionsTask::ChatCompletionsTask;
    void doTask() override;
public:
    int first = true;
    bool thinking = false;
};

class DslTask : public ChatCompletionsTask
{
public:
    using ChatCompletionsTask::ChatCompletionsTask;
    void doTask() override;
    std::string formatPrompt() override;
    std::string formatPromptV2() override;
};

class CompletionsCheckTask : public ChatCompletionsTask
{
public:
    using ChatCompletionsTask::ChatCompletionsTask;
    std::pair<int, std::string> initialize() override;
    void doTask() override;
    std::string formatPrompt() override;
    std::string formatPromptV2() override;
};

class StreamCompletionsCheckTask : public CompletionsCheckTask
{
public:
    using CompletionsCheckTask::CompletionsCheckTask;
    ~StreamCompletionsCheckTask();
    void doTask() override;
};

class ASRTask : public ModelTask
{
public:
    explicit ASRTask(ModelRunner *r, const std::string &content, const std::string &type, ModelProxy *m);
public:
    std::mutex genMtx;
    std::string text;
    std::condition_variable con;
    bool stop = false;
protected:
    void doTask() override;
protected:
    std::string input;
    std::string format;
    ModelProxy *model = nullptr;
};

GLOBAL_END_NAMESPACE

#endif // MODELTASKS_H
