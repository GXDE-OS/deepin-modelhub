// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "modeltasks.h"
#include "httpserver.h"
#include "embeddingproxy.h"
#include "llmproxy.h"
#include "asrproxy.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>
#include <regex>
#include <inja/inja.hpp>
#include <uuid/uuid.h>

using json = nlohmann::json;

GLOBAL_USE_NAMESPACE

HttpTask::HttpTask(ModelRunner *r, const std::string &req, ModelProxy *m, HttpContext *c)
    : ModelTask(r)
    , reqBody(req)
    , model(m)
    , ctx(c)
{
}

void EmbTask::doTask()
{
    EmbeddingProxy *emb = dynamic_cast<EmbeddingProxy *>(model);
    assert(emb);

    try {
        auto json_data = json::parse(reqBody);
        std::vector<std::string> prompts;
        if(json_data.contains("input") && json_data["input"].is_array()) {
            prompts = json_data["input"].get<std::vector<std::string>>();
        }

        std::string error;
        std::list<std::string> stdPrompts;
        for (size_t i = 0; i < prompts.size(); ++i) {
            size_t pmptSize = prompts[i].size();
            if (pmptSize > 5120) {
                std::stringstream ss;
                ss << "the input prompt " << i << " is too large: " << pmptSize;
                error = ss.str();
                std::cerr << error << std::endl;
                break;
            }
            stdPrompts.push_back(prompts[i]);
        }

        if (error.empty()) {
            json root;
            json array = json::array();
            
            if (!prompts.empty()) {
                auto tokens = emb->tokenize(stdPrompts);
                auto out = emb->embedding(tokens);

                int i = 0;
                for (auto it = out.begin(); it != out.end(); ++it) {
                    json embObj;
                    embObj["object"] = "embedding";
                    embObj["index"] = i++;
                    
                    json embValue = json::array();
                    for (const float &v : *it) {
                        embValue.push_back(v);
                    }
                    embObj["embedding"] = embValue;
                    array.push_back(embObj);
                }
            }

            root["data"] = array;
            root["model"] = emb->name();
            root["object"] = "list";
            
            HttpServer::setContent(ctx, root.dump());
        } else {
            HttpServer::setStatus(ctx, 403);
            json error_obj;
            error_obj["invalid_request_error"] = error;
            HttpServer::setContent(ctx, error_obj.dump());
        }
    } catch (const json::parse_error& e) {
        HttpServer::setStatus(ctx, 403);
        json error_obj;
        error_obj["invalid_request_error"] = "Invalid input content";
        HttpServer::setContent(ctx, error_obj.dump());
    }
}

ChatCompletionsTask::ChatCompletionsTask(ModelRunner *r, const json &json_data, ModelProxy *m)
    : ModelTask(r)
    , root(json_data)
    , model(m)
{
}

std::pair<int, std::string> ChatCompletionsTask::initialize()
{
    auto ver = runner->configVersion();
    if (ver >= VERSION_CHECK(0, 2, 0))
        curPrompt = formatPromptV2();
    else
        curPrompt = formatPrompt();

    if (curPrompt.empty()) {
        std::cerr << "fail to format prompt: " << ver << std::endl;
        nlohmann::json obj;
        obj["invalid_request_error"] = "Invalid input content.";
        return std::make_pair(403, obj.dump());
    }

    return std::make_pair(0, std::string());
}

void ChatCompletionsTask::doTask()
{
    LLMProxy *llm = dynamic_cast<LLMProxy *>(model);
    assert(llm);

    auto tokens = llm->tokenize(curPrompt, {});

#ifndef NDEBUG
    std::cerr << "chat token count:" << tokens.size() << " prompt:" << curPrompt << std::endl;
#endif

    auto checkEnd = [](const std::string &text, void *user) {
        ChatCompletionsTask *self = static_cast<ChatCompletionsTask *>(user);
        self->con.notify_all();
        return !self->stop;
    };

    std::map<std::string, std::string> params;
    params.insert(std::make_pair("stream_token", ""));
    {
        int max = root.value("max_tokens", -1);
        if (max > 0)
            params.insert(std::make_pair("predict", std::to_string(max)));
    }

    auto token = llm->generate(tokens, params, *checkEnd, this);
    std::string content = llm->detokenize(token, {});

    {
        std::lock_guard<std::mutex> lk(genMtx);
        if (!stop) {
#ifndef NDEBUG
    std::cerr << "token count:" << token.size() << " output:" << content << std::endl;
#endif
            std::string think;
            {
                std::string tmp;
                if (splitThink(content, tmp, think))
                    content = tmp;
            }

            json response;
            json choice;
            json message;
            message["role"] = "assistant";
            message["content"] = content;
            if (!think.empty())
                choice["reasoning_content"] = think;

            choice["index"] = 0;
            choice["message"] = message;
            response["choices"] = json::array({choice});
            response["finish_reason"] = "stop";
            text = response.dump();
            stop = true;
        } else {
#ifndef NDEBUG
    std::cerr << "chat canceled " << token.size() << " output:" << content << std::endl;
#endif
        }
    }

    con.notify_all();
}

void ChatCompletionsTask::cancel()
{
    stop = true;
}

std::string ChatCompletionsTask::formatPrompt() 
{
    auto messages = root["messages"];
    std::string prompt;
    auto temp = runner->chatTemplate();

    if (temp.empty()) {
        for (const auto& msg : messages) {
            prompt += msg["content"].get<std::string>() + "\n";
        }
        return prompt;
    }

    std::list<ChatHistory> prompts = parseMessage(messages);

    try {
        inja::Environment env;
        // add a space for line statement "##" to enable "###"
        env.set_line_statement(inja::LexerConfig().line_statement + " ");
        bool first = true;
        for (const ChatHistory &tmp: prompts) {
            inja::json json;
            if (first)
                json["System"] = tmp.system.has_value() ? tmp.system.value() : "";

            if (tmp.prompt.has_value())
                json["Prompt"] = tmp.prompt.value();

            if (tmp.response.has_value())
                json["Response"] = tmp.response.value();

            if (tmp.tool.has_value())
                json["Tool"] = tmp.tool.value();

            prompt += env.render(temp, json);
            first = false;
        }
    } catch (const std::exception &error) {
        std::cerr << "fail to format prompt, please check the template file of model " << error.what() << std::endl;
        return "";
    }

    return prompt;
}

std::string ChatCompletionsTask::formatPromptV2()
{
    inja::json json = parseMessageV2(root["messages"]);

    std::string prompt;
    auto temp = runner->chatTemplate();
    try {
        inja::Environment env;
        env.set_line_statement(inja::LexerConfig().line_statement + " ");
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);
        prompt = env.render(temp, json);
    } catch (const std::exception &error) {
        std::cerr << "fail to format prompt v2, please check the template file of model " << error.what() << std::endl;
        return "";
    }

    return prompt;
}

bool ChatCompletionsTask::splitThink(const std::string &input, std::string &content, std::string &think)
{
    const std::string begin_think = "<think>";
    const std::string end_think = "</think>";

    int pre = input.find(begin_think);
    if (pre != 0)
        return false;

    int end = input.find(end_think, begin_think.size());
    if (end == std::string::npos)
        return false;

    think = input.substr(begin_think.size(), end - begin_think.size());
    content = input.substr(end + end_think.size());
    return true;
}

nlohmann::json ChatCompletionsTask::parseMessageV2(const nlohmann::json &varList)
{
    json ret;
    auto msgs = json::array();
    for (const auto& msg : varList) {
        std::string role = msg.value("role", std::string());
        std::string content = msg.value("content", std::string());

        std::transform(role.begin(), role.end(), role.begin(), ::tolower);
        json line;
        if (role == "user") {
            line["role"] = role;
            line["content"] = content;
            msgs.push_back(line);
        } else if (role == "tool") {
            line["role"] = role;
            line["content"] = content;
            msgs.push_back(line);
        } else if (role == "assistant") {
            line["role"] = role;
            if (!content.empty()) {
                line["content"] = content;
            } else if (msg.contains("tool_calls")){
                auto calls = inja::json::array();
                for (auto tool : msg["tool_calls"]) {
                    if (tool.contains("function")) {
                        calls.push_back(tool);
                    }
                }
                if (!calls.empty())
                    line["tool_calls"] = calls;
            }

            msgs.push_back(line);
        } else if (role == "system") {
            ret["System"] = content;
        }
    }

    if (!msgs.empty())
        ret["Messages"] = msgs;
    return ret;
}

std::list<ChatCompletionsTask::ChatHistory> ChatCompletionsTask::parseMessage(const json &messages)
{
    std::list<ChatHistory> prompts;
    ChatHistory tmp;
    
    for (const auto& msg : messages) {
        std::string role = msg.value("role", std::string());
        std::string content = msg.value("content", std::string());
        
        std::transform(role.begin(), role.end(), role.begin(), ::tolower);

        if (role == "user") {
            if (tmp.prompt || tmp.response || tmp.tool) {
                prompts.push_back(tmp);
                tmp = ChatHistory();
            }
            tmp.prompt = content;
        } else if (role == "tool") {
            if (tmp.prompt || tmp.response || tmp.tool) {
                prompts.push_back(tmp);
                tmp = ChatHistory();
            }
            tmp.tool = content;
        } else if (role == "assistant") {
            if (tmp.response) {
                prompts.push_back(tmp);
                tmp = ChatHistory();
            }
            tmp.response = content;
        } else if (role == "system") {
            if (tmp.system || tmp.prompt || tmp.response || tmp.tool) {
                prompts.push_back(tmp);
                tmp = ChatHistory();
            }
            tmp.system = content;
        }
    }

    if (tmp.system || tmp.prompt || tmp.response || tmp.tool) {
        prompts.push_back(tmp);
    }
    return prompts;
}

void ChatStreamTask::doTask() 
{
    LLMProxy *llm = dynamic_cast<LLMProxy *>(model);
    assert(llm);

    auto tokens = llm->tokenize(curPrompt, {});

#ifndef NDEBUG
    std::cerr << "chat stream token count:" << tokens.size() << " prompt:" << curPrompt << std::endl;
#endif

    auto append = [](const std::string &text, void *user) {
        ChatStreamTask *self = static_cast<ChatStreamTask *>(user);

        std::unique_lock<std::mutex> lk(self->genMtx);
        self->text.append(text);
        lk.unlock();

        self->con.notify_all();
        return !self->stop;
    };

    std::map<std::string, std::string> params;
    {
        int max = root.value("max_tokens", -1);
        if (max > 0)
            params.insert(std::make_pair("predict", std::to_string(max)));
    }

    auto alltoken = llm->generate(tokens, params, *append, this);

    {
        std::lock_guard<std::mutex> lk(genMtx);
        stop = true;
    }
    con.notify_all();

#ifndef NDEBUG
    std::cerr << "stream token count:" << alltoken.size() << " output:" << llm->detokenize(alltoken, {}) << std::endl;
#endif
}

void ChatToolslTask::doTask()
{
    LLMProxy *llm = dynamic_cast<LLMProxy *>(model);
    assert(llm);

    bool stream = root.value("stream", false);
    auto tokens = llm->tokenize(curPrompt, {});

#ifndef NDEBUG
    std::cerr << "tools token count:" << tokens.size() << " prompt:" << curPrompt << std::endl;
#endif

    std::map<std::string, std::string> params;
    params.insert(std::make_pair("stream_token", ""));
    {
        int max = root.value("max_tokens", -1);
        if (max < 1)
            max = 100;

        params.insert(std::make_pair("predict", std::to_string(max)));
    }

    auto checkEnd = [](const std::string &text, void *user) {
        ChatCompletionsTask *self = static_cast<ChatCompletionsTask *>(user);
        self->con.notify_all();
        return !self->stop;
    };

    auto token = llm->generate(tokens, params, *checkEnd, this);
    std::string content = llm->detokenize(token, {});

    {
        std::lock_guard<std::mutex> lk(genMtx);
        if (!stop) {
#ifndef NDEBUG
            std::cerr << "tools token count:" << token.size() << " output:" << content << std::endl;
#endif
            json response;
            json choice;
            choice["index"] = 0;

            json message;
            message["role"] = "assistant";

            bool func = false;
            json funcDoc;
            try {
                funcDoc = json::parse(toolCall(content));
                func = funcDoc.contains("name");
                if (func && funcDoc.contains("arguments")) {
                    auto obj = funcDoc["arguments"];
                    if (obj.is_object()) {
#ifndef NDEBUG
                        std::cerr << "function calling arguments is a json object, convert to string." << std::endl;
#endif
                        funcDoc["arguments"] = obj.dump();
                    }
                }
            } catch (const json::parse_error& e) {
#ifndef NDEBUG
                std::cerr << "no tools to use." << std::endl;
#endif
            }

            if (func) {
                json toolCalls;
                // Generate UUID without Qt
                uuid_t uuid;
                uuid_generate(uuid);
                char uuid_str[37];
                uuid_unparse(uuid, uuid_str);

                toolCalls["id"] = uuid_str;
                toolCalls["type"] = "function";
                toolCalls["function"] = funcDoc;
                message["tool_calls"] = json::array({toolCalls});
                message["content"] = nullptr;
                response["finish_reason"] = "tool_calls";
            } else {
                message["content"] = content;
                response["finish_reason"] = "stop";
            }

            choice[stream ? "delta" : "message"] = message;
            response["choices"] = json::array({choice});

            text = (stream ? "data:" : "") + response.dump();
            stop = true;
        } else {
#ifndef NDEBUG
    std::cerr << "tools canceled " << token.size() << " output:" << content << std::endl;
#endif
        }
    }

    con.notify_all();
}

std::string ChatToolslTask::formatPrompt()
{
    std::string func = functionCalls();
    if (func.empty()) return "";

    std::string temp = runner->chatTemplate("functioncall");
    if (temp.empty()) return "";

    auto messages = root["messages"];
    std::list<ChatHistory> prompts = parseMessage(messages);

    std::string prompt = "";
    try {
        inja::Environment env;
        env.set_line_statement(inja::LexerConfig().line_statement + " ");
        bool first = true;

        for (const auto& hist : prompts) {
            inja::json data;
            if (first) {
                data["System"] = "";
            }
            if (hist.prompt) {
                data["Prompt"] = *hist.prompt;
            }
            if (hist.response) {
                data["Response"] = *hist.response;
            }
            if (hist.tool) {
                data["Tool"] = *hist.tool;
            }
            if (first && !func.empty()) {
                data["Functions"] = func;
            }
            prompt += env.render(temp, data);
            first = false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to format prompt: " << e.what() << std::endl;
        return "";
    }

    return prompt;
}

std::string ChatToolslTask::formatPromptV2()
{
    auto funcs = json::array();
    for (const auto& tool : root["tools"]) {
        if (tool["type"] == "function") {
            funcs.push_back(tool["function"]);
        }
    }

    if (funcs.empty())
        return "";

    inja::json json = parseMessageV2(root["messages"]);
    json["Tools"] = funcs;

    std::string prompt;
    auto temp = runner->chatTemplate("functioncall");

    try {
        inja::Environment env;
        env.set_line_statement(inja::LexerConfig().line_statement + " ");
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);
        prompt = env.render(temp, json);
    } catch (const std::exception &error) {
        std::cerr << "fail to format prompt v2, please check the template file of model " << error.what() << std::endl;
        return "";
    }

    return prompt;
}

std::string ChatToolslTask::functionCalls()
{
    if (!root.contains("tools") || !root["tools"].is_array()) {
        return "";
    }

    json funcs = json::array();
    for (const auto& tool : root["tools"]) {
        if (tool["type"] == "function") {
            funcs.push_back(tool["function"]);
        }
    }

    return funcs.empty() ? "" : funcs.dump(2);
}

std::string ChatToolslTask::toolCall(const std::string &in)
{
    std::regex tool_call_regex(R"(<tool_call>([\s\S]*?)</tool_call>)");
    std::smatch matches;
    if (std::regex_search(in, matches, tool_call_regex)) {
        std::string tool_call_content = matches[1].str();
#ifndef NDEBUG
        std::cerr << "output has too_call label, remove:" << tool_call_content << std::endl;
#endif
        return tool_call_content;
    }

    return in;
}

void DslTask::doTask()
{
    auto llm = dynamic_cast<LLMProxy*>(model);
    assert(llm);

    auto tokens = llm->tokenize(curPrompt, {});
#ifndef NDEBUG
    std::cerr << "DSL token count: " << tokens.size() << " prompt: " << curPrompt << std::endl;
#endif

    std::map<std::string, std::string> params;
    params.insert(std::make_pair("stream_token", ""));
    {
        int max = root.value("max_tokens", -1);
        if (max > 0)
            params.insert(std::make_pair("predict", std::to_string(max)));
    }

    auto checkEnd = [](const std::string &text, void *user) {
        DslTask *self = static_cast<DslTask *>(user);
        self->con.notify_all();
        return !self->stop;
    };

    auto output = llm->generate(tokens, params, *checkEnd, this);
    std::string content = llm->detokenize(output, {});

    {
        std::lock_guard<std::mutex> lk(genMtx);
        if (!stop) {
#ifndef NDEBUG
            std::cerr << "DSL output " << output.size() << " content:" << content << std::endl;
#endif
            json response;
            json array = json::array();
            {
                json contentObj;
                contentObj["object"] = "dsl";
                contentObj["index"] = 0;
                contentObj["dsl"] = content;
                array.push_back(contentObj);
            }

            response["model"] = llm->name();
            response["data"] = array;
            response["object"] = "list";
            text = response.dump();
            stop = true;
        } else {
#ifndef NDEBUG
            std::cerr << "DSL canceled " << output.size() << " content:" << content << std::endl;
#endif
        }
    }

    con.notify_all();
}

std::string DslTask::formatPrompt()
{
    std::string temp = runner->chatTemplate("dsl");
    if (temp.empty()) return "";

    std::string input = root.value("input", std::string());
    std::string prompt;
    
    try {
        inja::Environment env;
        env.set_line_statement(inja::LexerConfig().line_statement + " ");
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);

        inja::json data;
        data["Prompt"] = input;
        prompt = env.render(temp, data);
    } catch (const std::exception& e) {
        std::cerr << "Failed to format prompt: " << e.what() << std::endl;
        return "";
    }

    return prompt;
}

std::string DslTask::formatPromptV2()
{
    std::string temp = runner->chatTemplate("dsl");
    if (temp.empty()) return "";

    std::string input = root.value("input", std::string());
    std::string prompt;

    try {
        inja::Environment env;
        env.set_line_statement(inja::LexerConfig().line_statement + " ");
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);

        inja::json data;
        data["Prompt"] = input;
        prompt = env.render(temp, data);
    } catch (const std::exception& e) {
        std::cerr << "Failed to format prompt: " << e.what() << std::endl;
        return "";
    }

    return prompt;
}

std::pair<int, std::string> CompletionsCheckTask::initialize()
{
    ChatCompletionsTask::initialize();
    return std::make_pair(0, std::string());
}

void CompletionsCheckTask::doTask()
{
    LLMProxy *llm = dynamic_cast<LLMProxy *>(model);
    assert(llm);

    auto tokens = llm->tokenize(curPrompt, {});

    std::map<std::string, std::string> params;
    params.insert(std::make_pair("stream_token", ""));
    {
        int max = root.value("max_tokens", -1);
        if (max > 0)
            params.insert(std::make_pair("predict", std::to_string(max)));
    }

    auto checkEnd = [](const std::string &text, void *user) {
        ChatCompletionsTask *self = static_cast<ChatCompletionsTask *>(user);
        self->con.notify_all();
        return !self->stop;
    };

    auto token = llm->generate(tokens, params, *checkEnd, this);
    std::string content = llm->detokenize(token, {});

    {
        std::lock_guard<std::mutex> lk(genMtx);
        stop = true;
        text = R"({"finish_reason":"stop", "status":"success","reason":"success","content":")" +
                content + R"("})";
    }

    con.notify_all();
}

std::string CompletionsCheckTask::formatPrompt()
{
    if (!root.contains("dialogue"))
        return "";

    json trans;
    {
        for (auto &line: root["dialogue"]) {
            std::string role = line["role"];
            if (role == "model")
                line["role"] = "assistant";
            trans.push_back(line);
        }
    }

    std::string prompt = "";
    auto temp = runner->chatTemplate();
    std::list<ChatHistory> prompts = parseMessage(trans);

    try {
        inja::Environment env;
        // add a space for line statement "##" to enable "###"
        env.set_line_statement(inja::LexerConfig().line_statement + " ");
        bool first = true;
        for (const ChatHistory &tmp: prompts) {
            inja::json json;
            if (first)
                json["System"] = tmp.system.has_value() ? tmp.system.value() : "";

            if (tmp.prompt.has_value())
                json["Prompt"] = tmp.prompt.value();

            if (tmp.response.has_value())
                json["Response"] = tmp.response.value();

            if (tmp.tool.has_value())
                json["Tool"] = tmp.tool.value();

            prompt += env.render(temp, json);
            first = false;
        }
    } catch (const std::exception &error) {
        std::cerr << "fail to format prompt, please check the template file of model " << error.what() << std::endl;
        return "";
    }

    return prompt;
}

std::string CompletionsCheckTask::formatPromptV2()
{
    if (!root.contains("dialogue"))
        return "";

    json trans;
    {
        for (auto &line: root["dialogue"]) {
            std::string role = line["role"];
            if (role == "model")
                line["role"] = "assistant";
            trans.push_back(line);
        }
    }

    trans = parseMessageV2(trans);

    std::string prompt;
    auto temp = runner->chatTemplate();

    try {
        inja::Environment env;
        env.set_line_statement(inja::LexerConfig().line_statement + " ");
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);
        prompt = env.render(temp, trans);
    } catch (const std::exception &error) {
        std::cerr << "fail to format prompt v2, please check the template file of model " << error.what() << std::endl;
        return "";
    }

    return prompt;
}

StreamCompletionsCheckTask::~StreamCompletionsCheckTask()
{

}

void StreamCompletionsCheckTask::doTask()
{
    LLMProxy *llm = dynamic_cast<LLMProxy *>(model);
    assert(llm);

    auto tokens = llm->tokenize(curPrompt, {});
    auto append = [](const std::string &text, void *user) {
        StreamCompletionsCheckTask *self = static_cast<StreamCompletionsCheckTask *>(user);
        std::unique_lock<std::mutex> lk(self->genMtx);

        self->text.append(text);
        lk.unlock();

        self->con.notify_all();
        return !self->stop;
    };

    std::map<std::string, std::string> params;
    {
        int max = root.value("max_tokens", -1);
        if (max > 0)
            params.insert(std::make_pair("predict", std::to_string(max)));
    }

    llm->generate(tokens, params, *append, this);

    {
        std::lock_guard<std::mutex> lk(genMtx);
        stop = true;
    }

    con.notify_all();
}


ASRTask::ASRTask(ModelRunner *r, const std::string &content, const std::string &type, ModelProxy *m)
    : ModelTask(r)
    , input(content)
    , format(type)
    , model(m)
{

}

void ASRTask::doTask()
{
    ASRProxy *asr = dynamic_cast<ASRProxy *>(model);
    assert(asr);

    std::vector<double> pcmf32;
    asr->decodeContent(input, pcmf32, {{"format", format}});

    auto checkEnd = [](const std::string &text, void *user) {
        ASRTask *self = static_cast<ASRTask *>(user);
        self->con.notify_all();
        return !self->stop;
    };
    std::map<std::string, std::string> params;
    params.insert(std::make_pair("stream_token", ""));

    auto token = asr->transcriptions(pcmf32, params, *checkEnd, this);
    std::string content = asr->detokenize(token, {});

    {
        std::lock_guard<std::mutex> lk(genMtx);
        if (!stop) {
            json response;
            response["text"] = content;
            text = response.dump();
            stop = true;
        }
    }

    con.notify_all();
}
