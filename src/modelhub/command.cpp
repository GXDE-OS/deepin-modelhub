// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "command.h"
#include "modelrepo.h"
#include "backendloader.h"
#include "modelinfo.h"
#include "inferenceplugin.h"
#include "llmproxy.h"
#include "embeddingproxy.h"
#include "modelserver.h"
#include "runtimestate.h"
#include "modelrunner.h"
#include "util.h"
#include "configmanager.h"

#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <memory>
#include <ctime>
#include <cstring>

using json = nlohmann::json;

GLOBAL_USE_NAMESPACE

#ifndef VERSION
#define VERSION "0.0.1"  // 默认版本号
#endif

Command::Command()
{
    initOptions();
}

int Command::processCmd(int argc, char* argv[])
{
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.substr(0, 2) == "--") {
            std::string key = arg.substr(2);
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                m_options[key] = argv[++i];
            } else {
                m_options[key] = "";
            }
        }
    }

    int ret = 0;
    if (isSet("list")) {
        ret = listHandler();
    } else if (isSet("embeddings")) {
        ret = embeddings();
    } else if (isSet("run")) {
        ret = runServer();
    } else if (isSet("model")) {
        ret = llmGenerate();
    } else if (isSet("v") || isSet("version")) {
        // Get version from environment or compile-time define
        std::cout << VERSION << std::endl;
    } else if (isSet("stop")) {
        ret = stopServer();
    } else {
        showHelp(0);
    }

    return ret;
}

void Command::appExitHandler(int sig)
{
    std::cerr << "signal " << sig << " exit." << std::endl;

    // Clean up resources
    ModelServer::exitServer();
}

int Command::listHandler()
{
    bool info = isSet("info");
    std::string target = value("list");
    json output;

    if (target == "model") {
        ModelRepo repo;
        std::vector<std::string> models = repo.list();
        output["model"] = models;
        if (info) {
            json details;
            for (const std::string &name : models) {
                auto minfo = repo.modelInfo(name);
                if (minfo.get() != nullptr) {
                    json detail;
                    detail["formats"] = minfo->formats();
                    {
                        std::set<std::string> archs;
                        for (const std::string &fm : minfo->formats()) {
                            auto tmp = minfo->architectures(fm);
                            archs.insert(tmp.begin(), tmp.end());
                        }
                        detail["architectures"] = archs;
                    }
                    details[name] = detail;
                }
            }
            output["details"] = details;
        }
    } else if (target == "backend") {
        BackendLoader loader;
        loader.readBackends();
        std::vector<std::string> names;
        for (const auto& obj : loader.backends()) {
            names.push_back(obj->name());
        }
        output["backend"] = names;
    }
    else if (target == "server") {
        auto serverList = RuntimeState::listAll();
        
        if (info) {
            std::vector<std::map<std::string, std::string>> infos;
            for (const auto& server : serverList) {
                infos.push_back(server);
            }
            output["serverinfo"] = infos;
        } else {
            std::vector<std::string> names;
            for (const auto& server : serverList) {
                auto it = server.find("model");
                if (it != server.end() && !it->second.empty()) {
                    names.push_back(it->second);
                }
            }
            output["server"] = names;
        }
    } else {
        showHelp(1);
    }

    std::cout << output.dump() << std::endl;
    return 0;
}

bool Command::llmStreamOutput(const std::string &text, void *llm)
{
    std::cout << text;
    std::flush(std::cout);
    return true;
}

int64_t Command::getTimeUs()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec/1000;
}

void Command::warmUp(ModelProxy *model)
{
    if (auto llm = dynamic_cast<LLMProxy *>(model)) {
        std::string prompt = "1+1=";
        auto tokens = llm->tokenize(prompt, {});
        llm->generate(tokens, {{"predict", "10"}}, nullptr, llm);
    }
}

int Command::llmGenerate()
{
    std::string modelName = value("model");
    std::string prompt = value("prompt");
    std::string file = value("file");
    
    if (!file.empty()) {
        std::ifstream fp(file);
        if (fp.is_open()) {
            std::stringstream buffer;
            buffer << fp.rdbuf();
            prompt = buffer.str();
        }
    }

    bool timings = isSet("timings");

    if (modelName.empty() || prompt.empty()) {
        showHelp(1);
    }

    ModelRunner mr;
    int ret = loadModel(modelName, &mr);
    if (ret != 0)
        return ret;

    if (auto llm = dynamic_cast<LLMProxy *>(mr.modelProxy.get())) {
        if (timings) {
            warmUp(llm);

            int64_t startUs = getTimeUs();
            auto tokens = llm->tokenize(prompt, {});
#ifndef NDEBUG
            std::cerr << "llm token count:" << tokens.size() << " output:" << prompt << std::endl;
#endif
            double spendMs = (getTimeUs() - startUs) / 1e3;
            fprintf(stderr, "tokenize time = %.2f ms / %d tokens (%.2f ms per token, %.2f tokens per second)\n",
                    spendMs, tokens.size(), spendMs / (double)tokens.size(), tokens.size() / spendMs * 1e3);
            std::map<std::string, std::string> params;
            params.insert(std::make_pair("timings", "true"));
            auto token = llm->generate(tokens, params, llmStreamOutput, llm);
        } else {
            auto tokens = llm->tokenize(prompt, {});
#ifndef NDEBUG
            std::cerr << "llm token count:" << tokens.size() << " output:" << prompt << std::endl;
#endif
            auto token = llm->generate(tokens, {}, llmStreamOutput, llm);
        }
    } else {
        std::cerr << modelName << " do not support to generate" << std::endl;
    }

    return 0;
}

int Command::embeddings()
{
    std::string modelName = value("model");
    std::string prompt = value("prompt");

    if (modelName.empty() || prompt.empty()) {
        showHelp(1);
    }

    ModelRunner mr;
    int ret = loadModel(modelName, &mr);
    if (ret != 0)
        return ret;

    if (auto emb = dynamic_cast<EmbeddingProxy *>(mr.modelProxy.get())) {
        std::list<std::vector<int32_t>> tokens = emb->tokenize({prompt});
        std::list<std::vector<float>> out = emb->embedding(tokens);

        json root;
        json data = json::array();
        int i = 0;
        for (auto it = out.begin(); it != out.end(); ++it) {
            json embObj;
            embObj["object"] = "embedding";
            embObj["index"] = i++;
            embObj["embedding"] = *it;
            data.push_back(embObj);
        }

        root["data"] = data;
        root["model"] = modelName;
        root["object"] = "list";
        std::cout << root.dump() << std::endl;
    } else {
        std::cerr << modelName << " do not support to embedding" << std::endl;
    }

    return 0;
}

int Command::runServer()
{
    std::string modelName = value("run");
    if (modelName.empty())
        showHelp(1);
    int parallel = -1;
    {
        std::string strValue = value("parallel");
        int tmp = strValue.empty() ? -1 : std::stoi(strValue);
        parallel = tmp > 1 ? tmp : parallel;
        parallel = std::min(parallel, 16);
    }

    ModelRunner mr;
    int ret = loadModel(modelName, &mr, parallel);
    if (ret != 0)
        return ret;

    ModelServer serve;
    {
        serve.setHost(value("host"));
        int port = -1;
        if (isSet("port")) {
            try {
                port = std::stoi(value("port"));
            } catch (...) {
                port = -1;
            }
        }
        if (port > 0)
            serve.setPort(port);

        int pid = serve.instance(modelName);
        if (pid > 0) {
            std::cerr << "model: " << modelName << " is running as process: " << pid << std::endl;
            return 1;
        }
    }

    if (isSet("exit-idle")) {
        try {
            serve.setIdle(std::stoi(value("exit-idle")));
        } catch (...) {
            std::cerr << "Invalid exit-idle value" << std::endl;
            return 1;
        }
    }
    
    serve.run(&mr);
    return 0;
}

int Command::stopServer()
{
    std::string modelName = value("stop");
    if (modelName.empty())
        showHelp(1);
    
    RuntimeState rs(modelName);
    int pid = rs.pid();
    if (pid > 0) {
        std::cerr << "stop server " << modelName << " pid " << pid << std::endl;
        std::string cmd = "kill -3 " + std::to_string(pid);
        system(cmd.c_str());
    }
    return 0;
}

int Command::loadModel(const std::string &modelName, ModelRunner *runner, int parallel)
{
    ModelRepo repo;
    ModelInfoPointer stdInfo = repo.modelInfo(modelName);
    if (!stdInfo) {
        std::cerr << "no such model:" << modelName << std::endl;
        return 1;
    }

    BackendLoader bkLoader;
    bkLoader.readBackends();

    std::string format;  // 添加临时变量
    BackendMetaObjectPointer bmo = bkLoader.perfect(stdInfo, &format);
    if (!bmo) {
        std::cerr << "no backend for model: " << modelName << std::endl;
        return 1;
    }

    auto backend = bkLoader.load(bmo);
    if (!backend) {
        return 1;
    }

    VariantMap params;
    if (bmo->name() == "llama.cpp") {
        auto archs = stdInfo->architectures(format);
        bool isllm = std::find_if(archs.begin(), archs.end(), [](std::string val){
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            return val == std::string("llm");
        }) != archs.end();

        if (isllm) {
            runner->parallel = parallel < 1 ? 4 : parallel;
            runner->cacheDir = RuntimeState::stateDir() + "/cache/" + stdInfo->name();
            params.insert(std::make_pair("--parallel", std::to_string(runner->parallel)));     
            params.insert(std::make_pair("--temp-cache-dir", runner->cacheDir));
        }
    }

    backend->initialize({});
    ModelProxy *mp = backend->loadModel(stdInfo->name(),
                                        stdInfo->imagePath(format), params);

    if (!mp) {
        std::cerr << "backend " << bmo->name() << " unable to load model: " << modelName << " with format " << format << std::endl;
        return -1;
    }

    runner->modelInfo = stdInfo;
    runner->modelFormat = format;
    runner->backendmo = bmo;
    runner->backendIns = backend;
    runner->modelProxy.reset(mp);

    return 0;
}

void Command::initOptions()
{
    // No need for Qt option initialization
    // Options will be handled during command line parsing
}

bool Command::isSet(const std::string& option) const 
{
    return m_options.find(option) != m_options.end();
}

std::string Command::value(const std::string& option) const
{
    auto it = m_options.find(option);
    if (it != m_options.end()) {
        return it->second;
    }
    return "";
}

void Command::showHelp(int exitCode)
{
    std::cout << "Usage: deepin-modelhub [options]\n"
              << "Options:\n"
              << "  --list {model|backend|server} [--info]  List models/backends/servers\n"
              << "  --run <model>                           Run a model\n"
              << "  --host <host>                          Server host\n"
              << "  --port <port>                          Server port\n"
              << "  --stop <model>                         Stop a running model\n"
              << "  --model <name>                         Model name\n"
              << "  --embeddings                           Get embeddings for inputs\n"
              << "  --prompt <text>                        Input prompt for LLM\n"
              << "  --file <path>                          Input prompt file for LLM\n"
              << "  --info                                 Option for list\n"
              << "  --exit-idle <seconds>                  Exit when idle (min 10s)\n"
              << "  --timings                              Print inference time\n"
              << "  -v, --version                          Display version\n"
              << "  -h, --help                             Display this help\n";
    exit(exitCode);
}
