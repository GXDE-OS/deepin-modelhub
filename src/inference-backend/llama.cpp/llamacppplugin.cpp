// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "llamacppplugin.h"
#include "llamacppmodelconfig.h"
#include "llamallmproxy.h"
#include "llamaembproxy.h"
#include "util.h"

#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <thread>

#include "llama.h"
#include "common.h"
#include "log.h"

GLOBAL_USE_NAMESPACE

extern "C" { // 使用C链接，以避免C++名称修饰
    InferencePlugin* createInferencePlugin() {
        return new LlamacppPlugin();
    }
}

namespace fs = std::filesystem;

LlamacppPlugin::LlamacppPlugin() : InferencePlugin() {}

LlamacppPlugin::~LlamacppPlugin() {
    if (inited)
        llama_backend_free();
}

bool LlamacppPlugin::initialize(const VariantMap &params) {
    if (inited)
        return true;

    // disable llama.log
    common_log_pause(common_log_main());

    llama_backend_init();

    //todo
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    inited = true;
    return true;
}

ModelProxy* LlamacppPlugin::loadModel(const std::string &name,
                                    const std::string &imgDir,
                                    const VariantMap &params) {
    fs::path dir(imgDir);
    std::cerr << "loadModel: " << name << " " << imgDir << std::endl;
    LlamacppModelConfig cfg(dir / "config.json");
    
    LlamaModelWrapper *llamaModel = createModelWrapper(name, cfg.architectures());
    auto mParams = cfg.params();
    
    // Merge params
    for (const auto& [key, value] : params) {
        if (mParams.find(key) == mParams.end()) {
            std::string strValue = Util::findValue(params, key, std::string());
            if (!strValue.empty())
                mParams[key] = strValue;
        }
    }

    // Enable GPU offload if supported
    if (llama_supports_gpu_offload() && mParams.find("--n-gpu-layers") == mParams.end()) {
        mParams["--n-gpu-layers"] = std::string("999"); // gpu
    }

    // 限制CPU
    if (mParams.find("--threads") == mParams.end() && mParams.find("-t") == mParams.end()) {
        int num_core = cpu_get_num_physical_cores();
        int thread_num = std::thread::hardware_concurrency();
        num_core = thread_num > num_core ? num_core : num_core - 1;
        num_core = std::max(1, num_core);
        mParams["--threads"] = std::to_string(num_core);
    }

    if (!llamaModel->initialize((dir / cfg.bin()).string(), mParams)) {
        delete llamaModel;
        return nullptr;
    }

    return dynamic_cast<ModelProxy *>(llamaModel);
}

LlamaModelWrapper *LlamacppPlugin::createModelWrapper(const std::string &name,
                                                    const std::vector<std::string> &archs) {
    LlamaModelWrapper *ret = nullptr;
    
    auto compareIgnoreCase = [](const std::string &a, const std::string &b) {
        if (a.size() != b.size())
            return false;
        return strcasecmp(a.c_str(), b.c_str()) == 0;
    };

    for (const auto &arch : archs) {
        if (compareIgnoreCase(arch, "LLM")) {
            ret = new LlamaLLMProxy(name);
        } else if (compareIgnoreCase(arch, "Embedding")) {
            ret = new LlamaEmbProxy(name);
        }
    }

    return ret;
}
