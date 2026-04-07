// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "llamamodelwrapper.h"
#include "llama.h"
#include "common/common.h"
#include "common/arg.h"
#include <iostream>
#include <cstring>
#include <variant>
#include <thread>

GLOBAL_USE_NAMESPACE

LlamaModelWrapper::LlamaModelWrapper() {}

LlamaModelWrapper::~LlamaModelWrapper() {
    gModel = nullptr;
    gCtx = nullptr;

    delete gResult;
    gResult = nullptr;

    if (gParams)
        delete gParams;

    gParams = nullptr;
}

bool LlamaModelWrapper::initialize(const std::string &bin, 
                                 const std::unordered_map<std::string, std::string> &params) {
    if (gModel)
        return false;
        
    gParams = new common_params;
    
    // Convert params to argv style arguments
    int argc = 1;
    char argv[128][128] = {0};
    char *ptr[128] = {0};
    
    for (const auto &[key, value] : params) {
        if (argc >= 128) break;
        
        std::strncpy(argv[argc], key.c_str(), 127);
        ptr[argc] = argv[argc];
        argc++;
        
        if (!value.empty()) {
            std::strncpy(argv[argc], value.c_str(), 127);
            ptr[argc] = argv[argc];
            argc++;
        }
    }

    // Print system info
    std::string sysInfo = llama_print_system_info();
    sysInfo += "CUDA = " + std::to_string(llama_supports_gpu_offload()) + " | ";
    std::cerr << "system info: " << sysInfo << std::endl;

    if (!common_params_parse(argc, ptr, *gParams, LLAMA_EXAMPLE_SERVER))
        return false;

    gParams->model = bin;
    gResult = new common_init_result(common_init_from_params(*gParams));
    gModel = gResult->model.get();
    gCtx = gResult->context.get();
    gVocab = llama_model_get_vocab(gModel);

    return true;
}
