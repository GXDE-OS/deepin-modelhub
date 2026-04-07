// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef LLAMAMODELWRAPPER_H
#define LLAMAMODELWRAPPER_H

#include "modelproxy.h"

#include <string>
#include <unordered_map>
#include <mutex>

class llama_model;
class common_params;
class llama_context;
class llama_vocab;
class common_init_result;
GLOBAL_BEGIN_NAMESPACE
class LlamaModelWrapper
{
public:
    explicit LlamaModelWrapper();
    virtual ~LlamaModelWrapper();

    virtual bool initialize(const std::string &bin, const std::unordered_map<std::string, std::string> &params);
protected:
    llama_model *gModel = nullptr;
    common_params *gParams = nullptr;
    llama_context *gCtx = nullptr;
    const llama_vocab *gVocab = nullptr;
    common_init_result *gResult = nullptr;
    std::mutex gMtx;
};

GLOBAL_END_NAMESPACE

#endif // LLAMAMODELWRAPPER_H
