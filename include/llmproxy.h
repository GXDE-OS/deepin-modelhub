// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef LLMPROXY_H
#define LLMPROXY_H

#include "modelproxy.h"
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <functional>

GLOBAL_BEGIN_NAMESPACE

using generateStream = std::function<bool(const std::string&, void*)>;

class LLMProxy : public ModelProxy {
public:
    LLMProxy() : ModelProxy() {}
    virtual ~LLMProxy() {}
    ModelArchitecture architecture() const override {
        return LLM;
    }
    virtual std::vector<int32_t> tokenize(const std::string &prompt,
                                        const std::map<std::string, std::string> &params) = 0;
                                        
    virtual std::string detokenize(const std::vector<int32_t> &tokens,
                                 const std::map<std::string, std::string> &params) = 0;
                                 
    virtual std::vector<int32_t> generate(const std::vector<int32_t> &tokens,
                                        const std::map<std::string, std::string> &params = {},
                                        generateStream stream = nullptr,
                                        void *user = nullptr) = 0;
};

GLOBAL_END_NAMESPACE

#endif // LLMPROXY_H
