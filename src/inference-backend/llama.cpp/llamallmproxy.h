// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef LLAMALLMPROXY_H
#define LLAMALLMPROXY_H

#include "llmproxy.h"
#include "llamaserver.h"
#include <string>
#include <map>
#include <vector>
#include <cstdint>

GLOBAL_BEGIN_NAMESPACE

class LlamaLLMProxy : public LLMProxy, public LlamaServer
{
    struct LLMGenerateContext {
        std::string generatedText;
        size_t pushedPos = 0;
    };
public:
    explicit LlamaLLMProxy(const std::string &name);
    std::string name() const override;
    std::vector<int32_t> tokenize(const std::string &prompt, const std::map<std::string, std::string> &params) override;
    std::string detokenize(const std::vector<int32_t> &tokens, const std::map<std::string, std::string> &params) override;
    std::vector<int32_t> generate(const std::vector<int32_t> &tokens, const std::map<std::string, std::string> &params = {},
                                  generateStream stream = nullptr, void *user = nullptr) override;
protected:
    std::vector<int32_t> generateParallel(const std::vector<int32_t> &tokens, const std::map<std::string, std::string> &params ,
                          generateStream stream, void *user);
    std::vector<int32_t> generateSafe(const std::vector<int32_t> &tokens, const std::map<std::string, std::string> &params ,
                          generateStream stream, void *user);
    std::string processToken(int32_t token, LLMGenerateContext &slot) const;
};

GLOBAL_END_NAMESPACE

#endif // LLAMALLMPROXY_H
