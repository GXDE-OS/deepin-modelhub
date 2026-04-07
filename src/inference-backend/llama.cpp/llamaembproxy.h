// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef LLAMAEMBPROXY_H
#define LLAMAEMBPROXY_H

#include "embeddingproxy.h"
#include "llamamodelwrapper.h"
#include <string>
#include <list>
#include <unordered_map>

GLOBAL_BEGIN_NAMESPACE

class LlamaEmbProxy : public EmbeddingProxy, public LlamaModelWrapper
{
public:
    explicit LlamaEmbProxy(const std::string &name);
    std::string name() const override;
    std::list<std::vector<int32_t>> tokenize(const std::list<std::string> &prompt, const std::map<std::string, std::string> &params = {}) override;
    std::list<std::vector<float>> embedding(const std::list<std::vector<int32_t>> &tokens, const std::map<std::string, std::string> &params = {}) override;
    bool initialize(const std::string &bin, const std::unordered_map<std::string, std::string> &params) override;
protected:
    std::string modelName;
};
GLOBAL_END_NAMESPACE

#endif // LLAMAEMBPROXY_H
