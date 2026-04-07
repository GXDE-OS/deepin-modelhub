// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EMBEDDINGPROXY_H
#define EMBEDDINGPROXY_H

#include "modelproxy.h"
#include <string>
#include <list>
#include <vector>
#include <map>
#include <cstdint>

GLOBAL_BEGIN_NAMESPACE

class EmbeddingProxy : public ModelProxy {
public:
    EmbeddingProxy() : ModelProxy() {}
    virtual ~EmbeddingProxy() {}
    ModelArchitecture architecture() const override {
        return Eembedding;
    }
    virtual std::list<std::vector<int32_t>> tokenize(const std::list<std::string> &prompt,
                                                    const std::map<std::string, std::string> &params = {}) = 0;
                                                    
    virtual std::list<std::vector<float>> embedding(const std::list<std::vector<int32_t>> &tokens,
                                                  const std::map<std::string, std::string> &params = {}) = 0;
};

GLOBAL_END_NAMESPACE

#endif // EMBEDDINGPROXY_H
