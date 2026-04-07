// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef OVEMBPROXY_H
#define OVEMBPROXY_H

#include "embeddingproxy.h"

#include <openvino/openvino.hpp>

GLOBAL_BEGIN_NAMESPACE

class OVEmbProxy : public EmbeddingProxy
{
public:
    explicit OVEmbProxy(const std::string &name, ov::Core *core);
    ~OVEmbProxy();
    std::string name() const override;
    std::list<std::vector<int32_t>> tokenize(const std::list<std::string> &prompt, const std::map<std::string, std::string> &params = {}) override;
    std::list<std::vector<float>> embedding(const std::list<std::vector<int32_t>> &tokens, const std::map<std::string, std::string> &params = {}) override;
    bool initialize(const std::string &model, const std::string &tokenizer, const VariantMap &params);
protected:
    void embdNormalize(const float * inp, float * out, int n) const;
protected:
    std::string modelName;
    std::string device;
    int inputSize = -1;
    bool staticSize = false;
    ov::Core *ovCore = nullptr;
    std::shared_ptr<ov::Model> orgEmbModel;
    ov::CompiledModel embModel;
    ov::CompiledModel tokenizerModel;

};

GLOBAL_END_NAMESPACE

#endif // OVEMBPROXY_H
