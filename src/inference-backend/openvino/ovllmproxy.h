// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef OVLLMPROXY_H
#define OVLLMPROXY_H

#include "llmproxy.h"
#include "gen/ovtokenizer.h"

#include <openvino/openvino.hpp>

GLOBAL_BEGIN_NAMESPACE

class OvLLMProxy : public LLMProxy
{
public:
    explicit OvLLMProxy(const std::string &name, ov::Core *core);
    std::string name() const override;
    std::vector<int32_t> tokenize(const std::string &prompt, const std::map<std::string, std::string> &params) override;
    std::string detokenize(const std::vector<int32_t> &, const std::map<std::string, std::string> &params) override;
    std::vector<int32_t> generate(const std::vector<int32_t> &, const std::map<std::string,
                                  std::string> &params = {}, generateStream stream = nullptr, void *user = nullptr) override;
    bool initialize(const std::string &model, const std::string &tokenizer, const std::string &detokenizer, const VariantMap &params);
protected:
    std::vector<int64_t> pipelineWithStatic(const std::vector<int64_t> &, const std::map<std::string,
                                  std::string> &params = {}, generateStream stream = nullptr, void *user = nullptr);
//    std::vector<int64_t> generate(const std::vector<int64_t> &, const std::map<std::string,
//                                  std::string> &params = {}, generateStream stream = nullptr, void *user = nullptr);
    std::vector<int64_t> doPipeline(const std::vector<int64_t> &, const std::map<std::string,
                                  std::string> &params = {}, generateStream stream = nullptr, void *user = nullptr);
    std::string processToken(int64_t token, std::vector<int64_t> &tokenCache);
protected:
    std::string modelName;
    std::string device;
    int inputSize = -1;
    bool staticSize = false;
    ov::Core *ovCore = nullptr;
    std::shared_ptr<OvTokenizer> ovtokenizer;
    std::shared_ptr<ov::Model> orgModel;
    ov::CompiledModel chatModel;
    std::list<std::string> inputTensors;
};

GLOBAL_END_NAMESPACE

#endif // OVLLMPROXY_H
