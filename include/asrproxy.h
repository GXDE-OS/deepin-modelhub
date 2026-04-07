// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ASRPROXY_H
#define ASRPROXY_H

#include "modelproxy.h"

#include <vector>
#include <map>
#include <functional>

GLOBAL_BEGIN_NAMESPACE

using generateStream = std::function<bool(const std::string&, void*)>;

class ASRProxy : public ModelProxy
{
public:
    explicit ASRProxy() : ModelProxy() {}
    ModelArchitecture architecture() const override {
        return ASR;
    }

    virtual bool decodeContent(const std::string &content, std::vector<double> &pcmf32,
                               const std::map<std::string, std::string> &params = {}) = 0;
    virtual std::vector<int32_t> transcriptions(const std::vector<double> &pcmf32,
                                                const std::map<std::string, std::string> &params = {},
                                                generateStream stream = nullptr,
                                                void *user = nullptr) = 0;
    virtual std::string detokenize(const std::vector<int32_t> &tokens,
                                   const std::map<std::string, std::string> &params) = 0;
};

GLOBAL_END_NAMESPACE

#endif // ASRPROXY_H
