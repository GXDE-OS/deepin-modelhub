// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef OVMODELCONFIG_H
#define OVMODELCONFIG_H

#include "global_header.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>
#include <unordered_map>

GLOBAL_BEGIN_NAMESPACE

class OVModelConfig
{
public:
    explicit OVModelConfig(const std::string &file);
    inline std::string model() const {
        return read("model");
    }
    inline std::string tokenizer() const {
        return read("tokenizer");
    }
    inline std::string detokenizer() const {
        return read("detokenizer");
    }
    std::vector<std::string> architectures() const;
    VariantMap params() const;
protected:
    inline std::string read(const std::string &key) const {
        return configs.contains(key) ? configs[key].get<std::string>() : "";
    }
private:
    nlohmann::json configs;
};

GLOBAL_END_NAMESPACE

#endif // OVMODELCONFIG_H
