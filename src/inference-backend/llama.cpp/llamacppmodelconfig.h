// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef LLAMACPPMODELCONFIG_H
#define LLAMACPPMODELCONFIG_H

#include "global_header.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <unordered_map>

GLOBAL_BEGIN_NAMESPACE

class LlamacppModelConfig {
public:
    explicit LlamacppModelConfig(const std::string &file);
    
    inline std::string bin() const {
        return read("bin");
    }
    
    std::vector<std::string> architectures() const;
    std::unordered_map<std::string, std::string> params() const;

protected:
    inline std::string read(const std::string &key) const {
        return configs.contains(key) ? configs[key].get<std::string>() : "";
    }

private:
    nlohmann::json configs;
};

GLOBAL_END_NAMESPACE

#endif // LLAMACPPMODELCONFIG_H
