// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ovmodelconfig.h"

#include <fstream>
#include <iostream>

GLOBAL_USE_NAMESPACE

using json = nlohmann::json;

OVModelConfig::OVModelConfig(const std::string &file)
{
    std::ifstream f(file);
    if (!f.is_open()) {
        std::cerr << "Failed to open config file: " << file << std::endl;
        return;
    }

    try {
        f >> configs;
    } catch (const json::parse_error& e) {
        std::cerr << "Failed to parse config file: " << e.what() << std::endl;
    }
}

std::vector<std::string> OVModelConfig::architectures() const
{
    std::vector<std::string> archs;
    if (configs.contains("architectures") && configs["architectures"].is_array()) {
        for (const auto& arch : configs["architectures"]) {
            if (arch.is_string()) {
                archs.push_back(arch.get<std::string>());
            }
        }
    }
    return archs;
}

VariantMap OVModelConfig::params() const
{
    VariantMap parameters;
    if (configs.contains("params") && configs["params"].is_object()) {
        for (const auto& [key, value] : configs["params"].items()) {
            if (value.is_string()) {
                parameters[key] = value.get<std::string>();
            } else if (value.is_number()) {
                parameters[key] = value.get<double>();
            }
        }
    }
    return parameters;
}
