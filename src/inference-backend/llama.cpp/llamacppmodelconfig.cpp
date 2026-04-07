// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "llamacppmodelconfig.h"
#include <fstream>
#include <iostream>

GLOBAL_USE_NAMESPACE

using json = nlohmann::json;

LlamacppModelConfig::LlamacppModelConfig(const std::string &file) {
/**************** 
{
"verison":"0.1",
"architectures":["Qwen2ForCausalLM", "LLM"],
"bin":"yourong_1.5B_bf16_Q4_K_M.gguf",
"license":["LICENSE"],
"templates":{"default":"template","functioncall":"template_func", "dsl":"template_dsl"},
"params":{"--ctx-size":"8192","--temp":"0.1"}
}
**************** */
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

    f.close();
}

std::vector<std::string> LlamacppModelConfig::architectures() const {
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

std::unordered_map<std::string, std::string> LlamacppModelConfig::params() const {
    std::unordered_map<std::string, std::string> parameters;
    if (configs.contains("params") && configs["params"].is_object()) {
        for (const auto& [key, value] : configs["params"].items()) {
            if (value.is_string()) {
                parameters[key] = value.get<std::string>();
            } else if (value.is_number()) {
                // Convert numbers to string
                parameters[key] = std::to_string(value.get<double>());
            }
        }
    }
    return parameters;
}
