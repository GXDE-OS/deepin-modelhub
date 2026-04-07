// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "modelinfo.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <algorithm>

GLOBAL_USE_NAMESPACE

using json = nlohmann::json;
namespace fs = std::filesystem;

ModelInfo::ModelInfo(const std::string &path): rootPath(path)
{
}

std::string ModelInfo::name() const
{
    return fs::path(rootPath).filename().string();
}

std::string ModelInfo::path() const
{
    return rootPath;
}

std::vector<std::string> ModelInfo::formats() const
{
    std::vector<std::string> ret;
    
    for(const auto& entry : fs::directory_iterator(rootPath)) {
        if(fs::is_directory(entry) && fs::exists(entry.path() / "config.json")) {
            ret.push_back(entry.path().filename().string());
        }
    }
    
    return ret;
}

std::vector<std::string> ModelInfo::architectures(const std::string &format) const
{
    std::vector<std::string> ret;
    if (format.empty())
        return ret;

    std::ifstream f(imagePath(format) + "/config.json");
    if (f.is_open()) {
        json config = json::parse(f);
        f.close();

        if(config.contains("architectures") && config["architectures"].is_array()) {
            for(const auto& arch : config["architectures"]) {
                ret.push_back(arch.get<std::string>());
            }
        }
    }

    return ret;
}

std::string ModelInfo::imagePath(const std::string &format) const
{
    std::string formatLower = format;
    std::transform(formatLower.begin(), formatLower.end(), formatLower.begin(), ::tolower);
    return rootPath + "/" + formatLower;
}

std::string ModelInfo::chatTemplate(const std::string &format, const std::string &type) const
{
    std::string ret;
    if (format.empty() || type.empty())
        return ret;
    
    std::ifstream f(imagePath(format) + "/config.json");
    if (f.is_open()) {
        json config = json::parse(f);
        f.close();

        if(config.contains("templates") && 
           config["templates"].contains(type)) {
            std::string templateFile = config["templates"][type].get<std::string>();
            
            std::ifstream ft(imagePath(format) + "/" + templateFile);
            if (ft.is_open()) {
                ret = std::string((std::istreambuf_iterator<char>(ft)),
                                 std::istreambuf_iterator<char>());
                ft.close();
            }
        }
    }

    return ret;
}

std::string ModelInfo::version(const std::string &format) const
{
    std::string ret;
    if (format.empty())
        return ret;

    std::ifstream f(imagePath(format) + "/config.json");
    if (f.is_open()) {
        json config = json::parse(f);
        f.close();
        ret = config.value("verison", "0.1");
    }

    return ret;
}
