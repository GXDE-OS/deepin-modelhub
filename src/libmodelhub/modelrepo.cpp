// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "modelrepo_p.h"
#include "modelinfo.h"
#include "configmanager.h"

#include <filesystem>
#include <iostream>

GLOBAL_USE_NAMESPACE

namespace fs = std::filesystem;

ModelRepoPrivate::ModelRepoPrivate(ModelRepo *parent)
    : q(parent)
{
    // 首先从配置文件读取路径
    std::vector<std::string> configPaths = ConfigManager::instance()->modelRepoPaths();
    
    if (!configPaths.empty()) {
        // 如果配置文件中有设置路径,则使用配置的路径
        path = configPaths;
    } else {
        // 如果配置文件中没有设置路径,则使用默认路径
        // 默认路径为系统数据目录下的应用特定目录
        const char* xdg_data_home = std::getenv("XDG_DATA_HOME");
        std::string defaultPath;
        
        if (xdg_data_home) {
            defaultPath = std::string(xdg_data_home);
        } else {
            const char* home = std::getenv("HOME");
            if (home) {
                defaultPath = std::string(home) + "/.local/share";
            }
        }
        
        if (!defaultPath.empty()) {
            path.push_back(defaultPath + "/" + EXE_NAME + "/models");
        }
        
        // 添加系统级数据目录
        path.push_back("/usr/share/" + std::string(EXE_NAME) + "/models");
        path.push_back("/usr/local/share/" + std::string(EXE_NAME) + "/models");
    }

#ifndef NDEBUG
    // 调试模式下输出当前使用的路径
    std::cerr << "model repo paths:" << std::endl;
    for (const auto& p : path) {
        std::cerr << p << std::endl;
    }
#endif
}

ModelRepo::ModelRepo()
    : d(std::make_unique<ModelRepoPrivate>(this))
{
}

ModelRepo::~ModelRepo() = default;

void ModelRepo::setRepoPath(const std::vector<std::string> &path)
{
    d->path = path;
}

std::vector<std::string> ModelRepo::list() const
{
    std::vector<std::string> models;
    for (const std::string &path : d->path) {
        if (!fs::exists(path)) {
            continue;
        }
        
        for (const auto& entry : fs::directory_iterator(path)) {
            if (fs::is_directory(entry)) {
                auto str = entry.path().filename().string();
                if (str.find('.') == 0)
                    continue;

                models.push_back(entry.path().filename().string());
            }
        }
    }
    
    return models;
}

std::vector<ModelProperty> ModelRepo::show(const std::string &model) const
{
    return {};
}

std::shared_ptr<ModelInfo> ModelRepo::modelInfo(const std::string &model) const
{
    if (model.empty()) {
        return nullptr;
    }

    for (const std::string &path : d->path) {
        fs::path modelPath = fs::path(path) / model;
        if (fs::exists(modelPath)) {
            return std::make_shared<ModelInfo>(modelPath.string());
        }
    }

    return nullptr;
}
