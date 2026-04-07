// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "backendloader_p.h"
#include "inferenceplugin.h"
#include <dlfcn.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>

GLOBAL_USE_NAMESPACE

namespace fs = std::filesystem;

BackendMetaObjectPrivate::BackendMetaObjectPrivate(BackendMetaObject *parent) : q(parent)
{

}

std::string BackendMetaObjectPrivate::metaData(const std::string &key) const
{
    if (key.empty() || metaJson.is_null())
        return "";

    nlohmann::json &&dataJson = metaJson.value("MetaData", nlohmann::json::object());
    return dataJson.value(key, "");
}

BackendMetaObject::BackendMetaObject() : d(new BackendMetaObjectPrivate(this)) {}


std::string BackendMetaObject::name() const {
    return d->metaJson.value("name", "");
}

std::string BackendMetaObject::version() const {
    return d->metaJson.value("version", "");
}

std::vector<std::string> BackendMetaObject::supportedArchitectures() const {
    auto m = d->metaJson["model"];
    if (!m.contains("architectures"))
        return {};

    return m["architectures"].get<std::vector<std::string>>();
}

std::vector<std::string> BackendMetaObject::supportedFormats() const {
    auto m = d->metaJson["model"];
    if (!m.contains("formats"))
        return {};
    return m["formats"].get<std::vector<std::string>>();
}

ValueType BackendMetaObject::extra(const std::string &key, ValueType defaultValue) const {
    auto it = d->extra.find(key);
    return it != d->extra.end() ? it->second : defaultValue;
}

void BackendMetaObject::setExtra(const std::string &key, const ValueType &value) {
    d->extra[key] = value;
}

std::string BackendMetaObject::fileName() const {
    return d->fileName;
}

std::string BackendMetaObject::iid() const {
    // Assuming there's a metadata member similar to PluginMetadata
    return d->metaData("iid");  // Returns empty string if "iid" key doesn't exist
}

void BackendMetaObjectPrivate::setFileName(const std::string& fileName) {
    this->fileName = fileName;
    std::string noSuffixName = "";
    
    //去掉.so后缀
    if (fileName.size() > 3 && fileName.substr(fileName.size() - 3) == ".so") {
        noSuffixName = fileName.substr(0, fileName.size() - 3);
    }
    
    // 读取并解析JSON元数据文件
    std::string jsonPath = noSuffixName + ".json";
    std::ifstream f(jsonPath);
    if (f.is_open()) {
        try {
            f >> metaJson;
        } catch (...) {
            std::cerr << "Failed to parse metadata file: " << jsonPath << std::endl;
        }
    }
    else {
        std::cerr << "Failed to open metadata file: " << jsonPath << std::endl;
    }
}

