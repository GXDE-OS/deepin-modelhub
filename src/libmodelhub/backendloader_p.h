// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef BACKENDLOADER_P_H
#define BACKENDLOADER_P_H

#include "backendloader.h"
#include "backendmetaobject.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <dlfcn.h>
#include <nlohmann/json.hpp>
#include <variant>

GLOBAL_BEGIN_NAMESPACE

class BackendMetaObjectPrivate {

public:
    explicit BackendMetaObjectPrivate(BackendMetaObject* parent);
    VariantMap extra;
    void* handle;
    nlohmann::json metaJson;
    std::string fileName;
    std::string metaData(const std::string& key) const;
    void setFileName(const std::string& fileName);

private:
    BackendMetaObject* q;
};

class BackendLoaderPrivate {
public:
    explicit BackendLoaderPrivate(BackendLoader* parent);
    std::vector<std::shared_ptr<BackendMetaObject>> sorted() const;
    void preload(std::shared_ptr<BackendMetaObject> mo);
    void checkRuntime(std::shared_ptr<BackendMetaObject> mo);

public:
    std::vector<std::string> loadPaths;
    std::vector<std::shared_ptr<BackendMetaObject>> backends;
    
    std::string fixedBackend() const;
    std::string llamacppBackend() const;
    std::string configPath() const;

private:
    BackendLoader* q;
    std::string readConfigValue(const std::string& group, const std::string& key) const;
};

GLOBAL_END_NAMESPACE

#endif // BACKENDLOADER_P_H
