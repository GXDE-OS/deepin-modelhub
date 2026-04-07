// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef BACKENDLOADER_H
#define BACKENDLOADER_H

#include "global_header.h"
#include "inferenceplugin.h"
#include "backendmetaobject.h"
#include <string>
#include <memory>
#include <vector>

GLOBAL_BEGIN_NAMESPACE

class ModelInfo;
class BackendMetaObject;
class BackendLoaderPrivate;
class BackendLoader {
    friend class BackendLoaderPrivate;
public:
    explicit BackendLoader();

    void setPaths(const std::vector<std::string>& paths);
    void readBackends();
    std::vector<BackendMetaObjectPointer> backends() const;
    std::shared_ptr<InferencePlugin> load(std::shared_ptr<BackendMetaObject> mo) const;
    BackendMetaObjectPointer perfect(const std::shared_ptr<ModelInfo>& model,
                                   std::string* matchedFormat = nullptr,
                                   std::string* matchedArch = nullptr) const;
    bool isRuntimeSupported(std::shared_ptr<BackendMetaObject> mo) const;

private:
    std::shared_ptr<BackendLoaderPrivate> d;
};

GLOBAL_END_NAMESPACE

#endif // BACKENDLOADER_H
