// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef BACKENDMETAOBJECT_H
#define BACKENDMETAOBJECT_H

#include "global_header.h"
#include "inferenceplugin.h"
#include <string>
#include <vector>
#include <memory>

GLOBAL_BEGIN_NAMESPACE

class InferencePlugin;
class BackendMetaObjectPrivate;

class BackendMetaObject {
    friend class BackendMetaObjectPrivate;
    friend class BackendLoader;
public:
    std::string fileName() const;
    std::string iid() const;
    std::string name() const;
    std::string version() const;
    std::vector<std::string> supportedArchitectures() const;
    std::vector<std::string> supportedFormats() const;
    
    ValueType extra(const std::string &key, ValueType defaultValue = ValueType()) const;
    void setExtra(const std::string &key, const ValueType &value);
    
protected:
    explicit BackendMetaObject();
private:
    BackendMetaObjectPrivate *d;
};

using BackendMetaObjectPointer = std::shared_ptr<BackendMetaObject>;
GLOBAL_END_NAMESPACE

#endif // BACKENDMETAOBJECT_H
