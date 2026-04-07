// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INFERENCEPLUGIN_H
#define INFERENCEPLUGIN_H

#include "global_header.h"
#include <string>
#include <unordered_map>


GLOBAL_BEGIN_NAMESPACE
#define INFERENCE_PLUGIN_IID "org.deepin.plugin.modelhub.inference-backend"

class ModelProxy;

class InferencePlugin {
public:
    explicit InferencePlugin() {}
    virtual ~InferencePlugin() {}
    
    virtual bool initialize(const VariantMap &params) = 0;
    
    virtual ModelProxy* loadModel(const std::string &name,
                                const std::string &imgDir,
                                const VariantMap &params) = 0;
};

GLOBAL_END_NAMESPACE

typedef GLOBAL_NAMESPACE::InferencePlugin* (*createInferencePluginFunc)();

#endif // INFERENCEPLUGIN_H
