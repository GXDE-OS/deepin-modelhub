// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef LLAMACPPPLUGIN_H
#define LLAMACPPPLUGIN_H

#include "inferenceplugin.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

GLOBAL_BEGIN_NAMESPACE

class LlamacppModelConfig;
class LlamaModelWrapper;

class LlamacppPlugin : public InferencePlugin {
public:
    LlamacppPlugin();
    ~LlamacppPlugin();
    
    bool initialize(const VariantMap &params) override;
    ModelProxy* loadModel(const std::string &name, 
                         const std::string &imgDir,
                         const VariantMap &params) override;
    
    static LlamaModelWrapper *createModelWrapper(const std::string &name, 
                                               const std::vector<std::string> &archs);
private:
    bool inited = false;
};

GLOBAL_END_NAMESPACE

#endif // LLAMACPPPLUGIN_H
