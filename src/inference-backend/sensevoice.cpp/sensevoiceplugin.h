// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef SENSEVOICEPLUGIN_H
#define SENSEVOICEPLUGIN_H

#include "inferenceplugin.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

GLOBAL_BEGIN_NAMESPACE
class SenseVoiceConfig;
class SenseVoiceModelWrapper;

class SenseVoicePlugin : public InferencePlugin
{
public:
    SenseVoicePlugin();
    ~SenseVoicePlugin();
    bool initialize(const VariantMap &params) override;
    ModelProxy* loadModel(const std::string &name, const std::string &imgDir, const VariantMap &params) override;
    static SenseVoiceModelWrapper *createModelWrapper(const std::string &name, const std::vector<std::string> &archs);
};

GLOBAL_END_NAMESPACE

#endif // SENSEVOICEPLUGIN_H
