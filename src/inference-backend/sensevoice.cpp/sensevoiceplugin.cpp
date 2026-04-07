// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "sensevoiceplugin.h"
#include "sensevoiceconfig.h"
#include "sensevoiceasrproxy.h"
#include "util.h"
#include "modelinfo.h"

#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cctype>

#include "common.h"

GLOBAL_USE_NAMESPACE

extern "C" { // 使用C链接，以避免C++名称修饰
    InferencePlugin* createInferencePlugin() {
        return new SenseVoicePlugin();
    }
}

namespace fs = std::filesystem;

SenseVoicePlugin::SenseVoicePlugin() : InferencePlugin()
{
}

SenseVoicePlugin::~SenseVoicePlugin()
{
}

bool SenseVoicePlugin::initialize(const VariantMap &params)
{
    return true;
}

ModelProxy *SenseVoicePlugin::loadModel(const std::string &name, const std::string &imgDir, const VariantMap &params)
{
    fs::path dir(imgDir);
    SenseVoiceConfig cfg (dir / "config.json");
    SenseVoiceModelWrapper *sensevoiceModel = createModelWrapper(name, cfg.architectures());
    auto mParams = cfg.params();
    for (const auto& [key, value] : params) {
        if (mParams.find(key) == mParams.end()) {
            std::string strValue = Util::findValue(params, key, std::string());
            if (!strValue.empty())
                mParams[key] = strValue;
        }
    }

    if (!sensevoiceModel->initialize(dir / cfg.bin(), mParams)) {
        delete sensevoiceModel;
        return nullptr;
    }

    return dynamic_cast<ModelProxy *>(sensevoiceModel);
}

SenseVoiceModelWrapper *SenseVoicePlugin::createModelWrapper(const std::string &name, const std::vector<std::string> &archs)
{
    SenseVoiceModelWrapper *ret = nullptr;

    auto compareIgnoreCase = [](const std::string &a, const std::string &b) {
        if (a.size() != b.size())
            return false;
        return strcasecmp(a.c_str(), b.c_str()) == 0;
    };

    for (const std::string &arch : archs) {
        if (compareIgnoreCase(arch, "ASR")) {
            ret = new SenseVoiceASRProxy(name);
        }
    }
    return ret;
}
