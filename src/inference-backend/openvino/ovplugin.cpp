// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ovplugin.h"
#include "ovmodelconfig.h"
#include "ovembproxy.h"
#include "ovllmproxy.h"

#include <openvino/openvino.hpp>

#include <iostream>

GLOBAL_USE_NAMESPACE

extern "C" { // 使用C链接，以避免C++名称修饰
    InferencePlugin* createInferencePlugin() {
        return new OVPlugin();
    }
}

static bool compareArch(const std::string& src, const std::string& trgt) {
    if (src.length() != trgt.length()) {
        return false;
    }
    return strcasecmp(src.c_str(), trgt.c_str()) == 0;
}

OVPlugin::OVPlugin() : InferencePlugin()
{

}

OVPlugin::~OVPlugin()
{
    delete ovCore;
    ovCore = nullptr;
}

bool OVPlugin::initialize(const VariantMap &params)
{
    if (ovCore)
        return true;

    ovCore = new ov::Core;

    std::cerr << ov::get_openvino_version() << std::endl;
    std::vector<std::string> availableDevices = ovCore->get_available_devices();
    std::cerr << "Available devices: " << std::endl;
    for (auto&& device : availableDevices) {
        devices.push_back(device);

        // Query supported properties and print all of them
        std::cerr << device << " SUPPORTED_PROPERTIES: " << std::endl;
        auto supported_properties = ovCore->get_property(device, ov::supported_properties);
        for (auto&& property : supported_properties) {
            if (property != ov::supported_properties.name()) {
                std::cerr  << "\t" << (property.is_mutable() ? "Mutable: " : "Immutable: ") << property <<
                           " : " << ovCore->get_property(device, property).as<std::string>() << std::endl;
            }
        }
    }

    return true;
}

ModelProxy *OVPlugin::loadModel(const std::string &name, const std::string &imgDir, const VariantMap &params)
{
    ModelProxy *ret = nullptr;
    OVModelConfig cfg (imgDir + "/config.json");

    auto tmpParams = params;
    tmpParams["Devices"] = devices;
    auto cfgParms = cfg.params();
    for (auto it = cfgParms.begin(); it != cfgParms.end(); ++it) {
        if (tmpParams.find(it->first) != tmpParams.end())
            tmpParams[it->first] = it->second;
    }

    for (const std::string &arch : cfg.architectures()) {
        if (compareArch(arch, "LLM")) {
            auto model = new OvLLMProxy(name, ovCore);
            if (!model->initialize(imgDir + "/" + cfg.model(), imgDir + "/" + cfg.tokenizer(), imgDir + "/" + cfg.detokenizer(),tmpParams)) {
                delete model;
                model = nullptr;
            }
            ret = model;

        } else if (compareArch(arch, "Embedding")) {
            auto model = new OVEmbProxy(name, ovCore);
            if (!model->initialize(imgDir + "/" + cfg.model(), imgDir + "/" + cfg.tokenizer(), tmpParams)) {
                delete model;
                model = nullptr;
            }
            ret = model;
        }
    }

    return ret;
}
