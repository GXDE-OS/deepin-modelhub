// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef OVPLUGIN_H
#define OVPLUGIN_H

#include "inferenceplugin.h"

#include <list>

namespace ov {
class Core;
}

GLOBAL_BEGIN_NAMESPACE

class OVPlugin : public InferencePlugin
{
public:
    explicit OVPlugin();
    ~OVPlugin();
    bool initialize(const VariantMap &params) override;
    ModelProxy* loadModel(const std::string &name,
                          const std::string &imgDir,
                          const VariantMap &params);
private:
    ov::Core *ovCore = nullptr;
    std::vector<std::string> devices;
};

GLOBAL_END_NAMESPACE

#endif // OVPLUGIN_H
