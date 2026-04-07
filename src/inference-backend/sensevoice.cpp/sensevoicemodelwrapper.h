// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef SENSEVOICEMODELWRAPPER_H
#define SENSEVOICEMODELWRAPPER_H

#include "modelproxy.h"
#include "sensevoiceconfig.h"

#include "common.h"
#include "sense-voice.h"

#include <string>
#include <unordered_map>

GLOBAL_BEGIN_NAMESPACE

class SenseVoiceModelWrapper
{
public:
    explicit SenseVoiceModelWrapper();
    virtual ~SenseVoiceModelWrapper();

    virtual bool initialize(const std::string &bin, const std::unordered_map<std::string, std::string> &params);
protected:
    sense_voice_context *ctx = nullptr;//模型
    sense_voice_full_params wparams;//参数
};

GLOBAL_END_NAMESPACE

#endif // SENSEVOICEMODELWRAPPER_H
