// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef SENSEVOICEASRPROXY_H
#define SENSEVOICEASRPROXY_H

#include "asrproxy.h"
#include "sensevoicemodelwrapper.h"

GLOBAL_BEGIN_NAMESPACE

class SenseVoiceASRProxy : public ASRProxy, public SenseVoiceModelWrapper
{
    struct LLMGenerateContext {
        std::string generatedText;
        size_t pushedPos = 0;
    };
public:
    explicit SenseVoiceASRProxy(const std::string &name);
    std::string name() const override;
    bool decodeContent(const std::string &content, std::vector<double> &pcmf32,
                               const std::map<std::string, std::string> &params = {}) override;
    std::vector<int32_t> transcriptions(const std::vector<double> &pcmf32,
                                                const std::map<std::string, std::string> &params = {},
                                                generateStream stream = nullptr,
                                                void *user = nullptr) override;
    std::string detokenize(const std::vector<int32_t> &tokens,
                                   const std::map<std::string, std::string> &params) override;
protected:
    bool decodePcm(const std::string &content, std::vector<double> &pcmf32) const;
    bool decodeWav(const std::string &content, std::vector<double> &pcmf32) const;
protected:
    std::string modelName;
};

GLOBAL_END_NAMESPACE

#endif // SENSEVOICEASRPROXY_H
