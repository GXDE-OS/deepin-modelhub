// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef OVTOKENIZER_P_H
#define OVTOKENIZER_P_H

#include "ovtokenizer.h"

#include <nlohmann/json.hpp>

GLOBAL_BEGIN_NAMESPACE

class OvTokenizerPrivate
{
public:
    explicit OvTokenizerPrivate(ov::Core *core, OvTokenizer *qq);
    bool initialize(const std::string &tokenizerXml, const std::string &detokenizerXml, const std::string &configPath);
public:
    ov::Core *ovCore;
    std::string chatTemplate;
    ov::CompiledModel tokenizerModel;
    ov::CompiledModel detokenizerModel;

    int64_t padTokenId = -1;
    int64_t bosTokenId = -1;
    int64_t eosTokenId = -1;

    std::string padToken = "";
    std::string bosToken = "";
    std::string eosToken = "";
protected:
    void loadChatTemplate(const std::string &path);
    void loadTokenConfig(const std::string &path);
    nlohmann::json loadJson(const std::string &filePath);
private:
    OvTokenizer *q;
};

GLOBAL_END_NAMESPACE

#endif // OVTOKENIZER_P_H
