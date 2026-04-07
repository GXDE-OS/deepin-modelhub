// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef OVTOKENIZER_H
#define OVTOKENIZER_H

#include "global_header.h"

#include <openvino/openvino.hpp>

#include <string>
#include <vector>

GLOBAL_BEGIN_NAMESPACE

class OvTokenizerPrivate;
class OvTokenizer
{
    friend class OvTokenizerPrivate;
public:
    explicit OvTokenizer(ov::Core *core, const std::string &tokenizerXml,
                         const std::string &detokenizerXml, const std::string &configPath);
    virtual ~OvTokenizer();
    std::vector<int64_t> encode(const std::string &prompt);
    std::string decode(const std::vector<int64_t> &tokens);

    int64_t bosTokenId() const;
    int64_t eosTokenId() const;
    int64_t padTokenId() const;

    std::string bosToken() const;
    std::string eosToken() const;
    std::string padToken() const;
private:
    OvTokenizerPrivate *d = nullptr;
};

GLOBAL_END_NAMESPACE

#endif // OVTOKENIZER_H
