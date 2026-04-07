// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef UTIL_H
#define UTIL_H

#include "global_header.h"

GLOBAL_BEGIN_NAMESPACE

class Util
{
public:
    template<typename T>
    static T findValue(const VariantMap &map, const std::string &key, const T &defaultValue) {
        auto iter = map.find(key);
        if (iter != map.end()) {
            if (std::holds_alternative<T>(iter->second)) {
                return std::get<T>(iter->second);
            }
        }
        return defaultValue;
    }
protected:
    Util() {}
};

GLOBAL_END_NAMESPACE

#endif // UTIL_H
