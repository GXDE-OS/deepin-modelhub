// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MODELPROXY_H
#define MODELPROXY_H

#include "global_header.h"
#include <string>

GLOBAL_BEGIN_NAMESPACE

class ModelProxy {
public:
    ModelProxy() {}
    virtual ~ModelProxy() {}
    
    virtual std::string name() const = 0;
    virtual ModelArchitecture architecture() const = 0;
};

GLOBAL_END_NAMESPACE

#endif // MODELPROXY_H
