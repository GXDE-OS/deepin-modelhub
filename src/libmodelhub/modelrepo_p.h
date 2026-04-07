// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MODELREPO_P_H
#define MODELREPO_P_H

#include "modelrepo.h"

GLOBAL_BEGIN_NAMESPACE

class ModelRepoPrivate
{
public:
   explicit ModelRepoPrivate(ModelRepo *parent);
public:
    std::vector<std::string> path;
private:
    ModelRepo *q;
};

GLOBAL_END_NAMESPACE

#endif // MODELREPO_P_H
