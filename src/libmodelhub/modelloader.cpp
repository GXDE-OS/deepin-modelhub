// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "modelloader.h"

GLOBAL_USE_NAMESPACE

ModelLoader::ModelLoader() 
    : d(nullptr)
{
}

ModelLoader::~ModelLoader() {
    if (d) {
        delete d;
        d = nullptr;
    }
}
