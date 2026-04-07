// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MODELLOADER_H
#define MODELLOADER_H

#include "global_header.h"

GLOBAL_BEGIN_NAMESPACE

class ModelLoaderPrivate;

class ModelLoader {
    friend class ModelLoaderPrivate;
public:
    ModelLoader();
    ~ModelLoader();

    // 禁用拷贝构造和赋值操作符
    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;

private:
    ModelLoaderPrivate* d;  // PIMPL模式
};

GLOBAL_END_NAMESPACE

#endif // MODELLOADER_H
