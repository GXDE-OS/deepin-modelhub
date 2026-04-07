// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MODELINFO_H
#define MODELINFO_H

#include "global_header.h"
#include <string>
#include <vector>
#include <memory>

GLOBAL_BEGIN_NAMESPACE

class ModelInfo {
public:
    explicit ModelInfo(const std::string &path);
    std::string name() const;
    std::string path() const;
    std::vector<std::string> formats() const;
    std::vector<std::string> architectures(const std::string &format) const;
    std::string imagePath(const std::string &format) const;
    std::string chatTemplate(const std::string &format, const std::string &type) const;
    std::string version(const std::string &format) const;
private:
    std::string rootPath;
};

using ModelInfoPointer = std::shared_ptr<ModelInfo>;

GLOBAL_END_NAMESPACE

#endif // MODELINFO_H
