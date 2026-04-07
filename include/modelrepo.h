// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MODELREPO_H
#define MODELREPO_H

#include "global_header.h"
#include <string>
#include <vector>
#include <memory>
#include <utility>

GLOBAL_BEGIN_NAMESPACE

typedef std::pair<std::string, std::string> ModelProperty;

class ModelInfo;
class ModelRepoPrivate;

class ModelRepo {
    friend class ModelRepoPrivate;
public:
    explicit ModelRepo();
    ~ModelRepo();
    
    void setRepoPath(const std::vector<std::string> &path);
    std::vector<std::string> list() const;
    std::vector<ModelProperty> show(const std::string &model) const;
    std::shared_ptr<ModelInfo> modelInfo(const std::string &model) const;

private:
    std::unique_ptr<ModelRepoPrivate> d;
};

using ModelInfoPointer = std::shared_ptr<ModelInfo>;

GLOBAL_END_NAMESPACE

#endif // MODELREPO_H
