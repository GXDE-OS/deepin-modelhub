// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef CONFIGMANAGER_H
#define CONFIGMANAGER_H

#include "global_header.h"
#include <string>
#include <vector>

GLOBAL_BEGIN_NAMESPACE

/**
 * @brief 配置管理器类的私有实现
 */
class ConfigManagerPrivate;

/**
 * @brief 配置管理器类
 * 
 * 用于管理应用程序的配置信息,采用单例模式实现
 * 主要功能包括:
 * 1. 管理模型仓库路径
 * 2. 提供配置文件路径
 */
class ConfigManager
{
public:
    /**
     * @brief 获取ConfigManager的全局单例实例
     * @return ConfigManager单例对象指针
     */
    static ConfigManager* instance();

    /**
     * @brief 获取模型仓库路径列表
     * @return 从配置文件读取的模型路径列表,如果未配置则返回空列表
     */
    std::vector<std::string> modelRepoPaths() const;
    
    /**
     * @brief 设置模型仓库路径列表
     * @param paths 要保存到配置文件的路径列表
     */
    void setModelRepoPaths(const std::vector<std::string>& paths);

    /**
     * @brief 获取配置文件的完整路径
     * @return 配置文件路径,例如: ~/.config/deepin/[EXE_NAME]/config.conf
     */
    std::string configPath() const;

private:
    ConfigManager();
    ~ConfigManager();
    
    ConfigManagerPrivate* d;  // PIMPL模式的私有实现指针
    static ConfigManager* m_instance;  // 单例实例指针
};

GLOBAL_END_NAMESPACE

#endif // CONFIGMANAGER_H 