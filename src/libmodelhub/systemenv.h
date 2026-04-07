// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef SYSTEMENV_H
#define SYSTEMENV_H

#include "global_header.h"
#include <string>
#include <vector>

GLOBAL_BEGIN_NAMESPACE

/**
 * @class SystemEnv
 * @brief 系统环境信息获取类
 * 
 * 该类提供了一系列静态方法用于获取系统硬件和软件环境信息，
 * 包括CPU指令集、CPU型号、显卡信息、加速器信息以及库文件检查等功能
 */
class SystemEnv
{
public:
    /**
     * @brief 获取CPU支持的指令集列表
     * @return 包含所有CPU指令集的字符串向量
     * @details 通过读取/proc/cpuinfo文件获取CPU支持的指令集信息
     */
    static std::vector<std::string> cpuInstructions();

    /**
     * @brief 获取CPU型号名称
     * @return CPU型号名称字符串
     * @details 通过读取/proc/cpuinfo文件获取CPU型号信息
     */
    static std::string cpuModelName();

    /**
     * @brief 获取系统显卡信息
     * @return 显卡信息字符串
     * @details 通过执行lspci命令并过滤VGA相关信息获取显卡详情
     */
    static std::string vga();

    /**
     * @brief 获取系统加速器信息
     * @return 加速器信息字符串
     * @details 通过执行lspci命令并过滤Processing accelerators相关信息获取加速器详情
     */
    static std::string accelerators();

    /**
     * @brief 检查指定库文件的依赖是否完整
     * @param lib 待检查的库文件路径
     * @return true表示库文件依赖完整，false表示存在缺失依赖
     * @details 通过执行ldd命令检查库文件的依赖完整性
     */
    static bool checkLibrary(const std::string &lib, const std::vector<std::string> &ignore = {});
};

GLOBAL_END_NAMESPACE

#endif //SYSTEMENV_H

