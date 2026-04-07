// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "systemenv.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <filesystem>

GLOBAL_USE_NAMESPACE

std::vector<std::string> SystemEnv::cpuInstructions()
{
    std::vector<std::string> rets;
    // 打开/proc/cpuinfo文件
    std::ifstream file("/proc/cpuinfo");
    if (!file.is_open())
        return rets;

    // 逐行读取文件内容
    std::string line;
    while (std::getline(file, line)) {
        // 查找包含flags的行
        if (line.find("flags") == 0) {
            // 提取冒号后的内容并按空格分割
            std::istringstream iss(line.substr(line.find(":") + 1));
            std::string flag;
            while (iss >> flag) {
                rets.push_back(flag);
            }
            break;
        }
    }

    file.close();
    return rets;
}

std::string SystemEnv::cpuModelName()
{
    std::string ret;
    // 打开/proc/cpuinfo文件
    std::ifstream file("/proc/cpuinfo");
    if (!file.is_open())
        return ret;

    // 逐行读取文件内容
    std::string line;
    while (std::getline(file, line)) {
        // 查找包含model name的行
        if (line.find("model name") == 0) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                // 提取冒号后的内容
                ret = line.substr(pos + 1);
                // 去除开头的空白字符
                ret.erase(0, ret.find_first_not_of(" \t"));
            }
            break;
        }
    }

    file.close();
    return ret;
}

std::string SystemEnv::vga()
{
    // 构建lspci命令
    char cmd[512] = {0};
    snprintf(cmd, sizeof(cmd), "lspci|grep VGA");
    
    // 执行命令并获取输出
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        std::cerr << "ERROR: lspci|grep VGA " << std::endl;
        return "";
    }

    // 读取命令输出
    char buffer[1024] = {0};
    std::string ret;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        if (strstr(buffer, "VGA")) {
            ret.append(buffer);
        }
    }

    pclose(pipe);
    return ret;
}

std::string SystemEnv::accelerators()
{
    // 构建lspci命令
    char cmd[512] = {0};
    snprintf(cmd, sizeof(cmd), "lspci|grep 'Processing accelerators'");
    
    // 执行命令并获取输出
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        std::cerr << "ERROR: lspci|grep Processing accelerators " << std::endl;
        return "";
    }

    // 读取命令输出
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        if (strstr(buffer, "Processing accelerators")) {
            std::string str(buffer);
            pclose(pipe);
            return str;
        }
    }

    pclose(pipe);
    return "";
}

bool SystemEnv::checkLibrary(const std::string &lib, const std::vector<std::string> &ignore)
{
    // 构建ldd命令
    char cmd[512] = {0};
    snprintf(cmd, sizeof(cmd), "ldd %s", lib.c_str());
    
    // 执行命令并获取输出
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        std::cerr << "ERROR: ldd " << lib << std::endl;
        return false;
    }

    // 检查是否存在"not found"字样
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        if (strstr(buffer, "not found")) {
            bool cont = false;
            for (const std::string &so : ignore) {
                if (strstr(buffer, so.c_str())) {
                    cont = true;
                    break;
                }
            }
            if (cont)
                continue;

            // 输出错误信息，包含库文件名
            std::cerr << std::filesystem::path(lib).filename().string() << ":" << buffer;
            pclose(pipe);
            return false;
        }
    }

    pclose(pipe);
    return true;
}
