// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "configmanager.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <filesystem>

GLOBAL_BEGIN_NAMESPACE

/**
 * @brief ConfigManager的私有实现类
 */
class ConfigManagerPrivate 
{
public:
    /**
     * @brief 获取配置文件的标准路径
     * @return 返回用户配置目录下的应用配置文件完整路径
     */
    std::string configPath() const {
        std::string config_home;
        const char* xdg_config = std::getenv("XDG_CONFIG_HOME");
        if (xdg_config) {
            config_home = xdg_config;
        } else {
            const char* home = std::getenv("HOME");
            if (home) {
                config_home = std::string(home) + "/.config";
            }
        }
        return config_home + "/deepin/" + EXE_NAME + "/config.conf";
    }

    // 读取配置文件中的值
    std::vector<std::string> readValue(const std::string& section, const std::string& key) const {
        std::vector<std::string> result;
        std::ifstream file(configPath());
        if (!file.is_open()) {
            return result;
        }

        std::string line;
        bool in_section = false;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            if (line[0] == '[' && line.back() == ']') {
                in_section = (line.substr(1, line.size()-2) == section);
                continue;
            }

            if (in_section) {
                size_t pos = line.find('=');
                if (pos != std::string::npos) {
                    std::string k = line.substr(0, pos);
                    if (k == key) {
                        std::string value = line.substr(pos + 1);
                        std::stringstream ss(value);
                        std::string item;
                        while (std::getline(ss, item, ',')) {
                            if (!item.empty()) {
                                result.push_back(item);
                            }
                        }
                        break;
                    }
                }
            }
        }
        return result;
    }

    // 写入配置文件
    void writeValue(const std::string& section, const std::string& key, 
                   const std::vector<std::string>& values) {
        // 确保目录存在
        std::filesystem::path config_path(configPath());
        std::filesystem::create_directories(config_path.parent_path());

        // 构建值字符串
        std::string value_str;
        for (const auto& v : values) {
            if (!value_str.empty()) value_str += ",";
            value_str += v;
        }

        // 读取现有内容
        std::vector<std::string> lines;
        {
            std::ifstream file(configPath());
            if (file.is_open()) {
                std::string line;
                while (std::getline(file, line)) {
                    lines.push_back(line);
                }
            }
        }

        // 更新或添加配置
        bool section_found = false;
        bool key_found = false;
        
        for (size_t i = 0; i < lines.size(); ++i) {
            if (lines[i].empty() || lines[i][0] == '#') continue;

            if (lines[i][0] == '[' && lines[i].back() == ']') {
                if (lines[i].substr(1, lines[i].size()-2) == section) {
                    section_found = true;
                }
                continue;
            }

            if (section_found) {
                size_t pos = lines[i].find('=');
                if (pos != std::string::npos) {
                    std::string k = lines[i].substr(0, pos);
                    if (k == key) {
                        lines[i] = key + "=" + value_str;
                        key_found = true;
                        break;
                    }
                }
            }
        }

        if (!section_found) {
            lines.push_back("[" + section + "]");
            lines.push_back(key + "=" + value_str);
        } else if (!key_found) {
            lines.push_back(key + "=" + value_str);
        }

        // 写入文件
        std::ofstream file(configPath());
        if (file.is_open()) {
            for (const auto& line : lines) {
                file << line << std::endl;
            }
        }
    }
};

// 初始化静态单例指针
ConfigManager* ConfigManager::m_instance = nullptr;

ConfigManager* ConfigManager::instance()
{
    if (!m_instance) {
        m_instance = new ConfigManager();
    }
    return m_instance;
}

ConfigManager::ConfigManager()
    : d(new ConfigManagerPrivate())
{
}

ConfigManager::~ConfigManager()
{
    delete d;
}

std::string ConfigManager::configPath() const
{
    return d->configPath();
}

std::vector<std::string> ConfigManager::modelRepoPaths() const
{
    return d->readValue("repo", "model_paths");
}

void ConfigManager::setModelRepoPaths(const std::vector<std::string>& paths)
{
    d->writeValue("repo", "model_paths", paths);
}

GLOBAL_END_NAMESPACE 