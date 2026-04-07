// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runtimestate.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <unistd.h>
#include <limits.h>

GLOBAL_USE_NAMESPACE

namespace fs = std::filesystem;
using json = nlohmann::json;

RuntimeState::RuntimeState(const std::string& m) : model(m) {}

int RuntimeState::pid() const {
    // Get state file path
    auto runFile = stateFile();
    if (!fs::exists(runFile)) {
        return -1;
    }

    try {
        // Read and parse JSON file
        std::ifstream file(runFile);
        json state = json::parse(file);

        // Check if PID exists and process is running
        if (state.contains(kRuntime_state_pid)) {
            std::string pid = state.value(kRuntime_state_pid, std::string("-2"));
            std::string procPath = "/proc/" + pid;
            if (fs::exists(procPath)) {
                char path[PATH_MAX] = {0};
                std::string exe = EXE_NAME;
                readlink(std::string(procPath + "/exe").c_str(), path, sizeof(path) - 1);
                std::string pexe = path;
                auto it = std::search(pexe.begin(), pexe.end(), exe.begin(), exe.end());
                if (it != pexe.end())
                    return std::stoi(pid);
            }
        }
    } catch (...) {
        // Return -1 on any errors
        return -1;
    }

    return -1;
}

std::string RuntimeState::stateFile() const {
    return stateDir() + "/" + model + ".state";
}

std::string RuntimeState::stateDir() {
    // Get system temp directory
    std::string tempDir;
    if (const char* tmp = std::getenv("TMPDIR")) {
        tempDir = tmp;
    } else {
        tempDir = "/tmp";
    }

    // Append program-specific subdirectory
    return tempDir + "/" + std::string(EXE_NAME) + "-" + std::to_string(getuid());
}

void RuntimeState::mkpath() {
    auto dir = stateDir();
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
}

void RuntimeState::writeState(std::fstream& file, const std::map<std::string, std::string>& state) {
    if (!file.is_open()) {
        return;
    }

    // Convert map to JSON
    json j;
    for (const auto& [key, value] : state) {
        j[key] = value;
    }

    // Clear file and write new content
    file.seekp(0);
    file.clear();
    file << j.dump(4);
    file.flush();
}

std::vector<std::map<std::string, std::string>> RuntimeState::listAll() {
    std::vector<std::map<std::string, std::string>> ret;
    
    try {
        // Iterate through all .state files
        for (const auto& entry : fs::directory_iterator(stateDir())) {
            if (entry.path().extension() != ".state") {
                continue;
            }

            // Read and parse state file
            std::ifstream file(entry.path());
            json state = json::parse(file);

            // Check if process is still running
            if (state.contains(kRuntime_state_pid)) {
                std::string pid = state.value(kRuntime_state_pid, "-2");
                std::string procPath = "/proc/" + pid;
                
                if (fs::exists(procPath)) {
                    // Convert JSON to map
                    std::map<std::string, std::string> stateMap;
                    for (auto& [key, value] : state.items()) {
                        stateMap[key] = value.get<std::string>();
                    }
                    ret.push_back(stateMap);
                }
            }
        }
    } catch (...) {
        // Return empty vector on any errors
    }

    return ret;
}
