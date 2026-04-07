// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RUNTIMESTATE_H
#define RUNTIMESTATE_H

#include "global_header.h"

#include <string>
#include <vector>
#include <map>
#include <fstream>

GLOBAL_BEGIN_NAMESPACE

// Key for process ID in state file
inline constexpr char kRuntime_state_pid[]{"pid"};

/**
 * @brief Manages runtime state for model processes
 * 
 * This class handles reading and writing state information for model processes,
 * including their PIDs and other metadata. State is persisted to JSON files
 * in a temporary directory.
 */
class RuntimeState {
public:
    // Constructor taking model name
    explicit RuntimeState(const std::string& model);

    // Get PID of running model process, returns -1 if not running
    int pid() const;

    // Get full path to state file for this model
    std::string stateFile() const;

    // Get directory path for all state files
    static std::string stateDir();

    // Create state directory if it doesn't exist
    static void mkpath();

    // Write state data to file
    static void writeState(std::fstream& file, const std::map<std::string, std::string>& state);

    // List all running model states
    static std::vector<std::map<std::string, std::string>> listAll();

protected:
    std::string model;  // Model identifier
};

GLOBAL_END_NAMESPACE

#endif // RUNTIMESTATE_H
