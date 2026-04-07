// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef COMMAND_H
#define COMMAND_H

#include "global_header.h"
#include <string>
#include <map>
#include <vector>

GLOBAL_BEGIN_NAMESPACE

class ModelRunner;
class Command {
public:
    explicit Command();
    int processCmd(int argc, char* argv[]);
    static void appExitHandler(int);

protected:
    int listHandler();
    int llmGenerate();
    int embeddings();
    int runServer();
    int stopServer();
    int loadModel(const std::string &modelName, ModelRunner *runner, int parallel = 1);
    void showHelp(int exitCode);

private:
    static bool llmStreamOutput(const std::string &text, void *llm);
    static int64_t getTimeUs();
    static void warmUp(class ModelProxy *model);
    
protected:
    void initOptions();
    bool isSet(const std::string& option) const;
    std::string value(const std::string& option) const;

private:
    std::map<std::string, std::string> m_options;
    std::vector<std::string> m_args;
};

GLOBAL_END_NAMESPACE

#endif // COMMAND_H
