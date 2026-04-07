// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "global_header.h"
#include "command.h"
#include "../environments.h"

#include <signal.h>
#include <unistd.h>

GLOBAL_USE_NAMESPACE

int main(int argc, char *argv[])
{
    // 安全退出
    signal(SIGINT, Command::appExitHandler);
    signal(SIGQUIT, Command::appExitHandler);
    signal(SIGTERM, Command::appExitHandler);

    Command cmd;
    return cmd.processCmd(argc, argv);
}
