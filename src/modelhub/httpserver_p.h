// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef HTTPSERVER_P_H
#define HTTPSERVER_P_H

#include "httpserver.h"

#include "llama.cpp/examples/server/httplib.h"

#include <unordered_map>
#include <chrono>

GLOBAL_BEGIN_NAMESPACE

struct HttpContext
{
    const httplib::Request *req;
    httplib::Response *res;
};

class HttpServerPrivate
{
public:
    explicit HttpServerPrivate(HttpServer *parent);
    static bool isPortInUse(int port);
    bool inerApi(const httplib::Request & req, httplib::Response & res);
public:
    httplib::Server *hserve = nullptr;
    int lastWorkTime = -1;
private:
    HttpServer *q;

};

GLOBAL_END_NAMESPACE

#endif // HTTPSERVER_P_H
