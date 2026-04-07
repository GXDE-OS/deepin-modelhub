// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef HTTPSERVER_H
#define HTTPSERVER_H

#include "global_header.h"

#include <string>
#include <functional>
#include <memory>

GLOBAL_BEGIN_NAMESPACE

class HttpServerPrivate;
class HttpContext;

using chunckProvider = std::function<bool(std::string &out, bool &stop)>;
using chunckComplete = std::function<void(bool success)>;
using RequestCallback = std::function<void(void* ctx)>;

class HttpServer {
    friend class HttpServerPrivate;
public:
    enum ReqType {Get,Post};
    explicit HttpServer();
    ~HttpServer();
    
    bool initialize(std::string host = "", int port = -1);
    bool registerAPI(ReqType type, const std::string &api);
    static int randomPort();
    void setLastWorkTime(int seconds);
    void setRequestCallback(RequestCallback cb);

    static std::string getPath(HttpContext *ctx);
    static std::string getBody(HttpContext *ctx);
    static std::string getFileContent(HttpContext *ctx, const std::string &key);
    static std::string getFileType(HttpContext *ctx, const std::string &key);
    static void setContent(HttpContext *ctx, const std::string &content, 
                          const std::string &type = "application/json; charset=utf-8");
    static void setStatus(HttpContext *ctx, int st = 200);
    static void setChunckProvider(HttpContext *ctx, chunckProvider cp, 
                                 chunckComplete cc,
                                 const std::string &type = "text/event-stream");
    void stop();
    void run();
private:
    std::unique_ptr<HttpServerPrivate> d;
    RequestCallback request_callback;
};

GLOBAL_END_NAMESPACE

#endif // HTTPSERVER_H
