// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "httpserver_p.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>
#include <random>

GLOBAL_USE_NAMESPACE

using json = nlohmann::json;

HttpServerPrivate::HttpServerPrivate(HttpServer *parent) : q(parent)
{
}

bool HttpServerPrivate::isPortInUse(int port)
{
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "netstat -tuln | grep :%d", port);
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        std::cerr << "ERROR: popen(netstat) failed!" << std::endl;
        return false;
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        if (strstr(buffer, "LISTEN")) {
            pclose(pipe);
            return true;
        }
    }

    pclose(pipe);
    return false;
}

bool HttpServerPrivate::inerApi(const httplib::Request &req, httplib::Response &res)
{
    if (req.path == "/health") {
        res.status = 200;
        json response;
        auto now = std::chrono::system_clock::now();
        auto cur = std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count();
            
        response["status"] = "ok";
        response["time"] = cur;
        response["idle"] = lastWorkTime > 0 ? cur - lastWorkTime : -1;
        
        res.set_content(response.dump(), "application/json");
        return true;
    }
    return false;
}

HttpServer::HttpServer() : d(std::make_unique<HttpServerPrivate>(this))
{
}

bool HttpServer::initialize(std::string host, int port)
{
    if (d->hserve)
        return false;

    if (host.empty())
        host = "127.0.0.1";

    // create http server
    d->hserve = new httplib::Server();
    d->hserve->set_read_timeout(600);
    d->hserve->set_write_timeout(600);
#ifdef NDEBUG
    d->hserve->set_socket_options(nullptr); // default_socket_options sets SO_REUSEPORT
#endif

    if (!d->hserve->bind_to_port(host, port)) {
        std::cerr << "couldn't bind to server socket: hostname=" << host 
                  << " port=" << port << std::endl;
        return false;
    } else {
        std::cerr << "http server listen " << host << " " << port << std::endl;
    }

    d->hserve->set_error_handler([](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res.set_content("{\"error\":\"File Not Found\"}", 
                          "application/json; charset=utf-8");
        }
    });

    // api validation
    d->hserve->set_pre_routing_handler([this](const httplib::Request & req, 
                                             httplib::Response & res) {
        if (d->inerApi(req, res))
            return httplib::Server::HandlerResponse::Handled;
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // internal api
    registerAPI(Get, "/health");

    return true;
}

bool HttpServer::registerAPI(HttpServer::ReqType type, const std::string &api)
{
    if (api.empty())
        return false;

    auto handler = [this](const httplib::Request &req, httplib::Response & res){
        HttpContext ctx{&req, &res};
        // block，there is in work thread on http
        request_callback(&ctx);
    };
    if (type == Get)
       d->hserve->Get(api, handler);
    else
       d->hserve->Post(api, handler);

   return true;
}

int HttpServer::randomPort()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(30000, 39999);
    
    int count = 10;
    while (count--) {
        int port = dis(gen);
        if (!HttpServerPrivate::isPortInUse(port))
            return port;
    }
    return -1;
}

void HttpServer::setLastWorkTime(int seconds)
{
    d->lastWorkTime = seconds;
}

std::string HttpServer::getPath(HttpContext *ctx)
{
    return ctx->req->path;
}

std::string HttpServer::getBody(HttpContext *ctx)
{
    return ctx->req->body;
}

std::string HttpServer::getFileType(HttpContext *ctx, const std::string &key)
{
    return ctx->req->get_file_value(key).content_type;
}

std::string HttpServer::getFileContent(HttpContext *ctx, const std::string &key)
{
    return ctx->req->get_file_value(key).content;
}

void HttpServer::setContent(HttpContext *ctx, const std::string &content, const std::string &type)
{
    ctx->res->set_content(content, type);
}

void HttpServer::setStatus(deepin_modelhub::HttpContext *ctx, int st)
{
    ctx->res->status = st;
}

void HttpServer::setChunckProvider(HttpContext *ctx, chunckProvider cp, chunckComplete cc, const std::string &type)
{
    const auto chunked_content_provider = [cp](size_t, httplib::DataSink & sink) {
        while (true) {
            std::string out;
            bool stop = false;
            if (cp(out, stop)) {
                if (!sink.is_writable()) {
#ifndef NDEBUG
                    std::cerr << "fail to write, maybe the remote is closed." << std::endl;
#endif
                    return false;
                }

                if (!sink.write(out.c_str(), out.size()))
                    return false;

                if (stop)
                    break;
            } else {
                std::string error = R"(error:{""})";
                if (!sink.write(error.c_str(), error.size()))
                    return false;
                break;
            }
        }
        sink.done();
        return true;
    };

    ctx->res->set_chunked_content_provider(type, chunked_content_provider, cc);
}

void HttpServer::stop() {
    if (d->hserve)
        d->hserve->stop();
}

void HttpServer::setRequestCallback(RequestCallback cb) {
    request_callback = std::move(cb);
}

void HttpServer::run()
{
    std::cerr << "http server is running." << std::endl;
    bool ret = d->hserve->listen_after_bind();
    std::cerr << "http server exit." << ret << std::endl;
}

HttpServer::~HttpServer()
{
    stop();

    delete d->hserve;
    d->hserve = nullptr;
}

