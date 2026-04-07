// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "backendloader_p.h"
#include "inferenceplugin.h"
#include "systemenv.h"
#include "modelinfo.h"

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <regex>
#include <dirent.h>
#include <sys/stat.h>
#include <dlfcn.h>

GLOBAL_USE_NAMESPACE

#define GET_PROC_ADDR(handle, name) dlsym(handle, name)
typedef bool (*Initialize) (const std::unordered_map<std::string, ValueType> &params);

namespace fs = std::filesystem;


static bool contains_case_insensitive(const std::string& str, const std::string& substr) {
    auto it = std::search(
        str.begin(), str.end(),
        substr.begin(), substr.end(),
        [](char ch1, char ch2) {
            return std::tolower(ch1) == std::tolower(ch2);
        }
    );
    return it != str.end();
}

BackendLoaderPrivate::BackendLoaderPrivate(BackendLoader* parent) : q(parent) {}

std::vector<std::shared_ptr<BackendMetaObject>> BackendLoaderPrivate::sorted() const {
    std::vector<std::shared_ptr<BackendMetaObject>> ret = backends;
    std::stable_sort(ret.begin(), ret.end(), 
        [](const std::shared_ptr<BackendMetaObject>& t1, const std::shared_ptr<BackendMetaObject>& t2) {
            return std::get<float>(t1->extra(kInferenceBackendScore)) > 
                   std::get<float>(t2->extra(kInferenceBackendScore));
        });
    return ret;
}

void BackendLoaderPrivate::preload(std::shared_ptr<BackendMetaObject> mo) {
    // 处理llama.cpp后端的预加载

    bool isCuda = false;
    if (mo->name() == kInferenceBackendLlamaCpp) {
        std::string so = llamacppBackend();
        std::string soPath;
        
        // 如果用户配置了特定的llama库
        if (!so.empty()) {
            std::string soFilePath = std::string(PLUGIN_BACKEND_DIR) + "/llama.cpp/" + so;
            bool ok = std::get<bool>(mo->extra(fs::path(soFilePath).filename().string()));
            if (ok) {
                std::cerr << "load user setted libllama " << so << std::endl;
                soPath = fs::absolute(soFilePath).string();
            } else {
                std::cerr << "the libllama user setted is unavailable" << so << std::endl;
                so.clear();
            }
        }

        // 如果没有用户配置或配置无效,尝试默认库
        if (so.empty()) {
            // 按优先级顺序定义默认库列表
            static std::vector<std::string> defaultso = {"libllama-cuda.so", "libllama-avx2.so", "libllama.so"};
            for (const auto& fso : defaultso) {
                bool ok = std::get<bool>(mo->extra(fso, false));
                if (ok) {
                    //如果匹配到cuda库，则设置cuda为true
                    if (fso == "libllama-cuda.so") {
                        isCuda = true;
                    }
                    soPath = std::string(PLUGIN_BACKEND_DIR) + "/llama.cpp/" + fso;
                    so = fso;
                    break;
                }
            }
        }

        // 如果没有找到可用的库,则退出
        if (so.empty()) {
            std::cerr << "no libllama for backend " << mo->name() << std::endl;
            return;
        }

        std::string llamaPath = std::string(PLUGIN_BACKEND_DIR) + "/llama.cpp/";
        static std::vector<std::string> ggmlSo;
        if (isCuda) {
            ggmlSo = {"libggml-base", "libggml-cpu", "libggml-cuda","libggml"};
        } else {
            ggmlSo = {"libggml-base", "libggml-cpu", "libggml"};
        }

        for (std::string ggmlFile : ggmlSo) {
            // 构造ggml库路径并尝试加载
            ggmlFile = std::regex_replace(so, std::regex("libllama"), ggmlFile);
            void* ggmlHandle = dlopen(std::string(llamaPath + ggmlFile).c_str(), RTLD_LAZY);
            if (ggmlHandle == nullptr) {
                std::cerr << "fail to load ggml so:" << llamaPath << ggmlFile << std::endl;
                return;
            } else {
                std::cerr << "gmml " << ggmlFile << " loaded:" << (bool)ggmlHandle << std::endl;
            }
        }

        void* llamaHandle = dlopen(soPath.c_str(), RTLD_LAZY);
        bool loaded = (llamaHandle != nullptr);

        std::cerr << "load libllama " << soPath << " loaded:" << loaded << std::endl;
    } else if (mo->name() == kInferenceBackendSenseVoice) {
        std::string soPath = std::string(PLUGIN_BACKEND_DIR) + "/llama.cpp/";
        static std::vector<std::string> ggmlSo = {"libggml-base.so", "libggml-cpu.so", "libggml.so"};

        {
            auto flags = SystemEnv::cpuInstructions();
            if (flags.empty())
                std::cerr << "fail to get cpu flags by read /proc/cpuinfo" << std::endl;

            // 检查CPU是否支持AVX2指令集
            bool ok = std::find_if(flags.begin(), flags.end(),
                [](const std::string& flag) {
                    return strcasecmp(flag.c_str(), "avx2") == 0;
                }) != flags.end();
            if (!ok) {
                std::cerr << "cpu does not support avx2" << std::endl;
            } else {
                ggmlSo = {"libggml-base-avx2.so", "libggml-cpu-avx2.so", "libggml-avx2.so"};
            }
        }

        for (const std::string &ggmlFile : ggmlSo) {
            void* ggmlHandle = dlopen(std::string(soPath + ggmlFile).c_str(), RTLD_LAZY);
            if (ggmlHandle == nullptr) {
                std::cerr << "fail to load ggml so:" << soPath << ggmlFile << std::endl;
                return;
            } else {
                std::cerr << "gmml " << ggmlFile << " loaded:" << (bool)ggmlHandle << std::endl;
            }
        }
        {
            soPath = std::string(std::string(PLUGIN_BACKEND_DIR) + "/sensevoice.cpp/libsense-voice-core.so");
            void *core = dlopen(soPath.c_str(), RTLD_LAZY);
            if (core == nullptr) {
                std::cerr << "fail to load sense-voice-core so:" << soPath << std::endl;
                return;
            }
        }
    }
}

void BackendLoaderPrivate::checkRuntime(std::shared_ptr<BackendMetaObject> mo) {
    if (!mo) return;

    // 检查llama.cpp后端
    if (mo->name() == kInferenceBackendLlamaCpp) {
        std::vector<std::string> libs;
        float score = 0;
        
        // 扫描llama.cpp目录下的所有.so文件
        DIR* dir = opendir((std::string(PLUGIN_BACKEND_DIR) + "/llama.cpp").c_str());
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string name = entry->d_name;
                if (name.find(".so") == std::string::npos || name.find("libllama") != 0) continue;
                
                libs.push_back(name);

                // 检查AVX2支持
                if (name == "libllama-avx2.so") {
                    auto flags = SystemEnv::cpuInstructions();
                    if (flags.empty())
                        std::cerr << "fail to get cpu flags by read /proc/cpuinfo" << std::endl;

                    // 检查CPU是否支持AVX2指令集
                    bool ok = std::find_if(flags.begin(), flags.end(), 
                        [](const std::string& flag) {
                            return strcasecmp(flag.c_str(), "avx2") == 0;
                        }) != flags.end();
                    if (!ok)
                        std::cerr << "cpu does not support avx2" << std::endl;
                    mo->setExtra(name, ValueType(ok));

                    // 设置性能评分
                    if (score < 3 && ok) {
                        score = 3;
                        mo->setExtra(kInferenceBackendScore, ValueType(score));
                    }
                } 
                // 检查CUDA支持
                else if (name == "libllama-cuda.so") {
                    std::string ggml = std::string(PLUGIN_BACKEND_DIR) + "/llama.cpp/libggml-cuda-cuda.so";
                    bool enable = contains_case_insensitive(SystemEnv::vga(), "nvidia");
                    if (!enable) {
                        std::cerr << "no nvidia vga device." << std::endl;
                    } else {
                        enable = SystemEnv::checkLibrary(ggml, {"libggml-base"});
                        if (!enable)
                            std::cerr << "no cuda runtime librarys." << std::endl;
                    }

                    mo->setExtra(name, ValueType(enable));
                    if (score < 5 && enable) {
                        score = 5;
                        mo->setExtra(kInferenceBackendScore, ValueType(score));
                    }
                } 
                // 基础库总是可用
                else {
                    mo->setExtra(name, ValueType(true));
                    if (score < 1) {
                        score = 1;
                        mo->setExtra(kInferenceBackendScore, ValueType(score));
                    }
                }
            }
            closedir(dir);
        }
        
        mo->setExtra(kInferenceBackendLibs, ValueType(libs));
        
    } 
    // 检查OpenVINO后端
    else if (mo->name() == kInferenceBackendOpenvino) {
        // 检查硬件支持情况
        bool cpu = contains_case_insensitive(SystemEnv::cpuModelName(), "intel");
        bool gpu = contains_case_insensitive(SystemEnv::vga(), "intel");
        bool npu = contains_case_insensitive(SystemEnv::accelerators(), "intel");

        // 检查运行时库是否可用
        bool enable = (cpu || gpu || npu) ? SystemEnv::checkLibrary(mo->fileName()) : false;
        mo->setExtra(kInferenceBackendRTReady, ValueType(enable));

        // 根据可用硬件设置性能评分
        float score = 0;
        if (cpu) score = 2;
        if (gpu) score = 5;
        if (npu) score = 6;

        mo->setExtra(kInferenceBackendScore, ValueType(score));
    } 
    // 其他后端默认评分为1
    else {
        mo->setExtra(kInferenceBackendScore, ValueType(float(1.0)));
    }
}

std::string BackendLoaderPrivate::readConfigValue(const std::string& group, const std::string& key) const {
    std::string config = configPath();
    std::ifstream file(config);
    if (!file.is_open()) return "";

    std::string line;
    bool inGroup = false;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] == '[') {
            inGroup = line == "[" + group + "]";
            continue;
        }
        if (inGroup && line.find(key + "=") == 0) {
            return line.substr(key.length() + 1);
        }
    }
    return "";
}

std::string BackendLoaderPrivate::fixedBackend() const {
    return readConfigValue("backend", "plugin");
}

std::string BackendLoaderPrivate::llamacppBackend() const {
    return readConfigValue("backend", "llama.cpp");
}

std::string BackendLoaderPrivate::configPath() const {
    const char* home = getenv("HOME");
    if (!home) return "";
    return std::string(home) + "/.config/deepin/" + EXE_NAME + "/config.conf";
}

BackendLoader::BackendLoader() : d(new BackendLoaderPrivate(this)) {
    std::vector<std::string> paths{PLUGIN_BACKEND_DIR};
    d->loadPaths = paths;
}


void BackendLoader::setPaths(const std::vector<std::string>& paths) {
    d->loadPaths = paths;
}

void BackendLoader::readBackends() {
    d->backends.clear();

    for (const auto& path : d->loadPaths) {
        DIR* dir = opendir(path.c_str());
        if (!dir) continue;

        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string name = entry->d_name;
            if (name.find(".so") == std::string::npos) continue;

            BackendMetaObjectPointer metaObj(new BackendMetaObject);
            std::string fileName = path + "/" + name;
            
            // Add error checking
            if (!metaObj->d) {
                std::cerr << "Failed to initialize backend metadata for: " << fileName << std::endl;
                continue;
            }
            
            metaObj->d->setFileName(fileName);  // Make sure fileName is set
            
            // Add validation before adding to backends
            if (metaObj->iid() == INFERENCE_PLUGIN_IID) {
                try {
                    d->checkRuntime(metaObj);
                    d->backends.push_back(metaObj);
                } catch (const std::exception& e) {
                    std::cerr << "Error processing backend " << fileName << ": " << e.what() << std::endl;
                }
            }
        }
        closedir(dir);
    }
}

std::vector<std::shared_ptr<BackendMetaObject>> BackendLoader::backends() const {
    return d->backends;
}

std::shared_ptr<InferencePlugin> BackendLoader::load(std::shared_ptr<BackendMetaObject> mo) const {
    if (!mo)
        return nullptr;

    d->preload(mo);
    std::cerr << "load backend " << mo->d->fileName << std::endl;

    void* handle = dlopen(mo->d->fileName.c_str(), RTLD_NOW);
    if (!handle) {
        std::cerr << "Failed to load plugin: " << dlerror() << std::endl;
        return nullptr;
    }   

    createInferencePluginFunc createFunc = reinterpret_cast<createInferencePluginFunc>(dlsym(handle, "createInferencePlugin"));
    if (!createFunc) {
        std::cerr << "Failed to find createInferencePlugin symbol in " << mo->d->fileName << std::endl;
        return nullptr;
    }

    std::shared_ptr<InferencePlugin> plugin(createFunc());
    return plugin;
}

std::shared_ptr<BackendMetaObject> BackendLoader::perfect(
    const std::shared_ptr<ModelInfo>& model,
    std::string* matchedFormat,
    std::string* matchedArch) const {
    
    // 如果模型信息为空则返回nullptr
    if (!model) return nullptr;

    // 获取用户配置的固定后端和已排序的后端列表
    std::string fixed = d->fixedBackend();
    auto sortedBackends = d->sorted();

    // 检查是否有用户配置的固定后端
    if (!fixed.empty()) {
        for (auto& bk : sortedBackends) {
            // 查后端文件名是否匹配固定后端
            if (fs::path(bk->fileName()).filename() == fixed) {
                auto formats = bk->supportedFormats();
                // 遍历模型支持的格式
                for (const auto& fmt : model->formats()) {
                    // 检查后端是否持该格式(不区分大小写)
                    if (std::find_if(formats.begin(), formats.end(),
                        [&fmt](const std::string& f) {
                            return strcasecmp(f.c_str(), fmt.c_str()) == 0;
                        }) != formats.end()) {
                        
                        auto archs = bk->supportedArchitectures();
                        // 遍历该格式下支持的架构
                        for (const auto& arch : model->architectures(fmt)) {
                            // 检查后端是否支持该架构并且运行时环境可用
                            if (std::find_if(archs.begin(), archs.end(),
                                [&arch](const std::string& a) {
                                    return strcasecmp(a.c_str(), arch.c_str()) == 0;
                                }) != archs.end() && isRuntimeSupported(bk)) {
                                
                                // 找到匹配的后端,设匹配的格式和架构
                                if (matchedFormat) *matchedFormat = fmt;
                                if (matchedArch) *matchedArch = arch;
                                std::cerr << "using user configed backend " << fixed << std::endl;
                                return bk;
                            }
                        }
                    }
                }
                std::cerr << "user configed backend " << fixed 
                         << " does not support model " << model->name() << std::endl;
                break;
            }
        }
    }

    // 如果没有固定后端或固定后端不匹配,尝试所有后端
    for (auto& bk : sortedBackends) {
        auto formats = bk->supportedFormats();
        // 遍历模型支持的格式
        for (const auto& fmt : model->formats()) {
            // 检查后端是否支持该格式(不区分大小写)
            if (std::find_if(formats.begin(), formats.end(),
                [&fmt](const std::string& f) {
                    return strcasecmp(f.c_str(), fmt.c_str()) == 0;
                }) != formats.end()) {
                
                auto archs = bk->supportedArchitectures();
                // 遍历该格式下支持的架构
                for (const auto& arch : model->architectures(fmt)) {
                    // 检查后端是否支持该架构并且运行时环境可用
                    if (std::find_if(archs.begin(), archs.end(),
                        [&arch](const std::string& a) {
                            return strcasecmp(a.c_str(), arch.c_str()) == 0;
                        }) != archs.end() && isRuntimeSupported(bk)) {
                        
                        // 找到匹配的后端,设置匹配的格式和架构
                        if (matchedFormat) *matchedFormat = fmt;
                        if (matchedArch) *matchedArch = arch;
                        return bk;
                    }
                }
            }
        }
    }

    // 没有找到匹配的后端
    return nullptr;
}

bool BackendLoader::isRuntimeSupported(std::shared_ptr<BackendMetaObject> mo) const {
    if (mo->name() == kInferenceBackendLlamaCpp) {
        auto libs = std::get<std::vector<std::string>>(mo->extra(kInferenceBackendLibs));
        // Check if any lib is supported
        for (const auto& lib : libs) {
            if (std::get<bool>(mo->extra(lib)))
                return true;
        }
        return false;
    } else if (mo->name() == kInferenceBackendOpenvino) {
        return std::get<bool>(mo->extra(kInferenceBackendRTReady));
    }

    return true;
}




