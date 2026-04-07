// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef GLOBAL_HEADER_H
#define GLOBAL_HEADER_H

#include <string>
#include <variant>
#include <vector>
#include <unordered_map>

#define GLOBAL_NAMESPACE deepin_modelhub

#define GLOBAL_BEGIN_NAMESPACE namespace GLOBAL_NAMESPACE {
#define GLOBAL_END_NAMESPACE }
#define GLOBAL_USE_NAMESPACE using namespace GLOBAL_NAMESPACE;

GLOBAL_BEGIN_NAMESPACE

inline constexpr char kInferenceBackendUnknown[] { "unknownbackend" };
inline constexpr char kInferenceBackendLlamaCpp[] { "llama.cpp" };
inline constexpr char kInferenceBackendOpenvino[] { "openvino" };
inline constexpr char kInferenceBackendSenseVoice[] { "sensevoice.cpp" };


inline constexpr char kInferenceBackendRTReady[] { "rtready" };
inline constexpr char kInferenceBackendLibs[] { "libs" };
inline constexpr char kInferenceBackendScore[] { "score" };

enum ModelArchitecture {
    UnknownModel = 0,
    LLM = 100,
    Eembedding = 200,
    ASR = 300
};

using ValueType = std::variant<
        std::string,
        std::vector<std::string>,
        std::unordered_map<std::string, std::string>,
        int,
        float,
        double,
        bool
>;

using VariantMap = std::unordered_map<std::string, ValueType>;

GLOBAL_END_NAMESPACE

#endif   // GLOBAL_HEADER_H
