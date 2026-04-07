// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef SENSEVOICECONFIG_H
#define SENSEVOICECONFIG_H

#include "global_header.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include "common.h"

GLOBAL_BEGIN_NAMESPACE

struct sense_voice_params {
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors  = 1;
    int32_t offset_t_ms   = 0;
    int32_t offset_n      = 0;
    int32_t duration_ms   = 0;
    int32_t progress_step = 5;
    int32_t max_context   = -1;
    int32_t max_len       = 0;
    int32_t n_mel       = 80;
    int32_t best_of       = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size     = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    int32_t audio_ctx     = 0;

    float word_thold      =  0.01f;
    float entropy_thold   =  2.40f;
    float logprob_thold   = -1.00f;
    float grammar_penalty = 100.0f;
    float temperature     = 0.0f;
    float temperature_inc = 0.2f;

    bool debug_mode      = false;
    bool translate       = false;
    bool detect_language = false;
    bool diarize         = false;
    bool tinydiarize     = false;
    bool split_on_word   = false;
    bool no_fallback     = false;
    bool output_txt      = false;
    bool output_vtt      = false;
    bool output_srt      = false;
    bool output_wts      = false;
    bool output_csv      = false;
    bool output_jsn      = false;
    bool output_jsn_full = false;
    bool output_lrc      = false;
    bool no_prints       = false;
    bool print_special   = false;
    bool print_colors    = false;
    bool print_progress  = false;
    bool no_timestamps   = false;
    bool log_score       = false;
    bool use_gpu         = true;
    bool flash_attn      = false;
    bool use_itn         = false;

    std::string language  = "auto";
    std::string prompt;
    std::string model     = "models/ggml-base.en.bin";


    std::string openvino_encode_device = "CPU";


    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};

};

class SenseVoiceConfig
{
public:
    explicit SenseVoiceConfig(const std::string &file);
    inline std::string bin() const {
        return read("bin");
    }

    std::vector<std::string> architectures() const;
    std::unordered_map<std::string, std::string> params() const;

protected:
    inline std::string read(const std::string &key) const {
        return configs.contains(key) ? configs[key].get<std::string>() : "";
    }

private:
    nlohmann::json configs;
};

GLOBAL_END_NAMESPACE

#endif // SENSEVOICECONFIG_H
