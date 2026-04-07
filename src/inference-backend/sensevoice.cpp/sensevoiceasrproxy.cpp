// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "sensevoiceasrproxy.h"

#include <iostream>

GLOBAL_USE_NAMESPACE

#define TOKEN_OFFSET 4

SenseVoiceASRProxy::SenseVoiceASRProxy(const std::string &name)
    : ASRProxy()
    , SenseVoiceModelWrapper()
    , modelName(name)
{

}

std::string SenseVoiceASRProxy::name() const
{
    return modelName;
}

bool SenseVoiceASRProxy::decodeContent(const std::string &content, std::vector<double> &pcmf32, const std::map<std::string, std::string> &params)
{
    if (content.empty())
        return false;
    auto it = params.find("format");
    std::string format;
    if (it == params.end()) {
        std::cerr << "warring: content is not setted format.";
    } else {
        format = it->second;
    }

    if (format == "audio/pcm") {
        return decodePcm(content, pcmf32);
    } else {
        return decodeWav(content, pcmf32);
    }
}

std::vector<int32_t> SenseVoiceASRProxy::transcriptions(const std::vector<double> &pcmf32, const std::map<std::string, std::string> &params,
                                                        generateStream stream, void *user)
{
    std::vector<int32_t> ret;

    if (pcmf32.empty())
        return ret;

    const int chunk_size = 100;
    const int min_mute_chunks =  100 / chunk_size;
    const int max_nomute_chunks = 30000 / chunk_size;
    const bool streamToken = params.find("stream_token") != params.end();

    int32_t L_nomute = -1, L_mute = -1, R_mute = -1;  // [L_nomute, R_nomute)永远为需要解析的段落，[L_mute, R_mute)永远为最近一段静音空挡
    const int n_sample_step = chunk_size * 1e-3 * SENSE_VOICE_SAMPLE_RATE;
    const int keep_nomute_step = chunk_size * min_mute_chunks * 1e-3 * SENSE_VOICE_SAMPLE_RATE;
    const int max_nomute_step = chunk_size * max_nomute_chunks * 1e-3 * SENSE_VOICE_SAMPLE_RATE;

    auto pushStream = [this, streamToken, stream, user](std::vector<int32_t> &ret, sense_voice_context *pctx) -> bool {
        if (pctx->state->ids.size() < TOKEN_OFFSET)
            return true;
        ret.insert(ret.end(), pctx->state->ids.begin() + TOKEN_OFFSET, pctx->state->ids.end());
        if (stream) {
            if (streamToken) {
                return stream("", user);
            } else {
                return stream(detokenize(pctx->state->ids, {{"remove_prefix", ""}}), user);
            }
        } else {
            return true;
        }
    };

    std::vector<double> pcmf32_tmp;           // 记录需要解析的段落
    std::vector<double> pcmf32_chunk;         // 作为临时缓冲区
    pcmf32_chunk.resize(n_sample_step);

    for (int i = 0; i < pcmf32.size(); i += n_sample_step) {
        int R_this_chunk = std::min(i + n_sample_step, int(pcmf32.size()));
        bool isnomute = vad_energy_zcr<double>(pcmf32.begin() + i, R_this_chunk - i, SENSE_VOICE_SAMPLE_RATE);
        if (L_nomute >= 0 && R_this_chunk - L_nomute >= max_nomute_step) {
            int R_nomute = L_mute >= 0 && L_mute >= L_nomute ? L_mute : R_this_chunk;
            pcmf32_tmp.resize(R_nomute - L_nomute);
            std::copy(pcmf32.begin() + L_nomute, pcmf32.begin() + R_nomute, pcmf32_tmp.begin());
            if (sense_voice_full_parallel(ctx, wparams, pcmf32_tmp, pcmf32_tmp.size(), 1) != 0)
                return ret;
#ifndef NDEBUG
            fprintf(stderr, "[%.2f-%.2f]", L_nomute / (SENSE_VOICE_SAMPLE_RATE * 1.0), R_nomute / (SENSE_VOICE_SAMPLE_RATE * 1.0));
#endif
            if (!pushStream(ret, ctx))
                return ret;

            if (!isnomute)
                L_nomute = -1;
            else if (R_mute >= 0 && L_mute >= L_nomute)
                L_nomute = R_mute;
            else
                L_nomute = i;
            L_mute = R_mute = -1;
            continue;
        }
        if (isnomute) {
            if (L_nomute < 0)
                L_nomute = i;
        } else {
            if (R_mute != i)
                L_mute = i;
            R_mute = R_this_chunk;
            if (L_mute >= L_nomute && L_nomute >= 0 && R_this_chunk - L_mute >= keep_nomute_step) {
                pcmf32_tmp.resize(L_mute - L_nomute);
                std::copy(pcmf32.begin() + L_nomute, pcmf32.begin() + L_mute, pcmf32_tmp.begin());
#ifndef NDEBUG
                fprintf(stderr, "[%.2f-%.2f]", L_nomute / (SENSE_VOICE_SAMPLE_RATE * 1.0), L_mute / (SENSE_VOICE_SAMPLE_RATE * 1.0));
#endif
                if (sense_voice_full_parallel(ctx, wparams, pcmf32_tmp, pcmf32_tmp.size(), 1) != 0) {
                    return ret;
                }

                if (!pushStream(ret, ctx))
                    return ret;

                if (!isnomute)
                    L_nomute = -1;
                else if (R_mute >= 0 && L_mute >= L_nomute)
                    L_nomute = R_mute;
                else L_nomute = i;
                L_mute = R_mute = -1;
            }
        }
    }
    // 最后一段
    if (L_nomute >= 0) {
        int R_nomute = pcmf32.size();
        pcmf32_tmp.resize(R_nomute - L_nomute);
        std::copy(pcmf32.begin() + L_nomute, pcmf32.begin() + R_nomute, pcmf32_tmp.begin());
        if (sense_voice_full_parallel(ctx, wparams, pcmf32_tmp, pcmf32_tmp.size(), 1) != 0)
            return ret;

        if (!pushStream(ret, ctx))
            return ret;

        L_nomute = L_mute = R_mute = -1;
    }

#if 0
    std::cerr << "stream:";
    for (int32_t t : ret)
        std::cerr << " " << t;
    std::cerr << std::endl;

    sense_voice_full_parallel(ctx, wparams, *const_cast<std::vector<double>*>(&pcmf32), pcmf32.size(), 1);

    std::cerr << "full:";
    for (int32_t t : ctx->state->ids)
        std::cerr << " " << t;
    std::cerr << std::endl;
#endif

    return ret;
}

std::string SenseVoiceASRProxy::detokenize(const std::vector<int32_t> &tokens, const std::map<std::string, std::string> &params)
{
    const int offset = params.find("remove_prefix") != params.end() ? TOKEN_OFFSET : 0;
    std::string s = "";
    for(int i = offset; i < tokens.size(); i++) {
        int id = tokens[i];
        if (i > 0 && tokens[i - 1] == tokens[i])
            continue;

        if (id > 0)
            s += ctx->vocab.id_to_token[id].c_str();
    }

    return s;
}

bool SenseVoiceASRProxy::decodePcm(const std::string &content, std::vector<double> &pcmf32) const
{
    int numSamples = content.size() / 2;
    const char* rawData = content.c_str();
    const short* samples = reinterpret_cast<const short*>(rawData);
    const double normalizationFactor = 1.0; // 无需归一化
    for (int i = 0; i < numSamples; ++i) {
        pcmf32.push_back(samples[i] * normalizationFactor);
    }

    return true;
}

bool SenseVoiceASRProxy::decodeWav(const std::string &content, std::vector<double> &pcmf32) const
{
    WaveHeader header {};

    std::istringstream is(content);
    if (!is) {
        std::cerr << "Failed to initialize istringstream." << std::endl;
        return false;
    }

    if (content.size() < sizeof(header)) {
        std::cerr << "File content is too small to contain WAV header." << std::endl;
        return false;
    }

    is.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!is) {
        std::cerr << "Failed to read WAV header from memory. Stream state: "
                      << "eof: " << is.eof() << ", fail: " << is.fail() << ", bad: " << is.bad() << std::endl;
        return false;
    }

    if (!header.Validate()) {
        std::cerr << "Invalid WAV header." << std::endl;
        return false;
    }

    header.SeekToDataChunk(is);
    if (!is) {
        std::cerr << "Failed to seek to data chunk in WAV file." << std::endl;
        return false;
    }

    auto speech_len = header.subchunk2_size / 2;
    pcmf32.resize(speech_len);

    auto speech_buff = (int16_t *)malloc(sizeof(int16_t) * speech_len);

    if (speech_buff) {
        memset(speech_buff, 0, sizeof(int16_t) * speech_len);
        is.read(reinterpret_cast<char *>(speech_buff), header.subchunk2_size);
        if (!is) {
            std::cerr << "Failed to read WAV data from memory";
            free(speech_buff);
            return false;
        }

        float scale = 1.0;
        for (int32_t i = 0; i != speech_len; ++i) {
            pcmf32[i] = (double)speech_buff[i] / scale;
        }
        free(speech_buff);
        return true;
    } else {
        std::cerr << "Failed to allocate memory for speech buffer." << std::endl;
        free(speech_buff);
        return false;
    }
}
