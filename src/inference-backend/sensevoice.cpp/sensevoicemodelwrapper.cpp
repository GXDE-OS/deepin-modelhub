// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "sensevoicemodelwrapper.h"

#include <iostream>

GLOBAL_USE_NAMESPACE

const char *print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.


    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}

bool supports_gpu_offload(void) {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_reg_by_name("RPC") != nullptr;
}

struct callback_data {
    std::vector<uint8_t> data;
};

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        printf("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                printf("                                      ..., \n");
                i2 = ne[2] - n;
            }
            printf("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    printf("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                printf("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        printf("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v = 0;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        printf("fatal error");
                    }
                    printf("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) printf(", ");
                }
                printf("],\n");
            }
            printf("                                      ],\n");
        }
        printf("                                     ]\n");
        printf("                                     sum = %f\n", sum);
    }
}

static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return true; // Always retrieve data
    }

    char src1_str[128] = {0};
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    printf("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
           t->name, ggml_type_name(t->type), ggml_op_desc(t),
           src0->name, ggml_ne_string(src0).c_str(),
           src1 ? src1_str : "",
           ggml_ne_string(t).c_str());


    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type)) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        ggml_print_tensor(data, t->type, t->ne, t->nb, 3);
    }

    return true;
}

static char *sense_voice_param_turn_lowercase(char *in){
    int string_len = strlen(in);
    for (int i = 0; i < string_len; i++){
        *(in+i) = tolower((unsigned char)*(in+i));
    }
    return in;
}

static bool sense_voice_params_parse(int argc, char **argv, sense_voice_params &params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-"){
            params.fname_inp.push_back(arg);
            continue;
        }
        if (arg[0] != '-') {
            params.fname_inp.push_back(arg);
            continue;
        }
        else if (arg == "-t"    || arg == "--threads")         { params.n_threads       = std::stoi(argv[++i]); }
        else if (arg == "-p"    || arg == "--processors")      { params.n_processors    = std::stoi(argv[++i]); }
        else if (arg == "-ot"   || arg == "--offset-t")        { params.offset_t_ms     = std::stoi(argv[++i]); }
        else if (arg == "-on"   || arg == "--offset-n")        { params.offset_n        = std::stoi(argv[++i]); }
        else if (arg == "-d"    || arg == "--duration")        { params.duration_ms     = std::stoi(argv[++i]); }
        else if (arg == "-mc"   || arg == "--max-context")     { params.max_context     = std::stoi(argv[++i]); }
        else if (arg == "-ml"   || arg == "--max-len")         { params.max_len         = std::stoi(argv[++i]); }
        else if (arg == "-bo"   || arg == "--best-of")         { params.best_of         = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")       { params.beam_size       = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")       { params.audio_ctx       = std::stoi(argv[++i]); }
        else if (arg == "-wt"   || arg == "--word-thold")      { params.word_thold      = std::stof(argv[++i]); }
        else if (arg == "-et"   || arg == "--entropy-thold")   { params.entropy_thold   = std::stof(argv[++i]); }
        else if (arg == "-lpt"  || arg == "--logprob-thold")   { params.logprob_thold   = std::stof(argv[++i]); }
        else if (arg == "-tp"   || arg == "--temperature")     { params.temperature     = std::stof(argv[++i]); }
        else if (arg == "-tpi"  || arg == "--temperature-inc") { params.temperature_inc = std::stof(argv[++i]); }
        else if (arg == "-debug"|| arg == "--debug-mode")      { params.debug_mode      = true; }
        else if (arg == "-tr"   || arg == "--translate")       { params.translate       = true; }
        else if (arg == "-di"   || arg == "--diarize")         { params.diarize         = true; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")     { params.tinydiarize     = true; }
        else if (arg == "-sow"  || arg == "--split-on-word")   { params.split_on_word   = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")     { params.no_fallback     = true; }
        else if (arg == "-otxt" || arg == "--output-txt")      { params.output_txt      = true; }
        else if (arg == "-ovtt" || arg == "--output-vtt")      { params.output_vtt      = true; }
        else if (arg == "-osrt" || arg == "--output-srt")      { params.output_srt      = true; }
        else if (arg == "-owts" || arg == "--output-words")    { params.output_wts      = true; }
        else if (arg == "-olrc" || arg == "--output-lrc")      { params.output_lrc      = true; }
        else if (arg == "-ocsv" || arg == "--output-csv")      { params.output_csv      = true; }
        else if (arg == "-oj"   || arg == "--output-json")     { params.output_jsn      = true; }
        else if (arg == "-ojf"  || arg == "--output-json-full"){ params.output_jsn_full = params.output_jsn = true; }
        else if (arg == "-of"   || arg == "--output-file")     { params.fname_out.emplace_back(argv[++i]); }
        else if (arg == "-np"   || arg == "--no-prints")       { params.no_prints       = false; }
        else if (arg == "-ps"   || arg == "--print-special")   { params.print_special   = true; }
        else if (arg == "-pc"   || arg == "--print-colors")    { params.print_colors    = true; }
        else if (arg == "-pp"   || arg == "--print-progress")  { params.print_progress  = true; }
        else if (arg == "-nt"   || arg == "--no-timestamps")   { params.no_timestamps   = true; }
        else if (arg == "-l"    || arg == "--language")        { params.language        = sense_voice_param_turn_lowercase(argv[++i]); }
        else if (arg == "-dl"   || arg == "--detect-language") { params.detect_language = true; }
        else if (                  arg == "--prompt")          { params.prompt          = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")           { params.model           = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")            { params.fname_inp.emplace_back(argv[++i]); }
        else if (arg == "-oved" || arg == "--ov-e-device")     { params.openvino_encode_device = argv[++i]; }
        else if (arg == "-ls"   || arg == "--log-score")       { params.log_score       = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")          { params.use_gpu         = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")      { params.flash_attn      = true; }
        else if (                  arg == "--grammar-penalty") { params.grammar_penalty = std::stof(argv[++i]); }
        else if (arg == "-itn"   || arg == "--use-itn")        { params.use_itn         = true; }
    }
    return true;
}

SenseVoiceModelWrapper::SenseVoiceModelWrapper()
{

}

SenseVoiceModelWrapper::~SenseVoiceModelWrapper()
{
    if (ctx) {
        ggml_free(ctx->model.ctx);

        ggml_backend_buffer_free(ctx->model.buffer);

        sense_voice_free_state(ctx->state);

        delete ctx->model.model->encoder;
        delete ctx->model.model;
        delete ctx;
    }
}

bool SenseVoiceModelWrapper::initialize(const std::string &bin, const std::unordered_map<std::string, std::string> &initParams)
{
    if (ctx)
        return false;
    sense_voice_params params;

    int argc = 1; // 从第二个开始
    char** argv = new char*[128];
    for (int i = 0; i < 128; ++i) {
        argv[i] = nullptr;
    }

    size_t modelLength = strlen("--model");
    argv[argc] = new char[modelLength + 1];
    memcpy(argv[argc], "--model", modelLength + 1);
    argc++;

    size_t binLength = bin.length() + 1;
    argv[argc] = new char[binLength];
    memcpy(argv[argc], bin.c_str(), binLength);
    argc++;

    for (const auto &[key, value] : initParams) {
        if (argc >= 128) break;

        size_t keyLength = key.length() + 1;
        argv[argc] = new char[keyLength];
        memcpy(argv[argc], key.c_str(), keyLength);
        argc++;

        size_t valueLength = value.length() + 1;
        if (!value.empty()) {
            argv[argc] = new char[valueLength];
            memcpy(argv[argc], value.c_str(), valueLength);
            argc++;
        }
    }

    std::string sysInfo = print_system_info();
    sysInfo += "CUDA = " + std::to_string(supports_gpu_offload()) + " | ";
    std::cerr << "system info: " << sysInfo << std::endl;

    if (!sense_voice_params_parse(argc, argv, params)) {
        return false;
    }

    if (params.language != "auto" && sense_voice_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        exit(0);
    }

    // sense-voice init
    struct sense_voice_context_params cparams = sense_voice_context_default_params();

    callback_data cb_data;

    cparams.cb_eval = ggml_debug;
    cparams.cb_eval_user_data = &cb_data;

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;
    cparams.use_itn    = params.use_itn;

    ctx = sense_voice_small_init_from_file_with_params(params.model.c_str(), cparams);

    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize sense voice context\n");
        return false;
    }

    ctx->language_id = sense_voice_lang_id(params.language.c_str());

    wparams = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_GREEDY);
    // run inference
    {

        wparams.strategy = (params.beam_size > 1 ) ? SENSE_VOICE_SAMPLING_BEAM_SEARCH : SENSE_VOICE_SAMPLING_GREEDY;

        wparams.print_progress   = params.print_progress;
        wparams.print_timestamps = !params.no_timestamps;
        wparams.language         = params.language.c_str();
        wparams.n_threads        = params.n_threads;
        wparams.n_max_text_ctx   = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
        wparams.offset_ms        = params.offset_t_ms;
        wparams.duration_ms      = params.duration_ms;

        wparams.debug_mode       = params.debug_mode;

        wparams.greedy.best_of        = params.best_of;
        wparams.beam_search.beam_size = params.beam_size;

        wparams.no_timestamps    = params.no_timestamps;
    }

    return true;
}
