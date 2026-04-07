// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ovllmproxy.h"
#include "gen/ovutils.h"
#include "util.h"

#include <optional>
#include <filesystem>
#include <assert.h>


GLOBAL_USE_NAMESPACE

OvLLMProxy::OvLLMProxy(const std::string &name, ov::Core *core)
    : LLMProxy()
    , modelName(name)
    , ovCore(core)
{
    assert(core);
}

std::string OvLLMProxy::name() const
{
    return modelName;
}

std::vector<int32_t> OvLLMProxy::tokenize(const std::string &prompt, const std::map<std::string, std::string> &params)
{
    if (prompt.empty())
        return {};

    std::vector<int64_t> tokens = ovtokenizer->encode(prompt);
    std::vector<int32_t> out;
    out.resize(tokens.size());

    for (int i = 0; i < tokens.size(); ++i)
        out[i] = tokens[i];

    return out;
}

std::string OvLLMProxy::detokenize(const std::vector<int32_t> &tokens, const std::map<std::string, std::string> &params)
{
    if (tokens.empty())
        return "";

    std::vector<int64_t> input;
    input.resize(tokens.size());
    for (int i = 0; i < tokens.size(); ++i)
        input[i] = tokens[i];

    return ovtokenizer->decode(input);
}

std::vector<int32_t> OvLLMProxy::generate(const std::vector<int32_t> &tokens, const std::map<std::string, std::string> &params,
                                          generateStream stream, void *user)
{
    std::vector<int32_t> out;
    if (tokens.empty())
        return out;

    if (inputSize > 0) {
        if (tokens.size() > inputSize - 1) {
            std::cerr << "error: number of tokens in input line " << tokens.size() << " exceeds model context size " << inputSize << std::endl;
            return out;
        }
    }

    std::vector<int64_t> token64;
    {
        token64.resize(tokens.size());
        for (int i = 0; i < tokens.size(); ++i)
            token64[i] = tokens[i];
    }

    if (staticSize) {
        token64 = pipelineWithStatic(token64, params, stream, user);
    } else {
        token64 = doPipeline(token64, params, stream, user);
    }

    {
        out.resize(token64.size());
        for (int i = 0; i < token64.size(); ++i)
            out[i] = token64[i];
    }

    return out;
}

bool OvLLMProxy::initialize(const std::string &model, const std::string &tokenizer, const std::string &detokenizer, const VariantMap &params)
{
    if (model.empty() || tokenizer.empty() || detokenizer.empty())
        return false;

    ovCore->add_extension("libopenvino_tokenizers.so");

    std::vector<std::string> devices = Util::findValue(params, "Devices", std::vector<std::string>());
    device = "CPU";

    bool npu = OvUtils::containCaseInsensitive(devices, "NPU");
    bool gpu = OvUtils::containCaseInsensitive(devices, "GPU");

    if (gpu) {
        device = "GPU";
    } else if (npu) {
        device = "NPU";
        staticSize = true;
    }

    ovtokenizer.reset(new OvTokenizer(ovCore, tokenizer, detokenizer
                                      , std::filesystem::path(tokenizer).parent_path() / "modelconfig"));
    orgModel = ovCore->read_model(model);

    // print input and output
    {
        int idx = 0;
        for (auto input : orgModel->inputs()) {
            inputTensors.push_back(input.get_any_name());
            std::cerr << "model input " << idx++ << " " << input.get_any_name()
                      << " tensor partial shape " << input.get_partial_shape()
                      << " element type " << input.get_element_type() << std::endl;
        }

        idx = 0;
        for (auto output : orgModel->outputs()) {
            std::cerr << "model output " << idx++
                      << " tensor partial shape " << output.get_partial_shape()
                      << " element type " << output.get_element_type() << std::endl;
        }
    }



    bool pipeline = (std::find(inputTensors.begin(), inputTensors.end(), "input_ids") != inputTensors.end())
            && (std::find(inputTensors.begin(), inputTensors.end(), "attention_mask") != inputTensors.end())
            && (std::find(inputTensors.begin(), inputTensors.end(), "beam_idx") != inputTensors.end());

    if (!pipeline) {
        std::cerr << "model is not support pipeline." << std::endl;
        return false;
    }

    inputSize = Util::findValue(params, "input_size", -1);

    std::cerr << "params: input_size: " << inputSize << " devices " << device << std::endl;

    chatModel = ovCore->compile_model(orgModel, device);
    return true;
}

std::vector<int64_t> OvLLMProxy::pipelineWithStatic(const std::vector<int64_t> &, const std::map<std::string, std::string> &params, generateStream stream, void *user)
{
    return {};
}

std::vector<int64_t> OvLLMProxy::doPipeline(const std::vector<int64_t> &tokens, const std::map<std::string, std::string> &params, generateStream stream, void *user)
{
    std::shared_ptr<OvTimings> timings;
    if (params.find("timings") != params.end())
        timings.reset(new OvTimings);

    int predict = INT32_MAX;
    {
        auto pv = params.find("predict");
        if (pv != params.end()) {
            int value = std::stoi(pv->second);
            if (value > 0)
                predict = value;
        }
    }
    bool streamToken = params.find("stream_token") != params.end();

    ov::Tensor inputIds = ov::Tensor(ov::element::i64, ov::Shape{1, tokens.size()}, const_cast<int64_t *>(tokens.data()));
    ov::Tensor attentionMask = ov::Tensor(ov::element::i64, {1, tokens.size()});
    std::fill_n(attentionMask.data<int64_t>(), tokens.size(), 1);

    std::optional<ov::Tensor> positionIds = std::nullopt;
    if (std::find(inputTensors.begin(), inputTensors.end(), "position_ids") != inputTensors.end()) {
        positionIds = ov::Tensor{ov::element::i64, inputIds.get_shape()};
        OvUtils::initializePositionIds(*positionIds, attentionMask, 0);
    }

    ov::InferRequest model = chatModel.create_infer_request();

    std::vector<int64_t> tokenResults;
    std::vector<int64_t> streamCache;

    const size_t batchSize = 1;
    // greedy decoding
    {
        model.set_tensor("input_ids", inputIds);
        model.set_tensor("attention_mask", attentionMask);

        if (positionIds.has_value())
            model.set_tensor("position_ids", *positionIds);

        auto beam = model.get_tensor("beam_idx");
        beam.set_shape({batchSize});
        auto beamData = model.get_tensor("beam_idx").data<int32_t>();
        std::iota(beamData, beamData + batchSize, 0);

        auto startTime = OvUtils::getTimeMs();
        model.infer();
        if (timings.get()) {
            timings->pEvalTime = OvUtils::getTimeMs() - startTime;
            timings->pEval = tokens.size();
        }

        auto logits = model.get_tensor("logits");
        startTime = OvUtils::getTimeMs();

        auto outToken = OvUtils::argmax(logits, 0);

        if (timings.get())
            timings->sampleTime = OvUtils::getTimeMs() - startTime;

        tokenResults.push_back(outToken);

        if (stream) {
            if (!streamToken) {
                auto send = processToken(outToken, streamCache);
                if (!send.empty()) {
                    if (!stream(send, user)) {
                        OvUtils::printTimings(timings.get());
                        return tokenResults;
                    }
                }
            } else {
                if (!stream(std::to_string(outToken), user)) {
                    OvUtils::printTimings(timings.get());
                    return tokenResults;
                }
            }
        }

        // continue
        inputIds = ov::Tensor(ov::element::i64, ov::Shape{batchSize, 1});
        model.set_tensor("input_ids", inputIds);

        const size_t maxTokens = inputSize > 0 ? inputSize - tokens.size() - 1 : 1024 * 120;
        for (int i = 0; i < maxTokens; ++i) {
            if (outToken == ovtokenizer->eosTokenId())
                break;

            if (tokenResults.size() >= predict)
                break;

            model.get_tensor("input_ids").data<int64_t>()[0] = outToken;
            if (positionIds.has_value())
                OvUtils::updatePositionIds(model.get_tensor("position_ids"), model.get_tensor("attention_mask"));

            model.set_tensor("attention_mask", OvUtils::extendAttention(model.get_tensor("attention_mask")));

            startTime = OvUtils::getTimeMs();
            model.infer();

            if (timings.get()) {
                timings->evalTime += OvUtils::getTimeMs() - startTime;
                timings->eval++;
            }

            logits = model.get_tensor("logits");
            startTime = OvUtils::getTimeMs();

            outToken = OvUtils::argmax(logits, 0);

            if (timings.get())
                timings->sampleTime += OvUtils::getTimeMs() - startTime;

            tokenResults.push_back(outToken);

            if (stream) {
                if (streamToken) {
                    auto send = processToken(outToken, streamCache);
                    if (!send.empty()) {
                        if (!stream(send, user)) {
                            OvUtils::printTimings(timings.get());
                            return tokenResults;
                        }
                    }
                } else {
                    if (!stream(std::to_string(outToken), user)) {
                        OvUtils::printTimings(timings.get());
                        return tokenResults;
                    }
                }
            }
        }
    }

    OvUtils::printTimings(timings.get());
    return tokenResults;
}

/*
std::vector<int64_t> OvLLMProxy::generate(const std::vector<int64_t> &tokens, const std::map<std::string, std::string> &params, generateStream stream, void *user)
{
    std::vector<int64_t> inputTokens = tokens;
    std::vector<int64_t> tokenResults;
    std::vector<int64_t> streamCache;

    ov::InferRequest model = chatModel.create_infer_request();
    while (inputTokens.size() < inputSize - 1) {
        ov::Tensor inputIds = ov::Tensor(ov::element::i64, ov::Shape{1, inputTokens.size()}, inputTokens.data());

        ov::Tensor attentionMask = ov::Tensor(ov::element::i64, {1, inputTokens.size()});
        std::fill_n(attentionMask.data<int64_t>(), inputTokens.size(), 1);

        ov::Tensor positionIds = ov::Tensor(ov::element::i64, inputIds.get_shape());
        OvUtils::initializePositionIds(positionIds, attentionMask, 0);

        model.set_tensor("input_ids", inputIds);
        model.set_tensor("attention_mask", attentionMask);
        model.set_tensor("position_ids", positionIds);

        model.start_async();
        model.wait();

        auto logits = model.get_tensor("logits");
        auto outToken = OvUtils::argmax(logits, 0);
        tokenResults.push_back(outToken);
        inputTokens.push_back(outToken);

        if (stream) {
            auto send = processToken(outToken, streamCache);
            if (!send.empty()) {
                if (!stream(send, user))
                    break;
            }
        }

        if (outToken == ovtokenizer->eosTokenId())
            break;
    }

    return tokenResults;
}
*/

int findInvalidUTF8Chars(const std::string& input)
{
    size_t i = 0;
    while (i < input.size()) {
        unsigned char c = input[i];
        if ((c & 0x80) == 0) { // 1-byte character (ASCII)
            i++;
        } else if ((c & 0xE0) == 0xC0) { // 2-byte character
            if (i + 1 >= input.size() || (input[i + 1] & 0xC0) != 0x80) {
                return i;
            } else {
                i += 2;
            }
        } else if ((c & 0xF0) == 0xE0) { // 3-byte character
            if (i + 2 >= input.size() ||
                (input[i + 1] & 0xC0) != 0x80 ||
                (input[i + 2] & 0xC0) != 0x80) {
                return i;
            } else {
                i += 3;
            }
        } else if ((c & 0xF8) == 0xF0) { // 4-byte character
            if (i + 3 >= input.size() ||
                (input[i + 1] & 0xC0) != 0x80 ||
                (input[i + 2] & 0xC0) != 0x80 ||
                (input[i + 3] & 0xC0) != 0x80) {
                return i;
            } else {
                i += 4;
            }
        } else {
            return i;
        }
    }
    return -1;
}

std::string OvLLMProxy::processToken(int64_t token, std::vector<int64_t> &tokenCache)
{
    tokenCache.push_back(token);
    std::string text = ovtokenizer->decode(tokenCache);
    int invaildPos = findInvalidUTF8Chars(text);
    if (invaildPos >= 0 && (text.size() - invaildPos) < 5)
        return "";

    tokenCache.clear();
    return text;
}
