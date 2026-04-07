// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ovutils.h"

GLOBAL_USE_NAMESPACE

OvUtils::OvUtils()
{

}

double OvUtils::getTimeMs()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000) / 1e3;
}

void OvUtils::printTimings(OvTimings *tm)
{
    if (!tm)
        return;

    std::cerr << std::endl;
    //fprintf(stderr, "load time = %.2f ms\n", timings.t_load_ms);
    fprintf(stderr, "prompt eval time = %.2f ms / %d tokens (%.2f ms per token, %.2f tokens per second)\n",
            tm->pEvalTime, tm->pEval, tm->pEvalTime / (double)tm->pEval, 1e3 / tm->pEvalTime * tm->pEval);

    if (tm->eval > 0)
        fprintf(stderr, "eval time = %.2f ms / %d runs   (%.2f ms per token, %.2f tokens per second)\n",
                tm->evalTime, tm->eval, tm->evalTime / (double)tm->eval, 1e3 / tm->evalTime * tm->eval);

    fprintf(stderr, "sample time = %.2f ms / %d runs   (%.2f ms per token, %.2f tokens per second)\n",
            tm->sampleTime, tm->eval + 1, tm->sampleTime / (double)(tm->eval + 1), 1e3 / tm->sampleTime * (tm->eval + 1));
}

void OvUtils::initializePositionIds(ov::Tensor &positionIds, const ov::Tensor &attentionMask, int64_t startPos)
{
    const size_t batchSize = attentionMask.get_shape()[0];
    const size_t seqLength = attentionMask.get_shape()[1];

    const int64_t *attentionMaskData = attentionMask.data<int64_t>();
    int64_t *positionIdsData = positionIds.data<int64_t>();

    for (size_t batch = 0; batch < batchSize; batch++) {
        size_t sum = startPos;
        for (size_t i = 0; i < seqLength; i++) {
            const size_t elementOffset = batch * seqLength + i;
            positionIdsData[elementOffset] = sum;
            if (attentionMaskData[elementOffset] == 1) {
                sum += 1;
            }
        }
    }
}

int64_t OvUtils::argmax(const ov::Tensor& logits, const size_t batchIdx)
{
    if (logits.get_shape()[0] <= batchIdx)
        return -1;

    size_t vocabSize = logits.get_shape().back();
    size_t batchOffset = batchIdx * logits.get_shape()[1] * vocabSize;
    size_t sequenceOffset = (logits.get_shape()[1] - 1) * vocabSize;
    const float *logitsData = logits.data<const float>() + batchOffset + sequenceOffset;

    int64_t outToken = std::max_element(logitsData, logitsData + vocabSize) - logitsData;
    //float maxLogit = logitsData[outToken];

    return outToken;
}

void OvUtils::updatePositionIds(ov::Tensor &&positionIds, const ov::Tensor &&attentionMask)
{
    const size_t batch_size = attentionMask.get_shape().at(0);
    const size_t atten_length = attentionMask.get_shape().at(1);
    positionIds.set_shape({batch_size, 1});

    for (size_t batch = 0; batch < batch_size; batch++) {
        int64_t* start = attentionMask.data<int64_t>() + batch * atten_length;
        positionIds.data<int64_t>()[batch] = std::accumulate(start, start + atten_length, 0);
    }
}

ov::Tensor OvUtils::extendAttention(ov::Tensor attentionMask)
{
    auto shape = attentionMask.get_shape();
    auto batch_size = shape[0];
    auto seq_len = shape[1];

    ov::Tensor new_atten_mask = ov::Tensor{attentionMask.get_element_type(), {batch_size, seq_len + 1}};
    auto old_data = attentionMask.data<int64_t>();
    auto new_data = new_atten_mask.data<int64_t>();
    for (size_t batch = 0; batch < batch_size; ++batch) {
        std::memcpy(new_data + batch * (seq_len + 1), old_data + batch * seq_len, seq_len * sizeof(int64_t));
        new_data[batch * (seq_len + 1) + seq_len] = 1;
    }
    return new_atten_mask;
}

bool OvUtils::containCaseInsensitive(const std::vector<std::string> &list, const std::string &key)
{
    return std::any_of(list.begin(), list.end(),
            [&key](const std::string &value) {
                if (value.size() != key.size())
                    return false;

                return strcasecmp(value.c_str(), key.c_str()) == 0;
            });
}
