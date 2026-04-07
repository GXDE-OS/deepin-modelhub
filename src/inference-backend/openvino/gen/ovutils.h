// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef OVUTILS_H
#define OVUTILS_H

#include "global_header.h"

#include <openvino/openvino.hpp>

GLOBAL_BEGIN_NAMESPACE

struct OvTimings
{
    double sampleTime = 0;
    double pEvalTime = 0;
    int pEval = 0;
    double evalTime = 0;
    int eval = 0;
};

class OvUtils
{
public:
    static double getTimeMs();
    static void printTimings(OvTimings *tm);
    static void initializePositionIds(ov::Tensor &positionIds,
                                const ov::Tensor &attentionMask,
                                int64_t startPos);
    static int64_t argmax(const ov::Tensor &logits, const size_t batchIdx);
    static void updatePositionIds(ov::Tensor &&positionIds, const ov::Tensor &&attentionMask);
    static ov::Tensor extendAttention(ov::Tensor attentionMask);
    static bool containCaseInsensitive(const std::vector<std::string> &list, const std::string &key);
protected:
    OvUtils();
};

GLOBAL_END_NAMESPACE

#endif // OVUTILS_H
