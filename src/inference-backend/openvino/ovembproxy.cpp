// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ovembproxy.h"
#include "gen/ovutils.h"
#include "util.h"

#include <assert.h>

GLOBAL_USE_NAMESPACE

OVEmbProxy::OVEmbProxy(const std::string &name, ov::Core *core)
    : EmbeddingProxy()
    , modelName(name)
    , ovCore(core)
{
    assert(core);
}

OVEmbProxy::~OVEmbProxy()
{

}

std::string OVEmbProxy::name() const
{
    return modelName;
}

std::list<std::vector<int32_t>> OVEmbProxy::tokenize(const std::list<std::string> &prompt, const std::map<std::string, std::string> &params)
{
    auto tokenizer = tokenizerModel.create_infer_request();

    std::list<std::vector<int32_t>> out;
    for (std::string pmpt : prompt) {
        tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {1}, &pmpt});
        tokenizer.infer();

        auto tensor = tokenizer.get_tensor("input_ids");
        {
            int batch = tensor.get_shape().at(0);
            int dim = tensor.get_shape().at(1);
            for (int i = 0; i < batch; ++i) {
                std::vector<int32_t> tmp;
                tmp.resize(dim);
                int64_t *begin = tensor.data<int64_t>();
                for (int i = 0; i < dim; ++i)
                    tmp[i] = begin[i];
                out.push_back(tmp);
            }
        }
    }

    return out;
}

std::list<std::vector<float>> OVEmbProxy::embedding(const std::list<std::vector<int32_t>> &tokens, const std::map<std::string, std::string> &params)
{
    std::list<std::vector<float>> out;
    if (inputSize > 0) {
        for (auto it = tokens.begin(); it != tokens.end(); ++it) {
            if (it->size() > inputSize) {
                std::cerr << "error: number of tokens in input line " << it->size() << " exceeds model context size "
                            << inputSize << std::endl;
                return out;
            }
        }
    }

    ov::InferRequest model = embModel.create_infer_request();

    for (const std::vector<int32_t> &token : tokens) {
        ov::Tensor input_ids;
        ov::Tensor mask;
        if (staticSize) {
            {
                input_ids = ov::Tensor(ov::element::i64, {1, (size_t)inputSize});
                int64_t *paddedData =  input_ids.data<int64_t>();
                memset(paddedData, 0, sizeof(int64_t) * inputSize);
                for (int i = 0; i < token.size(); ++i)
                    paddedData[i] = token[i];
            }

            {
                mask = ov::Tensor(ov::element::i64, {1, (size_t)inputSize});
                int64_t *paddedData =  mask.data<int64_t>();
                memset(paddedData, 0, sizeof(int64_t) * inputSize);
                std::fill_n(paddedData, token.size(), 1);
            }
        } else {
            {
                input_ids = ov::Tensor(ov::element::i64, ov::Shape{1, token.size()});
                int64_t *paddedData =  input_ids.data<int64_t>();
                memset(paddedData, 0, sizeof(int64_t) * token.size());
                for (int i = 0; i < token.size(); ++i)
                    paddedData[i] = token[i];
            }
            {
                mask = ov::Tensor(ov::element::i64, {1, token.size()});
                int64_t *paddedData =  mask.data<int64_t>();
                memset(paddedData, 0, sizeof(int64_t) * token.size());
                std::fill_n(paddedData, token.size(), 1);
            }
        }

        model.set_tensor("input_ids", input_ids);
        model.set_tensor("attention_mask", mask);

        model.start_async();
        model.wait();
        const ov::Tensor& output_tensor = model.get_tensor("sentence_embedding");

        {
            auto shape = output_tensor.get_shape();
            int batch = shape.at(0);
            int dim = shape.at(1);
            auto strides = output_tensor.get_strides();
            char *begin = (char *)output_tensor.data();
            for (int i = 0; i < batch; ++i) {
                std::vector<float> tmp;
                tmp.resize(dim);
                memcpy(tmp.data(), begin + i * strides[0], strides[0]);
                out.push_back(tmp);
            }
        }
    }

    return out;
}

bool OVEmbProxy::initialize(const std::string &model, const std::string &tokenizer, const VariantMap &params)
{
    if (model.empty() || tokenizer.empty())
        return false;

    ovCore->add_extension("libopenvino_tokenizers.so");

    std::vector<std::string> devices = Util::findValue(params, "Devices", std::vector<std::string>());

    device = "CPU";

    bool npu = OvUtils::containCaseInsensitive(devices, "NPU");
    bool gpu = OvUtils::containCaseInsensitive(devices, "GPU");

    if (npu) {
        device = "NPU";
        staticSize = true;
    } else if (gpu) {
        device = "GPU";
    }

    tokenizerModel = ovCore->compile_model(ovCore->read_model(tokenizer, "CPU"));

    {
        int idx = 0;
        for (auto input : tokenizerModel.inputs()) {
            std::cerr << "tokenizer input " << idx++ << " " << input.get_any_name()
                      << " tensor partial shape " << input.get_partial_shape()
                      << " element type " << input.get_element_type() << std::endl;
        }

        idx = 0;
        for (auto output : tokenizerModel.outputs()) {
            std::cerr << "tokenizer output " << idx++ << " " << output.get_any_name()
                      << " tensor partial shape " << output.get_partial_shape()
                      << " element type " << output.get_element_type() << std::endl;
        }
    }

    orgEmbModel = ovCore->read_model(model);

    // print input and output
    {
        int idx = 0;
        for (auto input : orgEmbModel->inputs()) {
            std::cerr << "model input " << idx++ << " " << input.get_any_name()
                      << " tensor partial shape " << input.get_partial_shape()
                      << " element type " << input.get_element_type() << std::endl;
        }

        idx = 0;
        for (auto output : orgEmbModel->outputs()) {
            std::cerr << "model output " << idx++ << " " << output.get_any_name()
                      << " tensor partial shape" << output.get_partial_shape() << std::endl;
        }
    }

    inputSize = Util::findValue(params, "input_size", -1);

    std::cerr << "params: input_size: " << inputSize << " devices " << device << std::endl;

    if (staticSize) {
        if (inputSize < 1) {
            std::cerr << "model will load to gpu or npu, it need a static input size by params: input_size" << inputSize;
            return false;
        }

        std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;
        for (const ov::Output<ov::Node>& input : orgEmbModel->inputs()) {
            ov::PartialShape shape = input.get_partial_shape();
            shape = {1, inputSize};
            port_to_shape[input] = shape;
        }
        std::cerr << "reshape model input to {1," << inputSize << "}" << std::endl;
        orgEmbModel->reshape(port_to_shape);
        embModel = ovCore->compile_model(orgEmbModel, device);
    } else {
       embModel = ovCore->compile_model(orgEmbModel, device);
    }

    return true;
}

void OVEmbProxy::embdNormalize(const float *inp, float *out, int n) const
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += inp[i] * inp[i];
    }
    sum = sqrt(sum);

    const float norm = sum > 0.0 ? 1.0f / sum : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}
