// SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ovtokenizer_p.h"

#include <fstream>

GLOBAL_USE_NAMESPACE

OvTokenizerPrivate::OvTokenizerPrivate(ov::Core *core, OvTokenizer *qq)
    : ovCore(core)
    , q(qq)
{

}

bool OvTokenizerPrivate::initialize(const std::string &tokenizerXml, const std::string &detokenizerXml, const std::string &configPath)
{
    loadChatTemplate(configPath); // need jinja2cpp too weight
    loadTokenConfig(configPath);

    if (!tokenizerXml.empty())
        tokenizerModel = ovCore->compile_model(ovCore->read_model(tokenizerXml), "CPU");

    if (tokenizerModel) {
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

    if (!detokenizerXml.empty())
        detokenizerModel = ovCore->compile_model(ovCore->read_model(detokenizerXml), "CPU");
    if (detokenizerModel) {
        int idx = 0;
        for (auto input : detokenizerModel.inputs()) {
            std::cerr << "detokenizer input " << idx++ << " " << input.get_any_name()
                      << " tensor partial shape " << input.get_partial_shape()
                      << " element type " << input.get_element_type() << std::endl;
        }

        idx = 0;
        for (auto output : detokenizerModel.outputs()) {
            std::cerr << "detokenizer output " << idx++ << " " << output.get_any_name()
                      << " tensor partial shape " << output.get_partial_shape()
                      << " element type " << output.get_element_type() << std::endl;
        }
    }

    return true;
}

void OvTokenizerPrivate::loadChatTemplate(const std::string &path)
{
    chatTemplate.clear();

    std::string config = path + "/tokenizer_config.json";
    auto vars = loadJson(config);
    chatTemplate = vars.value("chat_template", "");

    return;
}

void OvTokenizerPrivate::loadTokenConfig(const std::string &path)
{
    {
        std::string config = path + "/config.json";
        auto vars = loadJson(config);

        padTokenId = vars.value("pad_token_id", -1);
        bosTokenId = vars.value("bos_token_id", -1);
        eosTokenId = vars.value("eos_token_id", -1);
    }

    if (padTokenId == -1 || bosTokenId == -1 || eosTokenId == 1
            || padToken.empty() || bosToken.empty() || eosToken.empty()) {
        std::string config = path + "/special_tokens_map.json";
        auto vars = loadJson(config);
        auto readToken = [&vars](const std::string &key, std::string &out) {
            if (out.empty() && vars.contains(key)) {
                auto tmp = vars[key];
                if (tmp.type() == nlohmann::detail::value_t::string)
                    out = tmp.get<std::string>();
                else {
                    out = tmp.value("content","");
                }
            }
        };
        readToken("pad_token", this->padToken);
        readToken("bos_token", this->bosToken);
        readToken("eos_token", this->eosToken);
    }

    if (padTokenId == -1 || bosTokenId == -1 || eosTokenId == 1
            || padToken.empty() || bosToken.empty() || eosToken.empty()) {

        std::string config = path + "/tokenizer_config.json";
        auto vars = loadJson(config);
        auto readToken = [&vars](const std::string &key, std::string &out) {
            if (out.empty() && vars.contains(key))
                out = vars[key].is_string() ? vars[key] : "";
        };

        readToken("pad_token", this->padToken);
        readToken("bos_token", this->bosToken);
        readToken("eos_token", this->eosToken);

        if (padToken.empty() && !eosToken.empty())
            padToken = eosToken;

        if (padTokenId != -1 && bosTokenId != -1 && eosTokenId != -1)
            return ;

        {
            auto atd = vars["added_tokens_decoder"];
            for (auto &item : atd.items()) {
                if (!item.value().is_object())
                    continue;
                int id = std::stoi(item.key());
                std::string content = item.value().value("content", "");
                if (padTokenId == -1 && content == padToken)
                    padTokenId = id;
                if (padTokenId == id && padToken.empty())
                    padToken = content;

                if (bosTokenId == -1 && content == bosToken)
                    bosTokenId = id;
                if (bosTokenId == id && bosToken.empty())
                    bosToken = content;

                if (eosTokenId == -1 && content == eosToken)
                    eosTokenId = id;
                if (eosTokenId == id && eosToken.empty())
                    eosToken = content;
            }
        }


        if (padTokenId == -1 && eosTokenId != -1)
             padTokenId = eosTokenId;
    }
}

nlohmann::json OvTokenizerPrivate::loadJson(const std::string &filePath)
{
    try {
          std::ifstream file(filePath);
          if (!file.is_open()) {
              // 返回一个空的JSON对象而不是抛出异常
              return nlohmann::json::object();
          }

          nlohmann::json jsonData = nlohmann::json::parse(file);
          return jsonData;
      } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "parse config error:" <<  filePath << " " << e.what() << std::endl;
        // 解析错误时返回空JSON对象
        return nlohmann::json::object();
    }
}

OvTokenizer::OvTokenizer(ov::Core *core, const std::string &tokenizerXml, const std::string &detokenizerXml, const std::string &configPath)
    : d(new OvTokenizerPrivate(core, this))
{
    d->initialize(tokenizerXml, detokenizerXml, configPath);
}

OvTokenizer::~OvTokenizer()
{
    delete d;
    d = nullptr;
}

std::vector<int64_t> OvTokenizer::encode(const std::string &prompt)
{
    std::string *prop = const_cast<std::string *>(&prompt);
    auto tokenizer = d->tokenizerModel.create_infer_request();

    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {1}, prop});
    tokenizer.infer();

    std::vector<int64_t> out;
    auto tensor = tokenizer.get_output_tensor(0);
    {
        int batch = tensor.get_shape().at(0);
        int dim = tensor.get_shape().at(1);
        const int stride = tensor.get_strides().at(0);
        if (batch > 0) {
            out.resize(dim);
            char *begin = (char *)tensor.data();
            memcpy(out.data(), begin, stride);
        }
    }

    return out;
}

std::string OvTokenizer::decode(const std::vector<int64_t> &tokens)
{
    auto detokenizer = d->detokenizerModel.create_infer_request();
    int64_t *tokenData  = const_cast<int64_t *>(tokens.data());
    detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {1, tokens.size()}, tokenData});
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

int64_t OvTokenizer::bosTokenId() const
{
    return d->bosTokenId;
}

int64_t OvTokenizer::eosTokenId() const
{
    return d->eosTokenId;
}

int64_t OvTokenizer::padTokenId() const
{
    return d->padTokenId;
}

std::string OvTokenizer::bosToken() const
{
    return d->bosToken;
}

std::string OvTokenizer::eosToken() const
{
    return d->eosToken;
}

std::string OvTokenizer::padToken() const
{
    return d->padToken;
}

