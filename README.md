# Deepin ModelHub

deepin-modelhub 是一个模型推理服务框架,用于在 Deepin 操作系统上运行和管理各类 AI 模型。它提供了统一的模型管理和推理接口,让用户可以方便地使用各种 AI 模型。

## 功能特性

- 支持多种推理后端，包括:
  - llama.cpp - 用于大语言模型推理
  - openvino - 用于计算机视觉等通用模型推理
- 提供统一的模型管理功能:
  - 模型加载和卸载
  - 模型版本管理 
  - 模型参数配置
- 硬件资源调度:
  - CPU/GPU 自动选择
  - 多设备负载均衡
- HTTP API 服务:
  - 兼容 OpenAI API 规范
  - RESTful 接口设计
  - 支持流式输出

## 系统要求

- 操作系统: Deepin/UOS
- 开发环境:
  - CMake >= 3.14
  - C++17 编译器支持
- 硬件要求:
  - CPU: 支持 AVX2 指令集
  - 内存: >= 8GB
  - 磁盘空间: >= 1GB

## 编译安装

1. 克隆代码仓库:
```
mkdir build
cd build
```

3. 配置和编译:

```bash
cmake ..
make
```

4. 安装:

```bash
sudo make install
```

## 项目结构

- 3rdparty - 第三方库
  - llama.cpp.tar.gz - llama.cpp 推理库
  - inja - 模板引擎，类似于 Python 的 Jinja2
  - nlohmann-json - json 库
- src/modelhub - 主程序
- src/libmodelhub - 核心库
- src/inference-backend - 推理后端实现
  - llama.cpp - LLaMA 模型推理后端
  - openvino - openvino 模型推理后端

## 开发文档

### 添加新的推理后端

1. 在 src/inference-backend 下创建新的后端目录
2. 实现推理插件接口
3. 在 CMake 中添加新后端的构建配置

### HTTP API

服务启动后默认监听本地端口,提供以下 API:

- POST /api/inference - 执行模型推理
- POST /api/chat/completions - 执行聊天推理
- GET /api/models - 获取可用模型列表
- POST /api/load - 加载模型

## 许可证

本项目采用 GPL-3.0 开源许可证。

## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。在提交代码前,请确保:

1. 代码符合项目代码规范
2. 添加必要的测试用例
3. 更新相关文档

## 联系方式

- 项目主页: https://github.com/linuxdeepin/deepin-modelhub
- Bug 报告: https://github.com/linuxdeepin/deepin-modelhub/issues
```
