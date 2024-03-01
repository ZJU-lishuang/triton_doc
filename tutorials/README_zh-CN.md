# Triton 教程


对于使用“Tensor in”和“Tensor out”方法进行深度学习推理的用户来说，开始使用Triton可能会导致许多问题。此存储库的目的是让用户熟悉Triton的功能，并提供指导和示例以简化迁移。有关功能说明，请参阅[Triton Inference Server文档][Triton Inference Server 文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html).

#### 入门清单
| [概述视频](https://www.youtube.com/watch?v=NQDtfSi5QF4) | [概念指南: 部署模型](Conceptual_Guide/Part_1-model_deployment/README.md) |
| ------------ | --------------- |

## 快速部署

这些示例的重点是演示使用各种框架训练的模型进行部署。这些都是在用户对 Triton 有些熟悉的情况下进行的快速演示。

### 部署一个 ...
| [PyTorch 模型](./Quick_Deploy/PyTorch/README.md) | [TensorFlow 模型](./Quick_Deploy/TensorFlow/README.md) | [ONNX 模型](./Quick_Deploy/ONNX/README.md) | [TensorRT 加速模型](https://github.com/NVIDIA/TensorRT/tree/main/quickstart/deploy_to_triton) | [vLLM 模型](./Quick_Deploy/vLLM/README.md)
| --------------- | ------------ | --------------- | --------------- | --------------- |

## LLM 教程
下表包含我们的教程中支持的一些流行模型
| 示例模型   | 教程链接 |
| :-------------: | :------------------------------: |
| [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main) |[TensorRT-LLM 教程](Popular_Models_Guide/Llama2/trtllm_guide.md) |
| [Persimmon-8B](https://www.adept.ai/blog/persimmon-8b) | [HuggingFace Transformers 教程](https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/HuggingFaceTransformers)  |
[Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) |[HuggingFace Transformers 教程](https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/HuggingFaceTransformers)   |

**注意:**
这不是Triton支持的详尽列表，只是教程中包含的内容。

## 此存储库包含哪些内容？
此存储库包含以下资源：
* [概念指南](./Conceptual_Guide/)：本指南侧重于建立对构建推理基础设施时面临的一般挑战的概念性理解，以及如何最好地使用 Triton Inference Server 来应对这些挑战。
* [快速部署](#quick-deploy)：这是一组关于将模型从首选框架部署到Triton推理服务器的指南。这些指南假设对Triton推理服务器有基本的了解。建议查看入门材料以获得完整的理解。
* [HuggingFace 指南](./HuggingFace/): 本指南的重点是引导用户使用 Triton 推理服务器部署 HuggingFace 模型的不同方法。
* [功能指南](./Feature_Guide/)：此文件夹旨在存放Triton的功能特定示例。
* [迁移指南](./Migration_Guide/migration_guide.md)：从现有解决方案迁移到 Triton Inference Server？了解可能最适合您的用例的总体架构。

## Triton 推理服务器资源导航

Triton Inference Server GitHub 组织包含多个存储库，其中包含 Triton Inference Server 的不同功能。以下内容并非对所有存储库的完整描述，而只是一个简单的指南，以建立直观的理解。

* [服务器](https://github.com/triton-inference-server/server)是Triton推理服务器库的主体部分。
* [客户端](https://github.com/triton-inference-server/client)包含创建Triton客户端所需的库和示例
* [后端](https://github.com/triton-inference-server/backend) 包含构建新 Triton 后端的核心脚本和实用程序。任何包含“后端”一词的存储库都是框架后端或如何创建后端的示例。
* [模型分析器](https://github.com/triton-inference-server/model_analyzer)和[模型导航器](https://github.com/triton-inference-server/model_navigator)等工具提供了工具来衡量性能或简化模型加速。

## 添加请求

打开一个问题并指定添加请求的详细信息。想要做出贡献吗？打开一个拉取请求并标记管理员。
