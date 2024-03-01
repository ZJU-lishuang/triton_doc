# 概念指南

| 相关页面 | [服务器文档](https://github.com/triton-inference-server/server/tree/main/docs#triton-inference-server-documentation) |
| ------------ | --------------- |

概念指南被设计为 Triton 推理服务器的入门体验。这些指南将涵盖：
* [第 1 部分：模型部署](Part_1-model_deployment/): 本指南介绍如何部署和管理多个模型。
* [第 2 部分：提高资源利用率](Part_2-improving_resource_utilization/): 本指南讨论了两种流行的功能/技术，用于在部署模型时最大限度地提高 GPU 的利用率。
* [第 3 部分：优化 Triton 配置](Part_3-optimizing_triton_configuration/): 每个部署都有特定于用例的要求。本指南将引导用户完成定制部署配置以匹配 SLA 的过程。
* [第 4 部分：加速模型](Part_4-inference_acceleration/): 实现更高吞吐量的另一种途径是加速基础模型。本指南介绍了可用于加速模型的 SDK 和工具。
* [第 5 部分：构建模型集合](./Part_5-Model_Ensembles/): 模型很少单独使用。本指南将介绍“如何构建深度学习推理流水线？”
* [第 6 部分：使用 BLS API 构建复杂的流水线](Part_6-building_complex_pipelines/): 通常情况下，流水线需要控制流。了解如何使用部署在不同后端的模型来处理复杂的流水线。