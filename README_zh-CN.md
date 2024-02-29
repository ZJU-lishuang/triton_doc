# Triton 推理服务器

Triton Inference Server是一个开源的推理服务软件，用于简化AI推理过程。Triton使团队能够从多个深度学习和机器学习框架中部署任何AI模型，包括TensorRT、TensorFlow、PyTorch、ONNX、OpenVINO、Python、RAPIDS FIL等。Triton Inference Server支持在云、数据中心、边缘和嵌入式设备上的NVIDIA GPU、x86和ARM CPU，或AWS Inferentia上进行推理。Triton Inference Server为多种查询类型提供了优化的性能，包括实时、批量、组合以及音频/视频流。Triton Inference Server是[NVIDIA AI Enterprise](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/)软件平台的一部分，该平台可加速数据科学管道并简化生产AI的开发和部署。

主要功能包括：

- [支持多种深度学习框架](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton)
- [支持多种机器学习框架](https://github.com/triton-inference-server/fil_backend)
- [并发模型执行](docs/user_guide/architecture.md#concurrent-model-execution)
- [动态批处理](docs/user_guide/model_configuration.md#dynamic-batcher)
- 有状态模型的[序列批处理](docs/user_guide/model_configuration.md#sequence-batcher)和[隐式状态管理](docs/user_guide/architecture.md#implicit-state-management)
- 提供[Backend API](https://github.com/triton-inference-server/backend)，允许添加自定义后端和预/后处理操作
- 支持用 python 编写自定义后端, 也称为[Python-based backends.](https://github.com/triton-inference-server/backend/blob/r24.01/docs/python_based_backends.md#python-based-backends)
- 模型流水线使用[集成](docs/user_guide/architecture.md#ensemble-models)或者[业务逻辑脚本 (BLS)](https://github.com/triton-inference-server/python_backend#business-logic-scripting)
- [HTTP/REST and GRPC inference protocols](docs/customization_guide/inference_protocols.md)基于社区开发的[KServe protocol](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2)
- [C API](docs/customization_guide/inference_protocols.md#in-process-triton-server-api)和[Java API](docs/customization_guide/inference_protocols.md#java-bindings-for-in-process-triton-server-api)允许 Triton 直接链接到您的应用程序，用于边缘和其他进程内用例
- [Metrics](docs/user_guide/metrics.md)表示GPU利用率，服务器吞吐量，服务器延时等

**不熟悉 Triton 推理服务器？** 利用[这些教程](https://github.com/triton-inference-server/tutorials)开始您的Triton之旅！

加入[Triton and TensorRT 社区](https://www.nvidia.com/en-us/deep-learning-ai/triton-tensorrt-newsletter/)可以随时了解最新的产品更新、错误修复、内容、最佳实践和更多内容。需要企业支持？NVIDIA 全球支持可用于具有[NVIDIA AI Enterprise 软件套件](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/)的Triton推理服务器。

##  3个简单步骤即可提供模型服务

```bash
# Step 1: Create the example model repository
git clone -b r24.01 https://github.com/triton-inference-server/server.git
cd server/docs/examples
./fetch_models.sh

# Step 2: Launch triton from the NGC Triton container
docker run --gpus=1 --rm --net=host -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.01-py3 tritonserver --model-repository=/models

# Step 3: Sending an Inference Request
# In a separate console, launch the image_client example from the NGC Triton SDK container
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:24.01-py3-sdk
/workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg

# Inference should return the following
Image '/workspace/images/mug.jpg':
    15.346230 (504) = COFFEE MUG
    13.224326 (968) = CUP
    10.422965 (505) = COFFEEPOT
```
关于这个例子的其他信息，请阅读[快速入门](docs/getting_started/quickstart.md)指南。快速入门指南还包含如何在[CPU-only 系统](docs/getting_started/quickstart.md#run-on-cpu-only-system)上启动 Triton的示例。Triton的新手，想知道从哪里开始？观看[入门视频](https://youtu.be/NQDtfSi5QF4)。

## 示例和教程

查看[NVIDIA LaunchPad](https://www.nvidia.com/en-us/data-center/products/ai-enterprise-suite/trial/)，使用NVIDIA基础设施上托管的Triton推理服务器免费访问一组动手实验。

流行的模型如 ResNet、BERT 和 DLRM的特定端到端示例位于GitHub上的[NVIDIA 深度学习示例](https://github.com/NVIDIA/DeepLearningExamples)中。[NVIDIA 开发专区](https://developer.nvidia.com/nvidia-triton-inference-server)包含其他文档、演示文稿和示例。

## 文档

### 构建和部署

构建和使用 Triton 推理服务器的推荐方法是使用 Docker 镜像。

- [Install Triton Inference Server with Docker containers](docs/customization_guide/build.md#building-with-docker) (*推荐*)
- [Install Triton Inference Server without Docker containers](docs/customization_guide/build.md#building-without-docker)
- [Build a custom Triton Inference Server Docker container](docs/customization_guide/compose.md)
- [Build Triton Inference Server from source](docs/customization_guide/build.md#building-on-unsupported-platforms)
- [Build Triton Inference Server for Windows 10](docs/customization_guide/build.md#building-for-windows-10)
- 在 [GCP](deploy/gcp/README.md)、[AWS](deploy/aws/README.md) 和 [NVIDIA FleetCommand](deploy/fleetcommand/README.md) 上使用 Kubernetes 和 Helm 部署 Triton 推理服务器的示例
- [安全部署注意事项](docs/customization_guide/deploy.md)

### 使用 Triton

#### 为 Triton 推理服务器准备模型

使用 Triton 为您的模型提供服务的第一步是放置一个或更多模型进入[模型存储库](docs/user_guide/model_repository.md)中。根据模型的类型以及要为模型启用的 Triton 功能，您可能需要为模型创建一个[模型配置](docs/user_guide/model_configuration.md)。

- [如果模型需要，将自定义操作添加到 Triton](docs/user_guide/custom_operations.md)
- 使用[模型集成](docs/user_guide/architecture.md#ensemble-models)和[业务逻辑脚本 (BLS)](https://github.com/triton-inference-server/python_backend#business-logic-scripting)启用模型流水线
- 优化模型，设置[调度和批处理](docs/user_guide/architecture.md#models-and-schedulers)参数以及[模型示例](docs/user_guide/model_configuration.md#instance-groups)
- 使用[模型分析工具](https://github.com/triton-inference-server/model_analyzer)通过分析帮助优化模型配置
- 了解如何[通过加载和卸载模型明确管理哪些模型可用](docs/user_guide/model_management.md)

#### 配置和使用 Triton 推理服务器

- 阅读[快速入门指南](docs/getting_started/quickstart.md)以在 GPU 和 CPU 上运行Triton 推理服务器
- Triton 支持多种执行引擎，称为
[backends](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton)，包括 
  [TensorRT](https://github.com/triton-inference-server/tensorrt_backend),
  [TensorFlow](https://github.com/triton-inference-server/tensorflow_backend),
  [PyTorch](https://github.com/triton-inference-server/pytorch_backend),
  [ONNX](https://github.com/triton-inference-server/onnxruntime_backend),
  [OpenVINO](https://github.com/triton-inference-server/openvino_backend),
  [Python](https://github.com/triton-inference-server/python_backend),和其它。
- 并非 Triton 支持的每个平台都支持上述所有后端。 查看[后端平台支持列表](https://github.com/triton-inference-server/backend/blob/r24.01/docs/backend_platform_support_matrix.md)，了解目标平台支持哪些后端。
- 了解如何使用[性能分析器](https://github.com/triton-inference-server/client/blob/r24.01/src/c++/perf_analyzer/README.md)和[模型分析器](https://github.com/triton-inference-server/model_analyzer)[优化性能](docs/user_guide/optimization.md)
- 学习如何在 Triton 中[管理加载和卸载模型](docs/user_guide/model_management.md)
- 使用[HTTP/REST JSON-based or gRPC protocols](docs/customization_guide/inference_protocols.md#httprest-and-grpc-protocols)直接向 Triton 发送请求

#### 客户端支持和示例

Triton *客户端*应用程序向 Triton 发送推理和其他请求。[Python and C++ 客户端库](https://github.com/triton-inference-server/client)提供了 API 来简化此通信。

- 查看你[C++](https://github.com/triton-inference-server/client/blob/r24.01/src/c%2B%2B/examples),
  [Python](https://github.com/triton-inference-server/client/blob/r24.01/src/python/examples),
  和 [Java](https://github.com/triton-inference-server/client/blob/r24.01/src/java/src/main/java/triton/client/examples)的客户端示例
- 配置 [HTTP](https://github.com/triton-inference-server/client#http-options)
  和 [gRPC](https://github.com/triton-inference-server/client#grpc-options)
  客户端选选项
- 在[无需任何额外的元数据的HTTP正文](https://github.com/triton-inference-server/server/blob/r24.01/docs/protocol/extension_binary_data.md#raw-binary-request)中将输入数据（例如jpeg图像）直接发送到Triton

### 扩展 Triton

[Triton Inference Server 的架构](docs/user_guide/architecture.md)是专门为模块化和灵活性而设计

- [自定义 Triton Inference Server 容器](docs/customization_guide/compose.md) 以满足您的使用需求
- 使用[C/C++](https://github.com/triton-inference-server/backend/blob/r24.01/README.md#triton-backend-api)或[Python](https://github.com/triton-inference-server/python_backend)[创建自定义后端](https://github.com/triton-inference-server/backend)
- 创建 [解耦的后端和模型](docs/user_guide/decoupled_models.md)，可以为请求发送多个响应，也可以不为请求发送任何响应
- 使用 [Triton 存储库代理](docs/customization_guide/repository_agents.md)添加在模型加载和卸载时运行的功能，例如身份验证、解密或转换
- 在 [Jetson 和 JetPack](docs/user_guide/jetson.md)上部署Triton
- [在AWS Inferentia上使用Triton](https://github.com/triton-inference-server/python_backend/tree/main/inferentia)

### 其他文档

- [常见问题](docs/user_guide/faq.md)
- [用户指南](docs/README.md#user-guide)
- [定制指南](docs/README.md#customization-guide)
- [发行说明](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html)
- [GPU、驱动程序和 CUDA 支持列表](https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html)

## 贡献

欢迎对 Triton 推理服务器做出贡献。请查看 [贡献指南](CONTRIBUTING.md) 以了解如何做出贡献。如果您有后端、客户端、示例或类似的贡献，但不修改 Triton 的核心，则应在 [contrib repo](https://github.com/triton-inference-server/contrib) 中提交 PR。

## 报告问题，提问

我们欢迎任何关于这个项目的反馈、问题或bug报告。
在 [GitHub 上发布问题](https://github.com/triton-inference-server/server/issues)时，
遵循[Stack Overflow文档](https://stackoverflow.com/help/mcve)中概述的过程。
确保张贴的示例是：
- 最小化 - 使用尽可能少的代码，但仍然会产生同样的问题
- 完整 - 提供再现问题所需的所有部分。检查您是否可以剥离外部依赖关系，并仍然显示问题。我们在再现问题上花费的时间越少，我们就越有时间修复它
- 可验证的 - 测试您即将提供的代码，以确保它重现问题。删除与您的请求/问题无关的所有其他问题。

对于问题，请使用提供的错误报告和功能请求模板。

如有疑问，我们建议您在我们的社区[GitHub 讨论区](https://github.com/triton-inference-server/server/discussions)中发布

## 了解更多信息

请参阅[NVIDIA Developer Triton 页面](https://developer.nvidia.com/nvidia-triton-inference-server)
了解更多信息。