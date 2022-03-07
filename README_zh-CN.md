# Triton推理服务器

Triton推理服务器提供了一个针对CPU和GPU优化的云端和边缘端推理解决方案 。Triton支持HTTP/REST和GRPC协议，允许远程客户端请求推理被服务器管理的任何模型。对于边缘部署，Triton可以作为有C接口的动态库，这允许Triton的全部功能被直接包含在一个应用程序中。

## 2.16.0的新特性

* 在FIL后端支持有[分类功能的LightGBM模型](https://github.com/triton-inference-server/fil_backend/tree/r21.11#categorical-feature-support)。

* 在文档中添加[Jetson示例](docs_zh-CN/examples/jetson)。

* 完成对[Inferentia支持](https://github.com/triton-inference-server/python_backend/tree/r21.11/inferentia#readme)概念证明。

* 模型分析增加对ARM的支持。

## 特性

* [多个深度学习框架](https://github.com/triton-inference-server/backend). Triton可以管理任意数量和格式的模型（受系统盘和内存资源限制）。Triton支持TensorRT, TensorFlow GraphDef,TensorFlow SavedModel, ONNX, PyTorch TorchScript和OpenVINO模型格式。TensorFlow 1.x和TensorFlow 2.x同时被支持。Triton也支持TensorFlow-TensorRT和ONNX-TensorRT的整合模型。

* [模型并发执行](docs_zh-CN/architecture.md#concurrent-model-execution). 多模型（或者同模型的多个实例）能同时运行在同一GPU或多个GPU上。

* [动态批处理](docs_zh-CN/architecture.md#models-and-schedulers).对支持批处理的模型，Triton实现了多种调度和批处理的算法，将单个推理请求组合在一起来提高推理的吞吐量。这些调度和批处理决策对请求推理的客户端是透明的。

* [可扩展后端](https://github.com/triton-inference-server/backend). 在深度学习框架之外，Triton提供一个*后端API*，允许用[Python](https://github.com/triton-inference-server/python_backend)或[C++](https://github.com/triton-inference-server/backend/blob/main/README.md#triton-backend-api)实现任何模型执行逻辑对Triton进行扩展，同时仍然受益于Triton提供的CPU和GPU支持，并发执行，动态批处理和其它特性。

* [模型管道](docs_zh-CN/architecture.md#ensemble-models). Triton*集合*代表一个或多个的模型管道，以及这些模型间输入和输出张量的连接。对集合的单个推理请求将触发整个管道的执行。

* [HTTP/REST和GRPC推理协议](docs_zh-CN/inference_protocols.md)基于社区开发的[KFServing协议](https://github.com/kubeflow/kfserving/tree/master/docs_zh-CN/predict-api/v2).

* [C接口](docs_zh-CN/inference_protocols.md#c-api)允许Triton直接链接到你的应用程序中，用于边缘和其它进程中的用例。

* [指标](docs_zh-CN/metrics.md)显示GPU利用率，服务器吞吐量和延时。指标以Prometheus数据格式提供。

## 文档

[Triton体系结构](docs_zh-CN/architecture.md)提供了对推理服务器结构和能力的高级概述。 这也有一个[FAQ](docs_zh-CN/faq.md)。另外的文档分为[*用户*](#user-documentation)和[*开发者*](#developer-documentation) 部分。*用户*文档描述了如何使用Triton作为一个推理解决方案，包括如何配置Triton，如何组织和配置你的模型，如何使用C++和Python的客户端等。*开发者*文档描述了如何构建和测试Triton，以及如何扩展Triton的新特性。

Triton的[发布说明](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html)和[支持矩阵](https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html)表明了NVIDIA驱动和CUDA版本的要求，以及支持的GPUs。

### 用户文档

* [快速开始](docs_zh-CN/quickstart.md)
  * [安装](docs_zh-CN/quickstart.md#install-triton-docker-image)
  * [运行](docs_zh-CN/quickstart.md#run-triton)
* [模型存储](docs_zh-CN/model_repository.md)
* [模型配置](docs_zh-CN/model_configuration.md)
* [模型管理](docs_zh-CN/model_management.md)
* [自定义操作](docs_zh-CN/custom_operations.md)
* [客户端的库和示例](https://github.com/triton-inference-server/client)
* [优化](docs_zh-CN/optimization.md)
  * [模型分析](docs_zh-CN/model_analyzer.md)
  * [性能分析](docs_zh-CN/perf_analyzer.md)
* [指标](docs_zh-CN/metrics.md)
* [速率限制器](docs_zh-CN/rate_limiter.md)
* [Jetson和JetPack](docs_zh-CN/jetson.md)

[快速入门](docs_zh-CN/quickstart.md)指导你所有的步骤，包括安装Triton，运行一个分类模型在Triton，使用该模型在客户端示例应用程序执行推理。快速入门也演示了[Triton的支持包括GPU的系统和仅有CPU的系统](docs_zh-CN/quickstart.md#run-triton).

使用Triton服务于你的模型的第一步是将一个或多个模型放在一个[模型存储](docs_zh-CN/model_repository.md)中。使用Triton服务于模型的第一步是将一个或多个模型放入[模型存储库](docs_zh-CN/model_repository.md)中。这是可选的，根据模型的类型以及您希望为模型启用的Triton功能，您可能需要为模型创建一个[模型配置](docs_zh-CN/model_configuration.md)。如果你的模型有[自定义操作](docs_zh-CN/custom_operations.md)，你需要确保它们被Triton正确加载。

在Triton中有了可用的模型之后，您将希望从*客户端*应用程序向Triton发送推断和其他请求。[Python和c++客户端库](https://github.com/triton-inference-server/client)提供了API来简化这种通信。还有大量[客户端示例](https://github.com/triton-inference-server/client)演示了如何使用这些库。您还可以使用[HTTP/REST基于json的协议](docs_zh-CN/inference_protocols.md#httprest-and-grpc-protocols)直接向Triton发送HTTP/REST请求，或者[为许多其他语言生成GRPC客户端](https://github.com/triton-inference-server/client)。

理解和[优化性能](docs_zh-CN/optimization.md)是部署模型的重要部分。Triton项目提供了[性能分析器](docs_zh-CN/perf_analyzer.md)和[模型分析器](docs_zh-CN/model_analyzer.md)来帮助您进行优化工作。具体来说，您需要为每个模型适当地优化[调度和批处理](docs_zh-CN/architecture.md#models-and-schedulers)以及[模型实例](docs_zh-CN/model_configuration.md#instance-groups)。您还可以使用[速率限制器](docs_zh-CN/rate_limiter.md)启用跨模型优先级，该速率限制器管理请求在模型实例上调度的速率。您可能还想[集成多个模型和预处理/后处理](docs_zh-CN/architecture.md#ensemble-models)到一个管道中。在某些情况下，您可能会发现在优化时[单个推理请求跟踪数据](docs_zh-CN/trace.md)很有用。[Prometheus指标节点](docs_zh-CN/metrics.md)允许您可视化和监控总体推理指标。

NVIDIA发布了许多使用Triton的[深度学习示例](https://github.com/NVIDIA/DeepLearningExamples)。

作为部署策略的一部分，您可能希望通过在运行的Triton服务器中[通过加载和卸载模型来显示的管理哪些模型可用](docs_zh-CN/model_management.md)。如果你正在使用Kubernetes进行部署，有一些简单的例子可以说明如何使用Kubernetes和Helm部署Triton，一个用于[GCP](deploy/gcp/README.md)，一个用于[AWS](deploy/aws/README.md)。

如果您从以前使用的版本1迁移到版本2，那么[版本1到版本2的迁移信息](docs_zh-CN/v1_to_v2.md)是很有帮助的。

### 开发者文档

* [构建](docs_zh-CN/build.md)
* [协议和APIs](docs_zh-CN/inference_protocols.md).
* [后端](https://github.com/triton-inference-server/backend)
* [存储库代理](docs_zh-CN/repository_agents.md)
* [测试](docs_zh-CN/test.md)

Triton可以[使用Docker构建](docs_zh-CN/build.md#building-triton-with-docker)，也可以[不使用Docker构建](docs_zh-CN/build.md#building-triton-without-docker)。在构建完之后，你应该[测试Triton](docs_zh-CN/test.md)。

也可以[创建一个包含自定义Triton的Docker镜像](docs_zh-CN/compose.md)，包含后端的一个子集。

Triton项目还提供了[Python和C++的客户端库](https://github.com/triton-inference-server/client)，使和服务器的通信变得容易。还有大量的[客户端示例](https://github.com/triton-inference-server/client)演示了如何使用这些库。您还可以开发自己的客户端，使用[HTTP/REST或GRPC协议](docs_zh-CN/inference_protocols.md)直接与Triton通信。还有一个[C API](docs_zh-CN/inference_protocols.md)，可以直接将Triton链接到你的应用程序中。

[Triton后端](https://github.com/triton-inference-server/backend)是执行模型的实现。后端可以与深度学习框架对接，如PyTorch、TensorFlow、TensorRT或ONNX Runtime;或者它可以与数据处理框架[DALI](https://github.com/triton-inference-server/dali_backend)对接;或者你可以通过[C/C++](https://github.com/triton-inference-server/backend/blob/main/README.md#triton-backend-api)或[Python](https://github.com/triton-inference-server/python_backend)来[编写自己的后端](https://github.com/triton-inference-server/backend)来扩展Triton。

[Triton存储库代理](docs_zh-CN/repository_agents.md)用新功能扩展了Triton，该功能在模型加载或卸载时运行。当加载模型时，您可以引入自己的代码来执行身份验证、解密、转换或类似的操作。

## 论文和演讲

* [Maximizing Deep Learning Inference Performance with NVIDIA Model
  Analyzer](https://developer.nvidia.com/blog/maximizing-deep-learning-inference-performance-with-nvidia-model-analyzer/).

* [High-Performance Inferencing at Scale Using the TensorRT Inference
  Server](https://developer.nvidia.com/gtc/2020/video/s22418).

* [Accelerate and Autoscale Deep Learning Inference on GPUs with
  KFServing](https://developer.nvidia.com/gtc/2020/video/s22459).

* [Deep into Triton Inference Server: BERT Practical Deployment on
  NVIDIA GPU](https://developer.nvidia.com/gtc/2020/video/s21736).

* [Maximizing Utilization for Data Center Inference with TensorRT
  Inference Server](https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=s9438-maximizing+utilization+for+data+center+inference+with+tensorrt+inference+server).

* [NVIDIA TensorRT Inference Server Boosts Deep Learning
  Inference](https://devblogs.nvidia.com/nvidia-serves-deep-learning-inference/).

* [GPU-Accelerated Inference for Kubernetes with the NVIDIA TensorRT
  Inference Server and
  Kubeflow](https://www.kubeflow.org/blog/nvidia_tensorrt/).

* [Deploying NVIDIA Triton at Scale with MIG and Kubernetes](https://developer.nvidia.com/blog/deploying-nvidia-triton-at-scale-with-mig-and-kubernetes/). 

## 贡献

我们非常欢迎对Triton Inference Server的贡献。遵循[CONTRIBUTING.md](CONTRIBUTING.md)中的指南概述提出一个pull请求来做出贡献。如果你有一个后端，客服端，例子或类似的贡献，且没有修改Triton的核心，你可以在[contrib repo](https://github.com/triton-inference-server/contrib)提交PR。

## 报告问题，提出问题

我们欢迎任何关于这个项目的反馈，问题或者bug报告。当需要代码方面的帮助时，遵循Stack Overflow (<https://stackoverflow.com/help/mcve>)文档中的流程概述。确保发布的例子如下：

* 最小化 – 使用尽可能少的代码任然产生问题。

* 完整的 – 提供重现问题所需的所有部分。检查是否可以剥离外部依赖项但问题仍然存在。我们花在重现问题的时间越少，解决问题的诗句就越多。

* 可验证 – 测试你提供的代码，确保能重现问题。删除与你的请求/问题无关的所有其它问题。
