# vLLM 后端

[vLLM](https://github.com/vllm-project/vllm) 的 Triton 后端旨在在 [vLLM 引擎](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py)上运行[受支持的模型](https://vllm.readthedocs.io/en/latest/models/supported_models.html)。 您可以在[后端存储库](https://github.com/triton-inference-server/backend)了解有关 Triton 后端的更多信息。


这是一个[Python-based 后端](https://github.com/triton-inference-server/backend/blob/main/docs/python_based_backends.md#python-based-backends)。使用此后端时，所有接收到的请求都放在 vLLM AsyncEngine。动态批处理和分页注意力由 vLLM 引擎处理。

我在哪里可以询问有关 Triton 和 Triton 后端的常见问题？ 请务必阅读以下所有信息以及在[server](https://github.com/triton-inference-server/server)存储库主干中提供的[通用 Triton 文档](https://github.com/triton-inference-server/server#triton-inference-server)。如果你在那里找不到答案，你可以在 Triton [issues page](https://github.com/triton-inference-server/server/issues)提出问题。

## 安装 vLLM 后端

有几种方法可以安装和部署 vLLM 后端。

### 选项 1. 使用预构建的 Docker 容器。

从[NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)注册表中拉取一个包含vLLM后端的容器`tritonserver:<xx.yy>-vllm-python-py3`。\<xx.yy\>是您要使用的 Triton 版本。请注意，Triton 的 vLLM 容器已从 23.10 版本开始引入。

```
docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-vllm-python-py3
```

### 选项 2. 从源代码生成自定义容器
你可以按照[Building With Docker](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-with-docker)指南中描述的步骤操作，并使用[build.py](https://github.com/triton-inference-server/server/blob/main/build.py)脚本。

下面显示了一个示例命令，用于构建启用了所有选项的Triton Server容器。您可以根据需要随意定制标志。

请使用[NGC 注册表](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags) 获取最新版本的 Triton vLLM 容器，该容器对应于[Triton 版本](https://github.com/triton-inference-server/server/releases)的最新 YY.MM（年.月）。


```
# YY.MM is the version of Triton.
export TRITON_CONTAINER_VERSION=<YY.MM>
./build.py -v  --enable-logging
                --enable-stats
                --enable-tracing
                --enable-metrics
                --enable-gpu-metrics
                --enable-cpu-metrics
                --enable-gpu
                --filesystem=gcs
                --filesystem=s3
                --filesystem=azure_storage
                --endpoint=http
                --endpoint=grpc
                --endpoint=sagemaker
                --endpoint=vertex-ai
                --upstream-container-version=${TRITON_CONTAINER_VERSION}
                --backend=python:r${TRITON_CONTAINER_VERSION}
                --backend=vllm:r${TRITON_CONTAINER_VERSION}
```

### 选项 3. 将 vLLM 后端添加到默认 Triton 容器

您可以将 vLLM 后端直接安装到 NGC Triton 容器中。在这种情况下，请先安装vLLM。您可以通过运行`pip install vllm==<vLLM_version>`来做到这一点。然后，使用以下命令在容器中设置 vLLM 后端：

```
mkdir -p /opt/tritonserver/backends/vllm
wget -P /opt/tritonserver/backends/vllm https://raw.githubusercontent.com/triton-inference-server/vllm_backend/main/src/model.py
```

## 使用 vLLM 后端

您可以在[示例](samples)文件夹中看到示例[model_repository](samples/model_repository)。您可以按原样使用它，并通过更改`model.json`中的`model`值来更改模型。 `model.json`表示初始化模型时输入到 vLLM 的 AsyncLLMEngine 的键值字典。您可以在 vLLM 的[arg_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py)中查看支持的参数。具体来说，[这里](https://github.com/vllm-project/vllm/blob/ee8217e5bee5860469204ee57077a91138c9af02/vllm/engine/arg_utils.py#L11)和[这里](https://github.com/vllm-project/vllm/blob/ee8217e5bee5860469204ee57077a91138c9af02/vllm/engine/arg_utils.py#L201)。

对于多 GPU 支持，可以在[model.json](samples/model_repository/vllm_model/1/model.json)中指定如tensor_parallel_size的EngineArgs。

注意：在默认设置下，vLLM 会贪婪地消耗高达 90% 的 GPU 内存。示例模型通过将 gpu_memory_utilization 设置为 50% 来更新此行为。您可以使用 gpu_memory_utilization 等字段和[model.json](samples/model_repository/vllm_model/1/model.json)中的其他设置来调整此行为。

### 启动 Triton 推理服务器

设置好模型存储库后，就可以启动 Triton 服务器了。在此示例中，我们将使用来自[NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)[预构建的支持vLLM后端的 Triton 容器](#option-1-use-the-pre-built-docker-container)。

```
docker run --gpus all -it --net=host --rm -p 8001:8001 --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/work -w /work nvcr.io/nvidia/tritonserver:<xx.yy>-vllm-python-py3 tritonserver --model-repository ./model_repository
```

将 \<xx.yy\> 替换为您要使用的 Triton 版本。请注意，Triton 的 vLLM 容器是从 23.10 版本开始首次发布的。

启动 Triton 后，您将在控制台上看到输出，显示服务器启动并加载模型。当您看到如下输出时，Triton 已准备好接受推理请求。

```
I1030 22:33:28.291908 1 grpc_server.cc:2513] Started GRPCInferenceService at 0.0.0.0:8001
I1030 22:33:28.292879 1 http_server.cc:4497] Started HTTPService at 0.0.0.0:8000
I1030 22:33:28.335154 1 http_server.cc:270] Started Metrics Service at 0.0.0.0:8002
```

### 发送您的第一个推理

使用[示例 model_repository](samples/model_repository)[启动 Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html)后 ，您可以使用[generate 端口](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md)快速运行第一个推理请求。

尝试下面的命令。

```
$ curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0}}'
```

成功后，您应该会看到来自服务器的响应，如下所示：
```
{"model_name":"vllm_model","model_version":"1","text_output":"What is Triton Inference Server?\n\nTriton Inference Server is a server that is used by many"}
```

在[示例](samples)文件夹中，您还可以找到示例客户端 [client.py](samples/client.py)，它使用 Triton 的 [asyncio gRPC 客户端库](https://github.com/triton-inference-server/client#python-asyncio-support-beta-1) 在 Triton 上运行推理。

### 运行最新的 vLLM 版本

您可以从[Framework Containers Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)检查 Triton Inference Server 中包含的 vLLM 版本。
*注意:* vLLM Triton Inference Server 容器是从 23.10 版本开始引入的。

您可以在容器内使用 `pip install ...` 来升级vLLM版本。


## 运行 Triton 服务器的多个实例

如果您正在运行具有基于Python的后端的Triton服务器的多个实例，则需要为每个服务器指定不同的`shm-region-prefix-name`。浏览[此处](https://github.com/triton-inference-server/python_backend#running-multiple-instances-of-triton-server)获取更多信息。

## 参考教程

您可以在[教程](https://github.com/triton-inference-server/tutorials/)存储库中的[vLLM 快速部署指南](https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/vLLM)中进一步阅读。
