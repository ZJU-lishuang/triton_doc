vLLM 后端使用 vLLM 进行推理。在[此处](https://blog.vllm.ai/2023/06/20/vllm.html)阅读有关 vLLM 的更多信息，在[此处](https://github.com/triton-inference-server/vllm_backend)阅读 vLLM 后端。

## 预构建说明

在本教程中，我们将使用带有预训练权重的 Llama2-7B HuggingFace 模型。请按照 [README.md](README.md) 获取预构建说明,和了解如何使用其他后端运行 Llama的链接。

## 安装

triton vLLM 容器可以从 [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) 中拉取

```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v $PWD/llama2vllm:/opt/tritonserver/model_repository/llama2vllm \
    nvcr.io/nvidia/tritonserver:23.11-vllm-python-py3
```
这将创建一个包含`llama2vllm`模型的路径为`/opt/tritonserver/model_repository`的文件夹。模型本身将从HuggingFace拉取

进入容器后，安装`huggingface-cli`并使用自己的账户登录
```bash
pip install --upgrade huggingface_hub
huggingface-cli login --token <your huggingface access token>
```


## 使用Triton提供服务

然后，您可以像往常一样运行tritonserver
```bash
tritonserver --model-repository model_repository
```
当您在控制台中看到以下输出时，服务器已成功启动：

```
I0922 23:28:40.351809 1 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0922 23:28:40.352017 1 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0922 23:28:40.395611 1 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

## 通过`generate`端口发送请求

作为一个确保服务器工作的简单示例，可以使用`generate`端口进行测试。有关generate端口的详细信息 在[此处](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md)。

```bash
$ curl -X POST localhost:8000/v2/models/llama2vllm/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0}}'
# returns (formatted for better visualization)
> {
    "model_name":"llama2vllm",
    "model_version":"1",
    "text_output":"What is Triton Inference Server?\nTriton Inference Server is a lightweight, high-performance"
  }
```

## 通过Triton客户端发送请求

Triton vLLM 后端存储库有一个[示例文件夹](https://github.com/triton-inference-server/vllm_backend/tree/main/samples)，其中包含用于测试 Llama2 模型的示例文件 client.py。

```bash
pip3 install tritonclient[all]
# Assuming Tritonserver server is running already
$ git clone https://github.com/triton-inference-server/vllm_backend.git
$ cd vllm_backend/samples
$ python3 client.py -m llama2vllm

```
以下步骤应产生具有以下内容的 `results.txt` 文件
```bash
Hello, my name is
I am a 20 year old student from the Netherlands. I am currently

=========

The most dangerous animal is
The most dangerous animal is the one that is not there.
The most dangerous

=========

The capital of France is
The capital of France is Paris.
The capital of France is Paris. The

=========

The future of AI is
The future of AI is in the hands of the people who use it.

=========
```