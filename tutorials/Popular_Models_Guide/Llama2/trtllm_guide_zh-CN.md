TensorRT-LLM 是 Nvidia 推荐的在Nvidia GPU 上运行大型语言模型（LLM）的解决方案。在[此处](https://github.com/NVIDIA/TensorRT-LLM)阅读有关 TensoRT-LLM 的更多信息，在[此处](https://github.com/triton-inference-server/tensorrtllm_backend)阅读 Triton 的 TensorRTLLM 后端。

*注意:* 如果本教程的某些部分不起作用，则可能是`tutorials`和`tensorrt_backend`存储库之间的某些版本不匹配。如有必要，参考[llama.md](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md)获得更详细的变更内容。


## 预构建说明

在本教程中，我们将使用带有预训练权重的 Llama2-7B HuggingFace 模型。 
克隆[此处](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main)的具有授权和权重模型的存储库。 
您需要获得 Llama2 存储库的权限，并访问 huggingface cli。要访问 huggingface cli，请转到此处：[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)。

## 安装

1. 安装从克隆 TensorRT-LLM 后端开始，并更新 TensorRT-LLM 子模块：
```bash
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git  --branch <release branch>
# Update the submodules
cd tensorrtllm_backend
# Install git-lfs if needed
apt-get update && apt-get install git-lfs -y --no-install-recommends
git lfs install
git submodule update --init --recursive
```

2. 带有TensorRT-LLM后端启动 Triton 容器。注意，为了简单起见，在容器中将挂载`tensorrtllm_backend`到`/tensorrtllm_backend`，Llama2模型到`/Llama-2-7b-hf`。在容器外创建一个`engines`文件夹以便将来在运行时重载引擎。
```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tensorrtllm_backend:/tensorrtllm_backend \
    -v /path/to/Llama2/repo:/Llama-2-7b-hf \
    -v /path/to/engines:/engines \
    nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3
```

或者，如果想构建专用的容器，可以按照[此处](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md)的说明使用Tensorrt-LLM后端构建Triton Server。

不要忘记在启动容器时允许 gpu 使用。

## 为每个模型创建引擎 [如果您已经有引擎，请跳过此步骤]
TensorRT-LLM 要求在运行之前针对您需要的配置编译每个模型。为此，在首次在Triton Server上运行模型之前，您需要通过以下步骤为模型创建所需的配置的TensorRT-LLM引擎：

1. 安装 Tensorrt-LLM python 包
   ```bash
    # Install CMake
    bash /tensorrtllm_backend/tensorrt_llm/docker/common/install_cmake.sh
    export PATH="/usr/local/cmake/bin:${PATH}"

    # PyTorch needs to be built from source for aarch64
    ARCH="$(uname -i)"
    if [ "${ARCH}" = "aarch64" ]; then TORCH_INSTALL_TYPE="src_non_cxx11_abi"; \
    else TORCH_INSTALL_TYPE="pypi"; fi && \
    (cd /tensorrtllm_backend/tensorrt_llm &&
        bash docker/common/install_pytorch.sh $TORCH_INSTALL_TYPE &&
        python3 ./scripts/build_wheel.py --trt_root=/usr/local/tensorrt &&
        pip3 install ./build/tensorrt_llm*.whl)
    ```

2.  编译模型引擎

    用于构建 Llama 模型的脚本位于[TensorRT-LLM 存储库](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples)。我们使用容器中的`/tensorrtllm_backend/tensorrt_llm/examples/llama/build.py`脚本。
    此命令使用动态批处理和1个GPU 编译模型。要使用更多 GPU 运行，您需要使用`--world_size X`来更改 build 命令。
    有关脚本的更多细节，参阅[此处](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama/README.md)的Llama示例文档。

    ```bash
    python /tensorrtllm_backend/tensorrt_llm/examples/llama/build.py --model_dir /Llama-2-7b-hf/ \
                    --dtype bfloat16 \
                    --use_gpt_attention_plugin bfloat16 \
                    --use_inflight_batching \
                    --paged_kv_cache \
                    --remove_input_padding \
                    --use_gemm_plugin bfloat16 \
                    --output_dir /engines/1-gpu/ \
                    --world_size 1
    ```

    > Optional: You can check test the output of the model with `run.py`
    > located in the same llama examples folder.
    >
    >   ```bash
    >    python3 /tensorrtllm_backend/tensorrt_llm/examples/llama/run.py --engine_dir=/engines/1-gpu/ --max_output_len 100 --tokenizer_dir /Llama-2-7b-hf --input_text "How do I count to ten in French?"
    >    ```

## Triton提供服务

最后一步是创建一个Triton可读模型。您可以在 [tensorrtllm_backend/all_models/inflight_batcher_llm](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main/all_models/inflight_batcher_llm) 中找到使用动态批处理的模型模板。 
要运行我们的 Llama2-7B 模型，您需要：


1. 复制动态批处理模型存储库

 ```bash
 cp -R /tensorrtllm_backend/all_models/inflight_batcher_llm /opt/tritonserver/.
 ```

2. 修改 config.pbtxt 进行预处理、后处理和处理步骤。有关详细信息，请参阅[文档](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md#create-the-model-repository):

    ```bash
    # preprocessing
    sed -i 's#${tokenizer_dir}#/Llama-2-7b-hf/#' /opt/tritonserver/inflight_batcher_llm/preprocessing/config.pbtxt
    sed -i 's#${tokenizer_type}#auto#' /opt/tritonserver/inflight_batcher_llm/preprocessing/config.pbtxt
    sed -i 's#${tokenizer_dir}#/Llama-2-7b-hf/#' /opt/tritonserver/inflight_batcher_llm/postprocessing/config.pbtxt
    sed -i 's#${tokenizer_type}#auto#' /opt/tritonserver/inflight_batcher_llm/postprocessing/config.pbtxt

    sed -i 's#${decoupled_mode}#false#' /opt/tritonserver/inflight_batcher_llm/tensorrt_llm/config.pbtxt
    sed -i 's#${engine_dir}#/engines/1-gpu/#' /opt/tritonserver/inflight_batcher_llm/tensorrt_llm/config.pbtxt
    ```
    此外，请确保将参数`gpt_model_type`设置为`inflight_fused_batching`

3.  启动Tritonserver

    使用 [launch_triton_server.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/release/0.5.0/scripts/launch_triton_server.py) 脚本。这将使用MPI启动`tritonserver`的多个实例。
    ```bash
    python3 /tensorrtllm_backend/scripts/launch_triton_server.py --world_size=<world size of the engine> --model_repo=/opt/tritonserver/inflight_batcher_llm
    ```

## 客户端

您可以使用以下方法测试运行结果：
1. [inflight_batcher_llm_client.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/client/inflight_batcher_llm_client.py) 脚本。

```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tensorrtllm_backend:/tensorrtllm_backend \
    -v /path/to/Llama2/repo:/Llama-2-7b-hf \
    -v /path/to/engines:/engines \
    nvcr.io/nvidia/tritonserver:23.10-py3-sdk
# Install extra dependencies for the script
pip3 install transformers sentencepiece
python3 /tensorrtllm_backend/inflight_batcher_llm/client/inflight_batcher_llm_client.py --request-output-len 200 --tokenizer_type llama --tokenizer_dir /Llama-2-7b-hf
```

2. 如果正在使用的Triton TensorRT-LLM后端容器版本大于`r23.10`，则可以使用 [generate 端口](https://github.com/triton-inference-server/tensorrtllm_backend/tree/release/0.5.0#query-the-server-with-the-triton-generate-endpoint)。


