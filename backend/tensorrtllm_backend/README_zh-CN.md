# TensorRT-LLM 后端
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)的Triton后端。
您可以在[backend repo](https://github.com/triton-inference-server/backend)中了解有关 Triton 后端的更多信息。
TensorRT-LLM 后端的目标是让您通过 Triton 推理服务器提供[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)模型服务。[inflight_batcher_llm](./inflight_batcher_llm/)目录包含后端的 C++ 实现，支持动态批处理、分页注意力等。

我在哪里可以询问有关 Triton 和 Triton 后端的常见问题？ 请务必阅读以下所有信息以及在[server](https://github.com/triton-inference-server/server)存储库主干中提供的[通用 Triton 文档](https://github.com/triton-inference-server/server#triton-inference-server)。如果你在那里找不到答案，你可以在 Triton [issues page](https://github.com/triton-inference-server/server/issues)提出问题。

## 访问 TensorRT-LLM 后端

有多种方法可以访问 TensorRT-LLM 后端。

**在 Triton 23.10 版本之前，请使用 [选项 3 通过 Docker 构建 TensorRT-LLM 后端](#option-3-build-via-docker)。**

### 运行预构建的 Docker 容器

从 Triton 23.10 版本开始，Triton 包含一个带有 TensorRT-LLM 后端和 Python 后端的容器。该容器应该具备运行 TensorRT-LLM 模型的一切。您可以在 [Triton NGC 页面](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)上找到此容器。

### 构建 Docker 容器

#### 选项 1. 在Server存储库中通过`build.py`脚本构建

从 Triton 23.10 版本开始，您可以按照[Building With Docker](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-with-docker)指南中描述的步骤并使用[build.py](https://github.com/triton-inference-server/server/blob/main/build.py)脚本构建 TRT-LLM 后端。

以下命令将构建与 NGC 上的容器相同的 Triton TRT-LLM 容器。

```bash
# Prepare the TRT-LLM base image using the dockerfile from tensorrtllm_backend.
cd tensorrtllm_backend
# Specify the build args for the dockerfile.
BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.01-py3-min
TRT_VERSION=9.2.0.5
TRT_URL_x86=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.linux.x86_64-gnu.cuda-12.2.tar.gz
TRT_URL_ARM=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.Ubuntu-22.04.aarch64-gnu.cuda-12.2.tar.gz

docker build -t trtllm_base \
             --build-arg BASE_IMAGE="${BASE_IMAGE}" \
             --build-arg TRT_VER="${TRT_VERSION}" \
             --build-arg RELEASE_URL_TRT_x86="${TRT_URL_x86}" \
             --build-arg RELEASE_URL_TRT_ARM="${TRT_URL_ARM}" \
             -f dockerfile/Dockerfile.triton.trt_llm_backend .

# Run the build script from Triton Server repo. The flags for some features or
# endpoints can be removed if not needed. Please refer to the support matrix to
# see the aligned versions: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
TRTLLM_BASE_IMAGE=trtllm_base
TENSORRTLLM_BACKEND_REPO_TAG=v0.7.2
PYTHON_BACKEND_REPO_TAG=r24.01

cd server
./build.py -v --no-container-interactive --enable-logging --enable-stats --enable-tracing \
              --enable-metrics --enable-gpu-metrics --enable-cpu-metrics \
              --filesystem=gcs --filesystem=s3 --filesystem=azure_storage \
              --endpoint=http --endpoint=grpc --endpoint=sagemaker --endpoint=vertex-ai \
              --backend=ensemble --enable-gpu --endpoint=http --endpoint=grpc \
              --no-container-pull \
              --image=base,${TRTLLM_BASE_IMAGE} \
              --backend=tensorrtllm:${TENSORRTLLM_BACKEND_REPO_TAG} \
              --backend=python:${PYTHON_BACKEND_REPO_TAG}
```

`TRTLLM_BASE_IMAGE`是将用于构建容器的基础映像。`TENSORRTLLM_BACKEND_REPO_TAG`和`PYTHON_BACKEND_REPO_TAG`是将用于构建容器的 TensorRT-LLM 后端和 Python 后端存储库的标签。您还可以通过删除相应的标志来删除不需要的功能或端口。

#### 选项 2. 通过 Docker 构建

此构建选项中使用的 Triton Server 版本可以在[Dockerfile](./dockerfile/Dockerfile.trt_llm_backend)中找到。

```bash
# Update the submodules
cd tensorrtllm_backend
git lfs install
git submodule update --init --recursive

# Use the Dockerfile to build the backend in a container
# For x86_64
DOCKER_BUILDKIT=1 docker build -t triton_trt_llm -f dockerfile/Dockerfile.trt_llm_backend .
# For aarch64
DOCKER_BUILDKIT=1 docker build -t triton_trt_llm --build-arg TORCH_INSTALL_TYPE="src_non_cxx11_abi" -f dockerfile/Dockerfile.trt_llm_backend .
```

## 使用 TensorRT-LLM 后端

下面是如何在 4-GPU 环境中使用 Triton TensorRT-LLM 后端来提供 TensorRT-LLM 模型服务的示例。该示例使用[TensorRT-LLM 存储库](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt)中的 GPT 模型。

### 准备 TensorRT-LLM 引擎

如果您已经准备好引擎，则可以跳过此步骤。请遵循TensorRT-LLM 存储库中的[指南](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt)，了解有关如何准备部署引擎的更多详细信息。

```bash
# Update the submodule TensorRT-LLM repository
git submodule update --init --recursive
git lfs install
git lfs pull

# TensorRT-LLM is required for generating engines. You can skip this step if
# you already have the package installed. If you are generating engines within
# the Triton container, you have to install the TRT-LLM package.
(cd tensorrt_llm &&
    bash docker/common/install_cmake.sh &&
    export PATH=/usr/local/cmake/bin:$PATH &&
    python3 ./scripts/build_wheel.py --trt_root="/usr/local/tensorrt" &&
    pip3 install ./build/tensorrt_llm*.whl)

# Go to the tensorrt_llm/examples/gpt directory
cd tensorrt_llm/examples/gpt

# Download weights from HuggingFace Transformers
rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2
pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd

# Convert weights from HF Tranformers to TensorRT-LLM checkpoint
python3 convert_checkpoint.py --model_dir gpt2 \
        --dtype float16 \
        --tp_size 4 \
        --output_dir ./c-model/gpt2/fp16/4-gpu

# Build TensorRT engines
trtllm-build --checkpoint_dir ./c-model/gpt2/fp16/4-gpu \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --paged_kv_cache enable \
        --gemm_plugin float16 \
        --output_dir engines/fp16/4-gpu
```

### 创建模型存储库

[`all_models/inflight_batcher_llm`](./all_models/inflight_batcher_llm/)目录中有五个模型将在本示例中使用：

#### 预处理

该模型用于tokenizing，即从prompts(string)到 input_ids(list of ints)的转换。

#### tensorrt_llm

该模型是 TensorRT-LLM 模型的包装器，用于推理。输入规范可以在[这里](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/inference_request.md)找到

#### 后处理

该模型用于de-tokenizing，即从output_ids(list of ints)）到outputs(string)的转换。

#### 集成

该模型可用于将预处理、tensorrt_llm 和后处理模型链接在一起。

#### tensorrt_llm_bls

该模型也可用于将预处理、tensorrt_llm 和后处理模型链接在一起。

BLS 模型有一个可选参数`accumulate_tokens`，可在流模式下使用该参数来调用具有所有累积令牌（而不是仅一个令牌）的后处理模型。这对于某些标记器可能是必要的。

BLS 模型支持推测解码。 目标和draft Triton 模型使用参数 `tensorrt_llm_model_name` `tensorrt_llm_draft_model_name` 设置。 通过在请求中设置`num_draft_tokens`来执行推测解码。 `use_draft_logits` 可以设置为使用 logits 比较推测解码。 请注意，使用推测解码时不支持`return_generation_logits`和`return_context_logits`。

BLS Inputs

| Name | Shape | Type | Description |
| :------------: | :---------------: | :-----------: | :--------: |
| `text_input` | [ -1 ] | `string` | Prompt text |
| `max_tokens` | [ -1 ] | `int32` | number of tokens to generate |
| `bad_words` | [2, num_bad_words] | `int32` | Bad words list |
| `stop_words` | [2, num_stop_words] | `int32` | Stop words list |
| `end_id` | [1] | `int32` | End token Id. If not specified, defaults to -1 |
| `pad_id` | [1] | `int32` | Pad token Id |
| `temperature` | [1] | `float32` | Sampling Config param: `temperature` |
| `top_k` | [1] | `int32` | Sampling Config param: `topK` |
| `top_p` | [1] | `float32` | Sampling Config param: `topP` |
| `len_penalty` | [1] | `float32` | Sampling Config param: `lengthPenalty` |
| `repetition_penalty` | [1] | `float` | Sampling Config param: `repetitionPenalty` |
| `min_length` | [1] | `int32_t` | Sampling Config param: `minLength` |
| `presence_penalty` | [1] | `float` | Sampling Config param: `presencePenalty` |
| `frequency_penalty` | [1] | `float` | Sampling Config param: `frequencyPenalty` |
| `random_seed` | [1] | `uint64_t` | Sampling Config param: `randomSeed` |
| `return_log_probs` | [1] | `bool` | When `true`, include log probs in the output |
| `return_context_logits` | [1] | `bool` | When `true`, include context logits in the output |
| `return_generation_logits` | [1] | `bool` | When `true`, include generation logits in the output |
| `beam_width` | [1] | `int32_t` | (Default=1) Beam width for this request; set to 1 for greedy sampling |
| `stream` | [1] | `bool` | (Default=`false`). When `true`, stream out tokens as they are generated. When `false` return only when the full generation has completed.  |
| `prompt_embedding_table` | [1] | `float16` (model data type) | P-tuning prompt embedding table |
| `prompt_vocab_size` | [1] | `int32` | P-tuning prompt vocab size |
| `lora_task_id` | [1] | `uint64` | Task ID for the given lora_weights.  This ID is expected to be globally unique.  To perform inference with a specific LoRA for the first time `lora_task_id` `lora_weights` and `lora_config` must all be given.  The LoRA will be cached, so that subsequent requests for the same task only require `lora_task_id`. If the cache is full the oldest LoRA will be evicted to make space for new ones.  An error is returned if `lora_task_id` is not cached |
| `lora_weights` | [ num_lora_modules_layers, D x Hi + Ho x D ] | `float` (model data type) | weights for a lora adapter. see [lora docs](lora.md#lora-tensor-format-details) for more details. |
| `lora_config` | [ num_lora_modules_layers, 3] | `int32t` | lora configuration tensor. `[ module_id, layer_idx, adapter_size (D aka R value) ]` see [lora docs](lora.md#lora-tensor-format-details) for more details. |
| `embedding_bias_words` | [-1] | `string` | Embedding bias words |
| `embedding_bias_weights` | [-1] | `float32` | Embedding bias weights |
| `num_draft_tokens` | [1] | int32 | number of tokens to get from draft model during speculative decoding |
| `use_draft_logits` | [1] | `bool` | use logit comparison during speculative decoding |

BLS Outputs

| Name | Shape | Type | Description |
| :------------: | :---------------: | :-----------: | :--------: |
| `text_output` | [-1] | `string` | text output |
| `cum_log_probs` | [-1] | `float` | cumulative probabilities for each output |
| `output_log_probs` | [beam_width, -1] | `float` | log probabilities for each output |
| `context_logits` | [-1, vocab_size] | `float` |  context logits for input |
| `generation_logtis` | [beam_width, seq_len, vocab_size] | `float` | generatiion logits for each output |

要了解有关集成和 BLS 模型的更多信息，请参阅 Triton 推理服务器文档的[集成模型](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models)和[Business Logic Scripting](https://github.com/triton-inference-server/python_backend#business-logic-scripting)部分。

```bash
# Create the model repository that will be used by the Triton server
cd tensorrtllm_backend
mkdir triton_model_repo

# Copy the example models to the model repository
cp -r all_models/inflight_batcher_llm/* triton_model_repo/

# Copy the TRT engine to triton_model_repo/tensorrt_llm/1/
cp tensorrt_llm/examples/gpt/engines/fp16/4-gpu/* triton_model_repo/tensorrt_llm/1
```

### 修改模型配置
下表显示了部署前可能需要修改的字段：

*triton_model_repo/preprocessing/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `tokenizer_dir` | The path to the tokenizer for the model. In this example, the path should be set to `/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2` as the tensorrtllm_backend directory will be mounted to `/tensorrtllm_backend` within the container |

*triton_model_repo/tensorrt_llm/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `gpt_model_type` | Mandatory. Set to `inflight_fused_batching` when enabling in-flight batching support. To disable in-flight batching, set to `V1` |
| `gpt_model_path` | Mandatory. Path to the TensorRT-LLM engines for deployment. In this example, the path should be set to `/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1` as the tensorrtllm_backend directory will be mounted to `/tensorrtllm_backend` within the container |
| `batch_scheduler_policy` | Mandatory. Set to `max_utilization` to greedily pack as many requests as possible in each current in-flight batching iteration. This maximizes the throughput but may result in overheads due to request pause/resume if KV cache limits are reached during execution. Set to `guaranteed_no_evict` to guarantee that a started request is never paused.|
| `decoupled` | Optional (default=`false`). Controls streaming. Decoupled mode must be set to `true` if using the streaming option from the client. |
| `max_beam_width` | Optional (default=1). The maximum beam width that any request may ask for when using beam search.|
| `max_tokens_in_paged_kv_cache` | Optional (default=unspecified). The maximum size of the KV cache in number of tokens. If unspecified, value is interpreted as 'infinite'. KV cache allocation is the min of max_tokens_in_paged_kv_cache and value derived from kv_cache_free_gpu_mem_fraction below. |
| `max_attention_window_size` | Optional (default=max_sequence_length). When using techniques like sliding window attention, the maximum number of tokens that are attended to generate one token. Defaults attends to all tokens in sequence. |
| `kv_cache_free_gpu_mem_fraction` | Optional (default=0.9). Set to a number between 0 and 1 to indicate the maximum fraction of GPU memory (after loading the model) that may be used for KV cache.|
| `enable_trt_overlap` | Optional (default=`false`). Set to `true` to partition available requests into 2 'microbatches' that can be run concurrently to hide exposed CPU runtime |
| `exclude_input_in_output` | Optional (default=`false`). Set to `true` to only return completion tokens in a response. Set to `false` to return the prompt tokens concatenated with the generated tokens  |
| `normalize_log_probs` | Optional (default=`true`). Set to `false` to skip normalization of `output_log_probs`  |
| `enable_chunked_context` | Optional (default=`false`). Set to `true` to enable context chunking. |
| `gpu_device_ids` | Optional (default=unspecified). Comma-separated list of GPU IDs to use for this model. If not provided, the model will use all visible GPUs. |
| `decoding_mode` | Optional. Set to one of the following: `{top_k, top_p, top_k_top_p, beam_search}` to select the decoding mode. The `top_k` mode exclusively uses Top-K algorithm for sampling, The `top_p` mode uses exclusively Top-P algorithm for sampling. The top_k_top_p mode employs both Top-K and Top-P algorithms, depending on the runtime sampling params of the request. Note that the `top_k_top_p option` requires more memory and has a longer runtime than using `top_k` or `top_p` individually; therefore, it should be used only when necessary. `beam_search` uses beam search algorithm. If not specified, the default is to use `top_k_top_p` if `max_beam_width == 1`; otherwise, `beam_search` is used. |
| `medusa_choices` | Optional. To specify Medusa choices tree in the format of e.g. "{0, 0, 0}, {0, 1}". By default, mc_sim_7b_63 choices are used. |

*triton_model_repo/postprocessing/config.pbtxt*

| Name | Description
| :----------------------: | :-----------------------------: |
| `tokenizer_dir` | The path to the tokenizer for the model. In this example, the path should be set to `/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2` as the tensorrtllm_backend directory will be mounted to `/tensorrtllm_backend` within the container |
| `tokenizer_type` | The type of the tokenizer for the model, `t5`, `auto` and `llama` are supported. In this example, the type should be set to `auto` |

### 启动 Triton 服务器

请遵循与您构建 TensorRT-LLM 后端的方式相对应的选项。

#### 选项  1. *在 Triton NGC 容器内*启动 Triton 服务器

```bash
docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v /path/to/tensorrtllm_backend:/tensorrtllm_backend nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3 bash
```

#### 选项 2. *在通过 build.py 脚本构建的 Triton 容器中*启动 Triton 服务器

```bash
docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v /path/to/tensorrtllm_backend:/tensorrtllm_backend tritonserver bash
```

#### 选项 3. *在通过 Docker 构建的 Triton 容器内*启动 Triton 服务器

```bash
docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v /path/to/tensorrtllm_backend:/tensorrtllm_backend triton_trt_llm bash
```

进入容器后，您可以使用以下命令启动 Triton 服务器：

```bash
cd /tensorrtllm_backend
# --world_size is the number of GPUs you want to use for serving
python3 scripts/launch_triton_server.py --world_size=4 --model_repo=/tensorrtllm_backend/triton_model_repo
```

要使用多个 TensorRT-LLM 模型，请使用`--multi-model`选项。`--world_size`必须为 1，因为 TensorRT-LLM 后端将根据需要动态启动 TensorRT-LLM 工作线程。

```bash
cd /tensorrtllm_backend
python3 scripts/launch_triton_server.py --model_repo=/tensorrtllm_backend/triton_model_repo --multi-model
```

使用`--multi-model`选项时，Triton 模型存储库可以包含多个 TensorRT-LLM 模型。当运行多个TensorRT-LLM模型时，`gpu_device_ids`参数应在模型配置文件`config.pbtxt`中指定。您需要确保分配的 GPU ID 之间不存在重叠。

部署成功后，服务器会生成类似以下的日志。
```
I0919 14:52:10.475738 293 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0919 14:52:10.475968 293 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0919 14:52:10.517138 293 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

### 使用 Triton generate端口查询服务器

从 Triton 23.10 版本开始，您可以使用 Triton 的[generate 端口](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md)以及基于客户端环境/容器内的以下通用格式的curl 命令来查询服务器：

```bash
curl -X POST localhost:8000/v2/models/${MODEL_NAME}/generate -d '{"{PARAM1_KEY}": "{PARAM1_VALUE}", ... }'
```

对于本示例中使用的模型，您可以将 MODEL_NAME 替换为`ensemble`或`tensorrt_llm_bls`。检查`ensemble`模型和`tensorrt_llm_bls`模型的 config.pbtxt 文件，您可以看到需要 4 个参数才能生成该模型的响应：

- "text_input": 输入文本以生成响应
- "max_tokens": 请求输出令牌的数量
- "bad_words": 坏词列表（可以为空）
- "stop_words": 停用词列表（可以为空）

因此，我们可以通过以下方式查询服务器：

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}'
```
如果使用`ensemble`模型
```
curl -X POST localhost:8000/v2/models/tensorrt_llm_bls/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}'
```
如果使用`tensorrt_llm_bls`模型

它应该返回类似于如下（为了便于阅读而格式化）的结果：
```json
{
  "model_name": "ensemble",
  "model_version": "1",
  "sequence_end": false,
  "sequence_id": 0,
  "sequence_start": false,
  "text_output": "What is machine learning?\n\nMachine learning is a method of learning by using machine learning algorithms to solve problems.\n\n"
}
```

### 利用提供的客户端脚本发送请求

您可以使用提供的[python 客户端脚本](./inflight_batcher_llm/client/inflight_batcher_llm_client.py)向 "tensorrt_llm" 模型发送请求， 如下所示：

```bash
python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py --request-output-len 200 --tokenizer-dir /workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2
```

结果应类似于以下内容：

```
Got completed request
output_ids =  [[28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257, 21221, 878, 3867, 284, 3576, 287, 262, 1903, 6303, 82, 13, 679, 468, 1201, 3111, 287, 10808, 287, 3576, 11, 6342, 11, 21574, 290, 968, 1971, 13, 198, 198, 1544, 318, 6405, 284, 262, 1966, 2746, 290, 14549, 11, 11735, 12, 44507, 11, 290, 468, 734, 1751, 11, 257, 4957, 11, 18966, 11, 290, 257, 3367, 11, 7806, 13, 198, 198, 50, 726, 263, 338, 3656, 11, 11735, 12, 44507, 11, 318, 257, 1966, 2746, 290, 14549, 13, 198, 198, 1544, 318, 11803, 416, 465, 3656, 11, 11735, 12, 44507, 11, 290, 511, 734, 1751, 11, 7806, 290, 18966, 13, 198, 198, 50, 726, 263, 373, 4642, 287, 6342, 11, 4881, 11, 284, 257, 4141, 2988, 290, 257, 2679, 2802, 13, 198, 198, 1544, 373, 15657, 379, 262, 23566, 38719, 293, 748, 1355, 14644, 12, 3163, 912, 287, 6342, 290, 262, 15423, 4189, 710, 287, 6342, 13, 198, 198, 1544, 373, 257, 2888, 286, 262, 4141, 8581, 286, 13473, 290, 262, 4141, 8581, 286, 11536, 13, 198, 198, 1544, 373, 257, 2888, 286, 262, 4141, 8581, 286, 13473, 290, 262, 4141, 8581, 286, 11536, 13, 198, 198, 50, 726, 263, 373, 257, 2888, 286, 262, 4141, 8581, 286, 13473, 290]]
Input: Born in north-east France, Soyer trained as a
Output:  chef before moving to London in the early 1990s. He has since worked in restaurants in London, Paris, Milan and New York.

He is married to the former model and actress, Anna-Marie, and has two children, a daughter, Emma, and a son, Daniel.

Soyer's wife, Anna-Marie, is a former model and actress.

He is survived by his wife, Anna-Marie, and their two children, Daniel and Emma.

Soyer was born in Paris, France, to a French father and a German mother.

He was educated at the prestigious Ecole des Beaux-Arts in Paris and the Sorbonne in Paris.

He was a member of the French Academy of Sciences and the French Academy of Arts.

He was a member of the French Academy of Sciences and the French Academy of Arts.

Soyer was a member of the French Academy of Sciences and
```

#### 提前停止
您还可以使用`--stop-after-ms`在几毫秒后发送停止请求的选项来提前停止生成过程：

```bash
python inflight_batcher_llm/client/inflight_batcher_llm_client.py --stop-after-ms 200 --request-output-len 200 --tokenizer-dir /workspace/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2
```

你会发现生成过程提前停止了，因此生成的token数量低于200个。你可以看一下客户端代码，看看提前停止是如何实现的。

#### 返回上下文 logits 和/或生成 logits
如果您想获取上下文 logits 和/或生成 logits，则需要在构建引擎时启用`--gather_context_logits`和/或`--gather_generation_logits`（或`--gather_all_token_logits`来同时启用两者）。有关这两个标志的更多设置详细信息，请参阅[build.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/commands/build.py)或[gpt_runtime](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/gpt_runtime.md)。

启动服务器后，您可以通过传递相应的参数`--return-context-logits`和/或`--return-generation-logits`在客户端脚本（[end_to_end_grpc_client.py](./inflight_batcher_llm/client/end_to_end_grpc_client.py)和[inflight_batcher_llm_client.py](./inflight_batcher_llm/client/inflight_batcher_llm_client.py)）中获取 logits 的输出。例如：
```bash
python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py --request-output-len 20 --tokenizer-dir /path/to/tokenizer/ \
--return-context-logits \
--return-generation-logits
```

结果应类似于以下内容：
```
Input sequence:  [28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257]
Got completed request
Input: Born in north-east France, Soyer trained as a
Output beam 0:  has since worked in restaurants in London,
Output sequence:  [21221, 878, 3867, 284, 3576, 287, 262, 1903, 6303, 82, 13, 679, 468, 1201, 3111, 287, 10808, 287, 3576, 11]
context_logits.shape: (1, 12, 50257)
context_logits: [[[ -65.9822     -62.267445   -70.08991   ...  -76.16964    -78.8893
    -65.90678  ]
  [-103.40278   -102.55243   -106.119026  ... -108.925415  -109.408585
   -101.37687  ]
  [ -63.971176   -64.03466    -67.58809   ...  -72.141235   -71.16892
    -64.23846  ]
  ...
  [ -80.776375   -79.1815     -85.50916   ...  -87.07368    -88.02817
    -79.28435  ]
  [ -10.551408    -7.786484   -14.524468  ...  -13.805856   -15.767286
     -7.9322424]
  [-106.33096   -105.58956   -111.44852   ... -111.04858   -111.994194
   -105.40376  ]]]
generation_logits.shape: (1, 1, 20, 50257)
generation_logits: [[[[-106.33096  -105.58956  -111.44852  ... -111.04858  -111.994194
    -105.40376 ]
   [ -77.867424  -76.96638   -83.119095 ...  -87.82542   -88.53957
     -75.64877 ]
   [-136.92282  -135.02484  -140.96051  ... -141.78284  -141.55045
    -136.01668 ]
   ...
   [-100.03721   -98.98237  -105.25507  ... -108.49254  -109.45882
     -98.95136 ]
   [-136.78777  -136.16165  -139.13437  ... -142.21495  -143.57468
    -134.94667 ]
   [  19.222942   19.127287   14.804495 ...   10.556551    9.685863
      19.625107]]]]
```


### *在基于 Slurm 的集群中*启动 Triton 服务器

#### 准备一些脚本

`tensorrt_llm_triton.sub`
```bash
#!/bin/bash
#SBATCH -o logs/tensorrt_llm.out
#SBATCH -e logs/tensorrt_llm.error
#SBATCH -J <REPLACE WITH YOUR JOB's NAME>
#SBATCH -A <REPLACE WITH YOUR ACCOUNT's NAME>
#SBATCH -p <REPLACE WITH YOUR PARTITION's NAME>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00

sudo nvidia-smi -lgc 1410,1410

srun --mpi=pmix \
    --container-image triton_trt_llm \
    --container-mounts /path/to/tensorrtllm_backend:/tensorrtllm_backend \
    --container-workdir /tensorrtllm_backend \
    --output logs/tensorrt_llm_%t.out \
    bash /tensorrtllm_backend/tensorrt_llm_triton.sh
```

`tensorrt_llm_triton.sh`
```bash
TRITONSERVER="/opt/tritonserver/bin/tritonserver"
MODEL_REPO="/tensorrtllm_backend/triton_model_repo"

${TRITONSERVER} --model-repository=${MODEL_REPO} --disable-auto-complete-config --backend-config=python,shm-region-prefix-name=prefix${SLURM_PROCID}_
```

#### 提交 Slurm 作业

```bash
sbatch tensorrt_llm_triton.sub
```

您可能需要联系集群管理员来帮助您自定义上述脚本。

### 杀死 Triton 服务器

```bash
pkill tritonserver
```

## Triton 指标
从 Triton 23.11 版本开始，用户现在可以通过查询 Triton 指标端口来获取 TRT LLM Batch Manager[统计信息](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/batch_manager.md#statistics)。这可以通过以上述任何方式启动 Triton 服务器（确保构建代码/容器为 23.11 或更高版本）并查询服务器来完成。收到成功响应后，您可以通过输入以下内容来查询指标端口：
```bash
curl localhost:8002/metrics
```
批量管理器统计信息由指标端口在前缀为`nv_trt_llm_`的字段中报告。这些字段的输出应类似于以下内容（假设您的模型是动态批处理模型）：
```bash
# HELP nv_trt_llm_request_metrics TRT LLM request metrics
# TYPE nv_trt_llm_request_metrics gauge
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="context",version="1"} 1
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="scheduled",version="1"} 1
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="max",version="1"} 512
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="active",version="1"} 0
# HELP nv_trt_llm_runtime_memory_metrics TRT LLM runtime memory metrics
# TYPE nv_trt_llm_runtime_memory_metrics gauge
nv_trt_llm_runtime_memory_metrics{memory_type="pinned",model="tensorrt_llm",version="1"} 0
nv_trt_llm_runtime_memory_metrics{memory_type="gpu",model="tensorrt_llm",version="1"} 1610236
nv_trt_llm_runtime_memory_metrics{memory_type="cpu",model="tensorrt_llm",version="1"} 0
# HELP nv_trt_llm_kv_cache_block_metrics TRT LLM KV cache block metrics
# TYPE nv_trt_llm_kv_cache_block_metrics gauge
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="tokens_per",model="tensorrt_llm",version="1"} 64
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="used",model="tensorrt_llm",version="1"} 1
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="free",model="tensorrt_llm",version="1"} 6239
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="max",model="tensorrt_llm",version="1"} 6239
# HELP nv_trt_llm_inflight_batcher_metrics TRT LLM inflight_batcher-specific metrics
# TYPE nv_trt_llm_inflight_batcher_metrics gauge
nv_trt_llm_inflight_batcher_metrics{inflight_batcher_specific_metric="micro_batch_id",model="tensorrt_llm",version="1"} 0
nv_trt_llm_inflight_batcher_metrics{inflight_batcher_specific_metric="generation_requests",model="tensorrt_llm",version="1"} 0
nv_trt_llm_inflight_batcher_metrics{inflight_batcher_specific_metric="total_context_tokens",model="tensorrt_llm",version="1"} 0
# HELP nv_trt_llm_general_metrics General TRT LLM metrics
# TYPE nv_trt_llm_general_metrics gauge
nv_trt_llm_general_metrics{general_type="iteration_counter",model="tensorrt_llm",version="1"} 0
nv_trt_llm_general_metrics{general_type="timestamp",model="tensorrt_llm",version="1"} 1700074049
```
相反，如果您启动了 V1 模型，您的输出将与上面的输出类似，只是与运行中批处理程序相关的字段将替换为类似于以下内容的内容：
```bash
# HELP nv_trt_llm_v1_metrics TRT LLM v1-specific metrics
# TYPE nv_trt_llm_v1_metrics gauge
nv_trt_llm_v1_metrics{model="tensorrt_llm",v1_specific_metric="total_generation_tokens",version="1"} 20
nv_trt_llm_v1_metrics{model="tensorrt_llm",v1_specific_metric="empty_generation_slots",version="1"} 0
nv_trt_llm_v1_metrics{model="tensorrt_llm",v1_specific_metric="total_context_tokens",version="1"} 5
```
请注意，23.12 版本之前的 Triton 版本不支持基本 Triton 指标。因此，以下字段将报告 0：
```bash
# HELP nv_inference_request_success Number of successful inference requests, all batch sizes
# TYPE nv_inference_request_success counter
nv_inference_request_success{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_request_failure Number of failed inference requests, all batch sizes
# TYPE nv_inference_request_failure counter
nv_inference_request_failure{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_count Number of inferences performed (does not include cached requests)
# TYPE nv_inference_count counter
nv_inference_count{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_exec_count Number of model executions performed (does not include cached requests)
# TYPE nv_inference_exec_count counter
nv_inference_exec_count{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_request_duration_us Cumulative inference request duration in microseconds (includes cached requests)
# TYPE nv_inference_request_duration_us counter
nv_inference_request_duration_us{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_queue_duration_us Cumulative inference queuing duration in microseconds (includes cached requests)
# TYPE nv_inference_queue_duration_us counter
nv_inference_queue_duration_us{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_compute_input_duration_us Cumulative compute input duration in microseconds (does not include cached requests)
# TYPE nv_inference_compute_input_duration_us counter
nv_inference_compute_input_duration_us{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_compute_infer_duration_us Cumulative compute inference duration in microseconds (does not include cached requests)
# TYPE nv_inference_compute_infer_duration_us counter
nv_inference_compute_infer_duration_us{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_compute_output_duration_us Cumulative inference compute output duration in microseconds (does not include cached requests)
# TYPE nv_inference_compute_output_duration_us counter
nv_inference_compute_output_duration_us{model="tensorrt_llm",version="1"} 0
# HELP nv_inference_pending_request_count Instantaneous number of pending requests awaiting execution per-model.
# TYPE nv_inference_pending_request_count gauge
nv_inference_pending_request_count{model="tensorrt_llm",version="1"} 0
```

## 测试 TensorRT-LLM 后端
请按照指南[`ci/README.md`](ci/README.md)了解如何运行 TensorRT-LLM 后端的测试。
