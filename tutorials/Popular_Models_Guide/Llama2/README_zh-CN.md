# 在 Triton 中部署 Hugging Face Transformer 系列模型

有多种方法可以使用 Tritonserver 运行 Llama2。
1. 使用[TensorRT-LLM Backend](trtllm_guide.md#infer-with-tensorrt-llm-backend)推理
2. 使用[vLLM Backend](vllm_guide.md#infer-with-vllm-backend)推理
3. 使用[Python-based Backends as a HuggingFace model](../../Quick_Deploy/HuggingFaceTransformers/README.md#deploying-hugging-face-transformer-models-in-triton)推理

## 预构建说明

在本教程中，我们假设 Llama2 模型、权重和令牌是从[此处](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main)的 Huggingface Llama2 存储库克隆而来的。 
要运行这些教程，您需要获得 Llama2 存储库的权限以及对 huggingface cli 的访问权限。 
cli 使用[用户访问令牌](https://huggingface.co/docs/hub/security-tokens)。令牌可以在这里找到：[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)。
