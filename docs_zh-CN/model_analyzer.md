# 模型分析器

Triton 模型分析器是一种工具，它使用[性能分析器](perf_analyzer.md)向您的模型发送请求，同时测量 GPU 内存和计算利用率。模型分析器特别适用于描述不同批处理和模型实例配置下模型的 GPU 内存要求。获得此 GPU 内存使用信息后，您可以更明智地决定如何在同一 GPU 上组合多个模型，同时保持在 GPU 的内存容量范围内。

有关更多信息，请参阅[模型分析器存储库](https://github.com/triton-inference-server/model_analyzer)以及[使用 NVIDIA 模型分析器最大化深度学习推理性能](https://developer.nvidia.com/blog/maximizing-deep-learning-inference-performance-with-nvidia-model-analyzer)中提供的详细说明。