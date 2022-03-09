# 指标

Triton 提供了指表示GPU 和请求统计信息的[Prometheus](https://prometheus.io/) 指标。默认情况下，这些指标可在http://localhost:8002/metrics获得。这些指标只能通过访问端点获得，不会推送或发布到任何远程服务器。度量格式是纯文本，因此您可以直接查看它们，例如：

```
$ curl localhost:8002/metrics
```

tritonserver --allow-metrics=false 选项可用于禁用所有指标报告，而 --allow-gpu-metrics=false 可用于仅禁用 GPU 利用率和 GPU 内存指标。--metrics-port 选项可用于选择不同的端口。

下表描述了可用的指标。

|Category      |Metric          |Description                            |Granularity|Frequency    |
|--------------|----------------|---------------------------------------|-----------|-------------|
|GPU Utilization |Power Usage   |GPU instantaneous power                |Per GPU    |Per second   |
|              |Power Limit     |Maximum GPU power limit                |Per GPU    |Per second   |
|              |Energy Consumption|GPU energy consumption in joules since Triton started|Per GPU|Per second|
|              |GPU Utilization |GPU utilization rate (0.0 - 1.0)       |Per GPU    |Per second   |
|GPU Memory    |GPU Total Memory|Total GPU memory, in bytes             |Per GPU    |Per second   |
|              |GPU Used Memory |Used GPU memory, in bytes              |Per GPU    |Per second   |
|Count         |Request Count   |Number of inference requests received by Triton (each request is counted as 1, even if the request contains a batch) |Per model  |Per request  |
|              |Inference Count |Number of inferences performed (a batch of "n" is counted as "n" inferences)|Per model|Per request|
|              |Execution Count |Number of inference batch executions (see [Count Metrics](#count-metrics))|Per model|Per request|
|Latency       |Request Time    |Cumulative end-to-end inference request handling time    |Per model  |Per request  |
|              |Queue Time      |Cumulative time requests spend waiting in the scheduling queue     |Per model  |Per request  |
|              |Compute Input Time|Cumulative time requests spend processing inference inputs (in the framework backend)     |Per model  |Per request  |
|              |Compute Time    |Cumulative time requests spend executing the inference model (in the framework backend)     |Per model  |Per request  |
|              |Compute Output Time|Cumulative time requests spend processing inference outputs (in the framework backend)     |Per model  |Per request  |

## 计数指标

对于不支持批处理的模型，*Request Count*, *Inference Count* 和 *Execution Count* 将相等，表示每个推理请求是单独执行的。

For models that support batching, the count metrics can be interpreted
to determine average batch size as *Inference Count* / *Execution
Count*. The count metrics are illustrated by the following examples:
对于支持批处理的模型，计数指标可以解释为将平均批处理大小确定为*Inference Count* / *Execution Count*。以下示例说明了计数指标：

* 客户端发送一个batch-1推理请求。 *Request Count* =1, *Inference Count* = 1, *Execution Count* = 1.

* 客户端发送一个batch-8推理请求。*Request Count* =1, *Inference Count* = 8, *Execution Count* = 1.

* 客户端发送两个请求: batch-1和batch-8。模型未启用动态批处理器。
*Request Count* = 2, *Inference Count* = 9,*Execution Count* = 2.

* 客户端发送两个请求: batch-1和batch-1。模型启用动态批处理器 ，两个请求被服务器动态批处理。 *Request Count* = 2, *Inference Count* = 2, *Execution Count* = 1.

* 客户端发送两个请求: batch-1和batch-8。模型启用动态批处理器 ，两个请求被服务器动态批处理。 *Request Count* = 2, *Inference Count* = 9, *Execution Count* = 1.

