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
|Count         |Success Count   |Number of successful inference requests received by Triton (each request is counted as 1, even if the request contains a batch) |Per model  |Per request  |
|              |Failure Count   |Number of failed inference requests received by Triton (each request is counted as 1, even if the request contains a batch) |Per model  |Per request  |
|              |Inference Count |Number of inferences performed (a batch of "n" is counted as "n" inferences, does not include cached requests)|Per model|Per request|
|              |Execution Count |Number of inference batch executions (see [Count Metrics](#count-metrics), does not include cached requests)|Per model|Per request|
|Latency       |Request Time    |Cumulative end-to-end inference request handling time (includes cached requests) |Per model  |Per request  |
|              |Queue Time      |Cumulative time requests spend waiting in the scheduling queue (includes cached requests) |Per model  |Per request  |
|              |Compute Input Time|Cumulative time requests spend processing inference inputs (in the framework backend, does not include cached requests)     |Per model  |Per request  |
|              |Compute Time    |Cumulative time requests spend executing the inference model (in the framework backend, does not include cached requests)     |Per model  |Per request  |
|              |Compute Output Time|Cumulative time requests spend processing inference outputs (in the framework backend, does not include cached requests)     |Per model  |Per request  |
|Response Cache|Total Cache Entry Count |Total number of responses stored in response cache across all models |Server-wide |Per second |
|              |Total Cache Lookup Count |Total number of response cache lookups done by Triton across all models |Server-wide |Per second |
|              |Total Cache Hit Count |Total number of response cache hits across all models |Server-wide |Per second |
|              |Total Cache Miss Count |Total number of response cache misses across all models |Server-wide |Per second |
|              |Total Cache Eviction Count |Total number of response cache evictions across all models |Server-wide |Per second |
|              |Total Cache Lookup Time |Cumulative time requests spend checking for a cached response across all models (microseconds) |Server-wide |Per second |
|              |Total Cache Utilization |Total Response Cache utilization rate (0.0 - 1.0) |Server-wide |Per second |
|              |Cache Hit Count |Number of response cache hits per model |Per model |Per request |
|              |Cache Hit Lookup Time |Cumulative time requests spend retrieving a cached response per model on cache hits (microseconds) |Per model |Per request |
|              |Cache Miss Count |Number of response cache misses per model |Per model |Per request |
|              |Cache Miss Lookup Time |Cumulative time requests spend looking up a request hash on a cache miss (microseconds) |Per model |Per request |
|              |Cache Miss Insertion Time |Cumulative time requests spend inserting responses into the cache on a cache miss (microseconds) |Per model |Per request |

## 响应缓存

上表中的计算延迟指标是针对模型推理后端所花费的时间计算的。如果为给定的模型启动响应缓存（查看[响应缓存](https://github.com/triton-inference-server/server/blob/main/docs/response_cache.md)文档获取更多的信息），总推理时间可能会收到响应缓存查找时间的影响。

在缓存命中时，"Cache Hit Lookup Time"表示查找响应所花费的时间，"Compute Input Time" /  "Compute Time" / "Compute Output Time"不记录。

在缓存未命中时，"Cache Miss Lookup Time"表示查找请求哈希所花费的时间，"Cache Miss Insertion Time"表示将计算的输出张量数据插入到缓存所花费的时间。另一方面，"Compute Input Time" /  "Compute Time" / "Compute Output Time"将照常记录。
## 计数指标

对于不支持批处理的模型，*Request Count*, *Inference Count* 和 *Execution Count* 将相等，表示每个推理请求是单独执行的。

对于支持批处理的模型，计数指标可以解释为将平均批处理大小确定为*Inference Count* / *Execution Count*。以下示例说明了计数指标：

* 客户端发送一个batch-1推理请求。 *Request Count* =1, *Inference Count* = 1, *Execution Count* = 1.

* 客户端发送一个batch-8推理请求。*Request Count* =1, *Inference Count* = 8, *Execution Count* = 1.

* 客户端发送两个请求: batch-1和batch-8。模型未启用动态批处理器。
*Request Count* = 2, *Inference Count* = 9,*Execution Count* = 2.

* 客户端发送两个请求: batch-1和batch-1。模型启用动态批处理器 ，两个请求被服务器动态批处理。 *Request Count* = 2, *Inference Count* = 2, *Execution Count* = 1.

* 客户端发送两个请求: batch-1和batch-8。模型启用动态批处理器 ，两个请求被服务器动态批处理。 *Request Count* = 2, *Inference Count* = 9, *Execution Count* = 1.

