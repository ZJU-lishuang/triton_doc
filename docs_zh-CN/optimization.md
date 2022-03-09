# 优化

Triton 推理服务器具有许多功能，可用于减少延迟并增加模型的吞吐量。本节讨论这些功能并演示如何使用它们来提高模型的性能。作为先决条件，您应该遵循[快速入门](quickstart.md)以使 Triton 和客户端示例与示例模型存储库一起运行。

本节重点了解单个模型的延迟和吞吐量权衡。[模型分析器](model_analyzer.md)部分介绍了一种工具，可帮助您了解模型的GPU 内存利用率，以便您决定如何在单个 GPU 上最好地运行多个模型。

除非您已经拥有适合在 Triton 上测量模型性能的客户端应用程序，否则您应该熟悉[性能分析器](perf_analyzer.md)。性能分析器是优化模型性能的重要工具。

作为演示优化功能和选项的运行示例，我们将使用 TensorFlow Inception 模型，您可以通过遵循[快速入门](quickstart.md)获得该模型。作为基线，我们使用 perf_analyzer 来确定模型的性能，该模型使用 [不启用任何性能特征的基本模型配置](examples/model_repository/inception_graphdef/config.pbtxt)。


```
$ perf_analyzer -m inception_graphdef --percentile=95 --concurrency-range 1:4
...
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 62.6 infer/sec, latency 21371 usec
Concurrency: 2, throughput: 73.2 infer/sec, latency 34381 usec
Concurrency: 3, throughput: 73.2 infer/sec, latency 50298 usec
Concurrency: 4, throughput: 73.4 infer/sec, latency 65569 usec
```

结果表明，我们的非优化模型配置提供了大约每秒 73 次推理的吞吐量。请注意，从一个并发请求到两个并发请求吞吐量显著增加，之后吞吐量趋于平稳。对于一个并发请求，Triton 在响应返回到客户端并且在服务器接收到下一个请求期间处于空闲状态。由于Triton 将一个请求的处理与另一个请求的通信重叠，吞吐量随着两个请求的并发性而增加。因为我们在与 Triton 相同的系统上运行 perf_analyzer，所以两个请求足以完全隐藏通信延迟。

## 优化设置

对于大多数模型，提供最大性能提升的 Triton 功能是[动态批处理](architecture.md#dynamic-batcher)。如果您的模型不支持批处理，那么您可以跳到[模型实例](#model-instances)。

### 动态批处理器

动态批处理器将单个推理请求组合成一个更大的批处理，该批处理通常比独立执行单个请求更有效。要启用动态批处理器，停止 Triton，将以下行添加到[inception_graphdef 的模型配置文件](examples/model_repository/inception_graphdef/config.pbtxt)的末尾，然后重新启动 Triton。

```
dynamic_batching { }
```

动态批处理器允许 Triton 处理更多的并发请求，因为这些请求被组合起来进行推理。为了启动该功能，请求并发数从 1 到 8去运行 perf_analyzer。

```
$ perf_analyzer -m inception_graphdef --percentile=95 --concurrency-range 1:8
...
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 66.8 infer/sec, latency 19785 usec
Concurrency: 2, throughput: 80.8 infer/sec, latency 30732 usec
Concurrency: 3, throughput: 118 infer/sec, latency 32968 usec
Concurrency: 4, throughput: 165.2 infer/sec, latency 32974 usec
Concurrency: 5, throughput: 194.4 infer/sec, latency 33035 usec
Concurrency: 6, throughput: 217.6 infer/sec, latency 34258 usec
Concurrency: 7, throughput: 249.8 infer/sec, latency 34522 usec
Concurrency: 8, throughput: 272 infer/sec, latency 35988 usec
```

与不使用动态批处理器相比，通过八个并发请求，动态批处理器允许 Triton 每秒提供 272 次推理，而不会增加延迟。

您还可以明确指定您希望动态批处理器在创建批处理时首选的批处理大小。例如，要表明您希望动态批处理器首选大小为 4 的批次，您可以像这样修改模型配置（可以给出多个首选大小，但在这种情况下我们只有一个）。

```
dynamic_batching { preferred_batch_size: [ 4 ]}
```

除了让 perf_analyzer 收集一系列请求并发值的数据外，我们还可以使用一些简单的规则，这些规则通常适用于 perf_analyzer 与 Triton 在同一系统上运行时。第一条规则是为了最小延迟，将请求并发设置为 1 并禁用动态批处理器并仅使用 1 个[模型实例](#model-instances)。第二条规则是为了获得最大吞吐量，请将请求并发设置为`2 * <preferred batch size> * <model instance count>`。我们将在[下面](#model-instances)讨论模型实例 ，现在我们正在使用一个模型实例。因此，对于首选批量大小 4，我们希望运行 perf_analyzer，请求并发为`2 * 4 * 1 = 8`。

```
$ perf_analyzer -m inception_graphdef --percentile=95 --concurrency-range 8
...
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 8, throughput: 267.8 infer/sec, latency 35590 usec
```

### 模型实例

Triton 允许您指定每个模型的多少副本可用于推理。默认情况下，您会获得每个模型的一份副本，但您可以通过[实例组](model_configuration.md#instance-groups)在模型配置中指定任意数量的实例。通常，一个模型拥有两个模型实例将提高性能，因为它允许内存传输操作（例如，CPU to/from GPU）与推理计算重叠。多个实例还允许在 GPU 上同时执行更多推理工作，从而提高 GPU 利用率。较小的模型可能会受益于两个以上的实例；您可以使用 perf_analyzer 进行实验。

要设置 inception_graphdef 模型有两个实例：停止 Triton，删除您之前可能添加到模型配置中的任何动态批处理设置（我们在下面讨论结合动态批处理器和多个模型实例），将下列代码添加到[模型配置文件](examples/model_repository/inception_graphdef/config.pbtxt)的末尾，然后重新启动 Triton。

```
instance_group [ { count: 2 }]
```

现在使用与基线相同的选项运行 perf_analyzer。

```
$ perf_analyzer -m inception_graphdef --percentile=95 --concurrency-range 1:4
...
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 70.6 infer/sec, latency 19547 usec
Concurrency: 2, throughput: 106.6 infer/sec, latency 23532 usec
Concurrency: 3, throughput: 110.2 infer/sec, latency 36649 usec
Concurrency: 4, throughput: 108.6 infer/sec, latency 43588 usec
```

在这种情况下，与一个实例相比，具有两个实例的模型将吞吐量从每秒约 73 次推理提高到每秒约 110 次推理。

可以同时启用动态批处理器和多个模型实例，例如，将以下内容添加到模型配置文件中。

```
dynamic_batching { preferred_batch_size: [ 4 ] }
instance_group [ { count: 2 }]
```

当我们使用与上述动态批处理器相同的选项运行 perf_analyzer 时。

```
$ perf_analyzer -m inception_graphdef --percentile=95 --concurrency-range 16
...
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 16, throughput: 289.6 infer/sec, latency 59817 usec
```

我们看到，与仅使用动态批处理器和一个实例相比，两个实例在增加延迟的同时并没有显着提高吞吐量。这是因为对于这个模型，单独的动态批处理器能够充分利用 GPU，因此添加额外的模型实例不会提供任何性能优势。一般来说，动态批处理器和多个实例的好处是特定于模型的，因此您应该尝试使用 perf_analyzer 以确定最能满足您的吞吐量和延迟要求的设置。

## 特定框架的优化

Triton有几个优化设置，它们只适用于受支持的模型框架的一个子集。这些优化设置由模型配置[优化策略](model_configuration.md#optimization-policy)控制。

### TensorRT优化的ONNX

一个特别强大的优化方法是将TensorRT与ONNX模型结合使用。作为TensorRT优化应用到ONNX模型的一个例子，我们将使用一个ONNX DenseNet模型，你可以通过[快速入门](quickstart.md)获得它。作为基线，我们使用perf_analyzer来确定[不启用任何性能特性的基本模型配置](examples/model_repository/densenet_onnx/config.pbtxt)的模型的性能。

```
$ perf_analyzer -m densenet_onnx --percentile=95 --concurrency-range 1:4
...
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, 113.2 infer/sec, latency 8939 usec
Concurrency: 2, 138.2 infer/sec, latency 14548 usec
Concurrency: 3, 137.2 infer/sec, latency 21947 usec
Concurrency: 4, 136.8 infer/sec, latency 29661 usec
```

要为模型启用TensorRT优化:停止Triton，在模型配置文件的末尾添加以下行，然后重新启动Triton。

```
optimization { execution_accelerators {
  gpu_execution_accelerator : [ {
    name : "tensorrt"
    parameters { key: "precision_mode" value: "FP16" }
    parameters { key: "max_workspace_size_bytes" value: "1073741824" }
    }]
}}
```

当Triton开始运行时，您应该检查控制台输出并等待，直到Triton打印出"Staring endpoints"消息。当TensorRT优化被启用时，ONNX模型加载可能会显著变慢。在生产中，你可以使用[模型预热](model_configuration.md#model-warmup)来避免这种模型启动/优化放缓。现在使用与基线相同的选项运行perf_analyzer。

```
$ perf_analyzer -m densenet_onnx --percentile=95 --concurrency-range 1:4
...
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, 190.6 infer/sec, latency 5384 usec
Concurrency: 2, 273.8 infer/sec, latency 7347 usec
Concurrency: 3, 272.2 infer/sec, latency 11046 usec
Concurrency: 4, 266.8 infer/sec, latency 15089 usec
```

TensorRT优化提供了2倍的吞吐量提升，同时将延迟减半。TensorRT提供的好处会因模型的不同而不同，但一般来说，它可以提供显著的性能提升。

### OpenVINO优化的ONNX

运行在CPU上的ONNX模型也可以通过使用[OpenVINO](https://docs.openvinotoolkit.org/latest/index.html)来加速。要为ONNX模型启用OpenVINO优化，请在模型配置文件的末尾添加以下代码行。

```
optimization { execution_accelerators {
  cpu_execution_accelerator : [ {
    name : "openvino"
  }]
}}
```

### TensorFlow with TensorRT Optimization

TensorRT optimization applied to a TensorFlow model works similarly to
TensorRT and ONNX described above. To enable TensorRT optimization you
must set the model configuration appropriately. For TensorRT
optimization of TensorFlow models there are several options that you
can enable, including selection of the compute precision.

```
optimization { execution_accelerators {
  gpu_execution_accelerator : [ {
    name : "tensorrt"
    parameters { key: "precision_mode" value: "FP16" }}]
}}
```

The options are described in detail in the
[ModelOptimizationPolicy](../src/core/model_configuration.proto)
section of the model configuration protobuf.

As an example of TensorRT optimization applied to a TensorFlow model,
we will use a TensorFlow Inception model that you can obtain by
following the [QuickStart](quickstart.md). As a baseline we use
perf_analyzer to determine the performance of the model using a [basic
model configuration that does not enable any performance
features](examples/model_repository/inception_graphdef/config.pbtxt).

```
$ perf_analyzer -m inception_graphdef --percentile=95 --concurrency-range 1:4
...
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 62.6 infer/sec, latency 21371 usec
Concurrency: 2, throughput: 73.2 infer/sec, latency 34381 usec
Concurrency: 3, throughput: 73.2 infer/sec, latency 50298 usec
Concurrency: 4, throughput: 73.4 infer/sec, latency 65569 usec
```

To enable TensorRT optimization for the model: stop Triton, add the
lines from above to the end of the model configuration file, and then
restart Triton. As Triton starts you should check the console output
and wait until the server prints the "Staring endpoints" message. Now
run perf_analyzer using the same options as for the baseline. Note
that the first run of perf_analyzer might timeout because the TensorRT
optimization is performed when the inference request is received and
may take significant time. In production you can use [model
warmup](model_configuration.md#model-warmup) to avoid this model
startup/optimization slowdown. For now, if this happens just run
perf_analyzer again.

```
$ perf_analyzer -m inception_graphdef --percentile=95 --concurrency-range 1:4
...
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 140 infer/sec, latency 8987 usec
Concurrency: 2, throughput: 195.6 infer/sec, latency 12583 usec
Concurrency: 3, throughput: 189 infer/sec, latency 19020 usec
Concurrency: 4, throughput: 191.6 infer/sec, latency 24622 usec
```

The TensorRT optimization provided 2.5x throughput improvement while
cutting latency by more than half. The benefit provided by TensorRT
will vary based on the model, but in general it can provide
significant performance improvement.

### TensorFlow Automatic FP16 Optimization

TensorFlow has an option to provide FP16 optimization that can be
enabled in the model configuration. As with the TensorRT optimization
described above, you can enable this optimization by using the
gpu_execution_accelerator property.

```
optimization { execution_accelerators {
  gpu_execution_accelerator : [
    { name : "auto_mixed_precision" }
  ]
}}
```

The options are described in detail in the
[ModelOptimizationPolicy](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)
section of the model configuration protobuf.

You can follow the steps described above for TensorRT to see how this
automatic FP16 optimization benefits a model by using perf_analyzer
to evaluate the model's performance with and without the optimization.

## NUMA优化

许多现代的cpu都是由多个核、内存和网络连接组成的，它们根据线程和数据的分配方式表现出不同的性能特征[引用https://www.kernel.org/doc/html/latest/vm/numa.html]。Triton允许您为您的系统设置描述此NUMA配置的主机策略，然后将模型实例分配给不同的主机策略，以利用这些NUMA属性。

### 主机策略

Triton允许您指定在启动时与策略名称关联的主机策略。如果在[实例组](model_configuration.md#instance-groups)中使用主机策略字段指定了具有相同策略名的实例，则主机策略将应用于模型实例。注意，如果没有指定，主机策略字段将根据实例属性设置为默认名称。

指定主机策略时，可以在命令行选项中指定如下参数:
```
--host-policy=<policy_name>,<setting>=<value>
```

目前支持的设置如下:

* *numa-node*: 主机策略绑定的NUMA节点id，主机策略限制内存分配给指定的节点。

* *cpu-cores*: 要运行的CPU内核，带有此主机策略集的实例将运行在这些CPU内核中的一个上。

假设配置GPU 0与CPU核数为0 ~ 15的NUMA节点绑定，设置“gpu_0”的NUMA -node和CPU -cores策略如下所示:

```
$ tritonserver --host-policy=gpu_0,numa-node=0 --host-policy=gpu_0,cpu-cores=0-15 ...
```
