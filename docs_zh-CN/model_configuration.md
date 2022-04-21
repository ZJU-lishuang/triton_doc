# 模型配置

[模型存储库](model_repository.md)中的每个模型都必须包含一个模型配置，该配置提供了关于模型的必需和可选信息。通常，该配置在指定为[模型配置protobuf](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)的config.pbtxt文件中提供。在某些情况下，如[自动生成模型配置](#auto-generated-model-configuration)中所讨论的，模型配置可以由Triton自动生成，因此不需要显式提供。

本节描述了最重要的模型配置属性，但是也应该参考[模型配置protobuf](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)中的文档。

## 最小的模型配置

最小的模型配置必须指定[*platform*和/或*backend*属性](https://github.com/triton-inference-server/backend/blob/main/README.md#backends)，*max_batch_size*属性以及输入和输出的张量。

例如，考虑一个TensorRT模型，有两个输入*input0*和*input1*，一个输出,*output0*，都是16个float32的张量。最小的配置如下：

```
  platform: "tensorrt_plan"
  max_batch_size: 8
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    },
    {
      name: "input1"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
```

### 名称，平台和后端

模型配置中*name*属性是可选的。如果模型的名称未在配置中指定，则假定它与包含模型的模型存储库目录相同。如果指定了*name*，它必须与包含模型的模型存储库目录的名称相匹配。*platform*和*backend*所需的值在 [后端文档](https://github.com/triton-inference-server/backend/blob/main/README.md#backends)中进行了描述。

### 最大批处理大小

*max_batch_size*属性表示模型支持的最大批处理大小，这些[批处理类型](architecture.md#models-and-schedulers)可以被 Triton 利用。如果模型的批处理维度是第一个维度，并且模型的所有输入和输出都具有此批处理维度，那么 Triton 可以使用其[动态批处理器](#dynamic-batcher)或[序列批处理器](#sequence-batcher)自动对模型使用批处理。在这种情况下，*max_batch_size*应设置为大于或等于1的值，该值表示Triton 应与模型一起使用的最大批量大小。

对于不支持批处理的模型，或者不支持上述特定方式的批处理，*max_batch_size* 必须设置为零。

### 输入和输出

每个模型输入和输出都必须指定名称、数据类型和形状。输入或输出张量指定的名称必须与模型预期的名称匹配。

#### PyTorch 后端的特殊约定

***命名约定:*** 由于TorchScript 模型中没有输入和输出名称，因此在配置中输入和输出的"name"属性必须遵循特定的命名约定，即"\<name\>__\<index\>"。其中\<name\>可以是任何字符串，而 \<index\> 指的是相应输入/输出的位置。这意味着如果有两个输入和两个输出，它们必须命名为：“INPUT__0”、“INPUT__1”和“OUTPUT__0”、“OUTPUT__1”，这样“INPUT__0”指的是第一个输入，INPUT__1 指的是第二个输入，等等。

***张量字典作为输入:*** PyTorch 后端支持以张量字典的形式将输入传递给模型。仅当模型是*单个*输入，且为从字符串到张量的映射的字典，才支持此功能。例如，如果有一个模型需要如下构成的输入：

```
{'A': tensor1, 'B': tensor2}
```

那么配置中的输入字段就不需要遵循"\<name\>__\<index\>"的约定了。相反，在这种情况下，输入的名称必须映射到该特定张量的字符串值'key'。在这种情况下，输入将是"A"和"B"，其中输入"A"代表对应于 tensor1 的值，"B"代表对应于 tensor2 的值。

<br>

输入和输出张量允许的数据类型因模型的类型而异。[数据类型](#datatypes)部分描述了允许的数据类型以及它们如何映射到每个类型的模型的数据类型。

输入形状表示模型和 Triton 在推理请求中预期的输入张量的形状。输出形状表示模型生成并由 Triton 响应推理请求返回的输出张量的形状。输入和输出形状的秩都必须大于或等于 1，即，不允许使用空形状 **[ ]** 。

输入和输出形状由*max_batch_size*和输入或输出*dims* 属性指定的尺寸组合来指定。对于*max_batch_size*大于 0 的模型，完整的形状为 [ -1 ] + *dims*。对于*max_batch_size*等于 0 的模型，完整的形状为*dims*。例如，对于以下配置，“input0”的形状为 [ -1, 16 ]，“output0”的形状为 [ -1, 4 ]。

```
  platform: "tensorrt_plan"
  max_batch_size: 8
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [ 4 ]
    }
  ]
```

对于除了*max_batch_size*等于 0 之外，其它相同的配置，“input0”的形状为 [16]，“output0”的形状为 [4]。

```
  platform: "tensorrt_plan"
  max_batch_size: 0
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [ 4 ]
    }
  ]
```

对于支持可变尺寸输入和输出张量的模型，这些尺寸可以在输入和输出配置中列为 -1。例如，如果一个模型需要一个二维输入张量，其中第一个维度的大小必须为 4，但第二个维度可以是任何大小，则该模型的输入配置将是*dims: [ 4, -1 ]*。Triton 随后将接受输入张量的第二维是大于或等于0的任何值的推理请求。模型配置可能比基础模型所允许的限制更多。例如，即使模型框架本身允许第二维为任意大小，模型配置也可以指定为*dims: [ 4, 4 ]*。 在这种情况下，Triton 只会接受输入张量的形状正好是 *[ 4, 4 ]* 的推理请求。

如果 Triton 在推理请求中接收到的输入形状与模型预期的输入形状不匹配，则必须使用 [*reshape* 属性](#reshape)。同样， 如果模型产生的输出形状与 Triton 在响应推理请求时返回的形状不匹配，则必须使用*reshape* 属性。

模型输入通过配置`allow_ragged_batch`，表示输入是[不规则输入](ragged_batching.md#ragged-batching)。该字段与[动态批处理器](#dynamic-batcher)一起使用，以允许在不强制输入在所有请求中具有相同形状的情况下进行批处理。

## 自动生成模型配置

默认情况下，每个模型都必须提供包含所需设置的模型配置文件。但是，如果 Triton 使用 --strict-model-config=false 选项启动，那么在某些情况下，模型配置文件的所需部分可以由 Triton 自动生成。模型配置的必需部分是[最小模型配置](#minimal-model-configuration)中显示的那些设置。具体来说，TensorRT、TensorFlow 保存的模型和 ONNX 模型不需要模型配置文件，因为 Triton 可以自动导出所有需要的设置。所有其他类型的模型必须提供模型配置文件。

使用 --strict-model-config=false 时，您可以查看使用 [模型配置节点](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_configuration.md)为模型生成的模型配置。最简单的方法是使用*curl*之类的工具：

```bash
$ curl localhost:8000/v2/models/<model name>/config
```

这将返回生成的模型配置的 JSON 表示。您可以从中获取 JSON格式 的 max_batch_size、输入和输出部分，并将其转换为 config.pbtxt 文件。Triton 只生成[模型配置的最小部分](#minimal-model-configuration)。您仍然必须通过编辑 config.pbtxt 文件来提供模型配置的可选部分。

## 数据类型

下表显示了 Triton 支持的张量数据类型。第一列显示模型配置文件中出现的数据类型的名称。接下来的四列显示了支持的模型框架的相应数据类型。如果模型框架没有给定数据类型的条目，则 Triton 不支持该模型的该数据类型。标记为“API”的第六列显示了 TRITONSERVER C API、TRITONBACKEND C API、HTTP/REST 协议和 GRPC 协议对应的数据类型。最后一列显示了 Python numpy 库对应的数据类型。

|Model Config  |TensorRT      |TensorFlow    |ONNX Runtime  |PyTorch  |API      |NumPy         |
|--------------|--------------|--------------|--------------|---------|---------|--------------|
|TYPE_BOOL     | kBOOL        |DT_BOOL       |BOOL          |kBool    |BOOL     |bool          |
|TYPE_UINT8    |              |DT_UINT8      |UINT8         |kByte    |UINT8    |uint8         |
|TYPE_UINT16   |              |DT_UINT16     |UINT16        |         |UINT16   |uint16        |
|TYPE_UINT32   |              |DT_UINT32     |UINT32        |         |UINT32   |uint32        |
|TYPE_UINT64   |              |DT_UINT64     |UINT64        |         |UINT64   |uint64        |
|TYPE_INT8     | kINT8        |DT_INT8       |INT8          |kChar    |INT8     |int8          |
|TYPE_INT16    |              |DT_INT16      |INT16         |kShort   |INT16    |int16         |
|TYPE_INT32    | kINT32       |DT_INT32      |INT32         |kInt     |INT32    |int32         |
|TYPE_INT64    |              |DT_INT64      |INT64         |kLong    |INT64    |int64         |
|TYPE_FP16     | kHALF        |DT_HALF       |FLOAT16       |         |FP16     |float16       |
|TYPE_FP32     | kFLOAT       |DT_FLOAT      |FLOAT         |kFloat   |FP32     |float32       |
|TYPE_FP64     |              |DT_DOUBLE     |DOUBLE        |kDouble  |FP64     |float64       |
|TYPE_STRING   |              |DT_STRING     |STRING        |         |BYTES    |dtype(object) |

对于 TensorRT，每个值都在 nvinfer1::DataType 命名空间中。例如，nvinfer1::DataType::kFLOAT 是 32 位浮点数据类型。

对于 TensorFlow，每个值都在 tensorflow 命名空间中。例如，tensorflow::DT_FLOAT 是 32 位浮点值。

对于 ONNX Runtime，每个值都以 ONNX_TENSOR_ELEMENT_DATA_TYPE_ 开头。例如，ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT 是 32 位浮点数据类型。

对于 PyTorch，每个值都在 torch 命名空间中。例如，torch::kFloat 是 32 位浮点数据类型。

对于 Numpy，每个值都在 numpy 模块中。例如，numpy.float32 是 32 位浮点数据类型。

## 重塑
<!-- reshape: { shape: [ ] }，reshape后接的shape是模型推理的形状，dims是推理API输入或输出的形状-->

模型配置输入或输出上的*ModelTensorReshape*属性用于表明推理 API 接受的输入或输出形状与底层模型框架或自定义后端预期或生成的输入或输出形状不同。

对于输入，*reshape*可用于将输入张量重塑为框架或后端所期望的不同形状。一个常见的用例是支持批处理的模型期望批处理输入具有形状 *[ batch-size ]*，这意味着批处理维度完全描述了形状。对于推理 API，必须指定等效形状 *[ batch-size, 1 ]*，因为每个输入都必须指定一个非空的*dims*。对于这种情况，输入应指定为：

```
  input [
    {
      name: "in"
      dims: [ 1 ]
      reshape: { shape: [ ] }
    }
```

对于输出，*reshape*可用于将框架或后端生成的输出张量重塑为推理 API 返回的不同形状。一个常见的用例是支持批处理的模型期望批处理输出具有形状 *[ batch-size]*，这意味着批处理维度完全描述了形状。对于推理 API，必须指定等效形状 *[ batch-size, 1 ]*， 因为每个输出都必须指定一个非空的 *dims*。对于这种情况，输出应指定为：

```
  output [
    {
      name: "in"
      dims: [ 1 ]
      reshape: { shape: [ ] }
    }
```

## 形状张量

对于支持形状张量的模型，必须为充当形状张量的输入和输出适当设置*is_shape_tensor*属性。以下显示了指定形状张量的示例配置。

```
  name: "myshapetensormodel"
  platform: "tensorrt_plan"
  max_batch_size: 8
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [ -1 ]
    },
    {
      name: "input1"
      data_type: TYPE_INT32
      dims: [ 1 ]
      is_shape_tensor: true
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [ -1 ]
    }
  ]
```

如上所述，Triton 假设批处理发生在输入或输出张量*dims*中未列出的第一个维度上。然而，对于形状张量，批处理发生在第一个形状值处。对于上面的示例，推理请求必须提供具有以下形状的输入。

```
  "input0": [ x, -1]
  "input1": [ 1 ]
  "output0": [ x, -1]
```

其中*x*是请求的批量大小。Triton 要求在使用批处理时将形状张量标记为模型中的形状张量。请注意，“input1”的形状为 *[ 1 ]* 而不是 *[ 2 ]*。在向模型发出请求之前， Triton 将在“input1”处添加形状值*x*。

## 版本政策

每个模型可以有一个或多个[版本](model_repository.md#model-versions)。模型配置的 *ModelVersionPolicy*属性用于设置以下策略之一。

* *All*: 模型存储库中可用的所有模型版本都可用于推理。```version_policy: { all: {}}```

* *Latest*: 只有存储库中模型的最新的‘n’版本可用于推理。模型的最新版本是数字上最大的版本号。```version_policy: { latest: { num_versions: 2}}```

* *Specific*: 只有模型中特别列出的版本可用于推理。```version_policy: { specific: { versions: [1,3]}}```

如果未指定版本策略，则使用*Latest*（n=1）作为默认值，表示 Triton 仅提供模型的最新版本。在所有情况下，从模型存储库中[添加或删除版本子目录](model_management.md)都可以更改在后续推理请求中使用的模型版本。

以下配置指定模型的所有版本都可从服务器获得。

```
  platform: "tensorrt_plan"
  max_batch_size: 8
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    },
    {
      name: "input1"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
  version_policy: { all { }}
```

## 实例组

Triton 可以提供多个[模型的实例](architecture.md#concurrent-model-execution)，以便可以同时处理该模型的多个推理请求。模型配置*ModelInstanceGroup*属性用于指定应该可用的执行实例的数量以及应该为这些实例使用的计算资源。

### 多个模型实例

默认情况下，为系统中可用的每个 GPU 创建模型的单个执行实例。实例组设置可用于在每个 GPU 上或仅在某些 GPU 上放置模型的多个执行实例。例如，以下配置将在每个系统 GPU 上放置模型的两个执行实例。

```
  instance_group [
    {
      count: 2
      kind: KIND_GPU
    }
  ]
```

以下配置将在 GPU 0 上放置一个执行实例，在 GPU 1 和 2 上放置两个执行实例。

```
  instance_group [
    {
      count: 1
      kind: KIND_GPU
      gpus: [ 0 ]
    },
    {
      count: 2
      kind: KIND_GPU
      gpus: [ 1, 2 ]
    }
  ]
```

### CPU 模型实例

实例组设置还用于启用模型在 CPU 上的执行。即使系统中有 GPU 可用，模型也可以在 CPU 上执行。下面将两个执行实例放在 CPU 上。

```
  instance_group [
    {
      count: 2
      kind: KIND_CPU
    }
  ]
```

### 宿主机策略

实例组设置与主机策略相关联。以下配置会将实例组设置创建的所有实例与主机策略“policy_0”相关联。默认情况下，主机策略将根据实例的设备类型进行设置，例如，KIND_CPU 为“cpu”，KIND_MODEL 为“model”，KIND_GPU 为“gpu_\<gpu_id\>”。

```
  instance_group [
    {
      count: 2
      kind: KIND_CPU
      host_policy: "policy_0"
    }
  ]
```

### 速率限制器配置

实例组可以选择指定[速率限制器](rate_limiter.md)配置，该配置控制速率限制器如何对组中的实例进行操作。如果速率限制关闭，速率限制器配置将被忽略。如果速率限制打开并且一个实例组不提供此配置，则属于该组的模型实例上的执行将不会受到速率限制器的任何限制。配置包括以下规范内容：

#### 资源

执行模型实例所需的一组[资源](rate_limiter.md#resources)。“name”字段标识资源，“count”字段是指组中的模型实例运行所需的资源副本数。“global”字段指定资源是按设备还是在系统中全局共享。加载的模型不能将同名的资源同时指定为全局和非全局资源。如果没有提供资源，则 triton 假定模型实例的执行不需要任何资源，并且将在模型实例可用时立即开始执行。

#### 优先级

优先级用作加权值，用于对所有模型的所有实例进行优先级排序。优先级为2的实例将获得优先级为1的实例的1/2调度机会。

以下示例指定组中的实例需要四个“R1”和两个“R2”资源才能执行。资源“R2”是全局资源。此外，实例组的速率限速器优先级为 2。

```
  instance_group [
    {
      count: 1
      kind: KIND_GPU
      gpus: [ 0, 1, 2 ]
      rate_limiter {
        resources [
          {
            name: "R1"
            count: 4
          },
          {
            name: "R2"
            global: True
            count: 2 
          }
        ] 
        priority: 2
      }
    }
  ]
```

上面的配置创建了 3 个模型实例，每个设备上都有一个（0、1 和 2）。这三个实例之间不会争用“R1”，因为“R1”对于它们自己的设备来说是本地的，但是，它们会争用“R2”，因为它被指定为全局资源，这意味着“R2”在整个系统中共享. 虽然这些实例在它们之间不竞争“R1”，但它们会与在同一设备上运行且资源需求中包含“R1”的其他模型实例竞争“R1”。

## 调度和批处理

Triton 通过允许单个推理请求指定一批输入来支持批量推理。同时执行一批输入的推理，这对于 GPU 尤其重要，因为它可以大大提高推理吞吐量。在许多用例中，单个推理请求没有被批处理，因此，它们不会从批处理的吞吐量优势中受益。

推理服务器包含多种调度和批处理算法，支持许多不同的模型类型和用例。关于模型类型和调度器的更多信息可以在 [模型和调度器](architecture.md#models-and-schedulers)中找到。

### 默认调度程序

如果在模型配置中未指定任何*scheduling_choice*属性，则默认调度程序用于该模型。默认调度程序只是将推理请求分发给模型配置的所有[模型实例](#instance-groups)。

### 动态批处理器

动态批处理是Triton 的一项功能，它允许服务器组合推理请求，从而动态创建批处理。创建一批请求通常会导致吞吐量增加。动态批处理器应该用于[无状态模型](architecture.md#stateless-models)。动态创建的批次分发给模型配置的所有[模型实例](#instance-groups) 。

使用模型配置中的*ModelDynamicBatching*属性为每个模型单独启用和配置动态批处理。这些设置控制动态创建的批处理的首选大小、请求可以在调度程序中延迟以允许其他请求加入动态批处理的最长时间，以及队列属性，例如队列大小、优先级和超时.

#### 推荐的配置过程

下面详细介绍各个设置。以下步骤是为每个模型调整动态批处理器的推荐过程。还可以使用[模型分析器](model_analyzer.md)自动搜索不同的动态批处理器配置。

* 确定模型的 [最大批量大小](#maximum-batch-size) 。

* 将以下内容添加到模型配置中以启用具有所有默认设置的动态批处理器。默认情况下，动态批处理器将创建尽可能大的批处理，直到最大批处理大小，并且在形成批处理时不会[延迟](#delayed-batching)。

```
  dynamic_batching { }
```

* 使用[性能分析器](perf_analyzer.md)确定默认动态批处理器配置提供的延迟和吞吐量。

* 如果默认配置导致延迟值在您的延迟预算范围内，请尝试以下一种或两种方法来权衡增加的延迟以提高吞吐量：

  * 增加最大批处理大小。

  * 将[批处理延迟](#delayed-batching)设置为非零值。尝试增加延迟值直到超过延迟预算以查看对吞吐量的影响。

* 大多数模型不应使用[首选批处理大小](#preferred-batch-sizes)。仅当该批处理大小导致性能明显高于其他批处理大小时，才应配置首选批处理大小。

#### 首选批处理大小

*preferred_batch_size*属性表示动态批处理器应该尝试创建的批处理大小。对大多数模型，不应该指定*preferred_batch_size*，如[推荐的配置过程](#recommended-configuration-process)中所述。一个例外是 TensorRT 模型，它为不同的批量大小指定了多个优化配置文件。在这种情况下，由于与其他优化配置文件相比，某些优化配置文件可能会显着提高性能，因此将*preferred_batch_size*用于那些高性能优化配置文件支持的批处理大小可能是有意义的。

以下示例显示了启用动态批处理的配置，首选批处理大小为 4 和 8。

```
  dynamic_batching {
    preferred_batch_size: [ 4, 8 ]
  }
```

当模型实例可用于推理时，动态批处理器将根据调度程序中可用的请求尝试创建批处理。按收到请求的顺序将请求添加到批处理中。如果动态批处理器可以形成首选大小的批处理，它将创建最大可能的首选大小的批处理并将其发送以进行推理。如果动态批处理器无法形成首选大小的批处理（或者如果动态批处理器未配置任何首选批处理大小），它将发送一个最大可能的批处理大小，其小于模型允许的最大批处理大小（但是，请参阅以下部分，了解更改此行为的延迟选项）。

生成的批处理大小可以使用[计数指标](metrics.md#count-metrics)进行汇总检查。

#### 延迟批处理

动态批处理器可以配置为允许请求在调度器中延迟有限的时间，以允许其他请求加入动态批处理。例如，以下配置将请求的最大延迟时间设置为 100 微秒。

```
  dynamic_batching {
    max_queue_delay_microseconds: 100
  }
```

当无法创建最大尺寸（或首选尺寸）批大小时，*max_queue_delay_microseconds*属性设置会改变动态批处理器行为。当无法从可用请求创建最大或首选大小的批处理时，只要请求的延迟时间没有超过配置的*max_queue_delay_microseconds*值，动态批处理器就会延迟发送批处理。如果在此延迟期间有新请求到达并允许动态批处理器形成最大或首选大小的批处理，则立即发送该批处理以进行推理。如果延迟到期，动态批处理器会按原样发送批处理，即使它不是最大或首选大小。

#### 保留顺序

*preserve_ordering*属性用于强制以与接收请求相同的顺序返回所有响应。有关详细信息，请参阅[protobuf文档](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)。

#### 优先等级

默认情况下，动态批处理器维护一个队列，其中包含一个模型的所有推理请求。请求按顺序处理和批处理。*priority_levels*属性可用于在动态批处理器中创建多个优先级，以便允许具有较高优先级的请求插队具有较低优先级的请求。相同优先级的请求按顺序处理。使用*default_priority_level*属性来调度未设置优先级的推理请求。

#### 队列策略

动态批处理器提供了几个设置来控制请求如何排队等待批处理。

当没有定义*priority_levels*时，单个队列的*ModelQueuePolicy*可以设置为*default_queue_policy*。当定义了*priority_levels*时，每个优先级可以有一个由*default_queue_policy*和*priority_queue_policy*指定的不同的 *ModelQueuePolicy*。

*ModelQueuePolicy*属性允许使用*max_queue_size*来设置最大队列大小。*timeout_action*、 *default_timeout_microseconds*和*allow_timeout_override*设置允许配置队列，以便单个请求在队列中的时间，超过指定的超时时间时，拒绝或延迟。

### 序列批处理

与动态批处理器一样，序列批处理器组合非批处理推理请求，从而动态创建批处理。与动态批处理器不同，序列批处理器应用于[有状态模型](architecture.md#stateful-models)，其中必须将一个序列的推理请求路由到同一模型实例。动态创建的批处理分发给模型配置的所有[模型实例](#instance-groups)。

使用模型配置中的*ModelSequenceBatching*属性为每个模型单独启用和配置序列批处理。这些设置控制序列超时以及配置 Triton 如何将控制信号发送到模型，表示序列开始、结束、就绪和相关 ID。有关更多信息和示例，请参阅[有状态模型](architecture.md#stateful-models)。

### 集成调度器

 集成调度器必须用于[集成模型](architecture.md#ensemble-models)，不能用于任何其他类型的模型。

使用模型配置中的*ModelEnsembleScheduling*属性为每个模型单独启用和配置集成调度程序。这些设置描述了集成中包含的模型以及模型之间的张量值流。有关更多信息和示例，请参阅[集成模型](architecture.md#ensemble-models)。

## 优化策略

模型配置*ModelOptimizationPolicy*属性用于指定模型的优化和优先级设置。这些设置控制模型是否/如何由后端优化，以及如何由 Triton 调度和执行。有关当前可用的设置，请参阅[模型配置protobuf](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto) 和 [优化](optimization.md#framework-specific-optimization)文档。

## 模型热身

当 Triton 加载模型时，相应的 [后端](https://github.com/triton-inference-server/backend/blob/main/README.md) 会为该模型初始化。对于某些后端，部分或全部初始化被推迟到模型接收到它的第一个推理请求（或前几个推理请求）。因此，由于延迟初始化，第一个或前几个推理请求可能会明显变慢。

为了避免这些初始时缓慢的推理请求，Triton 提供了一个配置选项，使模型能够“预热”，以便在收到第一个推理请求之前完全初始化它。在模型配置中定义*ModelWarmup*属性时，直到模型预热完成，Triton 才会将模型显示为已准备好进行推理。

模型配置*ModelWarmup*用于指定模型的预热设置。这些设置定义了 Triton 将创建的一系列推理请求，以预热每个模型实例。仅当模型实例成功完成请求时才会提供服务。请注意，预热模型的效果因框架后端而异，这会导致 Triton 对模型更新的响应速度较慢，因此用户应尝试并选择适合自己需要的配置。有关当前可用的设置，请参阅 protobuf 文档。

## 响应缓存

模型配置`response_cache`部分有一个`enable`布尔值，用于为此模型启用响应缓存。除了在模型配置中启用缓存外，启动服务器时还必须设置一个非零值的`--response-cache-byte-size`。

```
response_cache { 
  enable: True 
}
```

请参阅[响应缓存](https://github.com/triton-inference-server/server/blob/main/docs/response_cache.md)和[模型配置protobuf](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)。文档以获取更多信息。
