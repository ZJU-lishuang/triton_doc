# 模型存储库

Triton推理服务器为服务器声明时指定的一个或多个模型存储库中的模型提供服务。当Triton运行时，可以按照[模型管理](model_management.md)中的说明修改所服务的模型。

## 存储库布局

这些存储库的路径是在使用--model-repository选项启动Triton时指定的。--model-repository选项可以多次指定以包含来自多个存储库的模型。构成模型存储库的目录和文件必须遵循所需的布局。假设存储库路径指定如下。

```bash
$ tritonserver --model-repository=<model-repository-path>
```

相应的存储库布局必须是:

```
  <model-repository-path>/
    <model-name>/
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    <model-name>/
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    ...
```

在顶级模型存储库目录中必须有零个或多个`<model-name>`子目录。每个`<model-name>`子目录都包含相应模型的存储库信息。config.pbtxt文件描述了模型的[模型配置](model_configuration.md)。对于某些模型，config.pbtxt是必需的，而对于其他模型是可选的。参阅[自动生成的模型配置](#auto-generated-model-configuration)获取详细信息。

每个`<model-name>`目录必须至少有一个数字子目录，表示模型的一个版本。有关Triton如何处理模型版本的更多信息，请参见[模型版本](#model-versions)。每个模型都由特定的[后端](https://github.com/triton-inference-server/backend/blob/main/README.md)执行。在每个版本子目录中必须有该后端所需的文件。例如，使用框架后端如TensorRT, PyTorch, ONNX, OpenVINO和TensorFlow的模型必须提供[框架特定的模型文件](#model-files)。

## 模型存储库的位置

Triton可以从一个或多个本地可访问的文件路径、谷歌云存储、Amazon S3和Azure存储中访问模型。

### 本地文件系统

对于本地可访问的文件系统，必须指定绝对路径。

```bash
$ tritonserver --model-repository=/path/to/model/repository ...
```

### 谷歌云存储

对于驻留在谷歌云存储中的模型存储库，存储库路径必须以gs://作为前缀。

```bash
$ tritonserver --model-repository=gs://bucket/path/to/model/repository ...
```

### S3

对于驻留在Amazon S3中的模型存储库，路径必须以s3://作为前缀。

```bash
$ tritonserver --model-repository=s3://bucket/path/to/model/repository ...
```

对于S3的本地或私有实例，前缀s3://后面必须跟主机和端口(以分号分隔)，然后是bucket路径。

```bash
$ tritonserver --model-repository=s3://host:port/bucket/path/to/model/repository ...
```

默认情况下，Triton使用HTTP与S3实例通信。如果您的S3实例支持HTTPS，并且您希望Triton使用HTTPS协议与它通信，那么您可以在模型存储库路径中通过在主机名前加上https://来指定相同的协议。

```bash
$ tritonserver --model-repository=s3://https://host:port/bucket/path/to/model/repository ...
```

在使用S3时，凭据和默认区域可以通过使用[aws config](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)命令或通过相应的[环境变量](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html)传递。如果设置了环境变量，它们将具有更高的优先级，Triton将使用它们，而不是使用aws config命令设置凭据。

### Azure Storage

对于Azure Storage中的模型存储库，存储库路径必须以as://作为前缀。

```bash
$ tritonserver --model-repository=as://account_name/container_name/path/to/model/repository ...
```

使用Azure Storage时，您必须将`AZURE_STORAGE_ACCOUNT`和`AZURE_STORAGE_KEY`环境变量设置为能够访问Azure Storage存储库的帐户。

如果你不知道你的`AZURE_STORAGE_KEY`，也没有正确配置你的Azure CLI，下面是一个如何找到你的`AZURE_STORAGE_ACCOUNT`对应的密钥的例子:

```bash
$ export AZURE_STORAGE_ACCOUNT="account_name"
$ export AZURE_STORAGE_KEY=$(az storage account keys list -n $AZURE_STORAGE_ACCOUNT --query "[0].value")
```

## 模型版本

在模型存储库中，每个模型可以有一个或多个可用版本。每个版本都存储在自己的数字命名的子目录中，子目录的名称对应于模型的版本号。不是数字命名的子目录或名称以0开头的子目录将被忽略。每个模型配置都指定了一个[版本策略](model_configuration.md#version-policy)，该策略控制Triton在任何给定时间提供模型存储库中的哪些版本。

## 模型文件

每个模型版本子目录的内容由模型的类型和支持该模型的[后端](https://github.com/triton-inference-server/backend/blob/main/README.md)的要求决定。

### TensorRT模型
TensorRT模型定义称为*Plan*。TensorRT Plan是一个单独的文件，默认情况下必须命名为model.plan。这个默认名称可以在[模型配置](model_configuration.md)中使用*default_model_filename*属性重写。

TensorRT Plan是特定于GPU的[CUDA计算能力](https://developer.nvidia.com/cuda-gpus)。因此，TensorRT模型将需要在[模型配置](model_configuration.md)中设置*cc_model_filenames*属性，将每个Plan文件与相应的计算能力关联起来。

TensorRT模型的最小模型存储库是:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.plan
```

### ONNX模型

ONNX模型是单个文件或包含多个文件的目录。默认情况下，文件或目录必须命名为model.onnx。这个默认名称可以在[模型配置](model_configuration.md)中使用*default_model_filename*属性重写。

Triton支持被Triton使用的[ONNX Runtime](https://github.com/Microsoft/onnxruntime)版本所支持的所有ONNX模型。如果模型使用[陈旧的ONNX opset版本](https://github.com/Microsoft/onnxruntime/blob/master/docs/Versioning.md#version-matrix)或[包含不支持类型的操作符](https://github.com/microsoft/onnxruntime/issues/1122)，则不支持模型。

包含单个文件的ONNX模型的最小模型存储库是:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.onnx
```

由多个文件组成的ONNX模型必须包含在一个目录中。默认情况下，这个目录必须命名为model.onnx，但是可以使用[模型配置](model_configuration.md)中的*default_model_filename*属性重写。这个目录中的主模型文件必须命名为model.onnx。一个包含在目录中的ONNX模型的最小模型存储库是:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.onnx/
           model.onnx
           <other model files>
```

### TorchScript模型

一个TorchScript模型是一个单独的文件，默认情况下必须命名为model.pt。这个默认名称可以在[模型配置](model_configuration.md)中使用*default_model_filename*属性重写。由于底层opset的变化，使用不同版本的PyTorch追踪的一些模型可能不被Triton支持。

TorchScript模型的最小模型存储库是:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.pt
```

### TensorFlow模型

TensorFlow以两种格式保存模型:*GraphDef*或*SavedModel*。Triton支持这两种格式。

TensorFlow GraphDef是一个单独的文件，默认情况下必须命名为model.graphdef。一个TensorFlow SavedModel是一个包含多个文件的目录。默认情况下，目录必须命名为model.savedmodel。可以使用[模型配置](model_configuration.md)中的*default_model_filename*属性重写这些默认名称。

TensorFlow GraphDef模型的最小模型存储库是:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.graphdef
```

TensorFlow SavedModel模型的最小模型存储库是:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.savedmodel/
           <saved-model files>
```

### OpenVINO模型

一个OpenVINO模型由两个文件表示，一个*.xml文件和一个*.bin文件。默认情况下，*.xml文件必须命名为model.xml。这个默认名称可以在[模型配置](model_configuration.md)中使用*default_model_filename*属性重写。

OpenVINO模型的最小模型存储库是:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.xml
        model.bin
```

### Python模型

[Python后端](https://github.com/triton-inference-server/python_backend)允许您在Triton中运行Python代码作为模型。默认情况下，Python脚本必须命名为model.py，但这个默认名称可以在[模型配置](model_configuration.md)中使用*default_model_filename*属性重写。

Python模型的最小模型存储库是:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.py
```

### DALI模型

[DALI后端](https://github.com/triton-inference-server/dali_backend)允许你在Triton中运行一个[DALI管道](https://github.com/NVIDIA/DALI)作为模型。为了使用这个后端，您需要生成一个文件，默认情况下命名为`model.dali`，并将其包含在您的模型库中。有关如何生成`model.dali`，请参考[DALI后端文档](https://github.com/triton-inference-server/dali_backend#how-to-use)的描述。可以使用[模型配置](model_configuration.md)中的*default_model_filename*属性重写默认的模型文件名。

DALI模型的最小模型存储库是:

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.dali
```
