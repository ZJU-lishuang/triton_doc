# 自定义操作

Triton 推理服务器部分支持允许自定义操作的模型框架。自定义操作可以在构建时或启动时添加到 Triton，并且可用于所有加载的模型。

## TensorRT

TensorRT 允许用户创建[自定义层](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending) ，然后可以在 TensorRT 模型中使用。对于要在 Triton 中运行的模型，必须使自定义层可用。

为了使自定义层对 Triton 可用，TensorRT 自定义层实现必须编译到一个或多个共享库中，然后必须使用 LD_PRELOAD 将其加载到 Triton 中。例如，假设您的 TensorRT 自定义层被编译到 libtrtcustom.so 中，使用以下命令启动 Triton 会使这些自定义层可用于所有 TensorRT 模型。

```bash
$ LD_PRELOAD=libtrtcustom.so tritonserver --model-repository=/tmp/models ...
```

这种方法的一个限制是自定义层必须与模型存储库本身分开管理。更严重的是，如果多个共享库之间存在自定义层名称冲突，则目前无法处理。

在构建自定义层共享库时，使用与 Triton 中使用的相同版本的 TensorRT 非常重要。您可以在[Triton 发行说明](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html)中找到 TensorRT 版本。确保您使用正确版本的 TensorRT 的一种简单方法是使用与 Triton 容器对应的[NGC TensorRT容器](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)。例如，如果您使用的是 21.11 版本的 Triton，请使用 21.11 版本的 TensorRT 容器。

## TensorFlow

Tensorflow allows users to [add custom
operations](https://www.tensorflow.org/guide/extend/op) which can then
be used in TensorFlow models. By using LD_PRELOAD you can load your
custom TensorFlow operations into Triton. For example, assuming your
TensorFlow custom operations are compiled into libtfcustom.so,
starting Triton with the following command makes those operations
available to all TensorFlow models.

```bash
$ LD_PRELOAD=libtfcustom.so tritonserver --model-repository=/tmp/models ...
```

All TensorFlow custom operations depend on a TensorFlow shared library
that must be available to the custom shared library when it is
loading. In practice this means that you must make sure that
/opt/tritonserver/backends/tensorflow1 or
/opt/tritonserver/backends/tensorflow2 is on the library path before
issuing the above command. There are several ways to control the
library path and a common one is to use the LD_LIBRARY_PATH. You can
set LD_LIBRARY_PATH in the "docker run" command or inside the
container.

```bash
$ export LD_LIBRARY_PATH=/opt/tritonserver/backends/tensorflow1:$LD_LIBRARY_PATH
```

A limitation of this approach is that the custom operations must be
managed separately from the model repository itself. And more
seriously, if there are custom layer name conflicts across multiple
shared libraries there is currently no way to handle it.

When building the custom operations shared library it is important to
use the same version of TensorFlow as is being used in Triton. You can
find the TensorFlow version in the [Triton Release
Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html). A
simple way to ensure you are using the correct version of TensorFlow
is to use the [NGC TensorFlow
container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)
corresponding to the Triton container. For example, if you are using
the 21.11 version of Triton, use the 21.11 version of the TensorFlow
container.

## PyTorch

Torchscript allows users to [add custom
operations](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)
which can then be used in Torchscript models. By using LD_PRELOAD you
can load your custom C++ operations into Triton. For example, if you
follow the instructions in the
[pytorch/extension-script](https://github.com/pytorch/extension-script)
repository and your Torchscript custom operations are compiled into
libpytcustom.so, starting Triton with the following command makes
those operations available to all PyTorch models. Since all Pytorch
custom operations depend on one or more PyTorch shared libraries
that must be available to the custom shared library when it is
loading. In practice this means that you must make sure that
/opt/tritonserver/backends/pytorch is on the library path while
launching the server. There are several ways to control the library path
and a common one is to use the LD_LIBRARY_PATH.

```bash
$ LD_LIBRARY_PATH=/opt/tritonserver/backends/pytorch:$LD_LIBRARY_PATH LD_PRELOAD=libpytcustom.so tritonserver --model-repository=/tmp/models ...
```

A limitation of this approach is that the custom operations must be
managed separately from the model repository itself. And more
seriously, if there are custom layer name conflicts across multiple
shared libraries or the handles used to register them in PyTorch there
is currently no way to handle it.

Starting with the 20.07 release of Triton the [TorchVision
operations](https://github.com/pytorch/vision) will be included with
the PyTorch backend and hence they do not have to be explicitly added
as custom operations.

When building the custom operations shared library it is important to
use the same version of PyTorch as is being used in Triton. You can
find the PyTorch version in the [Triton Release
Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html). A
simple way to ensure you are using the correct version of PyTorch is
to use the [NGC PyTorch
container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
corresponding to the Triton container. For example, if you are using
the 21.11 version of Triton, use the 21.11 version of the PyTorch
container.

## ONNX

ONNX Runtime allows users to [add custom
operations](https://github.com/microsoft/onnxruntime/blob/master/docs/AddingCustomOp.md)
which can then be used in ONNX models. To register your custom
operations library you need to include it in the model configuration
as an additional field. For example, if you follow [this
example](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/shared_lib/test_inference.cc)
from the
[microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
repository and your ONNXRuntime custom operations are compiled into
libonnxcustom.so, adding the following to the model configuraion of
your model makes those operations available to that specific ONNX
model.

```bash
$ model_operations { op_library_filename: "/path/to/libonnxcustom.so" }
```

When building the custom operations shared library it is important to
use the same version of ONNXRuntime as is being used in Triton. You
can find the ONNXRuntime version in the [Triton Release
Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html).
