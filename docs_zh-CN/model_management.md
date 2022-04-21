# 模型管理

Triton 提供的模型管理 API 是一部分的[HTTP/REST 和 GRPC 协议，也是 C API 的一部分](inference_protocols.md)。Triton 以三种模型控制模式之一运行：NONE、EXPLICIT 或 POLL。模型控制模式决定了 Triton 如何处理对模型存储库的更改以及这些协议和 API 中的哪些可用。

## 模型控制模式 NONE

Triton 尝试在启动时加载模型存储库中的所有模型。Triton 无法加载的模型将被标记为 UNAVAILABLE 并且不可用于推理。

服务器运行时对模型存储库的更改将被忽略。使用[模型控制协议](protocol/extension_model_repository.md)的模型加载和卸载请求不会有任何影响，并且会返回错误响应。

通过在启动 Triton 时指定 --model-control-mode=none 来选择此模型控制模式。这是默认的模型控制模式。如[修改模型存储库](#modifying-the-model-repository)中所述，在 Triton 运行时更改模型存储库必须小心。

## 模型控制模式 EXPLICIT

在启动时，Triton 仅加载使用 --load-model 命令行选项明确指定的那些模型。如果未指定 --load-model 则在启动时不加载任何模型。Triton 无法加载的模型将被标记为 UNAVAILABLE 并且不可用于推理。

启动后，所有模型加载和卸载动作都必须使用[模型控制协议](protocol/extension_model_repository.md)显式启动。模型控制请求的响应状态表示加载或卸载动作的成功或失败。尝试重新加载已加载的模型时，如果由于任何原因重新加载失败，则已加载的模型将保持不变并保持加载状态。如果重新加载成功，新加载的模型将替换已经加载的模型，而不会损失模型的可用性。

通过指定 --model-control-mode=explicit 启用此模型控制模式。在 Triton 运行时更改模型存储库必须小心，如[修改模型存储库](#modifying-the-model-repository)中所述。

## 模型控制模式 POLL

Triton 尝试在启动时加载模型存储库中的所有模型。Triton 无法加载的模型将被标记为 UNAVAILABLE 并且不可用于推理。

将检测到模型存储库的更改，Triton 将根据这些更改尝试加载和卸载模型。尝试重新加载已加载的模型时，如果由于任何原因重新加载失败，则已加载的模型将保持不变并保持加载状态。如果重新加载成功，新加载的模型将替换已经加载的模型，而不会丢失模型的可用性。

由于 Triton 会定期轮询存储库，因此可能无法立即检测到对模型存储库的更改。您可以使用 --repository-poll-secs 选项控制轮询间隔。控制台日志或[模型就绪协议](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md)或[模型控制协议](protocol/extension_model_repository.md)的索引操作可用于确定模型存储库更改何时生效。

**警告：Triton 轮询模型存储库与您对存储库进行任何更改之间没有同步。因此，Triton 可以观察到会导致意外行为的部分和不完整的改变。因此，不建议在生产环境中使用 POLL 模式。**

使用[模型控制协议](protocols/extension_model_repository.md)的模型加载和卸载请求不会有任何影响，并且会返回错误响应。

通过指定 --model-control-mode=poll 并在启动 Triton 时将 --repository-poll-secs 设置为非零值来启用此模型控制模式。正如[修改模型存储库](#modifying-the-model-repository)中解释，在 Triton 运行时更改模型存储库必须小心。

在 POLL 模式下，Triton 响应以下模型存储库更改：

* 通过添加和删除相应的版本子目录，可以在模型中添加和删除版本。即使他们正在使用模型的已删除版本，Triton 也将允许执行中的请求完成。对已删除模型版本的新请求将失败。根据模型的[版本策略](model_configuration.md#version-policy)，对可用版本的更改可能会更改默认提供的模型版本。  

* 通过删除相应的模型目录，可以从存储库中删除现有模型。Triton 将允许即使存在使用已删除模型的任何版本的执行中的请求，也能完成删除。对已移除模型的新请求将失败。

* 可以通过添加新模型目录将新模型添加到存储库中。

* [模型配置文件](model_configuration.md) (config.pbtxt) 可以更改，Triton 将卸载并重新加载模型以获取新的模型配置。

* 为表示分类的输出提供标签的标签文件，可以添加、删除或修改，Triton 将卸载并重新加载模型以获取新标签。如果添加或删除标签文件，则必须同时对[模型配置](model_configuration.md)中对应的输出的*label_filename* 属性进行相应的编辑。

## 修改模型存储库

模型存储库中的每个模型都[位于自己的子目录](model_repository.md#repository-layout)中。模型子目录内容允许的活动取决于 Triton 使用该模型的方式。可以使用[模型元数据](inference_protocols.md#inference-protocols-and-apis)或 [存储库索引](protocol/extension_model_repository.md#index)API 来确定模型的状态。

* 如果模型正在加载或卸载，则不必添加、删除或修改该子目录中的任何文件或目录。

* 如果模型从未加载或完全卸载，则可以删除整个模型子目录，或者可以添加、删除或修改其任何内容。

* 如果模型已完全加载，除了实现模型后端的共享库，可以添加、删除或修改该子目录中的任何文件或目录。Triton 在加载模型时使用后端共享库，因此删除或修改它们可能会导致 Triton 崩溃。要更新模型的后端，您必须先完全卸载模型，修改后端共享库，然后重新加载模型。在某些操作系统上，也可以简单地将现有的共享库移动到模型存储库之外的另一个位置，复制进来新的共享库，然后重新加载模型。
