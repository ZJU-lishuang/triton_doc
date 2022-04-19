# 推理协议和API

客户端可以使用 HTTP/REST 或 GRPC 协议或通过 C API 与 Triton 进行通信。

## HTTP/REST 和 GRPC 协议

Triton 基于[KServe项目](https://github.com/kserve) 提出的[标准推理协议](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2)公开了 HTTP/REST 和 GRPC 端点。为了完全启用所有功能，Triton还实现了对KServe推理协议的一系列[HTTP/REST 和 GRPC 扩展](https://github.com/triton-inference-server/server/tree/main/docs/protocol)。

HTTP/REST 和 GRPC 协议提供端点来检查服务器和模型的健康状况、元数据和统计信息。其他端点允许模型加载和卸载以及推理。有关详细信息，请参阅 KServe 和扩展文档。

### HTTP选项
Triton 为使用 HTTP 协议的服务器-客户端网络事务提供以下配置选项。

#### 压缩

Triton 允许通过其客户端对 HTTP 上的请求/响应进行在线压缩。有关详细信息，请参阅[HTTP 压缩](https://github.com/triton-inference-server/client/tree/main#compression) 。

### GRPC选项
Triton 公开了用于配置服务器-客户端网络事务的各种 GRPC 参数。有关这些选项的用法，请参阅`tritonserver --help`的输出。

#### SSL/TLS

这些选项可用于配置通信的安全通道。服务器端选项包括：

* `--grpc-use-ssl`
* `--grpc-use-ssl-mutual`
* `--grpc-server-cert`
* `--grpc-server-key`
* `--grpc-root-cert`

有关客户端文档，请参阅[客户端GRPC SSL/TLS](https://github.com/triton-inference-server/client/tree/main#ssltls)

关于gRPC中身份验证的更多详细概述，请参考[这里](https://grpc.io/docs/guides/auth/)。

#### 压缩

通过在服务器端公开以下选项，Triton允许在线压缩请求/响应消息:

* `--grpc-infer-response-compression-level`

有关客户端文档，请参见[客户端GRPC压缩](https://github.com/triton-inference-server/client/tree/main#compression-1)

压缩可以用来减少服务器-客户端通信中使用的带宽量。要了解更多细节，请参见[gRPC压缩](https://grpc.github.io/grpc/core/md_doc_compression.html)。

#### GRPC KeepAlive

Triton使用[这里](https://github.com/grpc/grpc/blob/master/doc/keepalive.md)描述的客户端和服务器的默认值，公开GRPC KeepAlive参数。

这些选项可以用来配置KeepAlive设置:

* `--grpc-keepalive-time`
* `--grpc-keepalive-timeout`
* `--grpc-keepalive-permit-without-calls`
* `--grpc-http2-max-pings-without-data`
* `--grpc-http2-min-recv-ping-interval-without-data`
* `--grpc-http2-max-ping-strikes`

有关客户端文档，请参见[客户端 GRPC KeepAlive](https://github.com/triton-inference-server/client/blob/main/README.md#grpc-keepalive)。

## C API

Triton Inference Server提供了一个向后兼容的C API，允许将Triton直接链接到C/ C++应用程序中。这个API记录在[tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h)中。


在[simple.cc](../src/servers/simple.cc)中可以找到一个使用C API的简单示例。在源代码中可以找到一个更复杂的例子，它为Triton实现了HTTP/REST和GRPC端点。这些端点使用C API与Triton的核心进行通信。端点的主要源文件是[grpc_server.cc](../src/servers/grpc_server.cc)和[http_server.cc](../src/servers/http_server.cc)。
