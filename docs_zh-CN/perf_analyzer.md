# 性能分析器

优化模型的推理性能的一个关键部分是，当您使用不同的优化策略进行试验时，能够度量性能的变化。perf_analyzer应用程序(以前称为perf_client)为Triton Inference Server执行这个任务。perf_analyzer包含在客户端示例中，这些示例可以[从多个来源获得](https://github.com/triton-inference-server/client#getting-the-client-libraries-and-examples)。

<!-- 95-th percentile  在样本中，95%的个体都比它更低 -->
perf_analyzer应用程序向您的模型生成推理请求，并测量这些请求的吞吐量和延迟。为了获得有代表性的结果，perf_analyzer在一个时间窗口内测量吞吐量和延迟，然后重复测量，直到得到稳定的值。默认情况下，perf_analyzer使用平均延迟来确定稳定性，但您可以使用--percentile标志基于该置信度水平上来稳定结果。例如，如果使用--percentile=95，则使用95-th percentile请求延迟来稳定结果。例如,

```
$ perf_analyzer -m inception_graphdef --percentile=95
*** Measurement Settings ***
  Batch size: 1
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using p95 latency

Request concurrency: 1
  Client:
    Request count: 348
    Throughput: 69.6 infer/sec
    p50 latency: 13936 usec
    p90 latency: 18682 usec
    p95 latency: 19673 usec
    p99 latency: 21859 usec
    Avg HTTP time: 14017 usec (send/recv 200 usec + response wait 13817 usec)
  Server:
    Inference count: 428
    Execution count: 428
    Successful request count: 428
    Avg request latency: 12005 usec (overhead 36 usec + queue 42 usec + compute input 164 usec + compute infer 11748 usec + compute output 15 usec)

Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 69.6 infer/sec, latency 19673 usec
```

## 并发请求

默认情况下，perf_analyzer使用模型上尽可能低的负载来测量模型的延迟和吞吐量。为此，perf_analyzer向Triton发送一个推理请求并等待响应。当收到响应时，perf_analyzer立即发送另一个请求，然后在测量窗口中重复这个过程。未完成的推理请求的数目视为*并发请求*，因此默认情况下perf_analyzer使用并发请求数为1。

使用--concurrency-range \<start\>:\<end\>:\<step\>选项，可以让perf_analyzer收集请求并发级别范围的数据。使用--help选项查看此选项和其他选项的完整文档。例如，要查看从1到4的并发请求值的模型的延迟和吞吐量:

```
$ perf_analyzer -m inception_graphdef --concurrency-range 1:4
*** Measurement Settings ***
  Batch size: 1
  Measurement window: 5000 msec
  Latency limit: 0 msec
  Concurrency limit: 4 concurrent requests
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client:
    Request count: 339
    Throughput: 67.8 infer/sec
    Avg latency: 14710 usec (standard deviation 2539 usec)
    p50 latency: 13665 usec
...
Request concurrency: 4
  Client:
    Request count: 415
    Throughput: 83 infer/sec
    Avg latency: 48064 usec (standard deviation 6412 usec)
    p50 latency: 47975 usec
    p90 latency: 56670 usec
    p95 latency: 59118 usec
    p99 latency: 63609 usec
    Avg HTTP time: 48166 usec (send/recv 264 usec + response wait 47902 usec)
  Server:
    Inference count: 498
    Execution count: 498
    Successful request count: 498
    Avg request latency: 45602 usec (overhead 39 usec + queue 33577 usec + compute input 217 usec + compute infer 11753 usec + compute output 16 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 67.8 infer/sec, latency 14710 usec
Concurrency: 2, throughput: 89.8 infer/sec, latency 22280 usec
Concurrency: 3, throughput: 80.4 infer/sec, latency 37283 usec
Concurrency: 4, throughput: 83 infer/sec, latency 48064 usec
```

## 理解输出

对于每个并发级别请求，perf_analyzer报告从*客户端*看到的延迟和吞吐量(也就是说，是perf_analyzer看到的)，以及服务器上的平均请求延迟。

服务器延迟度量从服务器接收到请求到从服务器发送响应的总时间。由于用于实现服务器端点的HTTP和GRPC库，总的服务器延迟通常对HTTP请求更为精确，因为它测量从接收到的第一个字节到发送的最后一个字节的时间。对于HTTP和GRPC，总的服务器延迟被分解为以下组件:

- *queue*: 请求在推理调度队列中等待模型实例可使用的平均时间。
- *compute*: 执行实际推断所花费的平均时间，包括从或者向GPU复制数据所需的时间。

HTTP和GRPC的客户端延迟时间进一步细分如下:

- HTTP: *send/recv* 表示客户端发送请求和接收响应所花费的时间。 *response wait* 等待服务器响应的时间。
- GRPC: *(un)marshal request/response* 表示将请求数据编组到GRPC protobuf和从GRPC protobuf反编组响应数据所花费的时间。 *response wait* 表示向网络写入GRPC请求、等待响应、从网络读取GRPC响应的时间。

使用verbose (-v)选项perf_analyzer查看更多输出，包括为每个并发级别请求运行的稳定通道。

## 可视化 延时 vs. 吞吐量

perf_analyzer提供 -f选项来生成包含结果的CSV输出的文件。

```
$ perf_analyzer -m inception_graphdef --concurrency-range 1:4 -f perf.csv
$ cat perf.csv
Concurrency,Inferences/Second,Client Send,Network+Server Send/Recv,Server Queue,Server Compute Input,Server Compute Infer,Server Compute Output,Client Recv,p50 latency,p90 latency,p95 latency,p99 latency
1,69.2,225,2148,64,206,11781,19,0,13891,18795,19753,21018
3,84.2,237,1768,21673,209,11742,17,0,35398,43984,47085,51701
4,84.2,279,1604,33669,233,11731,18,1,47045,56545,59225,64886
2,87.2,235,1973,9151,190,11346,17,0,21874,28557,29768,34766
```

注意:CSV文件中的行按照吞吐量的递增顺序排序(推理/秒)。

您可以将CSV文件导入到电子表格中，以帮助可视化延迟与推断/秒的权衡，并查看延迟的一些组件。遵循以下步骤:

- 打开 [电子表格](https://docs.google.com/spreadsheets/d/1S8h0bWBBElHUoLd2SOvQPzZzRiQ55xjyqodm_9ireiw)
- 从文件菜单中"复制"进行复制
- 打开复制文件
- 在"原始数据"选项卡上选择A1单元格
- 从文件菜单找那个选择"导入"
- 选择"上传"开始上传文件Select "Upload" and upload the file
- 选择"替换选中单元格的数据"，并选择"导入数据"按钮

## 输入数据

Use the --help option to see complete documentation for all input
data options. By default perf_analyzer sends random data to all the
inputs of your model. You can select a different input data mode with
the --input-data option:

- *random*: (default) Send random data for each input.
- *zero*: Send zeros for each input.
- directory path: A path to a directory containing a binary file for each input, named the same as the input. Each binary file must contain the data required for that input for a batch-1 request. Each file should contain the raw binary representation of the input in row-major order.
- file path: A path to a JSON file containing data to be used with every inference request. See the "Real Input Data" section for further details. --input-data can be provided multiple times with different file paths to specific multiple JSON files.

For tensors with with STRING/BYTES datatype there are additional
options --string-length and --string-data that may be used in some
cases (see --help for full documentation).

For models that support batching you can use the -b option to indicate
the batch-size of the requests that perf_analyzer should send. For
models with variable-sized inputs you must provide the --shape
argument so that perf_analyzer knows what shape tensors to use. For
example, for a model that has an input called *IMAGE* that has shape [
3, N, M ], where N and M are variable-size dimensions, to tell
perf_analyzer to send batch-size 4 requests of shape [ 3, 224, 224 ]:

```
$ perf_analyzer -m mymodel -b 4 --shape IMAGE:3,224,224
```

## 真实的输入数据

The performance of some models is highly dependent on the data used.
For such cases you can provide data to be used with every inference
request made by analyzer in a JSON file. The perf_analyzer will use
the provided data in a round-robin order when sending inference
requests.

Each entry in the "data" array must specify all input tensors with the
exact size expected by the model from a single batch. The following
example describes data for a model with inputs named, INPUT0 and
INPUT1, shape [4, 4] and data type INT32:

```
  {
    "data" :
     [
        {
          "INPUT0" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "INPUT1" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        },
        {
          "INPUT0" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "INPUT1" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        },
        {
          "INPUT0" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "INPUT1" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        },
        {
          "INPUT0" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "INPUT1" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        ...
      ]
  }
```

Note that the [4, 4] tensor has been flattened in a row-major format
for the inputs. In addition to specifying explicit tensors, you can
also provide Base64 encoded binary data for the tensors. Each data
object must list its data in a row-major order. Binary data must be in
little-endian byte order. The following example highlights how this
can be acheived:

```
  {
    "data" :
     [
        {
          "INPUT0" : {"b64": "YmFzZTY0IGRlY29kZXI="},
          "INPUT1" : {"b64": "YmFzZTY0IGRlY29kZXI="}
        },
        {
          "INPUT0" : {"b64": "YmFzZTY0IGRlY29kZXI="},
          "INPUT1" : {"b64": "YmFzZTY0IGRlY29kZXI="}
        },
        {
          "INPUT0" : {"b64": "YmFzZTY0IGRlY29kZXI="},
          "INPUT1" : {"b64": "YmFzZTY0IGRlY29kZXI="}
        },
        ...
      ]
  }
```

In case of sequence models, multiple data streams can be specified in
the JSON file. Each sequence will get a data stream of its own and the
analyzer will ensure the data from each stream is played back to the
same correlation id. The below example highlights how to specify data
for multiple streams for a sequence model with a single input named
INPUT, shape [1] and data type STRING:

```
  {
    "data" :
      [
        [
          {
            "INPUT" : ["1"]
          },
          {
            "INPUT" : ["2"]
          },
          {
            "INPUT" : ["3"]
          },
          {
            "INPUT" : ["4"]
          }
        ],
        [
          {
            "INPUT" : ["1"]
          },
          {
            "INPUT" : ["1"]
          },
          {
            "INPUT" : ["1"]
          }
        ],
        [
          {
            "INPUT" : ["1"]
          },
          {
            "INPUT" : ["1"]
          }
        ]
      ]
  }
```

The above example describes three data streams with lengths 4, 3 and 2
respectively.  The perf_analyzer will hence produce sequences of
length 4, 3 and 2 in this case.

You can also provide an optional "shape" field to the tensors. This is
especially useful while profiling the models with variable-sized
tensors as input. Additionally note that when providing the "shape" field,
tensor contents must be provided separately in "content" field in row-major
order. The specified shape values will override default input shapes
provided as a command line option (see --shape) for variable-sized inputs.
In the absence of "shape" field, the provided defaults will be used. There
is no need to specify shape as a command line option if all the data steps
provide shape values for variable tensors. Below is an example json file
for a model with single input "INPUT", shape [-1,-1] and data type INT32:

```
  {
    "data" :
     [
        {
          "INPUT" :
                {
                    "content": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "shape": [2,8]
                }
        },
        {
          "INPUT" :
                {
                    "content": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "shape": [8,2]
                }
        },
        {
          "INPUT" :
                {
                    "content": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                }
        },
        {
          "INPUT" :
                {
                    "content": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "shape": [4,4]
                }
        }
        ...
      ]
  }
```

The following is the example to provide contents as base64 string with explicit shapes:

```
{
  "data": [{ 
      "INPUT": {
                 "content": {"b64": "/9j/4AAQSkZ(...)"},
                 "shape": [7964]
               }},
    (...)]
}
```

### 输出验证

When real input data is provided, it is optional to request perf analyzer to
validate the inference output for the input data.

Validation output can be specified in "validation_data" field in the same format
as "data" field for real input. Note that the entries in "validation_data" must
align with "data" for proper mapping. The following example describes validation
data for a model with inputs named, INPUT0 and INPUT1, outputs named, OUTPUT0
and OUTPUT1, all tensors have shape [4, 4] and data type INT32:

```
  {
    "data" :
     [
        {
          "INPUT0" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "INPUT1" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        ...
      ],
    "validation_data" :
     [
        {
          "OUTPUT0" : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "OUTPUT1" : [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        }
        ...
      ]
  }
```

Besides the above example, the validation outputs can be specified in the same
variations described in "real input data" section.

## 共享内存

默认情况下，perf_analyzer通过网络发送输入张量数据并接收输出张量数据。你可以让perf_analyzer使用系统共享内存或CUDA共享内存来通信张量数据。通过使用这些选项，您可以对通过在应用程序中使用共享内存实现的性能进行建模。使用--shared-memory=system使用系统(CPU)共享内存，使用--shared-memory=cuda使用cuda共享内存。

## 通信协议

默认情况下 perf_analyzer 使用 HTTP 与 Triton 通信。GRPC 协议可以使用 -i 选项指定。如果选择了 GRPC，也可以为 GRPC 流指定 --streaming 选项。

### SSL/TLS Support

perf_analyzer 可用于在启用 SSL/TLS 的端点后对 Triton 服务进行基准测试。这些选项可以帮助建立与端点的安全连接并配置服务器。

对于 gRPC，请参阅以下选项：

* `--ssl-grpc-use-ssl`
* `--ssl-grpc-root-certifications-file`
* `--ssl-grpc-private-key-file`
* `--ssl-grpc-certificate-chain-file`

更多详细信息: https://grpc.github.io/grpc/cpp/structgrpc_1_1_ssl_credentials_options.html

[推理协议 gRPC SSL/TLS 部分](inference_protocols.md#ssltls)描述了在 Triton 的 gRPC 端点中配置 SSL/TLS的服务器端选项。

对于 HTTPS，公开了以下选项：

* `--ssl-https-verify-peer`
* `--ssl-https-verify-host`
* `--ssl-https-ca-certificates-file`
* `--ssl-https-client-certificate-file`
* `--ssl-https-client-certificate-type`
* `--ssl-https-private-key-file`
* `--ssl-https-private-key-type`

通过`--help`查看完整的文档。

与 gRPC 不同，Triton 的 HTTP 服务器端点无法配置 SSL/TLS 支持。

注意：仅向 perf_analyzer 提供这些`--ssl-http-*`选项并不能确保在通信中使用 SSL/TLS。如果服务端点上未启用 SSL/TLS，则这些选项无效。向 perf_analyzer 用户公开这些选项的目的是允许他们配置 perf_analyzer 以在启用 SSL/TLS 的端点后对 Triton 服务进行基准测试。换句话说，如果 Triton 在 HTTPS 服务器代理后面运行，那么这些选项将允许 perf_analyzer 通过公开的 HTTPS 代理来分析 Triton。

## 直接通过 C API 对 Triton 进行基准测试

除了使用 HTTP 或 gRPC 服务器端点与 Triton 通信外，perf_analyzer 还允许用户直接使用 C API 对 Triton 进行基准测试。HTTP/gRPC 端点在管道中引入了额外的延迟，这对于在其应用程序中通过 C API 调用 Triton 的用户可能没用。具体来说，此功能可用于对最小Triton 进行基准测试，而不会产生 HTTP/gRPC 通信的额外开销。

### 先决条件
在目标机器上拉取 Triton SDK 和推理服务器容器镜像。由于您需要通过Tritonserver 安装，因此将 perf_analyzer 二进制文件复制到推理服务器容器可能会更容易。

### 需要的参数
使用 --help 选项查看支持的命令行参数的完整列表。默认情况下 perf_analyzer 期望 Triton 实例已经在运行。 您可以使用`--service-kind` 选项配置 C API 模式。此外，您需要使用`--triton-server-directory`选项将 perf_analyzer 指向 Triton 服务器库路径，并使用`--model-repository`选项指向模型存储库路径。如果服务器运行成功，有提示：“server is alive!” perf_analyzer 将照常打印统计信息。示例运行如下所示：
```
perf_analyzer -m graphdef_int32_int32_int32 --service-kind=triton_c_api --triton-server-directory=/opt/tritonserver --model-repository=/workspace/qa/L0_perf_analyzer_capi/models
```

### 不支持的功能
C API 中缺少一些功能。他们是：
1. 异步模式 (`-a`)
2. 使用共享内存模式(`--shared-memory=cuda` or `--shared-memory=system`)
3. 请求速率范围模式
4. 对于其他已知的非工作案例，请参阅[qa/L0_perf_analyzer_capi/test.sh](https://github.com/triton-inference-server/server/blob/main/qa/L0_perf_analyzer_capi/test.sh#L239-L277)
