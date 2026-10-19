# Models known to work

What has been served with Furnace, and how far it was checked. Tried a model? Add a row with a
pull request, including failures: they show what to fix, here or upstream.

**Checked** levels: *outputs match* = same outputs as a reference engine or as calling `forward`
directly; *serves* = builds and answers requests; *builds* = compiles into the server, outputs
not compared yet; *fails* = with the reason.

## Burn models (library API)

| Model | Tensors | Checked | Where |
|---|---|---|---|
| CNN, 28x28 images | 1 FP32 in, 1 out | outputs match `forward`, CPU and wgpu | [`examples/serve_burn_model`](../examples/serve_burn_model/models.rs) |
| Text classifier (embedding + mean + linear) | 2 INT64 in, 1 FP32 out | outputs match `forward`, CPU and wgpu | [`examples/serve_burn_model`](../examples/serve_burn_model/models.rs) |

## ONNX models (built in at build time)

From the [ONNX Model Zoo](https://github.com/onnx/models) unless noted.

| Model | File | Tensors | Checked |
|---|---|---|---|
| ResNet-18 | `resnet18-v1-7.onnx` | 1 FP32 in, 1 out | outputs match ONNX Runtime (same top class and logit) |
| MNIST | `mnist-8.onnx` | 1 FP32 in, 1 out | serves |
| MobileNetV2 | `mobilenetv2-12.onnx` | 1 FP32 in, 1 out | builds |
| EfficientNet-Lite4 | `efficientnet-lite4-11.onnx` | 1 FP32 in (NHWC), 1 out | builds |
| Tiny YOLOv2 | `tinyyolov2-8.onnx` | 1 FP32 in, 1 out | builds |
| YOLOv4 | `yolov4.onnx` | 1 FP32 in (NHWC), 3 out | builds |
| BERT-SQuAD | `bertsquad-12.onnx` | 4 INT64 in, 3 out | fails: the code burn-onnx generates for its `OneHot` node does not compile (the result's rank is not annotated) |

A model's I/O must be `float32`, `int64` or `int32` tensors with the batch first; see the
[README](../README.md#onnx-models). When burn-onnx cannot convert a graph,
[onnx-simplifier](https://github.com/daquexian/onnx-simplifier) often helps, and the failure
is worth reporting to [burn-onnx](https://github.com/tracel-ai/burn-onnx).
