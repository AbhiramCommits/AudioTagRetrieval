# ONNX vs PyTorch

Single-clip inference (`[1, 1, 128, 313]` mel), cnn model, CPU,
mean over 50 runs after warmup (torch with 2 intra-op threads,
onnxruntime 1.30.0 CPU provider).

| engine | mean latency (ms) | artifact |
|---|---|---|
| PyTorch (torch.inference_mode) | 13.54 | artifacts/cnn/best.pt |
| ONNX Runtime | 15.33 | artifacts/cnn/model.onnx |

Max absolute output difference across batch sizes 1-3 and time lengths
150-313 frames: 8.34e-06 (< 1e-4).

The API serves either engine via `AUDIOTAG_BACKEND=torch|onnx`.
