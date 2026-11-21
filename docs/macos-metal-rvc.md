# macOS arm64 RVC Metal preview

The offline RVC graph can send eligible FP32 projections to Metal Performance Shaders on Apple Silicon. It reuses the native VFRM loader and RVC stages. MatMul and the graph's fused LinearBias form accept contiguous input, immutable rank-two weights and a contiguous output. Both weight layouts use the corresponding MPS transpose setting. Constant weights stay in Metal buffers and projections share a command queue and scratch buffers per graph.

Convolution and transposed convolution, attention, RMVPE STFT and GRU, FAISS retrieval, dynamic MatMul and fused activation or residual paths use the existing native CPU implementations. The VFRM acoustic and voice bundles hold canonical tensors and have no Core ML executable program. MPSMatrixMultiplication accepts this subset directly, avoiding a separate conversion pipeline for every community voice. [Apple MPS matrix multiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)

```sh
cargo build --release --locked -p vocal-fast --features metal-projections
VOCAL_OFFLINE_METAL_PROJECTIONS=1 VOCAL_CPU_PROFILE=1 \
  VOCAL_OFFLINE_TRACE_DIR=/absolute/path/to/trace VOCAL_INDEX_RATIO=0.5 \
  target/release/vocal-fast model run-pipeline cpu complete.vfrm input.wav output.wav
```

Use a complete voice bundle containing Synth, HuBERT and RMVPE, with the FAISS index embedded for a nonzero index ratio. The output summary labels this policy `metal_projections_hybrid`. The per-segment `profile.json` lists projections actually executed by Metal with implementation `metal:mps-matrix-fp32`. A missing Metal device or a failed submitted command returns an explicit error. Ordinary CPU mode remains available in the same CLI.

The arm64 CI builds for macOS 13.0, checks the minimum version of every bundled binary, relocates the package, verifies its signatures, checks linkage against the system Metal frameworks and verifies the executable's relative loader path. It also compiles the numerical GPU test. On an Apple Silicon machine with an accessible Metal device, run the fixture to compare both matrix layouts and bias handling against FP64 and check placement across repeated inputs.

```sh
cargo test --release --locked -p vocal-cpu --features metal-projections \
  --test linear metal_fp32_projection_matches_reference_and_reports_placement -- --ignored
```

The preview remains opt-in pending physical-device operator parity, an indexed WAV run, latency measurement and memory measurement. The current CI packaging evidence covers Mach-O and loader behavior.
