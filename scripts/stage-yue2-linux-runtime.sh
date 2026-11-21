#!/usr/bin/env bash
set -euo pipefail

: "${TENSORRT_RTX_ROOT:?Point to the verified TensorRT-RTX SDK}"
: "${CARGO_TARGET_DIR:?Point to the Ubuntu 22.04 build output}"
stage=${1:?Pass an unused staging directory}
cuda=${CUDA_PATH:-/usr/local/cuda}
mkdir "$stage"
install -m 755 "$CARGO_TARGET_DIR/release/vocal-fast" "$stage/vocal-fast"
mkdir "$stage/licenses"
for library in libtensorrt_rtx.so.1 libtensorrt_onnxparser_rtx.so.1; do
    install -m 644 "$TENSORRT_RTX_ROOT/lib/$library" "$stage/$library"
done

# Use SONAMEs for DT_NEEDED and runtime-load names for TensorRT's JIT dependencies.
cuda_major=$("$cuda/bin/nvcc" --version | sed -nE 's/.*release ([0-9]+)\..*/\1/p')
case "$cuda_major" in
    12) cufft_major=11 ;;
    13) cufft_major=12 ;;
    *) echo "Unsupported CUDA runtime: $cuda_major" >&2; exit 1 ;;
esac
for library in "libcudart.so.$cuda_major" "libcufft.so.$cufft_major" \
    "libcublas.so.$cuda_major" "libcublasLt.so.$cuda_major" "libnvrtc.so.$cuda_major"; do
    install -m 644 "$cuda/lib64/$library" "$stage/$library"
done
shopt -s nullglob
for source in "$cuda"/lib64/libnvrtc-builtins.so."$cuda_major".*; do
    install -m 644 "$source" "$stage/${source##*/}"
done
for source in "$TENSORRT_RTX_ROOT"/doc/Acknowledgements.txt "$TENSORRT_RTX_ROOT"/doc/README.txt \
    "$cuda"/EULA.txt "$cuda"/LICENSE; do
    if [ -f "$source" ]; then install -m 644 "$source" "$stage/licenses/${source##*/}"; fi
done
install -m 644 LICENSE "$stage/licenses/vocal-fast-rs.txt"
readelf -d "$stage/vocal-fast" | grep -F 'Library runpath: [$ORIGIN]'
