#!/usr/bin/env bash
# Build a verified, pinned external source checkout; never fetch a moving branch.
set -euo pipefail

: "${ANDROID_NDK_HOME:?Set ANDROID_NDK_HOME to the installed NDK root}"
: "${LLAMA_CPP_DIR:?Set LLAMA_CPP_DIR to the pinned llama.cpp source checkout}"
: "${LLAMA_CPP_COMMIT:?Set LLAMA_CPP_COMMIT to its full 40-character commit}"

if [[ ! "$LLAMA_CPP_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LLAMA_CPP_COMMIT must be a full lowercase 40-character commit." >&2
  exit 2
fi
toolchain="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake"
[[ -f "$toolchain" ]] || { echo "Missing NDK toolchain: $toolchain" >&2; exit 1; }
actual_commit=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)
[[ "$actual_commit" == "$LLAMA_CPP_COMMIT" ]] || { echo "Runtime revision mismatch" >&2; exit 1; }
git -C "$LLAMA_CPP_DIR" diff --quiet HEAD -- || { echo "Runtime source is modified" >&2; exit 1; }

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$project_root/android/build/$LLAMA_CPP_COMMIT"
mkdir -p -- "$build_dir"
printf '%s\n' "$actual_commit" > "$build_dir/runtime-commit.txt"
cat "$ANDROID_NDK_HOME/source.properties" > "$build_dir/ndk-source.properties"
cmake -S "$LLAMA_CPP_DIR" -B "$build_dir" \
  -DCMAKE_TOOLCHAIN_FILE="$toolchain" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF \
  -DLLAMA_OPENSSL=OFF \
  2>&1 | tee "$build_dir/configure.log"
cmake --build "$build_dir" --config Release --parallel "${BUILD_JOBS:-2}" \
  --target llama-cli llama-bench 2>&1 | tee "$build_dir/build.log"
echo "CPU binaries: $build_dir/bin. Phone execution and actual offload remain unverified."
