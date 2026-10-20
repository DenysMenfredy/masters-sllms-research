# Android CPU path

Use the Android NDK and ADB for the required path. The host environment does not
include platform-tools or an NDK, and no phone measurement has been made.

1. Install Android SDK platform-tools and one exact NDK version outside this project.
2. Enable USB debugging, authorize the workstation, and confirm `adb devices -l`.
   Set `ANDROID_SERIAL` if more than one device is connected.
3. Run `bash android/inspect_device.sh results/raw/device-inventory-YYYYMMDD-NN`.
4. Review the captured firmware, ABI, and memory, then fill the **measured** section
   of `configs/device/realme_7_pro.yaml`. Keep specification values separate.
5. Check out llama.cpp at an explicitly selected commit under `external/llama.cpp`.
   Record it in the device config, then set `LLAMA_CPP_DIR`, `LLAMA_CPP_COMMIT`, and
   `ANDROID_NDK_HOME`. Run `bash android/build_cpu.sh`.
6. Preserve CMake logs/cache, NDK version, binary checksums, and runtime commit.
   Inspect `llama-cli --help` from that build before constructing inference commands.

The cross-compilation helper follows the official
[Android build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md).
It targets arm64-v8a/API 28, disables GPU backends, and avoids host-specific CPU
instructions. Verify the phone API level before using it. Neither script is a
benchmark runner; JSON constraints, templates, request timing, process RSS/PSS,
timeouts, cache behavior, and actual offload verification still need implementation.

CPU inference is mandatory. Optional Adreno/OpenCL investigation is limited to
one working day. `dumpsys battery` temperature must never be labeled CPU temperature.
