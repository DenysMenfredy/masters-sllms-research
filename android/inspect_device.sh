#!/usr/bin/env bash
# Save raw device observations. Battery temperature is not CPU temperature.
set -euo pipefail

if ! command -v adb >/dev/null 2>&1; then
  echo "adb is missing; install Android SDK platform-tools first." >&2
  exit 1
fi
if [[ $# -ne 1 ]]; then
  echo "Usage: bash android/inspect_device.sh NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
if [[ -e "$1" ]]; then
  echo "Refusing to overwrite an existing observation directory: $1" >&2
  exit 1
fi
adb get-state >/dev/null
mkdir -p -- "$1"
date -u +%Y-%m-%dT%H:%M:%SZ > "$1/collected_at_utc.txt"
adb version > "$1/adb-version.txt"
adb shell getprop > "$1/getprop.txt"
adb shell cat /proc/cpuinfo > "$1/cpuinfo.txt"
adb shell cat /proc/meminfo > "$1/meminfo.txt"
adb shell dumpsys battery > "$1/battery.txt"
if ! adb shell dumpsys thermalservice > "$1/thermalservice.txt" 2> "$1/thermalservice.stderr"; then
  echo "unavailable; see thermalservice.stderr" > "$1/thermal-status.txt"
fi
echo "Saved raw device inventory to $1. Inspect outputs; permission failures are not measurements."
