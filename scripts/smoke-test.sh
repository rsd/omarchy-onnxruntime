#!/usr/bin/env bash
# Smoke test for an installed ONNX Runtime (package onnxruntime-core): lists the
# available execution providers and runs a tiny model on each engine/device.
#
# Usage: scripts/smoke-test.sh [--prefix DIR] [--include DIR] [--libdir DIR]
#                              [--require LIST] [--only LIST] [--runs N] [--keep]
#
#   --prefix DIR    installation prefix (default /usr): headers in
#                   DIR/include/onnxruntime, libraries in DIR/lib
#   --include DIR   directory holding onnxruntime_c_api.h, overriding --prefix.
#                   A source/build tree also works: DIR/include/onnxruntime
#                   (the script looks in core/session/ below it)
#   --libdir DIR    directory holding libonnxruntime.so, overriding --prefix
#                   (e.g. a build tree's build/Linux/Release)
#   --require LIST  targets that must be "ok" (also "ok (no device)"), else exit 1
#                   (default: none)
#   --only LIST     test only these targets (default: all but migraphx-load)
#   --runs N        warm runs per target (default 5)
#   --keep          keep the temporary build directory and print its path
#
# Targets (LIST is comma-separated): cpu, cuda, tensorrt, openvino:CPU,
# openvino:GPU, openvino:NPU ("openvino" selects all three devices), dnnl
# (oneDNN, on the CPU), webgpu:nvidia, webgpu:intel ("webgpu" selects both,
# see below) and migraphx-load (AMD GPU, see below).
#
# The WebGPU targets register the plugin libonnxruntime_providers_webgpu.so
# found next to libonnxruntime.so and run on the GPU of the named vendor.  The
# GPU is steered by Dawn's power preference, not by the device ORT is given:
# webgpu:nvidia asks for high-performance (discrete GPUs first), webgpu:intel
# for low-power (integrated GPUs first).  A target is "unavailable" when the
# plugin is not installed or offers no GPU of that vendor.  A "device:" line
# names the GPU and what the kernel's per-process usage counters say: work
# counted only on another GPU, an Intel GPU this process never opened (no
# Intel Vulkan driver), or one whose counters did not grow during the runs
# ("work not seen": Mesa keeps the Intel GPU open after listing the GPUs), is a
# FAIL; the NVIDIA driver keeps no such counters, so there the GPU is inferred.  For an acceptance check run each WebGPU target
# alone (--require webgpu:intel --only webgpu:intel), so that its result cannot
# depend on a Dawn device left over from an earlier target in the same process.
#
# A target whose provider or vendor runtime is not installed is reported as
# "unavailable", which is not a failure unless it is in --require.  The default
# set is every target except migraphx-load, which runs only when named in --only
# or --require: it needs an AMD GPU, and on a machine with a broken HIP runtime
# it may abort the process.  It appends the MIGraphX provider by name and creates
# the session.  It is "unavailable" when libonnxruntime_providers_migraphx.so is
# not next to libonnxruntime.so (the core never lists MIGraphX among the
# available providers, so that list is not consulted) or when libmigraphx_c,
# libamdhip64 or libhsa is missing.  Without a device the provider fails with a
# HIP no-device error ("HIP failure 100: ... expr=hipGetDeviceCount",
# "no ROCm-capable device", hipErrorNoDevice, or "Invalid device ID: 0 ... and 0
# (exclusive)"): that is the pass "ok (no device)", the message is shown.  Any
# other error, including a HIP failure of another code (35, a broken driver) or call (hipMalloc), ("not
# supported in this build", an undefined symbol, ...) is a FAIL.  With a device the model runs and
# is checked ("ok").  A crash fails too, and then this script exits 1 naming the
# signal.
#
# Environment: CC selects the C compiler (default cc), PYTHON the interpreter
# that writes the model (default python3).
#
# Exit status: 0 success, 1 a test failed (or a required target is not ok, or
# the test program was killed by a signal), 2 usage error, 3 missing ONNX Runtime, compiler or Python.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
tool=$here/../tools/ort-smoke
cc=${CC:-cc}
python=${PYTHON:-python3}

prefix=/usr include='' libdir='' require='' only='' runs='' keep=0

die() { # die <status> <message>
    echo "smoke-test.sh: $2" >&2
    exit "$1"
}
usage() { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; }
need_value() { # need_value <option> <remaining arg count>
    (($2 >= 2)) || die 2 "$1 needs a value"
}

while (($# > 0)); do
    case $1 in
    -h | --help) usage; exit 0 ;;
    --prefix) need_value "$1" $#; prefix=$2; shift 2 ;;
    --include) need_value "$1" $#; include=$2; shift 2 ;;
    --libdir) need_value "$1" $#; libdir=$2; shift 2 ;;
    --require) need_value "$1" $#; require=$2; shift 2 ;;
    --only) need_value "$1" $#; only=$2; shift 2 ;;
    --runs) need_value "$1" $#; runs=$2; shift 2 ;;
    --keep) keep=1; shift ;;
    *) die 2 "unknown argument: $1 (see --help)" ;;
    esac
done
[[ -z $runs || $runs =~ ^[0-9]+$ ]] || die 2 "--runs needs a non-negative integer, got '$runs'"

include=${include:-$prefix/include/onnxruntime}
libdir=${libdir:-$prefix/lib}

# A source tree keeps the header under core/session/.
if [[ ! -f $include/onnxruntime_c_api.h && -f $include/core/session/onnxruntime_c_api.h ]]; then
    include=$include/core/session
fi
[[ -f $include/onnxruntime_c_api.h ]] ||
    die 3 "onnxruntime_c_api.h not found in $include: is onnxruntime-core installed? (see --prefix / --include)"
[[ -e $libdir/libonnxruntime.so ]] ||
    die 3 "libonnxruntime.so not found in $libdir: is onnxruntime-core installed? (see --prefix / --libdir)"
# ORT resolves a relative provider path against libonnxruntime's own directory,
# so every path the target builds from these options must be absolute.
include=$(realpath -sm -- "$include")
libdir=$(realpath -sm -- "$libdir")
command -v "$cc" >/dev/null || die 3 "C compiler '$cc' not found (set CC)"
command -v "$python" >/dev/null || die 3 "'$python' not found (set PYTHON)"

tmp=$(mktemp -d)
if ((keep)); then
    trap 'echo "kept: $tmp" >&2' EXIT
else
    trap 'rm -rf "$tmp"' EXIT
fi

"$python" "$tool/make-model.py" "$tmp/model.onnx"
"$cc" -std=c11 -O1 -Wall -Wextra -o "$tmp/smoke" "$tool/smoke.c" \
    -I"$include" -L"$libdir" -lonnxruntime -lm

args=("$tmp/model.onnx")
[[ -z $only ]] || args+=(--only "$only")
[[ -z $require ]] || args+=(--require "$require")
[[ -z $runs ]] || args+=(--runs "$runs")

# The system library path already finds an installed ORT; anything else needs help.
case $libdir in
/usr/lib | /usr/lib64) ;;
*) export LD_LIBRARY_PATH=$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} ;;
esac

status=0
"$tmp/smoke" "${args[@]}" || status=$?
# A status above 128 means the program died from a signal (a crash), not that
# it reported a result: say so and fail with the documented status 1.
if ((status > 128)); then
    signal=$(kill -l "$status" 2>/dev/null || echo "$((status - 128))")
    die 1 "the test program was killed by signal $((status - 128)) ($signal), exit status $status: the results above are not trustworthy"
fi
exit "$status"
