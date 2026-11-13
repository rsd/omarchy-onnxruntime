#!/usr/bin/env bash
# Hermetic tests for tools/ort-smoke and scripts/smoke-test.sh: no ONNX Runtime
# is needed, installed or fetched.
#   - make-model.py: deterministic bytes (pinned sha256), structurally valid
#     (decoded back by its own --check), corruption detected, constants in sync
#     with smoke.c
#   - smoke-test.sh: argument parsing, the missing-ORT error paths, and the
#     compile/run plumbing with a fake compiler and a fake test binary
#   - smoke.c: a syntax check only when an ONNX Runtime header is already on the
#     machine (SMOKE_ORT_INCLUDE, or /usr/include/onnxruntime); the header is not
#     vendored in this repository, so otherwise that check is skipped. The same
#     header builds smoke.c with -DSMOKE_UNIT_TEST (no linking) to test how a
#     setup error is classified: vendor runtime missing / OpenVINO device absent
#     are "unavailable", everything else a failure.
# Usage: scripts/tests/smoke-test-test.sh. Exits non-zero on any failure.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
tooldir=$here/../../tools/ort-smoke
smoke=$here/../smoke-test.sh
python=${PYTHON:-python3}

# Output of make-model.py: update deliberately when the model changes.
want_sha=0ba0fc7b40869ea274935852c01ec7ffefdd2a8cc245c911f26071a8473a922f

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fails=0 count=0
ok() { count=$((count + 1)); echo "ok   $1"; }
fail() { count=$((count + 1)); echo "FAIL $1"; fails=$((fails + 1)); }
skip() { echo "skip $1"; }
expect() { # expect <description> <want> <got>
    if [[ $3 == "$2" ]]; then ok "$1"; else fail "$1: got '$3', want '$2'"; fi
}
contains() { # contains <description> <needle> <haystack>
    if [[ $3 == *"$2"* ]]; then ok "$1"; else fail "$1: '$2' not in: $3"; fi
}
status_of() { # status_of <command...>: the exit status, output discarded
    local s=0
    "$@" >/dev/null 2>&1 || s=$?
    echo "$s"
}

# ---- make-model.py ----------------------------------------------------------

model=$tmp/model.onnx
"$python" "$tooldir/make-model.py" "$model"
"$python" "$tooldir/make-model.py" "$tmp/model2.onnx"
expect "make-model.py writes the pinned bytes" "$want_sha" "$(sha256sum "$model" | cut -d' ' -f1)"
if cmp -s "$model" "$tmp/model2.onnx"; then ok "make-model.py is deterministic"; else fail "make-model.py is deterministic"; fi

summary=$("$python" "$tooldir/make-model.py" --check "$model")
contains "--check accepts the model" "ok: " "$summary"
contains "--check reports the graph" "Conv->Relu->Add" "$summary"

head -c 400 "$model" >"$tmp/truncated.onnx"
expect "--check rejects a truncated model" 1 \
    "$(status_of "$python" "$tooldir/make-model.py" --check "$tmp/truncated.onnx")"
# Flip the first byte of the first float of the weight tensor payload region
# (the file tail holds the A initializer, then the value infos and opset).
"$python" - "$model" "$tmp/corrupt.onnx" <<'EOF'
import sys
b = bytearray(open(sys.argv[1], "rb").read())
b[len(b) // 2] ^= 0xFF
open(sys.argv[2], "wb").write(bytes(b))
EOF
expect "--check rejects a corrupted model" 1 \
    "$(status_of "$python" "$tooldir/make-model.py" --check "$tmp/corrupt.onnx")"
expect "make-model.py without arguments is a usage error" 2 \
    "$(status_of "$python" "$tooldir/make-model.py")"
expect "--check of a missing file fails" 1 \
    "$(status_of "$python" "$tooldir/make-model.py" --check "$tmp/none.onnx")"

# W_O, B_O and A_O must be the same numbers in make-model.py and smoke.c.
py_const() { sed -n "s/^$1 = (\\(.*\\))\$/\\1/p" "$tooldir/make-model.py" | tr -d ' '; }
c_const() { sed -n "s/^static const float $1\\[OUT_C\\] = {\\(.*\\)};\$/\\1/p" "$tooldir/smoke.c" | tr -d ' f'; }
for c in W_O B_O A_O; do
    py=$(py_const "$c") cc=$(c_const "$c")
    if [[ -n $py && $py == "$cc" ]]; then ok "$c matches in make-model.py and smoke.c"; else fail "$c differs: py='$py' c='$cc'"; fi
done

# ---- smoke-test.sh: arguments and the missing-ORT paths ---------------------

contains "--help prints the usage" "Usage: scripts/smoke-test.sh" "$("$smoke" --help)"
expect "--help exits 0" 0 "$(status_of "$smoke" --help)"
expect "an unknown option is a usage error" 2 "$(status_of "$smoke" --frobnicate)"
expect "--only without a value is a usage error" 2 "$(status_of "$smoke" --only)"
expect "--prefix without a value is a usage error" 2 "$(status_of "$smoke" --prefix)"
expect "--runs with a non-number is a usage error" 2 "$(status_of "$smoke" --runs many)"

empty=$tmp/empty
mkdir -p "$empty"
out=$("$smoke" --prefix "$empty" 2>&1 || true)
contains "missing headers: names the problem" "onnxruntime_c_api.h not found" "$out"
contains "missing headers: asks whether the package is installed" "is onnxruntime-full installed?" "$out"
expect "missing headers exit 3" 3 "$(status_of "$smoke" --prefix "$empty")"

fakeprefix=$tmp/prefix
mkdir -p "$fakeprefix/include/onnxruntime" "$fakeprefix/lib"
: >"$fakeprefix/include/onnxruntime/onnxruntime_c_api.h"
# Only the versioned file (what a runtime-only install ships): -lonnxruntime
# needs the unversioned name, so this must be the clear error, not a link error.
touch "$fakeprefix/lib/libonnxruntime.so.1"
out=$("$smoke" --prefix "$fakeprefix" 2>&1 || true)
contains "missing library: names the problem" "libonnxruntime.so not found" "$out"
contains "missing library: asks whether the package is installed" "is onnxruntime-full installed?" "$out"
expect "missing library exits 3" 3 "$(status_of "$smoke" --prefix "$fakeprefix")"
expect "a libdir with only libonnxruntime.so.1 exits 3" 3 "$(status_of "$smoke" --prefix "$fakeprefix")"
rm -f "$fakeprefix/lib/libonnxruntime.so.1"

# ---- smoke-test.sh: compile and run plumbing, with a fake compiler ----------

fakecc=$tmp/fakecc
cat >"$fakecc" <<'EOF'
#!/usr/bin/env bash
# Fake C compiler: records its arguments and writes a fake test binary.
out=''
for ((i = 1; i < $#; i++)); do
    if [[ ${!i} == -o ]]; then j=$((i + 1)); out=${!j}; fi
done
printf '%s\n' "$@" >"$FAKE_LOG.cc"
cat >"$out" <<'INNER'
#!/usr/bin/env bash
printf '%s\n' "$@" >"$FAKE_LOG.run"
printf 'LD=%s\n' "${LD_LIBRARY_PATH-unset}" >>"$FAKE_LOG.run"
printf '%s\n' "$0" >"$FAKE_LOG.bin"
[[ -z ${FAKE_SIGNAL:-} ]] || kill -"$FAKE_SIGNAL" $$
exit "${FAKE_RC:-0}"
INNER
chmod +x "$out"
EOF
chmod +x "$fakecc"
touch "$fakeprefix/lib/libonnxruntime.so"
export FAKE_LOG=$tmp/fake

status=0
CC=$fakecc "$smoke" --prefix "$fakeprefix" --only cuda,openvino --require cpu --runs 3 >/dev/null 2>&1 || status=$?
expect "fake run: exit status 0" 0 "$status"
cc_args=$(cat "$FAKE_LOG.cc")
run_args=$(cat "$FAKE_LOG.run")
contains "compile: C11" "-std=c11" "$cc_args"
contains "compile: header directory from the prefix" "-I$fakeprefix/include/onnxruntime" "$cc_args"
contains "compile: library directory from the prefix" "-L$fakeprefix/lib" "$cc_args"
contains "compile: links onnxruntime" "-lonnxruntime" "$cc_args"
contains "compile: builds smoke.c" "smoke.c" "$cc_args"
contains "run: --only forwarded" $'--only\ncuda,openvino' "$run_args"
contains "run: --require forwarded" $'--require\ncpu' "$run_args"
contains "run: --runs forwarded" $'--runs\n3' "$run_args"
contains "run: LD_LIBRARY_PATH points at the library directory" "LD=$fakeprefix/lib" "$run_args"
first=$(head -n1 "$FAKE_LOG.run")
if [[ -f $first ]]; then fail "the temporary directory is removed after the run"; else ok "the temporary directory is removed after the run"; fi

CC=$fakecc "$smoke" --prefix "$fakeprefix" >/dev/null 2>&1
run_args=$(cat "$FAKE_LOG.run")
if [[ $run_args == *--only* || $run_args == *--require* || $run_args == *--runs* ]]; then
    fail "run: no options forwarded by default: $run_args"
else
    ok "run: no options forwarded by default"
fi

expect "the test binary's exit status is passed through" 1 \
    "$(FAKE_RC=1 CC=$fakecc status_of "$smoke" --prefix "$fakeprefix")"

# A crash of the test program (here SIGSEGV after printing results) is a failure
# of the wrapper with a clear message, not a silent pass-through of 139.
out=$(FAKE_SIGNAL=SEGV CC=$fakecc "$smoke" --prefix "$fakeprefix" 2>&1 || true)
contains "a crash is reported: signal" "killed by signal 11 (SEGV)" "$out"
expect "a crash of the test program exits 1" 1 \
    "$(FAKE_SIGNAL=SEGV CC=$fakecc status_of "$smoke" --prefix "$fakeprefix")"

# A source-tree layout: header under core/session, library in a separate build dir.
tree=$tmp/srctree
mkdir -p "$tree/include/onnxruntime/core/session" "$tmp/build"
: >"$tree/include/onnxruntime/core/session/onnxruntime_c_api.h"
touch "$tmp/build/libonnxruntime.so.1.30.0" "$tmp/build/libonnxruntime.so"
CC=$fakecc "$smoke" --include "$tree/include/onnxruntime" --libdir "$tmp/build" >/dev/null 2>&1
cc_args=$(cat "$FAKE_LOG.cc")
contains "tree layout: finds the header under core/session" "-I$tree/include/onnxruntime/core/session" "$cc_args"
contains "tree layout: uses --libdir" "-L$tmp/build" "$cc_args"
contains "tree layout: LD_LIBRARY_PATH points at --libdir" "LD=$tmp/build" "$(cat "$FAKE_LOG.run")"

CC=$fakecc "$smoke" --prefix "$fakeprefix" --keep >/dev/null 2>"$tmp/keep.err"
kept=$(sed -n 's/^kept: //p' "$tmp/keep.err")
if [[ -n $kept && -f $kept/model.onnx ]]; then ok "--keep keeps the directory with the model"; else fail "--keep: no kept directory in: $(cat "$tmp/keep.err")"; fi
[[ -z $kept ]] || rm -rf "$kept"
expect "the kept model is the pinned one" "$want_sha" \
    "$(CC=$fakecc "$smoke" --prefix "$fakeprefix" --keep 2>&1 >/dev/null | sed -n 's/^kept: //p' | { read -r d; sha256sum "$d/model.onnx" | cut -d' ' -f1; rm -rf "$d"; })"

# ---- smoke.c: syntax check against a header already on the machine ----------

hdr=${SMOKE_ORT_INCLUDE:-/usr/include/onnxruntime}
if [[ ! -f $hdr/onnxruntime_c_api.h && -f $hdr/core/session/onnxruntime_c_api.h ]]; then
    hdr=$hdr/core/session
fi
if [[ -f $hdr/onnxruntime_c_api.h ]]; then
    if cc -std=c11 -Wall -Wextra -Werror -fsyntax-only -I"$hdr" "$tooldir/smoke.c"; then
        ok "smoke.c compiles cleanly against $hdr"
    else
        fail "smoke.c does not compile against $hdr"
    fi

    # The setup-failure classifier, compiled alone (nothing linked).
    classify=$tmp/classify
    if cc -std=c11 -Wall -Wextra -Werror -Wno-unused-function -DSMOKE_UNIT_TEST -I"$hdr" \
        "$tooldir/smoke.c" -lm -o "$classify"; then
        ok "smoke.c classifier test build compiles"
        dl_err="Failed to load library libonnxruntime_providers_cuda.so with error:"
        for t in cuda tensorrt openvino:CPU; do
            expect "$t: missing libcudnn is unavailable" \
                "unavailable: vendor runtime not installed: libcudnn.so.9" \
                "$("$classify" "$t" "append provider: $dl_err libcudnn.so.9: cannot open shared object file: No such file or directory")"
        done
        expect "tensorrt: missing libnvinfer is unavailable" \
            "unavailable: vendor runtime not installed: libnvinfer.so.10" \
            "$("$classify" tensorrt "Failed to load library libonnxruntime_providers_tensorrt.so with error: libnvinfer.so.10: cannot open shared object file: No such file or directory")"
        expect "openvino: missing libopenvino is unavailable" \
            "unavailable: vendor runtime not installed: libopenvino.so.2541" \
            "$("$classify" openvino:GPU "append provider: libopenvino.so.2541: cannot open shared object file: No such file or directory")"
        expect "a missing providers library is a packaging fault: fail" fail \
            "$("$classify" cuda "Failed to load library libonnxruntime_providers_cuda.so with error: libonnxruntime_providers_cuda.so: cannot open shared object file: No such file or directory")"
        expect "a load error that is not a missing file stays a failure" fail \
            "$("$classify" cuda "Failed to load library libonnxruntime_providers_cuda.so with error: undefined symbol: cudnnFoo")"
        expect "a dlopen failure outside a provider load stays a failure" fail \
            "$("$classify" cuda "CreateSession: libfoo.so: cannot open shared object file")"
        expect "a real runtime error stays a failure" fail \
            "$("$classify" cuda "CUDA failure 2: out of memory")"
        expect "cpu never classifies as unavailable" fail \
            "$("$classify" cpu "libcudnn.so.9: cannot open shared object file")"
        for d in GPU NPU CPU; do
            expect "openvino:$d: absent device is unavailable" \
                "unavailable: OpenVINO device $d not present" \
                "$("$classify" "openvino:$d" "append provider: [ERROR] [OpenVINO] Device $d is not available")"
        done
        expect "openvino: another device's absence does not match" fail \
            "$("$classify" openvino:GPU "[ERROR] [OpenVINO] Device NPU is not available")"
        expect "openvino: 'not supported' with 'device' is a failure" fail \
            "$("$classify" openvino:GPU "[OpenVINO] Selected device_type GPU is not supported with precision suffix")"
        expect "openvino: 'not found' with 'device' is a failure" fail \
            "$("$classify" openvino:NPU "Input not found on device NPU")"
        expect "openvino: invalid device_type is a failure" fail \
            "$("$classify" openvino:CPU "[ERROR] [OpenVINO] Invalid device_type is selected. Supported modes are AUTO/HETERO/MULTI")"
        expect "device-missing text does not apply to cuda" fail \
            "$("$classify" cuda "Device CPU is not available")"
    else
        fail "smoke.c classifier test build does not compile against $hdr"
    fi
else
    skip "smoke.c syntax check and classifier tests: no ONNX Runtime header found (set SMOKE_ORT_INCLUDE)"
fi

if ((fails > 0)); then
    echo "$fails of $count check(s) failed"
    exit 1
fi
echo "all $count checks passed"
