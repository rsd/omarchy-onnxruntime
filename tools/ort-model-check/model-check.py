#!/usr/bin/env python3
"""Run a real ONNX model on every ONNX Runtime execution provider and compare.

For each target the model runs in a session that holds only that execution
provider (the CPU EP fallback is disabled for every target but cpu, so a graph
the provider cannot take fails instead of quietly running on the CPU;
--allow-cpu-fallback lifts that and marks the rows "ok*").  The outputs are
compared with the cpu target's, and creation, first-run and steady-state times
are reported.  See README.md; the target names, the status
words and the "unavailable" classification mirror tools/ort-smoke/smoke.c.

--precision default runs each provider with its own defaults (TF32 on CUDA and
TensorRT, fp16 on the OpenVINO GPU and NPU) and judges those targets with
reduced-precision bounds; --precision fp32 asks every provider that can for
full fp32 arithmetic and judges it with tight bounds.

Exit status: 0 when no available target failed and every --require'd target is
ok; 1 otherwise; 2 on a usage error; 3 when onnxruntime or numpy cannot be
imported.

Only the standard library is imported before the arguments are parsed, so
--help works without onnxruntime or numpy installed.
"""

import argparse
import contextlib
import ctypes
import hashlib
import io
import json
import math
import os
import re
import sys
import time

# (id, provider name as reported by get_available_providers, OpenVINO device).
TARGETS = [
    ("cpu", "CPUExecutionProvider", None),
    ("cuda", "CUDAExecutionProvider", None),
    ("tensorrt", "TensorrtExecutionProvider", None),
    ("openvino:CPU", "OpenVINOExecutionProvider", "CPU"),
    ("openvino:GPU", "OpenVINOExecutionProvider", "GPU"),
    ("openvino:NPU", "OpenVINOExecutionProvider", "NPU"),
]
TARGET_IDS = [t[0] for t in TARGETS]
BY_ID = {t[0]: t for t in TARGETS}

PRECISIONS = ("default", "fp32")

# Bounds against the cpu target, as (arithmetic, max abs error, mean abs error
# or None).  A target passes when every bound holds.
#
# fp32 arithmetic: a tight bound on the maximum error.  Only summation order
# and fused kernels differ from the CPU; measured on the four reference models
# (uniform-noise input, ONNX Runtime 1.29.0) the worst is 2.8e-4 (CUDA with
# use_tf32=0, MODNet at 512x512).
FP32 = ("fp32", 1e-3, None)
# The cpu target is the reference, compared with itself.
EXACT = ("fp32", 0.0, None)
# TF32 (CUDA and TensorRT defaults) and fp16 (OpenVINO GPU and NPU defaults)
# keep a 10-bit mantissa, and a deep network fed uniform noise amplifies that
# rounding: on MODNet at 512x512, seeds 0-3, the measured maximum errors reach
# 9.7e-2 (CUDA), 1.1e-1 (TensorRT), 4.6e-1 (OpenVINO GPU) and 2.9e-1 (NPU) on
# outputs in [0, 1], while the mean errors stay at or below 1.4e-3 (TF32) and
# 5.4e-3 (fp16).  The maximum is therefore not a usable test there; the mean is
# the real check, bounded at about 3.5x (TF32) and 1.9x (fp16) the worst
# measured value, and the maximum keeps a loose bound (about 2x the worst
# measured) against isolated gross outliers.  A wrong result sits above the
# mean bounds: MODNet's output for a different noise input differs by a mean
# of 7.1e-2 (512x512) and 1.9e-2 (256x256), an all-zero output by 2.7e-1.
TF32 = ("TF32", 0.25, 5e-3)
FP16 = ("fp16", 1.0, 1e-2)

# What each target runs in, per --precision mode.  In fp32 mode CUDA gets
# use_tf32=0, TensorRT (which has no option for it in ONNX Runtime 1.29) gets
# TF32 off through NVIDIA_TF32_OVERRIDE=0 in the process environment, and the
# OpenVINO GPU gets INFERENCE_PRECISION_HINT=f32 through load_config (the EP's
# precision=FP32 option was observed to leave the GPU at fp16).  The NPU has no
# fp32 inference (an f32 hint fails to compile), so it stays at fp16.
BOUNDS = {
    "default": {"cpu": EXACT, "cuda": TF32, "tensorrt": TF32, "openvino:CPU": FP32,
                "openvino:GPU": FP16, "openvino:NPU": FP16},
    "fp32": {"cpu": EXACT, "cuda": FP32, "tensorrt": FP32, "openvino:CPU": FP32,
             "openvino:GPU": FP32, "openvino:NPU": FP16},
}
TF32_OVERRIDE = "NVIDIA_TF32_OVERRIDE"
FP32_NOTES = {
    "tensorrt": "TF32 off through %s=0 (the TensorRT EP has no option for it)" % TF32_OVERRIDE,
    "openvino:NPU": "the NPU has no fp32 inference: run at its fp16 default, judged with the fp16 bounds",
}

OK, FAIL, UNAVAILABLE = "ok", "FAIL", "unavailable"
OK_FALLBACK = "ok*"  # ok, but --allow-cpu-fallback let nodes run on the CPU
FALLBACK_NOTE = "CPU fallback allowed: some nodes may run on the CPU"

USAGE_ERROR, IMPORT_ERROR = 2, 3


def expand_targets(text, option):
    """Turn a comma-separated LIST into target ids in table order.

    A token matches an id case-insensitively, or the part of an id before the
    colon ("openvino" selects all three devices).
    """
    chosen = set()
    for tok in (s.strip() for s in text.split(",")):
        if not tok:
            continue
        hits = [
            tid
            for tid in TARGET_IDS
            if tid.lower() == tok.lower() or (":" in tid and tid.split(":")[0].lower() == tok.lower())
        ]
        if not hits:
            raise ValueError("%s: unknown target '%s' (known: cpu, cuda, tensorrt, openvino, %s)" % (
                option, tok, ", ".join(t for t in TARGET_IDS if ":" in t)))
        chosen.update(hits)
    return [tid for tid in TARGET_IDS if tid in chosen]


def parse_size(text):
    m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", text)
    if not m or int(m.group(1)) < 1 or int(m.group(2)) < 1:
        raise argparse.ArgumentTypeError("expected HxW with positive integers, e.g. 192x192, got '%s'" % text)
    return int(m.group(1)), int(m.group(2))


def build_parser():
    p = argparse.ArgumentParser(
        prog="model-check.py",
        description="Run an ONNX model on each ONNX Runtime execution provider, "
        "compare the outputs with the CPU provider's and time the runs.",
        epilog="targets: cpu, cuda, tensorrt, openvino:CPU, openvino:GPU, openvino:NPU "
        "('openvino' selects all three devices).  Exit status: 0 ok, 1 a target "
        "failed or a required target is not ok, 2 usage error, 3 onnxruntime or "
        "numpy cannot be imported.",
    )
    p.add_argument("model", metavar="MODEL.onnx", help="the model to run")
    p.add_argument("--targets", metavar="LIST", default=",".join(TARGET_IDS),
                   help="targets to report, comma-separated (default: all)")
    p.add_argument("--runs", type=int, default=50, metavar="N", help="timed runs per target (default 50)")
    p.add_argument("--warmup", type=int, default=3, metavar="N", help="untimed runs after the first (default 3)")
    p.add_argument("--size", type=parse_size, metavar="HxW",
                   help="height and width for dynamic spatial dimensions of the inputs")
    p.add_argument("--require", metavar="LIST", default="",
                   help="exit 1 unless these targets are ok (comma-separated)")
    p.add_argument("--json", metavar="FILE", dest="json_path", help="also write the results as JSON")
    p.add_argument("--trt-cache", metavar="DIR",
                   help="enable the TensorRT engine cache in DIR (default: cache off)")
    p.add_argument("--allow-cpu-fallback", action="store_true",
                   help="let non-cpu targets leave nodes to the CPU EP (rows are marked ok*)")
    p.add_argument("--precision", choices=PRECISIONS, default="default",
                   help="default: each provider's own arithmetic (TF32 on cuda and tensorrt, fp16 on "
                   "openvino:GPU and NPU), judged with reduced-precision bounds; fp32: request fp32 "
                   "arithmetic where the provider allows it, judged with tight bounds (default: default)")
    p.add_argument("--tol", metavar="TARGET=MAX[:MEAN]", action="append", default=[],
                   help="override the bounds on the maximum and mean absolute error for a target; an "
                   "empty part keeps the default, 'inf' removes a bound (repeatable; a comma-separated "
                   "list is also accepted)")
    p.add_argument("--seed", type=int, default=0, metavar="N", help="seed of the input data (default 0)")
    return p


# ---- inputs ----------------------------------------------------------------

def is_static(dim):
    return isinstance(dim, int) and dim > 0


def resolve_shape(name, shape, size):
    """Concrete shape for an input, or ValueError naming the input and its shape.

    Dynamic dimensions: the batch (first) dimension becomes 1; the spatial
    dimensions come from --size.  For a rank-4 input the layout is taken from
    where the channel dimension (1, 3 or 4) sits: NCHW when axis 1 is static and
    channel-like or the only static candidate, NHWC when axis 3 is.
    """
    shape = list(shape)
    out = list(shape)
    if shape and not is_static(shape[0]):
        out[0] = 1
    dynamic = [i for i in range(1, len(shape)) if not is_static(shape[i])]
    if not dynamic:
        return out

    spatial = None
    if len(shape) == 4:
        if is_static(shape[1]) and (shape[1] in (1, 3, 4) or not is_static(shape[3])):
            spatial = [2, 3]
        elif is_static(shape[3]):
            spatial = [1, 2]
    elif len(dynamic) == 2:
        spatial = dynamic
    if spatial is None or any(i not in spatial for i in dynamic):
        raise ValueError("input '%s' has shape %s: cannot tell which dimensions are spatial"
                         % (name, shape_text(shape)))
    if size is None:
        raise ValueError("input '%s' has shape %s with dynamic spatial dimensions: pass --size HxW"
                         % (name, shape_text(shape)))
    for i, v in zip(spatial, size):
        if not is_static(shape[i]):
            out[i] = v
    return out


def shape_text(shape):
    return "[" + ",".join(str(d) if is_static(d) else (str(d) if d else "?") for d in shape) + "]"


def concrete_text(shape):
    return "x".join(str(d) for d in shape)


# ---- classification (mirrors tools/ort-smoke/smoke.c) -----------------------

def first_line(text):
    for line in str(text).splitlines():
        if any(c.isalnum() for c in line):  # skips the "*****" rules of ORT's banners
            return line.strip().strip("* ") or line.strip()
    return str(text).strip() or "unknown error"


def unavailable_reason(target, message):
    """Why a session-creation error means "not usable on this machine", or None.

    - A dlopen failure ("<lib>: cannot open shared object file") means the
      provider's vendor runtime (CUDA, cuDNN, TensorRT, OpenVINO: optional
      dependencies, while every provider is compiled in) is not installed.
      Other load errors (an undefined symbol, say) stay FAIL.
    - OpenVINO's "Device <name> is not available" (the one message the OpenVINO
      EP in ORT 1.29.0 gives for an absent device, matched exactly, not looser):
      the device is not present.  Any other OpenVINO error is FAIL.
    """
    if target == "cpu":
        return None
    m = re.search(r"(\S+): cannot open shared object file", message)
    if m:
        return "vendor runtime not installed: %s" % m.group(1)
    device = BY_ID[target][2]
    if device and re.search(r"Device %s is not available" % re.escape(device), message, re.IGNORECASE):
        return "OpenVINO device %s not present" % device
    return None


def error_chain(exc):
    """The exception and everything in its __cause__/__context__ chain, outermost first."""
    seen, out, stack = set(), [], [exc]
    while stack:
        e = stack.pop(0)
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        out.append(e)
        stack.extend([e.__cause__, e.__context__])
    return out


def classify_error(target, exc):
    """(unavailable reason or None, ORT's message for the note).

    InferenceSession.__init__ answers a provider failure by retrying with the CPU
    EP, which then fails with "Conflicting session configuration ..." raised from
    the original error, so the message that says what happened sits somewhere in
    the __cause__/__context__ chain; every message in it is classified, and the
    note quotes the innermost (the original) one.
    """
    messages = [str(e) for e in error_chain(exc)]
    for message in messages:
        reason = unavailable_reason(target, message)
        if reason:
            return reason, message
    return None, messages[-1]


def preload_provider(ort, target):
    """Load the provider library so a missing vendor library is reported by dlopen.

    The provider bridge only logs a dlopen failure and the session then fails
    with its nodes assigned to the CPU EP, which says nothing about the cause.
    Returns None when the library loads (or the target is cpu), else a
    (status, note) pair: unavailable when the provider library does not exist
    ("provider not built") or a vendor library it needs is missing, FAIL for any
    other load error.
    """
    if target == "cpu":
        return None
    capi = os.path.join(os.path.dirname(os.path.abspath(ort.__file__)), "capi")
    shared = os.path.join(capi, "libonnxruntime_providers_shared.so")
    lib = os.path.join(capi, "libonnxruntime_providers_%s.so" % target.split(":")[0])
    if not os.path.exists(lib):
        return UNAVAILABLE, "provider not built: %s does not exist" % lib
    loads = [(lib, ctypes.RTLD_LOCAL)]
    if os.path.exists(shared):
        loads.insert(0, (shared, ctypes.RTLD_GLOBAL))
    for path, mode in loads:
        try:
            ctypes.CDLL(path, mode=mode)
        except OSError as e:
            reason = unavailable_reason(target, str(e))
            if reason:
                return UNAVAILABLE, "%s (dlopen said: %s)" % (reason, first_line(e))
            return FAIL, "loading %s: %s" % (os.path.basename(path), first_line(e))
    return None


# ---- running ---------------------------------------------------------------

def provider_config(target, trt_cache, precision="default"):
    _, provider, device = BY_ID[target]
    fp32 = precision == "fp32"
    if target == "cuda":
        opts = {"device_id": "0"}
        if fp32:
            opts["use_tf32"] = "0"
        return [(provider, opts)]
    if target == "tensorrt":
        opts = {"trt_fp16_enable": "False"}
        if trt_cache:
            opts.update({"trt_engine_cache_enable": "True", "trt_engine_cache_path": trt_cache})
        return [(provider, opts)]
    if device:
        opts = {"device_type": device}
        if fp32 and device == "GPU":
            opts["load_config"] = json.dumps({"GPU": {"INFERENCE_PRECISION_HINT": "f32"}})
        return [(provider, opts)]
    return [provider]


def make_session(ort, model_path, target, trt_cache, allow_cpu_fallback=False, precision="default"):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if target != "cpu" and not allow_cpu_fallback:
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    sess = ort.InferenceSession(model_path, sess_options=so, providers=provider_config(target, trt_cache, precision))
    if hasattr(sess, "disable_fallback"):
        sess.disable_fallback()  # a failing run must not retry on the CPU behind our back
    return sess


def percentile95(sorted_ms):
    return sorted_ms[max(0, math.ceil(0.95 * len(sorted_ms)) - 1)]


def timed_run(sess, feeds, n_warmup, n_runs, result):
    t0 = time.perf_counter()
    outputs = sess.run(None, feeds)
    result["first_ms"] = (time.perf_counter() - t0) * 1e3
    for _ in range(n_warmup):
        sess.run(None, feeds)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1e3)
    if times:
        times.sort()
        result["mean_ms"] = sum(times) / len(times)
        n = len(times)
        result["median_ms"] = times[n // 2] if n % 2 else (times[n // 2 - 1] + times[n // 2]) / 2
        result["p95_ms"] = percentile95(times)
    return outputs


def compare(np, outputs, reference):
    """(max abs error, mean abs error, problem text or None, outputs not compared).

    Floating-point tensors are compared by absolute error (the tolerance applies
    to those only); integer, bool and string tensors must match exactly; outputs
    that are not tensors (sequences, maps) are not compared and are listed.
    """
    if len(outputs) != len(reference):
        return None, None, "%d outputs, the cpu target has %d" % (len(outputs), len(reference)), []
    worst, total, count, skipped = 0.0, 0.0, 0, []
    for i, (got, ref) in enumerate(zip(outputs, reference)):
        if not isinstance(got, np.ndarray) or not isinstance(ref, np.ndarray):
            skipped.append(i)
            continue
        if got.shape != ref.shape:
            return None, None, "output %d has shape %s, the cpu target has %s" % (i, got.shape, ref.shape), skipped
        if got.dtype != ref.dtype:
            return None, None, "output %d has type %s, the cpu target has %s" % (i, got.dtype, ref.dtype), skipped
        if got.dtype.kind != "f":
            if not np.array_equal(got, ref):
                try:
                    differ = int((got != ref).sum())
                except Exception:  # noqa: BLE001
                    differ = got.size
                return None, None, "output %d (%s) differs from the cpu target in %d of %d elements" % (
                    i, got.dtype, differ, got.size), skipped
            continue
        got, ref = got.astype(np.float64), ref.astype(np.float64)
        if np.isnan(got).any():
            return None, None, "NaN in output %d" % i, skipped
        if np.isnan(ref).any():
            return None, None, "NaN in the cpu reference, output %d" % i, skipped
        with np.errstate(invalid="ignore"):  # inf - inf is nan, masked by got == ref
            diff = np.where(got == ref, 0.0, np.abs(got - ref))
        if diff.size:
            worst = max(worst, float(diff.max()))
            total += float(diff.sum())
            count += diff.size
    return worst, (total / count if count else 0.0), None, skipped


def run_target(ort, np, args, target, model_path, feeds, reference, bounds, available):
    arith, max_tol, mean_tol = bounds
    r = {"target": target, "status": FAIL, "precision": arith, "max_tolerance": max_tol,
         "mean_tolerance": mean_tol, "note": None, "providers": None,
         "create_ms": None, "first_ms": None, "mean_ms": None, "median_ms": None, "p95_ms": None,
         "max_err": None, "mean_err": None}
    outputs = None
    provider = BY_ID[target][1]
    if provider not in available:
        r["status"] = UNAVAILABLE
        r["note"] = "%s is not in the available providers" % provider
        return r, None

    sess = None
    printed = io.StringIO()  # InferenceSession prints its "EP Error ... Falling back" banner to stdout
    t0 = time.perf_counter()
    gate = preload_provider(ort, target)
    if gate:
        r["status"], r["note"] = gate
        return r, None
    try:
        with contextlib.redirect_stdout(printed):
            sess = make_session(ort, model_path, target, args.trt_cache, args.allow_cpu_fallback, args.precision)
        r["create_ms"] = (time.perf_counter() - t0) * 1e3
    except Exception as e:  # noqa: BLE001 - every ORT error is reported
        r["create_ms"] = None
        reason, message = classify_error(target, e)
        if reason:
            r["status"] = UNAVAILABLE
            r["note"] = "%s (ORT said: %s)" % (reason, first_line(message))
        else:
            r["note"] = "session creation: " + first_line(message)
        return r, None

    notes = [FP32_NOTES[target]] if args.precision == "fp32" and target in FP32_NOTES else []
    try:
        r["providers"] = list(sess.get_providers())
        if not r["providers"] or r["providers"][0] != provider:
            r["note"] = "session providers are %s, not %s first" % (" ".join(r["providers"]) or "none", provider)
            if printed.getvalue().strip():
                r["note"] += " (ORT printed: %s)" % first_line(printed.getvalue())
            return r, None
        outputs = timed_run(sess, feeds, args.warmup, args.runs, r)
    except Exception as e:  # noqa: BLE001
        r["note"] = "run: " + first_line(e)
        return r, None
    finally:
        del sess

    if reference is None and target != "cpu":
        r["note"] = "no cpu reference to compare with"
        return r, outputs
    worst, mean, problem, skipped = compare(np, outputs, outputs if target == "cpu" else reference)
    r["max_err"], r["mean_err"] = worst, mean
    if skipped:
        notes.append("output%s %s not compared (not a tensor)" % ("s" if len(skipped) > 1 else "", ", ".join(map(str, skipped))))
    exceeded = []
    if not problem and worst > max_tol:
        exceeded.append("max error %.2e exceeds tolerance %.2e" % (worst, max_tol))
    if not problem and mean_tol is not None and mean > mean_tol:
        exceeded.append("mean error %.2e exceeds tolerance %.2e" % (mean, mean_tol))
    if problem:
        r["note"] = "; ".join([problem] + notes)
    elif exceeded:
        r["note"] = "; ".join(exceeded + notes)
    else:
        r["status"] = OK
        if target != "cpu" and args.allow_cpu_fallback:
            r["status"] = OK_FALLBACK
            notes.insert(0, FALLBACK_NOTE)
        r["note"] = "; ".join(notes) or None
    return r, outputs


# ---- output ----------------------------------------------------------------

def fmt(v, spec):
    return "-" if v is None else spec % v


def render_table(results):
    lines = ["%-14s %-12s %-9s %10s %10s %10s %10s %10s %10s %10s" % (
        "target", "status", "precision", "create ms", "first ms", "mean ms", "median ms", "p95 ms",
        "max err", "mean err")]
    for r in results:
        lines.append("%-14s %-12s %-9s %10s %10s %10s %10s %10s %10s %10s" % (
            r["target"], r["status"], r["precision"], fmt(r["create_ms"], "%.1f"), fmt(r["first_ms"], "%.2f"),
            fmt(r["mean_ms"], "%.2f"), fmt(r["median_ms"], "%.2f"), fmt(r["p95_ms"], "%.2f"),
            fmt(r["max_err"], "%.2e"), fmt(r["mean_err"], "%.2e")))
    for r in results:
        if r["note"]:
            lines.append("    note: %s: %s" % (r["target"], r["note"]))
    return lines


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_or_none(v):
    return v if not isinstance(v, float) or math.isfinite(v) else None


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.runs < 0 or args.warmup < 0:
        parser.error("--runs and --warmup must not be negative")
    if not os.path.isfile(args.model):
        parser.error("model not found: %s" % args.model)
    try:
        reported = expand_targets(args.targets, "--targets")
        required = expand_targets(args.require, "--require") if args.require else []
        bounds = {tid: list(b) for tid, b in BOUNDS[args.precision].items()}
        for item in (s for entry in args.tol for s in entry.split(",") if s.strip()):
            name, sep, value = item.partition("=")
            if not sep:
                raise ValueError("--tol: expected TARGET=MAX[:MEAN], got '%s'" % item)
            ids = expand_targets(name, "--tol")
            parts = value.split(":")
            if len(parts) > 2 or not any(x.strip() for x in parts):
                raise ValueError("--tol: expected TARGET=MAX[:MEAN], got '%s'" % item)
            for slot, text in zip((1, 2), parts):
                if not text.strip():
                    continue  # an empty part keeps the default
                try:
                    v = float(text)
                except ValueError:
                    raise ValueError("--tol: '%s' is not a number in '%s'" % (text, item)) from None
                if not v >= 0:
                    raise ValueError("--tol: tolerance must be >= 0 in '%s'" % item)
                for tid in ids:
                    bounds[tid][slot] = None if slot == 2 and math.isinf(v) else v
    except ValueError as e:
        parser.error(str(e))

    # TensorRT has no TF32 option in ONNX Runtime 1.29; cuBLAS, cuDNN and
    # TensorRT read this variable when they initialize, so it is set before
    # onnxruntime (and through it any CUDA library) is loaded.
    inherited_override = os.environ.get(TF32_OVERRIDE)
    if args.precision == "fp32":
        os.environ[TF32_OVERRIDE] = "0"

    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as e:  # noqa: BLE001 - a broken install fails the same way
        print("model-check: cannot import onnxruntime and numpy: %s" % e, file=sys.stderr)
        return IMPORT_ERROR

    ort.set_default_logger_severity(3)
    available = list(ort.get_available_providers())
    rc = 0
    for tid in required:
        if tid not in reported:
            print("model-check: required target %s is not in --targets" % tid, file=sys.stderr)
            rc = 1

    # cpu runs first and is the reference, requested or not.
    to_run = ["cpu"] + [t for t in TARGET_IDS if t in reported and t != "cpu"]
    model_path = args.model

    # Input description comes from the cpu session, which every model must have.
    try:
        meta_sess = make_session(ort, model_path, "cpu", None)
        inputs_meta = [(i.name, list(i.shape), i.type) for i in meta_sess.get_inputs()]
        del meta_sess
    except Exception as e:  # noqa: BLE001
        print("model-check: the CPU provider cannot load %s: %s" % (args.model, first_line(e)), file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    feeds, inputs_desc = {}, []
    for name, shape, typ in inputs_meta:
        dtypes = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
        if typ not in dtypes:
            print("model-check: input '%s' has type %s: only float and float16 inputs are supported"
                  % (name, typ), file=sys.stderr)
            return USAGE_ERROR
        try:
            concrete = resolve_shape(name, shape, args.size)
        except ValueError as e:
            print("model-check: %s" % e, file=sys.stderr)
            return USAGE_ERROR
        data = rng.random(concrete, dtype=np.float32).astype(dtypes[typ])
        feeds[name] = data
        inputs_desc.append({"name": name, "shape": concrete, "dtype": str(data.dtype)})

    sha = sha256_of(model_path)
    print("ONNX Runtime %s; model %s (%s); inputs %s; available providers %s" % (
        ort.__version__, os.path.basename(model_path), sha[:12],
        ", ".join("%s %s %s" % (d["name"], concrete_text(d["shape"]), d["dtype"]) for d in inputs_desc),
        " ".join(available)))
    if args.precision == "fp32":
        print("precision fp32: fp32 arithmetic requested where the provider allows it, tight bounds")
    else:
        print("precision default: each provider's own arithmetic, reduced-precision bounds for TF32 and fp16")
    if args.precision == "default" and inherited_override is not None:
        print("%s=%s is set in the environment: it may turn TF32 off on cuda and tensorrt"
              % (TF32_OVERRIDE, inherited_override))
    print("%d warm-up and %d timed run(s) per target; CPU EP fallback %s for non-cpu targets\n"
          % (args.warmup, args.runs, "allowed" if args.allow_cpu_fallback else "disabled"))
    if args.trt_cache:
        os.makedirs(args.trt_cache, exist_ok=True)

    results, reference = [], None
    for tid in to_run:
        r, outputs = run_target(ort, np, args, tid, model_path, feeds, reference, tuple(bounds[tid]), available)
        if tid == "cpu":
            reference = outputs
            if outputs is None and r["status"] == FAIL:
                print("model-check: no cpu reference, so outputs cannot be compared", file=sys.stderr)
        if tid in reported:
            results.append(r)

    for line in render_table(results):
        print(line)

    for r in results:
        if r["status"] == FAIL or (r["target"] in required and r["status"] not in (OK, OK_FALLBACK)):
            rc = 1

    if args.json_path:
        doc = {
            "onnxruntime": ort.__version__,
            "model": {"name": os.path.basename(model_path), "sha256": sha},
            "inputs": inputs_desc,
            "available_providers": available,
            "warmup": args.warmup,
            "runs": args.runs,
            "seed": args.seed,
            "precision": args.precision,
            "allow_cpu_fallback": args.allow_cpu_fallback,
            "targets": [{k: finite_or_none(v) for k, v in r.items()} for r in results],
            "exit": rc,
        }
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, allow_nan=False)
            f.write("\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
