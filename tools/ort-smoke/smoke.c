/*
 * ort-smoke: list the ONNX Runtime execution providers and run a tiny model
 * on each engine/device, checking the numbers.
 *
 *   smoke MODEL.onnx [--only LIST] [--require LIST] [--runs N] [--log-level N]
 *
 * MODEL.onnx is the model written by make-model.py.  LIST is a comma-separated
 * set of targets: cpu, cuda, tensorrt, openvino:CPU, openvino:GPU,
 * openvino:NPU, dnnl, webgpu:nvidia, webgpu:intel, migraphx-load
 * (case-insensitive; "openvino" selects all three OpenVINO devices, "webgpu"
 * both WebGPU GPUs).
 *
 * The WebGPU provider is a plugin library (libonnxruntime_providers_webgpu.so,
 * a separate package): the main library neither lists it nor appends it by
 * name.  The program registers it from next to libonnxruntime.so, asks for its
 * devices and appends it for the GPU of the target's vendor.  In ORT 1.29.0
 * that device does not choose the GPU: the provider ignores it and Dawn takes
 * the first adapter in its order for the requested power preference, discrete
 * GPUs first for "high-performance" (the provider's default) and integrated
 * ones first for "low-power".  So webgpu:nvidia asks for high-performance and
 * webgpu:intel for low-power, which picks the right GPU on a machine with an
 * NVIDIA card and an Intel integrated GPU.  A target is "unavailable" when the
 * provider library is not installed or when it reports no GPU of that vendor.
 * Since the API does not say which adapter Dawn took, the kernel's per-process
 * GPU usage counters (/proc/self/fdinfo) are read after the runs: work
 * counted on another GPU and none on the target's is a failure, and so is an
 * Intel GPU whose device file this process never opened (its Vulkan driver is
 * missing, so the work went elsewhere) and a GPU whose driver keeps counters
 * but counted no work during the runs (Mesa keeps the Intel device file open
 * after listing the GPUs, so an open file alone does not show the work ran
 * there).  The NVIDIA driver publishes no such counters, so for it the line
 * says the adapter is inferred.
 *
 * A target is "unavailable" (not a failure) when its provider is not in the
 * library, when the provider's vendor runtime is not installed (CUDA, cuDNN,
 * TensorRT, OpenVINO, oneDNN and MIGraphX are optional dependencies, so every
 * provider is compiled in), when the oneDNN, WebGPU or MIGraphX provider
 * library (libonnxruntime_providers_dnnl.so / _webgpu.so / _migraphx.so,
 * separate packages) is not installed, or when OpenVINO reports the device
 * absent.
 *
 * migraphx-load is the one target nobody can be sure to run: it needs an AMD
 * GPU.  It appends the MIGraphX provider by name and creates the session.  The
 * main library does not list MIGraphX among its available providers (the
 * provider is loaded on demand), so the usual "provider is listed" check does
 * not apply: the target is "unavailable" when libonnxruntime_providers_migraphx.so
 * is not installed next to libonnxruntime.so, or when libmigraphx_c, libamdhip64
 * or libhsa is missing.  Without a device the provider's own option parsing
 * fails with "HIP failure 100: ... expr=hipGetDeviceCount(...)" (hipErrorNoDevice,
 * "no ROCm-capable device is detected"), or with "Invalid device
 * ID: 0, must be between 0 (inclusive) and 0 (exclusive)"; only those signatures
 * are the pass "ok (no device)" (the message follows).  A HIP failure of any
 * other code (e.g. 35, a broken driver) or call (hipMalloc, hipSetDevice, ...) is a failure.
 * Any other error, an old library ("Unknown provider name"), a library built
 * without MIGraphX ("not supported in this build"), an undefined symbol in the
 * provider library (soname or ABI drift) and the like, is a failure.  With a
 * device the model runs and is checked like any other target ("ok").  A crash
 * is not caught here: the wrapper reports the signal.  Because a broken HIP
 * runtime may abort the process, migraphx-load is not in the default set of
 * targets: it runs only when named in --only or --require.
 *
 * Exit status: 0 when every selected target that is available passed and every
 * --require'd target is ok; 1 otherwise; 2 on a usage error; 3 when the ONNX
 * Runtime API cannot be initialised.
 *
 * Only the stable C API is used (no C++ API), so the program builds against
 * any ONNX Runtime from 1.17 on; at run time it asks the library for the newest
 * API table both sides understand.
 *
 * Build:  cc -std=c11 smoke.c -I<dir with onnxruntime_c_api.h> -lonnxruntime -lm
 */
#define _GNU_SOURCE /* dladdr, for the location of libonnxruntime.so */

#include <onnxruntime_c_api.h>

#include <ctype.h>
#include <dirent.h>
#ifndef SMOKE_UNIT_TEST
#include <dlfcn.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

/* Oldest API table we accept: SessionOptionsAppendExecutionProvider_OpenVINO_V2
 * (the newest entry point used here) appeared in ORT 1.17. */
#define MIN_API_VERSION 17
/* RegisterExecutionProviderLibrary, GetEpDevices and
 * SessionOptionsAppendExecutionProvider_V2, which the WebGPU targets need,
 * appeared in ORT 1.22: an older library makes those targets unavailable. */
#define EP_LIBRARY_API_VERSION 22

/*
 * The model, see make-model.py:  Y = Add(Relu(Conv3x3(X, W, B)), A)
 *   X [1,3,16,16] -> Y [1,4,16,16], W[o,:,:,:] = W_O[o], pads 1.
 * W_O, B_O and A_O must match make-model.py (a hermetic test compares them).
 * The input pattern and weights are multiples of 1/8, so every product and
 * partial sum is a multiple of 1/64 below 32: exact in fp32 and in fp16.
 * The expected output is therefore computed here in plain double precision.
 */
enum { IN_C = 3, OUT_C = 4, H = 16, W = 16, K = 3 };
#define IN_COUNT (IN_C * H * W)
#define OUT_COUNT (OUT_C * H * W)
static const float W_O[OUT_C] = {0.5f, -0.25f, 0.125f, 1.0f};
static const float B_O[OUT_C] = {0.25f, -0.5f, 0.0f, -1.0f};
static const float A_O[OUT_C] = {1.0f, 0.5f, -0.25f, 2.0f};

static float g_input[IN_COUNT];
static float g_expected[OUT_COUNT];

static void build_reference(void)
{
    for (int c = 0; c < IN_C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                g_input[(c * H + h) * W + w] = (float)(((c * 7 + h * 3 + w * 5) % 11) - 5) / 8.0f;

    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            /* Every output channel sees the same window sum: the weights are
             * constant within a channel. Zero padding drops out-of-range taps. */
            double s = 0.0;
            for (int c = 0; c < IN_C; c++)
                for (int kh = 0; kh < K; kh++)
                    for (int kw = 0; kw < K; kw++) {
                        int y = h + kh - 1, x = w + kw - 1;
                        if (y >= 0 && y < H && x >= 0 && x < W)
                            s += g_input[(c * H + y) * W + x];
                    }
            for (int o = 0; o < OUT_C; o++) {
                double v = (double)W_O[o] * s + (double)B_O[o];
                if (v < 0.0)
                    v = 0.0;
                g_expected[(o * H + h) * W + w] = (float)(v + (double)A_O[o]);
            }
        }
    }
}

/* ---- targets ------------------------------------------------------------- */

typedef enum { EP_CPU, EP_CUDA, EP_TENSORRT, EP_OPENVINO, EP_DNNL, EP_WEBGPU, EP_MIGRAPHX } EpKind;

typedef struct {
    const char *id;       /* name used on the command line and in the report */
    EpKind kind;
    const char *provider; /* name reported by GetAvailableProviders, or the plugin's EP name */
    const char *device;   /* OpenVINO device_type, WebGPU vendor key, else NULL */
    int opt_in;           /* not part of the default set: runs only when named */
    /*
     * Accepted error per element: atol + rtol * |expected|.  The reference
     * values are exact in fp16, so a correct fp16 or fp32 path reproduces them
     * exactly; the tolerances only leave room for what the engine may do:
     *  - CPU: fp32 throughout; 1e-5 only absorbs summation order.
     *  - CUDA: cuDNN may use TF32 tensor cores (10-bit mantissa) by default.
     *  - TensorRT, OpenVINO: may legitimately run fp16 (GPU, NPU) or bf16
     *    (OpenVINO CPU on AMX hardware, 8-bit mantissa, ~0.4% relative); 1e-2
     *    catches any wrong result (the output spans 0.25..18) without
     *    flagging reduced precision.
     *  - oneDNN: fp32 on the CPU by default; 1e-3 leaves room for fused
     *    kernels' summation order, as for CUDA.
     *  - MIGraphX: fp32 unless fp16 is requested, which it is not; 1e-2 as for
     *    the other GPU engines (never measured: no AMD GPU was available).
     *  - WebGPU: fp32 shaders, but the kernels are chosen per GPU vendor and
     *    may use reduced-precision paths; 1e-2 as for the other GPU engines.
     */
    double atol, rtol;
} Target;

static const Target TARGETS[] = {
    {"cpu", EP_CPU, "CPUExecutionProvider", NULL, 0, 1e-5, 1e-5},
    {"cuda", EP_CUDA, "CUDAExecutionProvider", NULL, 0, 1e-3, 1e-3},
    {"tensorrt", EP_TENSORRT, "TensorrtExecutionProvider", NULL, 0, 1e-2, 1e-2},
    {"openvino:CPU", EP_OPENVINO, "OpenVINOExecutionProvider", "CPU", 0, 1e-2, 1e-2},
    {"openvino:GPU", EP_OPENVINO, "OpenVINOExecutionProvider", "GPU", 0, 1e-2, 1e-2},
    {"openvino:NPU", EP_OPENVINO, "OpenVINOExecutionProvider", "NPU", 0, 1e-2, 1e-2},
    {"dnnl", EP_DNNL, "DnnlExecutionProvider", NULL, 0, 1e-3, 1e-3},
    {"webgpu:nvidia", EP_WEBGPU, "WebGpuExecutionProvider", "nvidia", 0, 1e-2, 1e-2},
    {"webgpu:intel", EP_WEBGPU, "WebGpuExecutionProvider", "intel", 0, 1e-2, 1e-2},
    {"migraphx-load", EP_MIGRAPHX, "MIGraphXExecutionProvider", NULL, 1, 1e-2, 1e-2},
};
#define NTARGETS (sizeof TARGETS / sizeof TARGETS[0])

/*
 * The GPU vendors a WebGPU target can name.  The PCI vendor id is what ORT's
 * device discovery reports for a GPU (it leaves the vendor name empty on
 * Linux).  power is the Dawn power preference that makes Dawn take that
 * vendor's GPU, from the adapter order described at the top: NVIDIA GPUs are
 * discrete (ORT's own discovery assumes so), Intel's are taken as integrated.
 * An Intel discrete card (Arc) next to an NVIDIA one would break that
 * assumption; the usage counters below would then report the mismatch.
 * opens_drm_node: the vendor's Vulkan driver (Mesa's) keeps the GPU's DRM
 * device file open, and its kernel driver (i915, xe) counts the work there.
 */
typedef struct {
    const char *key, *name;
    uint32_t pci_vendor;
    const char *power;
    int opens_drm_node;
} WebGpuVendor;

static const WebGpuVendor WEBGPU_VENDORS[] = {
    {"nvidia", "NVIDIA", 0x10de, "high-performance", 0},
    {"intel", "Intel", 0x8086, "low-power", 1},
};

static const WebGpuVendor *webgpu_vendor(const Target *t)
{
    for (size_t i = 0; t->kind == EP_WEBGPU && i < sizeof WEBGPU_VENDORS / sizeof WEBGPU_VENDORS[0]; i++)
        if (strcmp(t->device, WEBGPU_VENDORS[i].key) == 0)
            return &WEBGPU_VENDORS[i];
    return NULL;
}

static const char *pci_vendor_name(uint32_t id)
{
    switch (id) {
    case 0x10de:
        return "NVIDIA";
    case 0x8086:
        return "Intel";
    case 0x1002:
        return "AMD";
    }
    return NULL;
}

/* A LIST token selects a target when it equals the id (case-insensitive) or is
 * the id's prefix before the colon ("openvino" selects all openvino:*). */
static int token_matches(const char *tok, size_t len, const Target *t)
{
    size_t idlen = strlen(t->id), colon = strcspn(t->id, ":");
    if (len == idlen && strncasecmp(tok, t->id, len) == 0)
        return 1;
    return t->id[colon] == ':' && len == colon && strncasecmp(tok, t->id, len) == 0;
}

/* selected[i] |= 1 for every target named in list; returns -1 on an unknown token. */
static int mark_list(const char *list, unsigned char *selected)
{
    const char *p = list;
    while (*p) {
        size_t len = strcspn(p, ",");
        int hit = 0;
        for (size_t i = 0; len > 0 && i < NTARGETS; i++)
            if (token_matches(p, len, &TARGETS[i])) {
                selected[i] = 1;
                hit = 1;
            }
        if (!hit) {
            fprintf(stderr, "smoke: unknown target '%.*s'\n", (int)len, p);
            return -1;
        }
        p += len;
        if (*p == ',')
            p++;
    }
    return 0;
}

/* ---- results ------------------------------------------------------------- */

typedef enum { ST_OK, ST_FAIL, ST_UNAVAILABLE } Status;
typedef enum { PH_SETUP, PH_RUN } Phase;

typedef struct {
    Status status;
    Phase phase;
    double create_ms, first_ms, warm_ms, max_err;
    int have_timing;
    int no_device; /* status is ST_OK because the provider failed cleanly (migraphx-load) */
    char msg[1024];
    char device[512]; /* WebGPU: the GPU chosen and what the usage counters say */
} Result;

static const OrtApi *g_api;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* Record an OrtStatus as the failure message and release it. */
static void fail_status(Result *r, const char *stage, OrtStatus *st)
{
    snprintf(r->msg, sizeof r->msg, "%s: %s", stage, g_api->GetErrorMessage(st));
    g_api->ReleaseStatus(st);
    r->status = ST_FAIL;
}

/* Evaluate an ORT call; on error record it and jump to the cleanup label. */
#define TRY(stage, call)                                  \
    do {                                                  \
        OrtStatus *st_ = (call);                          \
        if (st_ != NULL) {                                \
            fail_status(r, (stage), st_);                 \
            goto done;                                    \
        }                                                 \
    } while (0)

static int contains_ci(const char *hay, const char *needle)
{
    size_t n = strlen(needle);
    for (; *hay; hay++)
        if (strncasecmp(hay, needle, n) == 0)
            return 1;
    return 0;
}

/* The OpenVINO EP reports an absent device as an ordinary setup error; in
 * ORT 1.29.0 (openvino_provider_factory.cc) the one message for it is
 * "[ERROR] [OpenVINO] Device <name> is not available".  Match exactly that and
 * nothing looser: any other message (an unsupported precision, an unsupported
 * graph, a bad option...) is a failure, even when it mentions a device. */
static int says_device_missing(const char *msg, const char *device)
{
    char phrase[64];
    snprintf(phrase, sizeof phrase, "Device %s is not available", device);
    return contains_ci(msg, phrase);
}

/* A provider is compiled into the library but its vendor runtime (CUDA, cuDNN,
 * TensorRT, OpenVINO...) is an optional dependency that may be absent.  ORT then
 * fails the provider's setup with a dlopen error such as
 *   Failed to load library libonnxruntime_providers_cuda.so with error:
 *   libcudnn.so.9: cannot open shared object file: No such file or directory
 * Copy the missing library's name (the word before ": cannot open shared object
 * file") into soname.  Returns 0, leaving soname empty, when the message is not
 * of that form.  A missing libonnxruntime_providers_*.so itself is a packaging
 * fault, not a missing vendor runtime, so it does not count. */
static int missing_vendor_library(const char *msg, char *soname, size_t n)
{
    static const char tail[] = ": cannot open shared object file";
    const char *hit = strstr(msg, tail);
    soname[0] = '\0';
    if (hit == NULL)
        return 0;
    const char *b = hit;
    while (b > msg && b[-1] != ' ' && b[-1] != '\t')
        b--;
    size_t len = (size_t)(hit - b);
    if (len == 0 || len >= n)
        return 0;
    memcpy(soname, b, len);
    soname[len] = '\0';
    if (strncmp(soname, "libonnxruntime_providers", 24) == 0 || strncmp(soname, "libonnxruntime.", 15) == 0) {
        soname[0] = '\0';
        return 0;
    }
    /* Only a provider load error counts, not any dlopen failure. */
    return contains_ci(msg, "Failed to load library") || contains_ci(msg, "libonnxruntime_providers") ||
           strncmp(soname, "libcu", 5) == 0 || strncmp(soname, "libnv", 5) == 0 ||
           strncmp(soname, "libopenvino", 11) == 0 || strncmp(soname, "libdnnl", 7) == 0 ||
           strncmp(soname, "libmigraphx", 11) == 0 || strncmp(soname, "libamdhip", 9) == 0 ||
           strncmp(soname, "libhsa", 6) == 0;
}

/* Only the no-device answers of ORT's MIGraphX provider are a clean pass.  Both
 * come from MIGraphXExecutionProviderInfo (migraphx_execution_provider_info.cc,
 * v1.29.0) parsing device_id: hipGetDeviceCount failing, which HIP_CALL turns
 * into "HIP failure <code>: <hipGetErrorString>; GPU=...; expr=hipGetDeviceCount(
 * &num_devices); " (hipErrorNoDevice reads "no ROCm-capable device is detected"),
 * or hipGetDeviceCount succeeding with 0 devices: "Invalid device ID: 0, must be
 * between 0 (inclusive) and 0 (exclusive)."  A "HIP failure" counts only as code 100
 * (hipErrorNoDevice) together with "hipGetDeviceCount": any other code (35,
 * hipErrorInsufficientDriver, a broken driver) fails, and so does the same message shape from any other HIP call
 * (hipMalloc out of memory, hipSetDevice, no binary for the GPU) means a device
 * is there and something is wrong.  Nothing looser. */
static int says_no_device(const char *msg)
{
    return (contains_ci(msg, "HIP failure 100:") && contains_ci(msg, "hipGetDeviceCount")) ||
           contains_ci(msg, "hipErrorNoDevice") ||
           contains_ci(msg, "no ROCm-capable device") ||
           (contains_ci(msg, "Invalid device ID: 0") && contains_ci(msg, "and 0 (exclusive)"));
}

/* The provider library of the engines that ship as separate packages
 * (onnxruntime-provider-dnnl, -migraphx, -webgpu), else NULL. */
static const char *own_provider_library(EpKind kind)
{
    if (kind == EP_DNNL)
        return "libonnxruntime_providers_dnnl.so";
    if (kind == EP_MIGRAPHX)
        return "libonnxruntime_providers_migraphx.so";
    if (kind == EP_WEBGPU)
        return "libonnxruntime_providers_webgpu.so";
    return NULL;
}

/* Decide whether a setup failure means "this engine is not usable on this
 * machine" (unavailable) rather than a bug.  On yes, writes the reason. */
static int setup_unavailable_reason(const Target *t, const char *msg, char *reason, size_t n)
{
    char soname[128], needle[160];
    const char *own = own_provider_library(t->kind);
    if (t->kind == EP_CPU)
        return 0;
    if (own != NULL) {
        /* The provider library itself is absent: its package is not installed. */
        snprintf(needle, sizeof needle, "%s: cannot open shared object file", own);
        if (strstr(msg, needle) != NULL) {
            snprintf(reason, n, "provider library not installed: %s", own);
            return 1;
        }
    }
    if (missing_vendor_library(msg, soname, sizeof soname)) {
        snprintf(reason, n, "vendor runtime not installed: %s", soname);
        return 1;
    }
    if (t->kind == EP_OPENVINO && says_device_missing(msg, t->device)) {
        snprintf(reason, n, "OpenVINO device %s not present", t->device);
        return 1;
    }
    return 0;
}

typedef enum { SETUP_FAIL, SETUP_UNAVAILABLE, SETUP_NO_DEVICE } SetupOutcome;

/* What a failure while appending the provider or creating the session means.
 * Only migraphx-load has a "no device" outcome, and only for the signatures in
 * says_no_device; every other message, "not supported in this build" and
 * "undefined symbol" included, is a failure. */
static SetupOutcome classify_setup(const Target *t, const char *msg, char *reason, size_t n)
{
    if (setup_unavailable_reason(t, msg, reason, n))
        return SETUP_UNAVAILABLE;
    if (t->kind == EP_MIGRAPHX && says_no_device(msg))
        return SETUP_NO_DEVICE;
    return SETUP_FAIL;
}

static int provider_listed(char **providers, int n, const char *name)
{
    for (int i = 0; i < n; i++)
        if (strcmp(providers[i], name) == 0)
            return 1;
    return 0;
}

/* The path of a separately packaged provider library next to the
 * libonnxruntime.so that is loaded, which is where ORT looks for the ones it
 * loads itself.  Returns 0 when that location is unknown, leaving the bare name
 * in path: the dynamic loader then searches its usual directories. */
static int provider_library_path(const char *name, char *path, size_t n)
{
#ifndef SMOKE_UNIT_TEST
    Dl_info info;
    if (dladdr((void *)(size_t)&OrtGetApiBase, &info) != 0 && info.dli_fname != NULL) {
        const char *slash = strrchr(info.dli_fname, '/');
        int len = slash == NULL ? snprintf(path, n, "%s", name)
                                : snprintf(path, n, "%.*s/%s", (int)(slash - info.dli_fname),
                                           info.dli_fname, name);
        if (len >= 0 && (size_t)len < n)
            return 1;
    }
#endif
    snprintf(path, n, "%s", name);
    return 0;
}

/* ---- WebGPU: which GPU did the work -------------------------------------- */

/*
 * The kernel's GPU usage counters for this process, per GPU.  Each DRM file
 * descriptor has a /proc/self/fdinfo entry; drivers that keep per-client
 * statistics (i915, xe, amdgpu) write "drm-pdev: <PCI address>" and one
 * counter per engine, "drm-engine-<name>: <ns> ns" (i915, amdgpu) or
 * "drm-cycles-<name>: <cycles>" (xe).  The counters only grow while this
 * process's work runs, so a difference across the runs says which GPU ran it.
 * "drm-engine-capacity-*" is a number of engines and "drm-total-cycles-*" the
 * GPU's own clock, which grows anyway: neither is work.
 */
enum { MAX_DRM_DEVICES = 16 };
typedef struct {
    char pdev[32];
    unsigned long long busy; /* sum of the work counters over its descriptors */
    int fds;                 /* descriptors open on it */
    int counters;            /* work counters seen: its driver keeps them */
} DrmDevice;
typedef struct {
    int n;
    DrmDevice dev[MAX_DRM_DEVICES];
} DrmUsage;

static DrmDevice *drm_device(DrmUsage *u, const char *pdev)
{
    for (int i = 0; i < u->n; i++)
        if (strcmp(u->dev[i].pdev, pdev) == 0)
            return &u->dev[i];
    if (u->n == MAX_DRM_DEVICES)
        return NULL;
    DrmDevice *d = &u->dev[u->n++];
    memset(d, 0, sizeof *d);
    snprintf(d->pdev, sizeof d->pdev, "%s", pdev);
    return d;
}

static const DrmDevice *drm_find(const DrmUsage *u, const char *pdev)
{
    for (int i = 0; i < u->n; i++)
        if (strcmp(u->dev[i].pdev, pdev) == 0)
            return &u->dev[i];
    return NULL;
}

/* Read every fdinfo entry in dir (normally /proc/self/fdinfo) into u. */
static void drm_usage(const char *dir, DrmUsage *u)
{
    char path[4096], line[256];
    memset(u, 0, sizeof *u);
    DIR *d = opendir(dir);
    if (d == NULL)
        return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.')
            continue;
        snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
        FILE *f = fopen(path, "r");
        if (f == NULL)
            continue;
        char pdev[32] = "";
        unsigned long long busy = 0;
        int counters = 0;
        while (fgets(line, sizeof line, f) != NULL) {
            char *colon = strchr(line, ':');
            if (colon == NULL || strncmp(line, "drm-", 4) != 0)
                continue;
            *colon = '\0';
            const char *v = colon + 1;
            while (*v == ' ' || *v == '\t')
                v++;
            if (strcmp(line, "drm-pdev") == 0)
                snprintf(pdev, sizeof pdev, "%.*s", (int)strcspn(v, " \t\n"), v);
            else if ((strncmp(line, "drm-engine-", 11) == 0 && strncmp(line, "drm-engine-capacity-", 20) != 0) ||
                     strncmp(line, "drm-cycles-", 11) == 0) {
                busy += strtoull(v, NULL, 10);
                counters++;
            }
        }
        fclose(f);
        DrmDevice *dev = pdev[0] ? drm_device(u, pdev) : NULL;
        if (dev != NULL) {
            dev->busy += busy;
            dev->fds++;
            dev->counters += counters;
        }
    }
    closedir(d);
}

static unsigned long long drm_delta(const DrmUsage *before, const DrmDevice *after)
{
    const DrmDevice *b = drm_find(before, after->pdev);
    unsigned long long was = b == NULL ? 0 : b->busy;
    return after->busy > was ? after->busy - was : 0;
}

typedef enum { GPU_CONFIRMED, GPU_INFERRED, GPU_ELSEWHERE, GPU_NOT_OPENED, GPU_IDLE } GpuVerdict;

/* What the counters say about the runs for a target GPU at PCI address pdev
 * (empty when ORT did not report one).  Work counted on the target is a
 * confirmation; work counted only on another GPU (copied into other) means
 * Dawn took the wrong adapter; for a vendor whose Vulkan driver keeps the
 * device file open, a target this process never opened means that driver is
 * missing and the work ran elsewhere; a target whose driver keeps counters
 * (the descriptors it has open show them) but counted nothing means the work
 * ran on a GPU that keeps none, such as the NVIDIA one: Mesa keeps the Intel
 * device file open after listing the GPUs, so being open is not enough.
 * Anything else (the NVIDIA driver keeps no counters) leaves the adapter
 * inferred from the power preference. */
static GpuVerdict gpu_verdict(const char *pdev, int opens_drm_node, const DrmUsage *before,
                              const DrmUsage *after, char *other, size_t n)
{
    other[0] = '\0';
    if (pdev[0] == '\0')
        return GPU_INFERRED;
    const DrmDevice *target = drm_find(after, pdev);
    if (target != NULL && drm_delta(before, target) > 0)
        return GPU_CONFIRMED;
    for (int i = 0; i < after->n; i++)
        if (strcmp(after->dev[i].pdev, pdev) != 0 && drm_delta(before, &after->dev[i]) > 0) {
            snprintf(other, n, "%s", after->dev[i].pdev);
            return GPU_ELSEWHERE;
        }
    if (opens_drm_node && (target == NULL || target->fds == 0))
        return GPU_NOT_OPENED;
    if (target != NULL && target->counters > 0)
        return GPU_IDLE;
    return GPU_INFERRED;
}

/* Register the WebGPU plugin library with the environment, once per process
 * (a second registration under the same name is an error).  A failed attempt
 * leaves nothing registered, so the next WebGPU target tries again and
 * reports the same error. */
static OrtStatus *webgpu_register(OrtEnv *env)
{
    static int registered;
    char path[4096];
    if (registered)
        return NULL;
    provider_library_path(own_provider_library(EP_WEBGPU), path, sizeof path);
    OrtStatus *st = g_api->RegisterExecutionProviderLibrary(env, "WebGpuExecutionProvider", path);
    if (st == NULL)
        registered = 1;
    return st;
}

/* Pick the WebGPU provider's device on a GPU of the target's vendor: the
 * provider offers one per GPU ORT found.  On success the GPU is described in
 * r->device and its PCI address copied into pdev.  When there is none, *dev
 * stays NULL and r says the target is unavailable, naming the GPUs that were
 * offered.  With two GPUs of the vendor the first one is taken. */
static OrtStatus *webgpu_pick(const OrtEnv *env, const Target *t, const OrtEpDevice **dev, char *pdev,
                              size_t pn, Result *r)
{
    const OrtEpDevice *const *all = NULL;
    const WebGpuVendor *v = webgpu_vendor(t);
    char seen[384] = "";
    size_t n = 0, used = 0;
    *dev = NULL;
    pdev[0] = '\0';
    OrtStatus *st = g_api->GetEpDevices(env, &all, &n);
    if (st != NULL)
        return st;
    for (size_t i = 0; i < n; i++) {
        const char *ep = g_api->EpDevice_EpName(all[i]);
        const OrtHardwareDevice *hw = g_api->EpDevice_Device(all[i]);
        if (ep == NULL || strcmp(ep, t->provider) != 0 || hw == NULL ||
            g_api->HardwareDevice_Type(hw) != OrtHardwareDeviceType_GPU)
            continue;
        const OrtKeyValuePairs *md = g_api->HardwareDevice_Metadata(hw);
        const char *virt = md == NULL ? NULL : g_api->GetKeyValue(md, "is_virtual");
        if (virt != NULL && strcmp(virt, "1") == 0)
            continue; /* a stand-in for compile-only sessions, no GPU behind it */
        const char *bus = md == NULL ? NULL : g_api->GetKeyValue(md, "pci_bus_id");
        const char *vendor = g_api->HardwareDevice_Vendor(hw);
        uint32_t vid = g_api->HardwareDevice_VendorId(hw), did = g_api->HardwareDevice_DeviceId(hw);
        if (vendor == NULL || vendor[0] == '\0')
            vendor = pci_vendor_name(vid);
        char desc[160];
        snprintf(desc, sizeof desc, "%s GPU %04x:%04x at %s", vendor != NULL ? vendor : "unknown vendor's",
                 (unsigned)vid, (unsigned)did, bus != NULL ? bus : "an unknown PCI address");
        if (*dev == NULL && vid == v->pci_vendor) {
            *dev = all[i];
            snprintf(r->device, sizeof r->device, "%s", desc);
            snprintf(pdev, pn, "%s", bus != NULL ? bus : "");
        } else if (used < sizeof seen) {
            int k = snprintf(seen + used, sizeof seen - used, "%s%s", used ? "; " : "", desc);
            used += k > 0 ? (size_t)k : 0;
        }
    }
    if (*dev == NULL) {
        r->status = ST_UNAVAILABLE;
        snprintf(r->msg, sizeof r->msg, "the WebGPU provider offers no %s GPU (it offers: %s)", v->name,
                 seen[0] ? seen : "none");
    }
    return NULL;
}

/* Append the target's execution provider to opts.  env and dev are for the
 * WebGPU plugin, which is appended for one of its devices. */
static OrtStatus *append_provider(const Target *t, OrtSessionOptions *opts, OrtEnv *env, const OrtEpDevice *dev)
{
    OrtStatus *st = NULL;
    switch (t->kind) {
    case EP_CPU:
        return NULL; /* the CPU EP is always registered */
    case EP_CUDA: {
        /* The _V2 entry points exist from ORT 1.11 through 1.31+ and keep
         * ownership of the options struct inside ORT, unlike the old
         * struct-based SessionOptionsAppendExecutionProvider_CUDA. */
        OrtCUDAProviderOptionsV2 *o = NULL;
        if ((st = g_api->CreateCUDAProviderOptions(&o)) != NULL)
            return st;
        const char *keys[] = {"device_id"}, *vals[] = {"0"};
        st = g_api->UpdateCUDAProviderOptions(o, keys, vals, 1);
        if (st == NULL)
            st = g_api->SessionOptionsAppendExecutionProvider_CUDA_V2(opts, o);
        g_api->ReleaseCUDAProviderOptions(o);
        return st;
    }
    case EP_TENSORRT: {
        /* Engine and timing caches off: every run builds its engine, so the
         * result never depends on a stale cache. Expect create to take seconds. */
        OrtTensorRTProviderOptionsV2 *o = NULL;
        if ((st = g_api->CreateTensorRTProviderOptions(&o)) != NULL)
            return st;
        const char *keys[] = {"device_id", "trt_engine_cache_enable", "trt_timing_cache_enable"};
        const char *vals[] = {"0", "0", "0"};
        st = g_api->UpdateTensorRTProviderOptions(o, keys, vals, 3);
        if (st == NULL)
            st = g_api->SessionOptionsAppendExecutionProvider_TensorRT_V2(opts, o);
        g_api->ReleaseTensorRTProviderOptions(o);
        return st;
    }
    case EP_OPENVINO: {
        const char *keys[] = {"device_type"}, *vals[] = {t->device};
        return g_api->SessionOptionsAppendExecutionProvider_OpenVINO_V2(opts, keys, vals, 1);
    }
    case EP_DNNL: {
        /* Default options (arena on), through the options object so that the
         * struct layout stays ORT's business (ORT 1.15 and later). */
        OrtDnnlProviderOptions *o = NULL;
        if ((st = g_api->CreateDnnlProviderOptions(&o)) != NULL)
            return st;
        st = g_api->SessionOptionsAppendExecutionProvider_Dnnl(opts, o);
        g_api->ReleaseDnnlProviderOptions(o);
        return st;
    }
    case EP_MIGRAPHX: {
        /* The generic, by-name call: the legacy OrtMIGraphXProviderOptions
         * struct changes size between versions, which a program built against
         * one header and run on another library must not depend on.  ORT 1.29.0
         * accepts the name when built with USE_MIGRAPHX or
         * USE_MIGRAPHX_PROVIDER_INTERFACE (provider_registration.cc); a core
         * without either answers "not supported in this build", a failure. */
        const char *keys[] = {"device_id"}, *vals[] = {"0"};
        return g_api->SessionOptionsAppendExecutionProvider(opts, "MIGraphX", keys, vals, 1);
    }
    case EP_WEBGPU: {
        /* The one option that steers Dawn's choice of GPU (see the top); the
         * plugin reads it as ep.webgpuexecutionprovider.powerPreference. */
        const char *keys[] = {"powerPreference"}, *vals[] = {webgpu_vendor(t)->power};
        return g_api->SessionOptionsAppendExecutionProvider_V2(opts, env, &dev, 1, keys, vals, 1);
    }
    }
    return NULL;
}

static void run_target(const Target *t, OrtEnv *env, const OrtMemoryInfo *mem,
                       const char *model_path, int runs, Result *r)
{
    OrtSessionOptions *opts = NULL;
    OrtSession *session = NULL;
    OrtValue *input = NULL;
    OrtValue *output = NULL;
    OrtTensorTypeAndShapeInfo *info = NULL;
    const char *in_names[] = {"X"}, *out_names[] = {"Y"};
    const int64_t shape[4] = {1, IN_C, H, W};
    const OrtEpDevice *dev = NULL;
    char pdev[32] = "";
    DrmUsage before, after;

    memset(r, 0, sizeof *r);
    r->status = ST_OK;
    r->phase = PH_SETUP;

    double t0 = now_ms();
    if (t->kind == EP_WEBGPU) {
        TRY("RegisterExecutionProviderLibrary", webgpu_register(env));
        TRY("GetEpDevices", webgpu_pick(env, t, &dev, pdev, sizeof pdev, r));
        if (dev == NULL)
            goto done; /* unavailable: no GPU of that vendor, r says which were offered */
        drm_usage("/proc/self/fdinfo", &before);
    }
    TRY("CreateSessionOptions", g_api->CreateSessionOptions(&opts));
    if (t->kind != EP_CPU) {
        /* Without this, a graph the EP cannot take (or an EP that registers
         * but cannot start) silently runs on the CPU EP and would pass. */
        TRY("disable_cpu_ep_fallback",
            g_api->AddSessionConfigEntry(opts, "session.disable_cpu_ep_fallback", "1"));
    }
    TRY("append provider", append_provider(t, opts, env, dev));
    TRY("CreateSession", g_api->CreateSession(env, model_path, opts, &session));
    r->create_ms = now_ms() - t0;

    r->phase = PH_RUN;
    TRY("CreateTensorWithDataAsOrtValue",
        g_api->CreateTensorWithDataAsOrtValue(mem, g_input, sizeof g_input, shape, 4,
                                              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input));

    /* Run 0 is the "first run" (lazy initialisation, kernel compilation); the
     * following runs-1... are averaged as "warm". Every output is checked. */
    double warm_total = 0.0;
    for (int i = 0; i <= runs; i++) {
        double t1 = now_ms();
        TRY("Run", g_api->Run(session, NULL, in_names, (const OrtValue *const *)&input, 1,
                              out_names, 1, &output));
        double dt = now_ms() - t1;
        if (i == 0)
            r->first_ms = dt;
        else
            warm_total += dt;
        r->have_timing = 1;

        int is_tensor = 0;
        TRY("IsTensor", g_api->IsTensor(output, &is_tensor));
        if (!is_tensor) {
            snprintf(r->msg, sizeof r->msg, "output is not a tensor");
            r->status = ST_FAIL;
            goto done;
        }
        TRY("GetTensorTypeAndShape", g_api->GetTensorTypeAndShape(output, &info));
        size_t count = 0;
        ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        TRY("GetTensorShapeElementCount", g_api->GetTensorShapeElementCount(info, &count));
        TRY("GetTensorElementType", g_api->GetTensorElementType(info, &et));
        g_api->ReleaseTensorTypeAndShapeInfo(info);
        info = NULL;
        if (count != OUT_COUNT || et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            snprintf(r->msg, sizeof r->msg, "unexpected output: %zu elements (want %d), element type %d",
                     count, OUT_COUNT, (int)et);
            r->status = ST_FAIL;
            goto done;
        }

        float *y = NULL;
        TRY("GetTensorMutableData", g_api->GetTensorMutableData(output, (void **)&y));
        for (int k = 0; k < OUT_COUNT; k++) {
            double err = fabs((double)y[k] - (double)g_expected[k]);
            double allowed = t->atol + t->rtol * fabs((double)g_expected[k]);
            if (!(err <= allowed)) { /* also catches NaN */
                snprintf(r->msg, sizeof r->msg,
                         "output mismatch at element %d (run %d): got %g, expected %g",
                         k, i, (double)y[k], (double)g_expected[k]);
                r->status = ST_FAIL;
                r->max_err = isnan(err) ? INFINITY : fmax(r->max_err, err);
                goto done;
            }
            if (err > r->max_err)
                r->max_err = err;
        }
        g_api->ReleaseValue(output);
        output = NULL;
    }
    if (runs > 0)
        r->warm_ms = warm_total / runs;

    if (t->kind == EP_WEBGPU) {
        /* Read while the session is open: its device files are still there. */
        const WebGpuVendor *v = webgpu_vendor(t);
        char other[32];
        size_t len = strlen(r->device);
        drm_usage("/proc/self/fdinfo", &after);
        GpuVerdict verdict = gpu_verdict(pdev, v->opens_drm_node, &before, &after, other, sizeof other);
        snprintf(r->device + len, sizeof r->device - len, ", power preference %s: %s", v->power,
                 verdict == GPU_CONFIRMED ? "the kernel counted the work on this GPU"
                 : verdict == GPU_INFERRED ? "Dawn's adapter is inferred from the power preference "
                                             "(no kernel usage counters for this GPU)"
                                           : "the kernel counted no work on this GPU");
        if (verdict == GPU_ELSEWHERE) {
            snprintf(r->msg, sizeof r->msg, "the work ran on the GPU at %s, not on the %s GPU at %s", other,
                     v->name, pdev);
            r->status = ST_FAIL;
        } else if (verdict == GPU_IDLE) {
            snprintf(r->msg, sizeof r->msg,
                     "work not seen on the %s GPU at %s: its driver keeps usage counters and they did "
                     "not grow during the runs, so the work ran on a GPU that keeps none", v->name, pdev);
            r->status = ST_FAIL;
        } else if (verdict == GPU_NOT_OPENED) {
            snprintf(r->msg, sizeof r->msg,
                     "this process never opened the %s GPU at %s, so the work ran on another GPU: "
                     "is its Vulkan driver installed?", v->name, pdev);
            r->status = ST_FAIL;
        }
    }

done:
    if (info != NULL)
        g_api->ReleaseTensorTypeAndShapeInfo(info);
    if (output != NULL)
        g_api->ReleaseValue(output);
    if (input != NULL)
        g_api->ReleaseValue(input);
    if (session != NULL)
        g_api->ReleaseSession(session);
    if (opts != NULL)
        g_api->ReleaseSessionOptions(opts);

    if (r->status == ST_FAIL && r->phase == PH_SETUP) {
        char reason[256], full[sizeof r->msg];
        SetupOutcome outcome = classify_setup(t, r->msg, reason, sizeof reason);
        if (outcome == SETUP_UNAVAILABLE) {
            snprintf(full, sizeof full, "%.256s (ORT said: %.700s)", reason, r->msg);
            memcpy(r->msg, full, sizeof r->msg);
            r->status = ST_UNAVAILABLE;
        } else if (outcome == SETUP_NO_DEVICE) {
            r->status = ST_OK;
            r->no_device = 1;
        }
    }
}

/* ---- main ---------------------------------------------------------------- */

#ifdef SMOKE_UNIT_TEST
/*
 * Test build (-DSMOKE_UNIT_TEST, only the ORT header needed, nothing linked):
 *   classify TARGET MESSAGE
 * prints "unavailable: REASON", "no device" or "fail" for a setup error MESSAGE
 * on TARGET.
 *   verdict PDEV OPENS_DRM_NODE BEFORE_DIR AFTER_DIR
 * prints what GPU usage counters read from two fdinfo-like directories say
 * about a WebGPU target's GPU at PCI address PDEV: "confirmed", "inferred",
 * "elsewhere OTHER_PDEV", "not opened" or "idle".
 */
int main(int argc, char **argv)
{
    char reason[256];
    if (argc == 6 && !strcmp(argv[1], "verdict")) {
        static DrmUsage before, after;
        char other[32];
        drm_usage(argv[4], &before);
        drm_usage(argv[5], &after);
        switch (gpu_verdict(argv[2], atoi(argv[3]), &before, &after, other, sizeof other)) {
        case GPU_CONFIRMED:
            puts("confirmed");
            break;
        case GPU_INFERRED:
            puts("inferred");
            break;
        case GPU_ELSEWHERE:
            printf("elsewhere %s\n", other);
            break;
        case GPU_NOT_OPENED:
            puts("not opened");
            break;
        case GPU_IDLE:
            puts("idle");
            break;
        }
        return 0;
    }
    if (argc != 3)
        return 2;
    for (size_t i = 0; i < NTARGETS; i++)
        if (!strcmp(argv[1], TARGETS[i].id)) {
            switch (classify_setup(&TARGETS[i], argv[2], reason, sizeof reason)) {
            case SETUP_UNAVAILABLE:
                printf("unavailable: %s\n", reason);
                break;
            case SETUP_NO_DEVICE:
                puts("no device");
                break;
            default:
                puts("fail");
            }
            return 0;
        }
    return 2;
}
#else

static int g_api_version; /* of the table in g_api */

/* Whether the provider library of a separately packaged engine sits next to
 * the libonnxruntime.so that is loaded, where ORT looks for it.  An unknown
 * location counts as present: the load error then says what is missing. */
static int provider_library_present(const char *name)
{
    char path[4096];
    if (!provider_library_path(name, path, sizeof path))
        return 1;
    return access(path, F_OK) == 0;
}

static void print_oneline(const char *s)
{
    /* Collapse newlines and control characters; cap the length. */
    int n = 0;
    for (; *s && n < 700; s++, n++)
        putchar(isprint((unsigned char)*s) ? *s : ' ');
    if (*s)
        fputs("...", stdout);
}

static void usage(FILE *f)
{
    fputs("usage: smoke MODEL.onnx [--only LIST] [--require LIST] [--runs N] [--log-level N]\n"
          "targets: cpu, cuda, tensorrt, openvino:CPU, openvino:GPU, openvino:NPU, dnnl,\n"
          "         webgpu:nvidia, webgpu:intel, migraphx-load (\"openvino\" selects the three\n"
          "         OpenVINO devices, \"webgpu\" both WebGPU GPUs)\n"
          "default set: all but migraphx-load, which runs only when named in --only or --require\n"
          "webgpu:*: unavailable without libonnxruntime_providers_webgpu.so or a GPU of that vendor;\n"
          "         the line below the result names the GPU and what the kernel counted on it\n"
          "migraphx-load: unavailable without libonnxruntime_providers_migraphx.so or its AMD\n"
          "         libraries; \"ok (no device)\" only for a hipGetDeviceCount or no-device error, any other error fails\n",
          f);
}

int main(int argc, char **argv)
{
    const char *model = NULL;
    unsigned char only[NTARGETS] = {0}, require[NTARGETS] = {0};
    int have_only = 0, runs = 5, log_level = ORT_LOGGING_LEVEL_WARNING;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int takes_value = !strcmp(a, "--only") || !strcmp(a, "--require") || !strcmp(a, "--runs") ||
                          !strcmp(a, "--log-level");
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            return 0;
        } else if (takes_value) {
            if (i + 1 >= argc) {
                fprintf(stderr, "smoke: %s needs a value\n", a);
                usage(stderr);
                return 2;
            }
            const char *v = argv[++i];
            char *end = NULL;
            if (!strcmp(a, "--only")) {
                have_only = 1;
                if (mark_list(v, only) != 0)
                    return 2;
            } else if (!strcmp(a, "--require")) {
                if (mark_list(v, require) != 0)
                    return 2;
            } else {
                long n = strtol(v, &end, 10);
                int is_runs = !strcmp(a, "--runs");
                if (*v == '\0' || *end != '\0' || n < 0 || n > (is_runs ? 100000 : 4)) {
                    fprintf(stderr, "smoke: invalid value '%s' for %s\n", v, a);
                    return 2;
                }
                if (is_runs)
                    runs = (int)n;
                else
                    log_level = (int)n;
            }
        } else if (a[0] == '-') {
            fprintf(stderr, "smoke: unknown option %s\n", a);
            usage(stderr);
            return 2;
        } else if (model == NULL) {
            model = a;
        } else {
            fprintf(stderr, "smoke: unexpected argument %s\n", a);
            usage(stderr);
            return 2;
        }
    }
    if (model == NULL) {
        usage(stderr);
        return 2;
    }

    /* Ask for the newest API table that both this header and the library know. */
    const OrtApiBase *base = OrtGetApiBase();
    int api_version = ORT_API_VERSION;
    for (; api_version >= MIN_API_VERSION; api_version--) {
        g_api = base->GetApi((uint32_t)api_version);
        if (g_api != NULL)
            break;
    }
    if (g_api == NULL) {
        fprintf(stderr, "smoke: ONNX Runtime %s does not provide API %d or newer\n",
                base->GetVersionString(), MIN_API_VERSION);
        return 3;
    }
    g_api_version = api_version;
    printf("ONNX Runtime %s (API %d, built against headers for API %d)\n", base->GetVersionString(),
           api_version, ORT_API_VERSION);
    printf("build: ");
    print_oneline(g_api->GetBuildInfoString());
    putchar('\n');

    build_reference();

    OrtEnv *env = NULL;
    OrtMemoryInfo *mem = NULL;
    char **providers = NULL;
    int nproviders = 0, rc = 0;
    OrtStatus *st;

    if ((st = g_api->CreateEnv((OrtLoggingLevel)log_level, "ort-smoke", &env)) != NULL) {
        fprintf(stderr, "smoke: CreateEnv: %s\n", g_api->GetErrorMessage(st));
        g_api->ReleaseStatus(st);
        return 3;
    }
    if ((st = g_api->GetAvailableProviders(&providers, &nproviders)) != NULL) {
        fprintf(stderr, "smoke: GetAvailableProviders: %s\n", g_api->GetErrorMessage(st));
        g_api->ReleaseStatus(st);
        /* The environment is deliberately not released: see "cleanup" below. */
        return 3;
    }
    printf("available providers:");
    for (int i = 0; i < nproviders; i++)
        printf(" %s", providers[i]);
    putchar('\n');

    if ((st = g_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem)) != NULL) {
        fprintf(stderr, "smoke: CreateCpuMemoryInfo: %s\n", g_api->GetErrorMessage(st));
        g_api->ReleaseStatus(st);
        rc = 3;
        goto cleanup;
    }

    printf("model: %s, %d warm run(s) per target, CPU EP fallback disabled for non-CPU targets\n\n",
           model, runs);
    printf("%-14s %-15s %10s %10s %10s %10s\n", "target", "status", "create ms", "first ms",
           "warm ms", "max error");

    for (size_t i = 0; i < NTARGETS; i++)
        if (require[i] && have_only && !only[i]) {
            fprintf(stderr, "smoke: required target %s is excluded by --only\n", TARGETS[i].id);
            rc = 1;
        }

    int nok = 0, nfail = 0, nunavail = 0, nskipped = 0;
    for (size_t i = 0; i < NTARGETS; i++) {
        const Target *t = &TARGETS[i];
        /* --only picks exactly; otherwise the default set plus anything required. */
        if (have_only ? !only[i] : (t->opt_in && !require[i])) {
            if (!have_only)
                nskipped++;
            continue;
        }

        Result r;
        /* The main library never lists MIGraphX (its provider is loaded on
         * demand) nor the WebGPU plugin, so for those the provider library on
         * disk decides. */
        const char *own = own_provider_library(t->kind);
        int on_disk = t->kind == EP_MIGRAPHX || t->kind == EP_WEBGPU;
        if (on_disk ? !provider_library_present(own) : !provider_listed(providers, nproviders, t->provider)) {
            memset(&r, 0, sizeof r);
            r.status = ST_UNAVAILABLE;
            if (on_disk)
                snprintf(r.msg, sizeof r.msg, "provider library not installed: %s", own);
            else
                snprintf(r.msg, sizeof r.msg, "%s is not in the available providers", t->provider);
        } else if (t->kind == EP_WEBGPU && g_api_version < EP_LIBRARY_API_VERSION) {
            memset(&r, 0, sizeof r);
            r.status = ST_UNAVAILABLE;
            snprintf(r.msg, sizeof r.msg, "API %d cannot register a plugin provider (needs API %d)",
                     g_api_version, EP_LIBRARY_API_VERSION);
        } else {
            run_target(t, env, mem, model, runs, &r);
        }

        const char *label = r.status == ST_OK ? (r.no_device ? "ok (no device)" : "ok")
                            : r.status == ST_FAIL ? "FAIL" : "unavailable";
        printf("%-14s %-15s", t->id, label);
        if (r.have_timing)
            printf(" %10.1f %10.2f %10.2f %10.2e", r.create_ms, r.first_ms, r.warm_ms, r.max_err);
        else if (r.create_ms > 0)
            printf(" %10.1f %10s %10s %10s", r.create_ms, "-", "-", "-");
        else
            printf(" %10s %10s %10s %10s", "-", "-", "-", "-");
        putchar('\n');
        if (r.device[0]) {
            /* Which GPU a WebGPU target ran on, whatever the outcome. */
            fputs("    device: ", stdout);
            print_oneline(r.device);
            putchar('\n');
        }
        if ((r.status != ST_OK || r.no_device) && r.msg[0]) {
            printf("    %s: %s", r.status == ST_FAIL ? "error" : "note", r.no_device ? "no device: " : "");
            print_oneline(r.msg);
            putchar('\n');
        }
        /* A later target may take the process down (a vendor runtime aborting):
         * what is known by now must already be out when stdout is a pipe. */
        fflush(stdout);

        if (r.status == ST_OK)
            nok++;
        else if (r.status == ST_FAIL)
            nfail++;
        else
            nunavail++;
        if (r.status == ST_FAIL || (require[i] && r.status != ST_OK))
            rc = 1;
    }
    printf("\n%d ok, %d failed, %d unavailable\n", nok, nfail, nunavail);
    if (nskipped > 0)
        printf("%d target(s) not run by default: name them in --only or --require (see --help)\n", nskipped);
cleanup:
    if (mem != NULL)
        g_api->ReleaseMemoryInfo(mem);
    if ((st = g_api->ReleaseAvailableProviders(providers, nproviders)) != NULL) {
        fprintf(stderr, "smoke: ReleaseAvailableProviders: %s\n", g_api->GetErrorMessage(st));
        g_api->ReleaseStatus(st);
    }
    /* The OrtEnv is deliberately never released. With ONNX Runtime 1.29.0 and
     * OpenVINO 2026.4, releasing it after an OpenVINO session had been created
     * made the process crash at exit (SIGSEGV) although every target had
     * reported ok. Releasing the environment makes ONNX Runtime unload
     * libonnxruntime_providers_openvino.so, and with it OpenVINO's CPU plugin
     * (libopenvino_intel_cpu_plugin.so); the crash happens afterwards, when
     * OpenVINO/TBB worker threads or thread-local destructors run in code that
     * is no longer mapped. An LD_PRELOAD shim that skipped only the dlclose of
     * either of those two libraries made the exit clean; skipping the TBB
     * libraries did not. The cause inside OpenVINO was not investigated further.
     * The process is ending, so the operating system reclaims the environment;
     * the sessions, options, values and memory info above are still released. */
    return rc;
}
#endif /* SMOKE_UNIT_TEST */
