/*
 * ort-smoke: list the ONNX Runtime execution providers and run a tiny model
 * on each engine/device, checking the numbers.
 *
 *   smoke MODEL.onnx [--only LIST] [--require LIST] [--runs N] [--log-level N]
 *
 * MODEL.onnx is the model written by make-model.py.  LIST is a comma-separated
 * set of targets: cpu, cuda, tensorrt, openvino:CPU, openvino:GPU,
 * openvino:NPU (case-insensitive; "openvino" selects all three devices).
 *
 * A target is "unavailable" (not a failure) when its provider is not in the
 * library, when the provider's vendor runtime is not installed (CUDA, cuDNN,
 * TensorRT and OpenVINO are optional dependencies, so every provider is compiled
 * in), or when OpenVINO reports the device absent.
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
#define _POSIX_C_SOURCE 200809L

#include <onnxruntime_c_api.h>

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

/* Oldest API table we accept: SessionOptionsAppendExecutionProvider_OpenVINO_V2
 * (the newest entry point used here) appeared in ORT 1.17. */
#define MIN_API_VERSION 17

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

typedef enum { EP_CPU, EP_CUDA, EP_TENSORRT, EP_OPENVINO } EpKind;

typedef struct {
    const char *id;       /* name used on the command line and in the report */
    EpKind kind;
    const char *provider; /* name reported by GetAvailableProviders */
    const char *device;   /* OpenVINO device_type, else NULL */
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
     */
    double atol, rtol;
} Target;

static const Target TARGETS[] = {
    {"cpu", EP_CPU, "CPUExecutionProvider", NULL, 1e-5, 1e-5},
    {"cuda", EP_CUDA, "CUDAExecutionProvider", NULL, 1e-3, 1e-3},
    {"tensorrt", EP_TENSORRT, "TensorrtExecutionProvider", NULL, 1e-2, 1e-2},
    {"openvino:CPU", EP_OPENVINO, "OpenVINOExecutionProvider", "CPU", 1e-2, 1e-2},
    {"openvino:GPU", EP_OPENVINO, "OpenVINOExecutionProvider", "GPU", 1e-2, 1e-2},
    {"openvino:NPU", EP_OPENVINO, "OpenVINOExecutionProvider", "NPU", 1e-2, 1e-2},
};
#define NTARGETS (sizeof TARGETS / sizeof TARGETS[0])

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
    char msg[1024];
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
           strncmp(soname, "libopenvino", 11) == 0;
}

/* Decide whether a setup failure means "this engine is not usable on this
 * machine" (unavailable) rather than a bug.  On yes, writes the reason. */
static int setup_unavailable_reason(const Target *t, const char *msg, char *reason, size_t n)
{
    char soname[128];
    if (t->kind == EP_CPU)
        return 0;
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

static int provider_listed(char **providers, int n, const char *name)
{
    for (int i = 0; i < n; i++)
        if (strcmp(providers[i], name) == 0)
            return 1;
    return 0;
}

/* Append the target's execution provider to opts. */
static OrtStatus *append_provider(const Target *t, OrtSessionOptions *opts)
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
    }
    return NULL;
}

static void run_target(const Target *t, const OrtEnv *env, const OrtMemoryInfo *mem,
                       const char *model_path, int runs, Result *r)
{
    OrtSessionOptions *opts = NULL;
    OrtSession *session = NULL;
    OrtValue *input = NULL;
    OrtValue *output = NULL;
    OrtTensorTypeAndShapeInfo *info = NULL;
    const char *in_names[] = {"X"}, *out_names[] = {"Y"};
    const int64_t shape[4] = {1, IN_C, H, W};

    memset(r, 0, sizeof *r);
    r->status = ST_OK;
    r->phase = PH_SETUP;

    double t0 = now_ms();
    TRY("CreateSessionOptions", g_api->CreateSessionOptions(&opts));
    if (t->kind != EP_CPU) {
        /* Without this, a graph the EP cannot take (or an EP that registers
         * but cannot start) silently runs on the CPU EP and would pass. */
        TRY("disable_cpu_ep_fallback",
            g_api->AddSessionConfigEntry(opts, "session.disable_cpu_ep_fallback", "1"));
    }
    TRY("append provider", append_provider(t, opts));
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
        if (setup_unavailable_reason(t, r->msg, reason, sizeof reason)) {
            snprintf(full, sizeof full, "%.256s (ORT said: %.700s)", reason, r->msg);
            memcpy(r->msg, full, sizeof r->msg);
            r->status = ST_UNAVAILABLE;
        }
    }
}

/* ---- main ---------------------------------------------------------------- */

#ifdef SMOKE_UNIT_TEST
/*
 * Test build (-DSMOKE_UNIT_TEST, only the ORT header needed, nothing linked):
 *   classify TARGET MESSAGE
 * prints "unavailable: REASON" or "fail" for a setup error MESSAGE on TARGET.
 */
int main(int argc, char **argv)
{
    char reason[256];
    if (argc != 3)
        return 2;
    for (size_t i = 0; i < NTARGETS; i++)
        if (!strcmp(argv[1], TARGETS[i].id)) {
            if (setup_unavailable_reason(&TARGETS[i], argv[2], reason, sizeof reason))
                printf("unavailable: %s\n", reason);
            else
                puts("fail");
            return 0;
        }
    return 2;
}
#else

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
          "targets: cpu, cuda, tensorrt, openvino:CPU, openvino:GPU, openvino:NPU\n"
          "         (\"openvino\" selects the three OpenVINO devices)\n",
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
        g_api->ReleaseEnv(env);
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
    printf("%-14s %-12s %10s %10s %10s %10s\n", "target", "status", "create ms", "first ms",
           "warm ms", "max error");

    for (size_t i = 0; i < NTARGETS; i++)
        if (require[i] && have_only && !only[i]) {
            fprintf(stderr, "smoke: required target %s is excluded by --only\n", TARGETS[i].id);
            rc = 1;
        }

    int nok = 0, nfail = 0, nunavail = 0;
    for (size_t i = 0; i < NTARGETS; i++) {
        const Target *t = &TARGETS[i];
        if (have_only && !only[i])
            continue;

        Result r;
        if (!provider_listed(providers, nproviders, t->provider)) {
            memset(&r, 0, sizeof r);
            r.status = ST_UNAVAILABLE;
            snprintf(r.msg, sizeof r.msg, "%s is not in the available providers", t->provider);
        } else {
            run_target(t, env, mem, model, runs, &r);
        }

        const char *label = r.status == ST_OK ? "ok" : r.status == ST_FAIL ? "FAIL" : "unavailable";
        printf("%-14s %-12s", t->id, label);
        if (r.have_timing)
            printf(" %10.1f %10.2f %10.2f %10.2e", r.create_ms, r.first_ms, r.warm_ms, r.max_err);
        else if (r.create_ms > 0)
            printf(" %10.1f %10s %10s %10s", r.create_ms, "-", "-", "-");
        else
            printf(" %10s %10s %10s %10s", "-", "-", "-", "-");
        putchar('\n');
        if (r.status != ST_OK && r.msg[0]) {
            printf("    %s: ", r.status == ST_FAIL ? "error" : "note");
            print_oneline(r.msg);
            putchar('\n');
        }

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
cleanup:
    if (mem != NULL)
        g_api->ReleaseMemoryInfo(mem);
    if ((st = g_api->ReleaseAvailableProviders(providers, nproviders)) != NULL) {
        fprintf(stderr, "smoke: ReleaseAvailableProviders: %s\n", g_api->GetErrorMessage(st));
        g_api->ReleaseStatus(st);
    }
    g_api->ReleaseEnv(env);
    return rc;
}
#endif /* SMOKE_UNIT_TEST */
