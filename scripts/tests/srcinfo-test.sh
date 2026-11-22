#!/usr/bin/env bash
# Tests for scripts/srcinfo.sh and the committed onnxruntime-split/.SRCINFO:
# --check passes on the tree; on a copy of the script and the recipe (under
# mktemp) an altered .SRCINFO or PKGBUILD fails it, with a diff, and
# --write mends it; the output is the same whatever ORTF_* variables the
# caller has; --aur copies into the top of a git clone only; the committed
# .SRCINFO has the AUR's default names (CUDA 13, the rtx preset, the oneDNN
# package), tensorrt in makedepends, the MIGraphX optdepend naming its pkgbase, onnxruntime-core's
# conflicts on the cuda12 line's packages built without training, and this
# pkgrel; git neither ignores it nor would the licence guard refuse it. The
# same, with --recipe onnxruntime-split-cuda12, for the cuda12 line's
# .SRCINFO: its pkgbase, its one package (onnxruntime-provider-cuda12-legacy,
# CUDA 12 names on this host whatever its CUDA), its pin pair on the main
# line's onnxruntime-core (>=1.29.0-7, <1.29.1) and provides and conflicts,
# its pkgrel and makedepends; and with --recipe onnxruntime-split-migraphx,
# for the MIGraphX line's (its one package, its pin pair, no CUDA);
# --recipe combines with --write, --check and --aur, and an unknown recipe
# is a usage error.
# Usage: scripts/tests/srcinfo-test.sh. Exits non-zero on any failure.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
top=$(cd "$here/../.." && pwd)
srcinfo_file=$top/onnxruntime-split/.SRCINFO
srcinfo12_file=$top/onnxruntime-split-cuda12/.SRCINFO
r12=(--recipe onnxruntime-split-cuda12)
srcinfo_mgx_file=$top/onnxruntime-split-migraphx/.SRCINFO
rmgx=(--recipe onnxruntime-split-migraphx)

fails=0 count=0
ok() { count=$((count + 1)); echo "ok   $1"; }
fail() { count=$((count + 1)); echo "FAIL $1"; fails=$((fails + 1)); }

if ! command -v makepkg >/dev/null 2>&1; then
    echo "skip: makepkg not found"
    exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "== the tree"
if out=$("$top/scripts/srcinfo.sh" --check 2>&1); then ok "--check passes on the tree"; else
    fail "--check on the tree: $out"; fi
# The caller's environment does not reach the .SRCINFO.
if out=$(ORTF_EPS=cpu ORTF_CUDA_ARCHS=120 ORTF_PYTHON=0 ORTF_TENSORRT_HOME=/x "$top/scripts/srcinfo.sh" --check 2>&1); then
    ok "--check ignores the ORTF_* variables of the environment"
else
    fail "--check with ORTF_* set: $out"
fi
if out=$("$top/scripts/srcinfo.sh") && [[ $out == "$(<"$srcinfo_file")" ]]; then
    ok "without arguments it prints the .SRCINFO"
else
    fail "the printed .SRCINFO differs from the committed one"
fi

echo "== the committed .SRCINFO"
T=$'\t'
for line in 'pkgbase = onnxruntime-split' "${T}pkgrel = 8" 'pkgname = onnxruntime-core' \
    'pkgname = onnxruntime-provider-cuda13-rtx' 'pkgname = onnxruntime-provider-tensorrt-cuda13' \
    'pkgname = onnxruntime-provider-openvino' 'pkgname = onnxruntime-provider-dnnl' \
    'pkgname = onnxruntime-provider-webgpu' 'pkgname = python-onnxruntime-core' \
    "${T}makedepends = tensorrt" "${T}makedepends = cuda" "${T}makedepends = cudnn" \
    "${T}makedepends = spirv-headers" "${T}makedepends = vulkan-headers" "${T}makedepends = python-jinja" \
    "${T}source = ortf-dep-dawn.zip::https://github.com/google/dawn/archive/refs/tags/v20260714.215939.zip" \
    "${T}conflicts = onnxruntime-provider-cuda12-legacy<1.29.0-2" \
    "${T}conflicts = onnxruntime-provider-cuda12-legacy-bin<1.29.0-2"; do
    if grep -qxF -- "$line" "$srcinfo_file"; then ok "has [$line]"; else fail "lacks [$line]"; fi
done
if grep -n 'cudax' "$srcinfo_file"; then fail "a name with the old fallback cudax (above)"; else
    ok "no cudax name"; fi
if grep -nE '^pkgname = onnxruntime-provider-migraphx$|makedepends = (migraphx|hip-runtime-amd)$' "$srcinfo_file"; then
    fail "the default build has no MIGraphX package or ROCm build input (above)"
else
    ok "no MIGraphX package or ROCm build input"
fi
# The MIGraphX optdepend is in every build: onnxruntime-core's and
# python-onnxruntime-core's, naming the AUR pkgbase that makes it.
mgx_opt="${T}optdepends = onnxruntime-provider-migraphx: the MIGraphX execution provider (AMD GPUs; AUR pkgbase onnxruntime-split-migraphx, untested on AMD hardware)"
if [[ $(grep -cxF -- "$mgx_opt" "$srcinfo_file") == 2 && $(grep -c 'optdepends = onnxruntime-provider-migraphx' "$srcinfo_file") == 2 ]]; then
    ok "the MIGraphX optdepend, twice, naming onnxruntime-split-migraphx and untested"
else
    fail "the MIGraphX optdepends: $(grep -n 'optdepends = onnxruntime-provider-migraphx' "$srcinfo_file")"
fi
# Tracked or untracked, never ignored; and nothing the licence guard
# refuses (a .SRCINFO is plain metadata).
if git -C "$top" check-ignore -q onnxruntime-split/.SRCINFO; then
    fail "git ignores onnxruntime-split/.SRCINFO: $(git -C "$top" check-ignore -v onnxruntime-split/.SRCINFO)"
else
    ok "git does not ignore onnxruntime-split/.SRCINFO"
fi
if [[ -z $(git -C "$top" ls-files --others --ignored --exclude-standard -- onnxruntime-split/.SRCINFO) ]]; then
    ok "onnxruntime-split/.SRCINFO is not among the ignored files"
else
    fail "onnxruntime-split/.SRCINFO is listed as ignored"
fi
# The guard's own content patterns, which a literal copy here would trip.
if ! grep -qiE "$(sed -n "s/^    -e '\(copyright[^']*\)'.*/\1/p" "$top/scripts/check-no-vendor.sh" | paste -sd '|')" "$srcinfo_file" &&
    ! LC_ALL=C grep -q '[^[:print:][:space:]]' "$srcinfo_file" && (($(wc -c <"$srcinfo_file") < 1048576)); then
    ok "the .SRCINFO is small plain text without a vendor copyright line"
else
    fail "the .SRCINFO would trip the licence guard"
fi

echo "== the cuda12 line's .SRCINFO (--recipe onnxruntime-split-cuda12)"
if out=$("$top/scripts/srcinfo.sh" "${r12[@]}" --check 2>&1) && [[ $out == 'onnxruntime-split-cuda12/.SRCINFO is up to date' ]]; then
    ok "--recipe onnxruntime-split-cuda12 --check passes on the tree"
else
    fail "--recipe onnxruntime-split-cuda12 --check on the tree: $out"
fi
if out=$("$top/scripts/srcinfo.sh" --check "${r12[@]}" 2>&1); then ok "--recipe also after --check"; else
    fail "--check --recipe onnxruntime-split-cuda12: $out"; fi
if out=$(ORTF_EPS='cuda tensorrt openvino dnnl' ORTF_CUDA_ARCHS=rtx ORTF_PYTHON=1 "$top/scripts/srcinfo.sh" "${r12[@]}" --check 2>&1); then
    ok "--recipe onnxruntime-split-cuda12 --check ignores the ORTF_* variables"
else
    fail "--recipe onnxruntime-split-cuda12 --check with ORTF_* set: $out"
fi
if out=$("$top/scripts/srcinfo.sh" "${r12[@]}") && [[ $out == "$(<"$srcinfo12_file")" ]]; then
    ok "--recipe onnxruntime-split-cuda12 alone prints its .SRCINFO"
else
    fail "the printed cuda12 .SRCINFO differs from the committed one"
fi
for line in 'pkgbase = onnxruntime-split-cuda12' "${T}pkgrel = 2" "${T}pkgver = 1.29.0" \
    'pkgname = onnxruntime-provider-cuda12-legacy' "${T}depends = onnxruntime-core>=1.29.0-7" \
    "${T}depends = onnxruntime-core<1.29.1" \
    "${T}provides = onnxruntime-provider-cuda=1.29.0" "${T}provides = onnxruntime-provider-cuda12=1.29.0" \
    "${T}conflicts = onnxruntime-provider-cuda" "${T}conflicts = onnxruntime-provider-cuda12" \
    "${T}depends = nvidia-utils" "${T}depends = cuda>=12" "${T}depends = cuda<13" \
    "${T}depends = cudnn>=9.10" "${T}depends = cudnn<9.11" "${T}makedepends = cuda-12.9" \
    "${T}makedepends = cudnn9.10-cuda12.9" "${T}makedepends = gcc14" "${T}makedepends = python"; do
    if grep -qxF -- "$line" "$srcinfo12_file"; then ok "cuda12: has [$line]"; else fail "cuda12: lacks [$line]"; fi
done
# No bare cuda or cudnn (an AUR helper would take Arch's CUDA 13 and cuDNN
# 9.11 or later); the main line's .SRCINFO keeps its bare ones.
if ! grep -nxE "${T}(make)?depends = (cuda|cudnn)" "$srcinfo12_file" &&
    grep -qxF "${T}depends = cuda" "$top/onnxruntime-split/.SRCINFO" &&
    grep -qxF "${T}depends = cudnn" "$top/onnxruntime-split/.SRCINFO" &&
    grep -qxF "${T}makedepends = cuda" "$top/onnxruntime-split/.SRCINFO" &&
    grep -qxF "${T}makedepends = cudnn" "$top/onnxruntime-split/.SRCINFO"; then
    ok "cuda12: no bare cuda or cudnn in (make)depends; the main line has them"
else
    fail "cuda12: a bare cuda or cudnn (above), or the main line lacks its own"
fi
if (($(grep -c '^pkgname = ' "$srcinfo12_file") == 1)) &&
    ! grep -nE 'makedepends = (tensorrt|openvino|python-.*|migraphx|hip-runtime-amd)$|cuda13|onnxruntime-core=|dnnl' "$srcinfo12_file"; then
    ok "cuda12: one package; no TensorRT, OpenVINO, Python-module or ROCm build input; no CUDA 13 name; no exact core pin; no oneDNN package"
else
    fail "cuda12: the .SRCINFO has more (above)"
fi
if git -C "$top" check-ignore -q onnxruntime-split-cuda12/.SRCINFO; then
    fail "git ignores onnxruntime-split-cuda12/.SRCINFO"
else
    ok "git does not ignore onnxruntime-split-cuda12/.SRCINFO"
fi
if ! grep -qiE "$(sed -n "s/^    -e '\(copyright[^']*\)'.*/\1/p" "$top/scripts/check-no-vendor.sh" | paste -sd '|')" "$srcinfo12_file" &&
    ! LC_ALL=C grep -q '[^[:print:][:space:]]' "$srcinfo12_file"; then
    ok "the cuda12 .SRCINFO is plain text without a vendor copyright line"
else
    fail "the cuda12 .SRCINFO would trip the licence guard"
fi

echo "== the MIGraphX line's .SRCINFO (--recipe onnxruntime-split-migraphx)"
if out=$("$top/scripts/srcinfo.sh" "${rmgx[@]}" --check 2>&1) && [[ $out == 'onnxruntime-split-migraphx/.SRCINFO is up to date' ]]; then
    ok "--recipe onnxruntime-split-migraphx --check passes on the tree"
else
    fail "--recipe onnxruntime-split-migraphx --check on the tree: $out"
fi
if out=$(ORTF_EPS='cuda tensorrt openvino dnnl' ORTF_PYTHON=1 "$top/scripts/srcinfo.sh" "${rmgx[@]}" --check 2>&1); then
    ok "--recipe onnxruntime-split-migraphx --check ignores the ORTF_* variables"
else
    fail "--recipe onnxruntime-split-migraphx --check with ORTF_* set: $out"
fi
for line in 'pkgbase = onnxruntime-split-migraphx' "${T}pkgrel = 1" 'pkgname = onnxruntime-provider-migraphx' \
    "${T}depends = onnxruntime-core>=1.29.0-7" "${T}depends = onnxruntime-core<1.29.1" \
    "${T}depends = migraphx" "${T}depends = hip-runtime-amd" "${T}makedepends = migraphx" "${T}makedepends = hip-runtime-amd"; do
    if grep -qxF -- "$line" "$srcinfo_mgx_file"; then ok "migraphx: has [$line]"; else fail "migraphx: lacks [$line]"; fi
done
# The python- names: the provider's optdepend on its Python add-on, and
# the add-on's block (it links the provider's library into the main line's
# Python module, and so depends on python); the provider has no python.
mgx_opt="${T}optdepends = python-onnxruntime-provider-migraphx: MIGraphX from Python (MIGraphXExecutionProvider in python-onnxruntime-core)"
if [[ $(sed -n 's/^pkgname = //p' "$srcinfo_mgx_file" | tr '\n' ' ') == 'onnxruntime-provider-migraphx python-onnxruntime-provider-migraphx ' ]] &&
    ! grep -vxF -- "$mgx_opt" "$srcinfo_mgx_file" | sed '/^pkgname = python-onnxruntime-provider-migraphx$/,/^$/d' |
    grep -niE "cuda|-bin\b|openvino|tensorrt|python-|^${T}depends = python" &&
    grep -qxF -- "$mgx_opt" "$srcinfo_mgx_file" &&
    sed -n '/^pkgname = python-onnxruntime-provider-migraphx$/,/^$/p' "$srcinfo_mgx_file" | grep -qxF -- "${T}depends = python"; then
    ok "migraphx: the provider and its Python add-on; no CUDA, -bin, OpenVINO, TensorRT, Python module or python outside the add-on but its optdepend"
else
    fail "migraphx: the .SRCINFO has more (above)"
fi
if git -C "$top" check-ignore -q onnxruntime-split-migraphx/.SRCINFO; then fail "git ignores onnxruntime-split-migraphx/.SRCINFO"; else
    ok "git does not ignore onnxruntime-split-migraphx/.SRCINFO"; fi
# The WebGPU provider is the main line's: neither line gains its package,
# Dawn's archive or Dawn's build inputs.
for f in "$srcinfo12_file" "$srcinfo_mgx_file"; do
    if bad=$(grep -niE 'webgpu|dawn|vulkan|spirv|python-jinja|python-markupsafe' "$f"); then
        fail "${f#"$top"/}: WebGPU content: $bad"
    else
        ok "${f#"$top"/}: no WebGPU package, Dawn archive or Dawn build input"
    fi
done

echo "== a copy of the script and the recipe"
copy=$tmp/repo
mkdir -p "$copy/scripts" "$copy/onnxruntime-split" "$copy/onnxruntime-split-cuda12" "$copy/onnxruntime-split-migraphx"
cp "$top/scripts/srcinfo.sh" "$copy/scripts/"
cp "$top/onnxruntime-split/PKGBUILD" "$srcinfo_file" "$copy/onnxruntime-split/"
cp "$top/onnxruntime-split-cuda12/PKGBUILD" "$srcinfo12_file" "$copy/onnxruntime-split-cuda12/"
cp "$top/onnxruntime-split-migraphx/PKGBUILD" "$srcinfo_mgx_file" "$copy/onnxruntime-split-migraphx/"
if "$copy/scripts/srcinfo.sh" --check >/dev/null 2>&1; then ok "the copy checks out"; else
    fail "the unaltered copy fails --check"; fi
sed -i 's/^\tpkgrel = 8$/\tpkgrel = 5/' "$copy/onnxruntime-split/.SRCINFO"
status=0
out=$("$copy/scripts/srcinfo.sh" --check 2>&1) || status=$?
if ((status == 1)) && grep -qxF -- "-${T}pkgrel = 5" <<<"$out" && grep -qxF -- "+${T}pkgrel = 8" <<<"$out" &&
    grep -qF 'is stale: run srcinfo.sh --write' <<<"$out"; then
    ok "an altered .SRCINFO fails --check (exit 1) with a diff"
else
    fail "an altered .SRCINFO (exit $status): $out"
fi
if "$copy/scripts/srcinfo.sh" --write >/dev/null && "$copy/scripts/srcinfo.sh" --check >/dev/null &&
    cmp -s "$copy/onnxruntime-split/.SRCINFO" "$srcinfo_file"; then
    ok "--write mends it, to the committed one"
else
    fail "--write did not restore the .SRCINFO"
fi
sed -i 's/^_ortf_default_cuda_archs=.*/_ortf_default_cuda_archs=all/' "$copy/onnxruntime-split/PKGBUILD"
status=0
out=$("$copy/scripts/srcinfo.sh" --check 2>&1) || status=$?
if ((status == 1)) && grep -qxF -- '+pkgname = onnxruntime-provider-cuda13-all' <<<"$out"; then
    ok "a PKGBUILD that no longer matches its .SRCINFO fails --check"
else
    fail "an altered PKGBUILD (exit $status): $out"
fi
rm "$copy/onnxruntime-split/.SRCINFO"
if ! out=$("$copy/scripts/srcinfo.sh" --check 2>&1) && grep -qF 'does not exist' <<<"$out"; then
    ok "a missing .SRCINFO fails --check"
else
    fail "a missing .SRCINFO: $out"
fi
cp "$top/onnxruntime-split/PKGBUILD" "$srcinfo_file" "$copy/onnxruntime-split/"
for args in '--bogus' '--aur' '--write extra' '--recipe' '--recipe onnxruntime-split-bin --check' \
    '--recipe onnxruntime-split --recipe onnxruntime-split --check' '--write --check' '--recipe onnxruntime-split-cuda12 --aur' '--recipe onnxruntime-split-migraphx-bin --check'; do
    status=0
    # shellcheck disable=SC2086 # the arguments are words
    "$copy/scripts/srcinfo.sh" $args >/dev/null 2>&1 || status=$?
    if ((status == 2)); then ok "usage error for [$args]"; else fail "[$args] exits $status, not 2"; fi
done

echo "== a copy: the cuda12 recipe"
c12=$copy/onnxruntime-split-cuda12
if "$copy/scripts/srcinfo.sh" "${r12[@]}" --check >/dev/null 2>&1; then ok "cuda12: the copy checks out"; else
    fail "cuda12: the unaltered copy fails --check"; fi
sed -i 's/^\tpkgrel = 2$/\tpkgrel = 7/' "$c12/.SRCINFO"
status=0
out=$("$copy/scripts/srcinfo.sh" "${r12[@]}" --check 2>&1) || status=$?
if ((status == 1)) && grep -qxF -- "-${T}pkgrel = 7" <<<"$out" && grep -qxF -- "+${T}pkgrel = 2" <<<"$out" &&
    grep -qF 'onnxruntime-split-cuda12/.SRCINFO is stale: run srcinfo.sh --recipe onnxruntime-split-cuda12 --write' <<<"$out"; then
    ok "cuda12: an altered .SRCINFO fails --check (exit 1) with a diff, naming the recipe"
else
    fail "cuda12: an altered .SRCINFO (exit $status): $out"
fi
if "$copy/scripts/srcinfo.sh" --check >/dev/null 2>&1; then ok "cuda12: the main recipe's --check is not affected"; else
    fail "cuda12: a stale cuda12 .SRCINFO fails the main recipe's --check"; fi
if "$copy/scripts/srcinfo.sh" "${r12[@]}" --write >/dev/null && cmp -s "$c12/.SRCINFO" "$srcinfo12_file" &&
    cmp -s "$copy/onnxruntime-split/.SRCINFO" "$srcinfo_file"; then
    ok "cuda12: --write mends its .SRCINFO, and only that one"
else
    fail "cuda12: --write did not restore the .SRCINFO"
fi
sed -i 's/^pkgrel=2$/pkgrel=3/' "$c12/PKGBUILD"
status=0
out=$("$copy/scripts/srcinfo.sh" "${r12[@]}" --check 2>&1) || status=$?
if ((status == 1)) && grep -qxF -- "+${T}pkgrel = 3" <<<"$out"; then ok "cuda12: a PKGBUILD that no longer matches fails --check"; else
    fail "cuda12: an altered PKGBUILD (exit $status): $out"; fi
cp "$top/onnxruntime-split-cuda12/PKGBUILD" "$c12/"
aur12=$tmp/aur/onnxruntime-split-cuda12
mkdir -p "$aur12"
git -C "$aur12" init -q
if out=$("$copy/scripts/srcinfo.sh" --aur "$aur12" "${r12[@]}" 2>&1) && cmp -s "$aur12/PKGBUILD" "$top/onnxruntime-split-cuda12/PKGBUILD" &&
    cmp -s "$aur12/.SRCINFO" "$srcinfo12_file" && [[ -z $(git -C "$aur12" diff --cached --name-only) ]]; then
    ok "cuda12: --aur copies its PKGBUILD and .SRCINFO into the clone, staging nothing"
else
    fail "cuda12: --aur into a clone: $out"
fi
status=0
out=$("$copy/scripts/srcinfo.sh" "${r12[@]}" --aur "$tmp/none" 2>&1) || status=$?
if ((status == 1)) && grep -qF 'git clone ssh://aur@aur.archlinux.org/onnxruntime-split-cuda12.git' <<<"$out"; then
    ok "cuda12: --aur to a missing directory fails, naming its AUR repository"
else
    fail "cuda12: --aur to a missing directory (exit $status): $out"
fi
rm "$c12/.SRCINFO"
if ! out=$("$copy/scripts/srcinfo.sh" "${r12[@]}" --check 2>&1) &&
    grep -qF 'does not exist: run srcinfo.sh --recipe onnxruntime-split-cuda12 --write' <<<"$out"; then
    ok "cuda12: a missing .SRCINFO fails --check"
else
    fail "cuda12: a missing .SRCINFO: $out"
fi
cp "$srcinfo12_file" "$c12/"

echo "== a copy: the MIGraphX recipe"
cmgx=$copy/onnxruntime-split-migraphx
if "$copy/scripts/srcinfo.sh" "${rmgx[@]}" --check >/dev/null 2>&1; then ok "migraphx: the copy checks out"; else
    fail "migraphx: the unaltered copy fails --check"; fi
sed -i 's/^pkgrel=1$/pkgrel=2/' "$cmgx/PKGBUILD"
status=0
out=$("$copy/scripts/srcinfo.sh" "${rmgx[@]}" --check 2>&1) || status=$?
if ((status == 1)) && grep -qxF -- "+${T}pkgrel = 2" <<<"$out" &&
    grep -qF 'onnxruntime-split-migraphx/.SRCINFO is stale: run srcinfo.sh --recipe onnxruntime-split-migraphx --write' <<<"$out"; then
    ok "migraphx: a PKGBUILD that no longer matches fails --check, naming the recipe"
else
    fail "migraphx: an altered PKGBUILD (exit $status): $out"
fi
if "$copy/scripts/srcinfo.sh" --check >/dev/null 2>&1 && "$copy/scripts/srcinfo.sh" "${r12[@]}" --check >/dev/null 2>&1; then
    ok "migraphx: the other recipes' --check is not affected"
else
    fail "migraphx: a stale migraphx recipe fails another recipe's --check"
fi
cp "$top/onnxruntime-split-migraphx/PKGBUILD" "$cmgx/"
aurmgx=$tmp/aur/onnxruntime-split-migraphx
mkdir -p "$aurmgx"
git -C "$aurmgx" init -q
if out=$("$copy/scripts/srcinfo.sh" "${rmgx[@]}" --aur "$aurmgx" 2>&1) && cmp -s "$aurmgx/PKGBUILD" "$top/onnxruntime-split-migraphx/PKGBUILD" &&
    cmp -s "$aurmgx/.SRCINFO" "$srcinfo_mgx_file" && [[ -z $(git -C "$aurmgx" diff --cached --name-only) ]]; then
    ok "migraphx: --aur copies its PKGBUILD and .SRCINFO into the clone, staging nothing"
else
    fail "migraphx: --aur into a clone: $out"
fi

echo "== --aur"
status=0
out=$("$copy/scripts/srcinfo.sh" --aur "$tmp/none" 2>&1) || status=$?
if ((status == 1)) && grep -qF 'is not the top of a git clone' <<<"$out"; then ok "--aur to a missing directory fails"; else
    fail "--aur to a missing directory (exit $status): $out"; fi
mkdir -p "$tmp/plain"
status=0
out=$("$copy/scripts/srcinfo.sh" --aur "$tmp/plain" 2>&1) || status=$?
if ((status == 1)) && [[ ! -e $tmp/plain/PKGBUILD ]]; then ok "--aur to a directory outside git fails, copies nothing"; else
    fail "--aur to a plain directory (exit $status): $out"; fi
aur=$tmp/aur/onnxruntime-split
mkdir -p "$aur/sub"
git -C "$aur" init -q
status=0
out=$("$copy/scripts/srcinfo.sh" --aur "$aur/sub" 2>&1) || status=$?
if ((status == 1)) && [[ ! -e $aur/sub/PKGBUILD ]]; then ok "--aur to a directory inside a clone fails, copies nothing"; else
    fail "--aur to a subdirectory (exit $status): $out"; fi
if out=$("$copy/scripts/srcinfo.sh" --aur "$aur" 2>&1) && cmp -s "$aur/PKGBUILD" "$top/onnxruntime-split/PKGBUILD" &&
    cmp -s "$aur/.SRCINFO" "$srcinfo_file" && [[ -z $(git -C "$aur" diff --cached --name-only) ]]; then
    ok "--aur copies PKGBUILD and .SRCINFO into the clone, staging nothing"
else
    fail "--aur into a clone: $out"
fi
rm -f "$aur/PKGBUILD" "$aur/.SRCINFO"
sed -i 's/^\tpkgrel = 8$/\tpkgrel = 5/' "$copy/onnxruntime-split/.SRCINFO"
status=0
out=$("$copy/scripts/srcinfo.sh" --aur "$aur" 2>&1) || status=$?
if ((status == 1)) && [[ ! -e $aur/PKGBUILD ]]; then ok "--aur with a stale .SRCINFO fails, copies nothing"; else
    fail "--aur with a stale .SRCINFO (exit $status): $out"; fi

if ((fails > 0)); then
    echo "$fails of $count check(s) failed"
    exit 1
fi
echo "all $count checks passed"
