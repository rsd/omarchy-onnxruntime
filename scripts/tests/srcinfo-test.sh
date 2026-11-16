#!/usr/bin/env bash
# Tests for scripts/srcinfo.sh and the committed onnxruntime-split/.SRCINFO:
# --check passes on the tree; on a copy of the script and the recipe (under
# mktemp) an altered .SRCINFO or PKGBUILD fails it, with a diff, and
# --write mends it; the output is the same whatever ORTF_* variables the
# caller has; --aur copies into the top of a git clone only; the committed
# .SRCINFO has the AUR's default names (CUDA 13, the rtx preset), tensorrt
# in makedepends, no MIGraphX optdepend and this pkgrel; git neither ignores it nor would the
# licence guard refuse it. The same, with --recipe onnxruntime-split-cuda12,
# for the cuda12 line's .SRCINFO: its pkgbase, its one package
# (onnxruntime-provider-cuda12-legacy, CUDA 12 names on this host whatever
# its CUDA), its onnxruntime-core=<pkgver> pin and provides and conflicts,
# its pkgrel and makedepends; --recipe combines with --write, --check and
# --aur, and an unknown recipe is a usage error.
# Usage: scripts/tests/srcinfo-test.sh. Exits non-zero on any failure.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
top=$(cd "$here/../.." && pwd)
srcinfo_file=$top/onnxruntime-split/.SRCINFO
srcinfo12_file=$top/onnxruntime-split-cuda12/.SRCINFO
r12=(--recipe onnxruntime-split-cuda12)

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
for line in 'pkgbase = onnxruntime-split' "${T}pkgrel = 6" 'pkgname = onnxruntime-core' \
    'pkgname = onnxruntime-provider-cuda13-rtx' 'pkgname = onnxruntime-provider-tensorrt-cuda13' \
    'pkgname = onnxruntime-provider-openvino' 'pkgname = python-onnxruntime-core' \
    "${T}makedepends = tensorrt" "${T}makedepends = cuda" "${T}makedepends = cudnn"; do
    if grep -qxF -- "$line" "$srcinfo_file"; then ok "has [$line]"; else fail "lacks [$line]"; fi
done
if grep -n 'cudax' "$srcinfo_file"; then fail "a name with the old fallback cudax (above)"; else
    ok "no cudax name"; fi
if grep -nE '^pkgname = onnxruntime-provider-migraphx$|makedepends = (migraphx|hip-runtime-amd)$|optdepends = onnxruntime-provider-migraphx' "$srcinfo_file"; then
    fail "the default build has no MIGraphX package, ROCm build input or MIGraphX optdepend (above)"
else
    ok "no MIGraphX package, ROCm build input or MIGraphX optdepend"
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
if out=$(ORTF_EPS='cuda tensorrt openvino' ORTF_CUDA_ARCHS=rtx ORTF_PYTHON=1 "$top/scripts/srcinfo.sh" "${r12[@]}" --check 2>&1); then
    ok "--recipe onnxruntime-split-cuda12 --check ignores the ORTF_* variables"
else
    fail "--recipe onnxruntime-split-cuda12 --check with ORTF_* set: $out"
fi
if out=$("$top/scripts/srcinfo.sh" "${r12[@]}") && [[ $out == "$(<"$srcinfo12_file")" ]]; then
    ok "--recipe onnxruntime-split-cuda12 alone prints its .SRCINFO"
else
    fail "the printed cuda12 .SRCINFO differs from the committed one"
fi
for line in 'pkgbase = onnxruntime-split-cuda12' "${T}pkgrel = 1" "${T}pkgver = 1.29.0" \
    'pkgname = onnxruntime-provider-cuda12-legacy' "${T}depends = onnxruntime-core=1.29.0" \
    "${T}provides = onnxruntime-provider-cuda=1.29.0" "${T}provides = onnxruntime-provider-cuda12=1.29.0" \
    "${T}conflicts = onnxruntime-provider-cuda" "${T}conflicts = onnxruntime-provider-cuda12" \
    "${T}depends = nvidia-utils" "${T}makedepends = cuda" "${T}makedepends = cudnn" "${T}makedepends = python"; do
    if grep -qxF -- "$line" "$srcinfo12_file"; then ok "cuda12: has [$line]"; else fail "cuda12: lacks [$line]"; fi
done
if (($(grep -c '^pkgname = ' "$srcinfo12_file") == 1)) &&
    ! grep -nE 'makedepends = (tensorrt|openvino|python-.*|migraphx|hip-runtime-amd)$|cuda13|onnxruntime-core=1\.29\.0-' "$srcinfo12_file"; then
    ok "cuda12: one package; no TensorRT, OpenVINO, Python-module or ROCm build input; no CUDA 13 name; no pkgrel pin"
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

echo "== a copy of the script and the recipe"
copy=$tmp/repo
mkdir -p "$copy/scripts" "$copy/onnxruntime-split" "$copy/onnxruntime-split-cuda12"
cp "$top/scripts/srcinfo.sh" "$copy/scripts/"
cp "$top/onnxruntime-split/PKGBUILD" "$srcinfo_file" "$copy/onnxruntime-split/"
cp "$top/onnxruntime-split-cuda12/PKGBUILD" "$srcinfo12_file" "$copy/onnxruntime-split-cuda12/"
if "$copy/scripts/srcinfo.sh" --check >/dev/null 2>&1; then ok "the copy checks out"; else
    fail "the unaltered copy fails --check"; fi
sed -i 's/^\tpkgrel = 6$/\tpkgrel = 5/' "$copy/onnxruntime-split/.SRCINFO"
status=0
out=$("$copy/scripts/srcinfo.sh" --check 2>&1) || status=$?
if ((status == 1)) && grep -qxF -- "-${T}pkgrel = 5" <<<"$out" && grep -qxF -- "+${T}pkgrel = 6" <<<"$out" &&
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
    '--recipe onnxruntime-split --recipe onnxruntime-split --check' '--write --check' '--recipe onnxruntime-split-cuda12 --aur'; do
    status=0
    # shellcheck disable=SC2086 # the arguments are words
    "$copy/scripts/srcinfo.sh" $args >/dev/null 2>&1 || status=$?
    if ((status == 2)); then ok "usage error for [$args]"; else fail "[$args] exits $status, not 2"; fi
done

echo "== a copy: the cuda12 recipe"
c12=$copy/onnxruntime-split-cuda12
if "$copy/scripts/srcinfo.sh" "${r12[@]}" --check >/dev/null 2>&1; then ok "cuda12: the copy checks out"; else
    fail "cuda12: the unaltered copy fails --check"; fi
sed -i 's/^\tpkgrel = 1$/\tpkgrel = 7/' "$c12/.SRCINFO"
status=0
out=$("$copy/scripts/srcinfo.sh" "${r12[@]}" --check 2>&1) || status=$?
if ((status == 1)) && grep -qxF -- "-${T}pkgrel = 7" <<<"$out" && grep -qxF -- "+${T}pkgrel = 1" <<<"$out" &&
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
sed -i 's/^pkgrel=1$/pkgrel=2/' "$c12/PKGBUILD"
status=0
out=$("$copy/scripts/srcinfo.sh" "${r12[@]}" --check 2>&1) || status=$?
if ((status == 1)) && grep -qxF -- "+${T}pkgrel = 2" <<<"$out"; then ok "cuda12: a PKGBUILD that no longer matches fails --check"; else
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
sed -i 's/^\tpkgrel = 6$/\tpkgrel = 5/' "$copy/onnxruntime-split/.SRCINFO"
status=0
out=$("$copy/scripts/srcinfo.sh" --aur "$aur" 2>&1) || status=$?
if ((status == 1)) && [[ ! -e $aur/PKGBUILD ]]; then ok "--aur with a stale .SRCINFO fails, copies nothing"; else
    fail "--aur with a stale .SRCINFO (exit $status): $out"; fi

if ((fails > 0)); then
    echo "$fails of $count check(s) failed"
    exit 1
fi
echo "all $count checks passed"
