#!/usr/bin/env bash
# Tests for scripts/srcinfo.sh and the committed onnxruntime-split/.SRCINFO:
# --check passes on the tree; on a copy of the script and the recipe (under
# mktemp) an altered .SRCINFO or PKGBUILD fails it, with a diff, and
# --write mends it; the output is the same whatever ORTF_* variables the
# caller has; --aur copies into the top of a git clone only; the committed
# .SRCINFO has the AUR's default names (CUDA 13, the rtx preset), tensorrt
# in makedepends, no MIGraphX optdepend and this pkgrel; git neither ignores it nor would the
# licence guard refuse it.
# Usage: scripts/tests/srcinfo-test.sh. Exits non-zero on any failure.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
top=$(cd "$here/../.." && pwd)
srcinfo_file=$top/onnxruntime-split/.SRCINFO

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

echo "== a copy of the script and the recipe"
copy=$tmp/repo
mkdir -p "$copy/scripts" "$copy/onnxruntime-split"
cp "$top/scripts/srcinfo.sh" "$copy/scripts/"
cp "$top/onnxruntime-split/PKGBUILD" "$srcinfo_file" "$copy/onnxruntime-split/"
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
for args in '--bogus' '--aur' '--write extra'; do
    status=0
    # shellcheck disable=SC2086 # the arguments are words
    "$copy/scripts/srcinfo.sh" $args >/dev/null 2>&1 || status=$?
    if ((status == 2)); then ok "usage error for [$args]"; else fail "[$args] exits $status, not 2"; fi
done

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
