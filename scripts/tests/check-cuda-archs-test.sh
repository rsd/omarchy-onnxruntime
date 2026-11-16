#!/usr/bin/env bash
# Tests for scripts/check-cuda-archs.sh with a fake cuobjdump (a script under
# mktemp that prints the listing kept in the file named by $FAKE_ELF and
# $FAKE_PTX, in cuobjdump's format: "ELF file    1: <name>.sm_80.cubin",
# "PTX file    1: <name>.sm_60.ptx"; the real tool is never run). The cases:
# an exact match (the preset legacy: 60 61 70 75 80 86 89 120 60-virtual);
# a missing SASS arch; an extra one; no PTX when one is expected; a wrong
# PTX arch; an extra PTX when none is expected (and none expected, none
# listed: a match, also with the "does not contain" message and exit status
# of the real tool); sm_120a counted as 120; N-real as N; the input as a
# .so and as a .pkg.tar.zst (built with bsdtar; one without the library is
# exit 3); a missing file, a missing cuobjdump and a failing cuobjdump
# (exit 3); usage errors (exit 2).
# Needs bsdtar; without it it says so and skips.
# Usage: scripts/tests/check-cuda-archs-test.sh. Exits non-zero on any failure.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
tool=$here/../check-cuda-archs.sh

if ! command -v bsdtar >/dev/null 2>&1; then
    echo "skip: bsdtar not found"
    exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fails=0 count=0
ok() { count=$((count + 1)); echo "ok   $1"; }
fail() { count=$((count + 1)); echo "FAIL $1"; fails=$((fails + 1)); }

fake=$tmp/cuobjdump
cat >"$fake" <<'EOF'
#!/usr/bin/env bash
# The fake: --list-elf / --list-ptx print the listing files; an empty one
# prints what the real tool says and exits 1.
case $1 in
    --list-elf) list=$FAKE_ELF ;;
    --list-ptx) list=$FAKE_PTX ;;
    *) exit 64 ;;
esac
if [[ -s $list ]]; then
    cat "$list"
else
    echo "cuobjdump info    : File '$2' does not contain device code"
    exit 1
fi
EOF
chmod +x "$fake"
export FAKE_ELF=$tmp/elf FAKE_PTX=$tmp/ptx

# listing <kind> <sm names...>: the listing of the fake tool, e.g. 80 120a.
listing() {
    local kind=$1 ext i=1 a
    shift
    [[ $kind == ELF ]] && ext=cubin || ext=ptx
    for a in "$@"; do
        printf '%s file %4d: libonnxruntime_providers_cuda.%d.sm_%s.%s\n' "$kind" "$i" "$i" "$a" "$ext"
        i=$((i + 1))
    done
}
set_fake() { # set_fake "<sass sms>" "<ptx sms>"
    # shellcheck disable=SC2086 # the lists are words
    listing ELF $1 >"$FAKE_ELF"
    # shellcheck disable=SC2086
    listing PTX $2 >"$FAKE_PTX"
}

# The library: a .so and a package holding it (with its versioned symlink
# name, as the build installs it, and another file that must be ignored).
so=$tmp/libonnxruntime_providers_cuda.so
echo 'fake elf' >"$so"
root=$tmp/root
mkdir -p "$root/usr/lib"
echo 'fake elf' >"$root/usr/lib/libonnxruntime_providers_cuda.so"
echo 'other' >"$root/usr/lib/libother.so"
(cd "$root" && bsdtar --zstd -cf "$tmp/good.pkg.tar.zst" usr)
mkdir -p "$tmp/empty/usr/lib"
echo 'other' >"$tmp/empty/usr/lib/libother.so"
(cd "$tmp/empty" && bsdtar --zstd -cf "$tmp/nolib.pkg.tar.zst" usr)

legacy=(60 61 70 75 80 86 89 120 60-virtual)
# run <args...>: the tool with the fake; sets status and out.
run() {
    status=0
    out=$("$tool" --cuobjdump "$fake" "$@" 2>&1) || status=$?
}
has() { grep -qE -- "$1" <<<"$out"; }
expect() { # expect <description> <status> <regex> <args...>
    local what=$1 want=$2 re=$3
    shift 3
    run "$@"
    if ((status == want)) && has "$re"; then
        ok "$what"
    else
        fail "$what (exit $status, want $want): $out"
    fi
}

set_fake "60 61 70 75 80 86 89 120a" "60"
expect "an exact match on a .so (sm_120a counts as 120)" 0 'SASS found: 60 61 70 75 80 86 89 120' "$so" "${legacy[@]}"
if has 'PTX  found: 60' && ! has MISMATCH; then ok "the match prints found and want, no mismatch"; else fail "match output: $out"; fi
expect "an exact match on a .pkg.tar.zst" 0 'PTX  found: 60' "$tmp/good.pkg.tar.zst" "${legacy[@]}"
expect "N-real is SASS for N" 0 'SASS found: 60 61 70 75 80 86 89 120' "$so" 60-real 61-real 70 75 80 86 89 120-real 60-virtual
set_fake "60 61 70 75 80 86 89 120a" "60"
expect "an arch listed twice counts once" 0 'SASS found' "$so" "${legacy[@]}" 60

set_fake "60 61 70 75 80 86 89" "60"
expect "a missing SASS arch" 1 'MISMATCH' "$so" "${legacy[@]}"
set_fake "60 61 70 75 80 86 89 90a 120a" "60"
expect "an extra SASS arch (sm_90a)" 1 'SASS found: 60 61 70 75 80 86 89 90 120' "$so" "${legacy[@]}"
set_fake "60 61 70 75 80 86 89 120a" ""
expect "no PTX where one is expected" 1 'PTX  found: none' "$so" "${legacy[@]}"
set_fake "60 61 70 75 80 86 89 120a" "100"
expect "the wrong PTX arch" 1 'PTX  found: 100' "$so" "${legacy[@]}"
set_fake "60 61 70 75 80 86 89 120a" "120 100"
expect "an extra PTX arch" 1 'PTX  found: 100 120' "$so" "${legacy[@]}"
set_fake "60 61 70 75 80 86 89 120a" "60"
expect "PTX where none is expected" 1 'PTX  want:  none' "$so" 60 61 70 75 80 86 89 120
set_fake "75 80" ""
expect "no PTX expected, none listed (the tool says 'does not contain' and exits 1)" 0 'PTX  found: none' "$so" 75 80
set_fake "" ""
expect "no code at all against an expectation" 1 'SASS found: none' "$so" 80

set_fake "60 61 70 75 80 86 89 120a" "60"
expect "a package without the library: exit 3" 3 'no usr/lib/libonnxruntime_providers_cuda\.so\* in it' "$tmp/nolib.pkg.tar.zst" "${legacy[@]}"
expect "a missing file: exit 3" 3 'no such file' "$tmp/none.so" "${legacy[@]}"
status=0
out=$("$tool" --cuobjdump "$tmp/no-such-tool" "$so" "${legacy[@]}" 2>&1) || status=$?
if ((status == 3)) && has 'cuobjdump not found'; then ok "a missing cuobjdump: exit 3"; else fail "missing cuobjdump (exit $status): $out"; fi
printf '#!/usr/bin/env bash\necho "cuobjdump fatal : cannot read" >&2\nexit 1\n' >"$tmp/broken"
chmod +x "$tmp/broken"
status=0
out=$("$tool" --cuobjdump "$tmp/broken" "$so" "${legacy[@]}" 2>&1) || status=$?
if ((status == 3)) && has 'failed \(exit 1\)'; then ok "a failing cuobjdump: exit 3"; else fail "failing cuobjdump (exit $status): $out"; fi

# Usage errors.
status=0
out=$("$tool" 2>&1) || status=$?
if ((status == 2)) && has 'Usage|check-cuda-archs\.sh'; then ok "no arguments: usage, exit 2"; else fail "no arguments (exit $status): $out"; fi
expect "a file but no arch: exit 2" 2 'check-cuda-archs\.sh' "$so"
expect "an arch that is none: exit 2" 2 'x: not an arch' "$so" 80 x
expect "an arch with a letter (80a): exit 2" 2 '80a: not an arch' "$so" 80a
expect "an unknown option: exit 2" 2 'unknown option: --nope' --nope "$so" 80
status=0
out=$("$tool" --cuobjdump 2>&1) || status=$?
if ((status == 2)) && has 'needs a path'; then ok "--cuobjdump without a path: exit 2"; else fail "--cuobjdump alone (exit $status): $out"; fi
status=0
out=$("$tool" --help 2>&1) || status=$?
if ((status == 0)) && has 'cuobjdump --list-elf'; then ok "--help: the header, exit 0"; else fail "--help (exit $status): $out"; fi

echo "$((count - fails))/$count passed"
((fails == 0))
