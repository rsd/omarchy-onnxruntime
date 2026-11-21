#!/usr/bin/env bash
# Tests for scripts/make-source-archive.sh on a small fixture repository
# under mktemp (never the real mirror) with a fake PKGBUILD (--pkgbuild).
# The cases: a re-run gives identical bytes; the .sha256 file verifies;
# git get-tar-commit-id gives the commit; every path has the prefix
# onnxruntime-<pkgver>/; PROVENANCE.md names the commit and the hashes; a tag
# that does not resolve to _ortf_commit (also an annotated one resolving to
# it: accepted) exits 1 and leaves nothing in DIR (not even the directory);
# bad options and a missing repository exit 2.
# Usage: scripts/tests/make-source-archive-test.sh. Exits non-zero on any failure.
# shellcheck disable=SC2016 # the bash -c bodies are single-quoted on purpose
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
tool=$here/../make-source-archive.sh

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fails=0 count=0
ok() { count=$((count + 1)); echo "ok   $1"; }
fail() { count=$((count + 1)); echo "FAIL $1"; fails=$((fails + 1)); }
check() { # check <name> <command...>
    local name=$1
    shift
    if "$@"; then ok "$name"; else fail "$name"; fi
}

export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@example.com GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@example.com
export GIT_AUTHOR_DATE='2026-01-01T00:00:00Z' GIT_COMMITTER_DATE='2026-01-01T00:00:00Z'

repo=$tmp/repo
git init -q "$repo"
mkdir -p "$repo/sub"
echo one >"$repo/a.txt"
echo two >"$repo/sub/b.txt"
git -C "$repo" add -A
git -C "$repo" commit -q -m first
c1=$(git -C "$repo" rev-parse HEAD)
echo three >"$repo/c.txt"
git -C "$repo" add -A
git -C "$repo" commit -q -m second
c2=$(git -C "$repo" rev-parse HEAD)
git -C "$repo" tag v9.9.9 "$c1"

pb() { printf "pkgbase=x\npkgver=9.9.9\n_ortf_commit=%s\n" "$1" >"$tmp/PKGBUILD.$2"; }
pb "$c1" good
pb "$c2" bad

out=$tmp/out
status=0
"$tool" --repo "$repo" --out "$out" --pkgbuild "$tmp/PKGBUILD.good" >"$tmp/log" 2>&1 || status=$?
check "run exits 0" test "$status" -eq 0
f=$out/onnxruntime-9.9.9.tar.gz
check "tarball, .sha256 and PROVENANCE.md exist" test -f "$f" -a -f "$f.sha256" -a -f "$out/PROVENANCE.md"
check "the .sha256 file verifies" bash -c 'cd "$1" && sha256sum -c onnxruntime-9.9.9.tar.gz.sha256 >/dev/null' _ "$out"
check "get-tar-commit-id gives the commit" test "$(gzip -dc "$f" | git get-tar-commit-id)" = "$c1"
check "every path has the prefix" bash -c '! gzip -dc "$1" | tar -t | grep -qv "^onnxruntime-9.9.9/"' _ "$f"
check "the tree is that of the commit, not of HEAD" bash -c 'gzip -dc "$1" | tar -t | grep -q "sub/b.txt" && ! gzip -dc "$1" | tar -t | grep -q c.txt' _ "$f"
check "no gzip name or time in the header" test "$(head -c 10 "$f" | od -An -tx1 | tr -d ' \n')" = 1f8b0800000000000203
sha=$(sha256sum "$f" | cut -d' ' -f1)
check "PROVENANCE.md names the commit and the sha256" bash -c 'grep -q "$2" "$1" && grep -q "$3" "$1"' _ "$out/PROVENANCE.md" "$c1" "$sha"
check "PROVENANCE.md says the lightweight tag is lightweight" grep -q 'v9.9.9. (lightweight)' "$out/PROVENANCE.md"
check "the output names the sha256" grep -q "$sha" "$tmp/log"
check "the output prints the gh command" grep -q -- 'gh release create source-v9.9.9 .*--latest=false' "$tmp/log"

sleep 1
out2=$tmp/out2
"$tool" --repo "$repo" --out "$out2" --pkgbuild "$tmp/PKGBUILD.good" >/dev/null 2>&1
check "a re-run gives identical bytes" cmp -s "$f" "$out2/onnxruntime-9.9.9.tar.gz"
"$tool" --repo "$repo" --out "$out" --pkgbuild "$tmp/PKGBUILD.good" >/dev/null 2>&1
check "a re-run over the same DIR leaves no temporary file" test "$(find "$out" -mindepth 1 | wc -l)" -eq 3

git -C "$repo" tag -a -m annotated v8.8.8 "$c1"
printf "pkgver=8.8.8\n_ortf_commit=%s\n" "$c1" >"$tmp/PKGBUILD.ann"
status=0
"$tool" --repo "$repo" --out "$tmp/out3" --pkgbuild "$tmp/PKGBUILD.ann" >/dev/null 2>&1 || status=$?
check "an annotated tag at the commit is accepted" test "$status" -eq 0
check "PROVENANCE.md says the annotated tag is annotated" grep -q 'v8.8.8. (annotated)' "$tmp/out3/PROVENANCE.md"

status=0
"$tool" --repo "$repo" --out "$tmp/outbad" --pkgbuild "$tmp/PKGBUILD.bad" >/dev/null 2>&1 || status=$?
check "a mismatching tag exits 1" test "$status" -eq 1
check "a mismatching tag leaves nothing" test ! -e "$tmp/outbad"
mkdir "$tmp/outbad2"
"$tool" --repo "$repo" --out "$tmp/outbad2" --pkgbuild "$tmp/PKGBUILD.bad" >/dev/null 2>&1 || true
check "a mismatching tag leaves DIR empty" test -z "$(ls -A "$tmp/outbad2")"
printf "pkgver=7.7.7\n_ortf_commit=%s\n" "$c1" >"$tmp/PKGBUILD.notag"
status=0
"$tool" --repo "$repo" --out "$tmp/outbad3" --pkgbuild "$tmp/PKGBUILD.notag" >/dev/null 2>&1 || status=$?
check "a missing tag exits 1" test "$status" -eq 1

status=0
"$tool" --bogus >/dev/null 2>&1 || status=$?
check "a bad option exits 2" test "$status" -eq 2
status=0
"$tool" --out >/dev/null 2>&1 || status=$?
check "an option without value exits 2" test "$status" -eq 2
status=0
"$tool" --repo "$tmp/nonexistent" --out "$tmp/o4" --pkgbuild "$tmp/PKGBUILD.good" >/dev/null 2>&1 || status=$?
check "a missing repository exits 2" test "$status" -eq 2
status=0
"$tool" --repo "$repo" --out "$tmp/o5" --pkgbuild "$tmp/none" >/dev/null 2>&1 || status=$?
check "a missing PKGBUILD exits 2" test "$status" -eq 2

echo "$((count - fails))/$count passed"
exit $((fails > 0))
