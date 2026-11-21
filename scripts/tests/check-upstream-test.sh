#!/usr/bin/env bash
# Tests for scripts/check-upstream.sh on fixture repositories under mktemp,
# reached with --url (a path and a file:// URL; no network) and a fake
# PKGBUILD (ORTF_PKGBUILD). The cases: a lightweight and an annotated tag at
# our commit exit 0; a tag at another commit exits 1 (and says which); no
# such tag exits 1; a repository that does not exist exits 3; a bad option
# exits 2; a global git config (insteadOf) is ignored; a malformed
# _ortf_commit exits 2.
# Usage: scripts/tests/check-upstream-test.sh. Exits non-zero on any failure.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
tool=$here/../check-upstream.sh

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fails=0 count=0
ok() { count=$((count + 1)); echo "ok   $1"; }
fail() { count=$((count + 1)); echo "FAIL $1"; fails=$((fails + 1)); }

export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@example.com GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@example.com

mkrepo() { # mkrepo <dir>: two commits, sets c1 (ours) and c2
    git init -q "$1"
    echo one >"$1/a"
    git -C "$1" add -A
    git -C "$1" commit -q -m first
    c1=$(git -C "$1" rev-parse HEAD)
    echo two >"$1/b"
    git -C "$1" add -A
    git -C "$1" commit -q -m second
    c2=$(git -C "$1" rev-parse HEAD)
}
mkrepo "$tmp/light"
git -C "$tmp/light" tag v9.9.9 "$c1"
mkrepo "$tmp/annot"
git -C "$tmp/annot" tag -a -m annotated v9.9.9 "$c1"
mkrepo "$tmp/moved"
git -C "$tmp/moved" tag v9.9.9 "$c2"
mkrepo "$tmp/notag"
git -C "$tmp/notag" tag v1.0.0 "$c1"
printf 'pkgver=9.9.9\n_ortf_commit=%s\n' "$c1" >"$tmp/PKGBUILD"
export ORTF_PKGBUILD=$tmp/PKGBUILD

run() { # run <expected status> <name> <args...>
    local want=$1 name=$2 rc=0
    shift 2
    "$tool" "$@" >"$tmp/log" 2>&1 || rc=$?
    if [[ $rc -eq $want ]]; then ok "$name"; else fail "$name (exit $rc, want $want): $(head -n 2 "$tmp/log")"; fi
}
run 0 "lightweight tag at our commit" --url "$tmp/light"
run 0 "annotated tag at our commit (peeled)" --url "$tmp/annot"
run 0 "file:// URL" --url "file://$tmp/light"
run 1 "tag at another commit" --url "$tmp/moved"
if grep -q "$c2" "$tmp/log"; then ok "it names the other commit"; else fail "it names the other commit"; fi
run 1 "no such tag" --url "$tmp/notag"
run 3 "nonexistent repository" --url "$tmp/nonexistent"
run 2 "bad option" --bogus
run 2 "option without value" --url

# The user's git setup is ignored: a global insteadOf that rewrites the URL to
# a nonexistent path must not matter, and the HOME is not read.
mkdir "$tmp/fakehome"
printf '[url "%s/nonexistent/"]\n\tinsteadOf = %s/\n' "$tmp" "$tmp" >"$tmp/fakehome/gitconfig"
rc=0
HOME=$tmp/fakehome GIT_CONFIG_GLOBAL=$tmp/fakehome/gitconfig "$tool" --url "$tmp/light" >"$tmp/log" 2>&1 || rc=$?
if [[ $rc -eq 0 ]]; then ok "a global insteadOf is ignored (the fixture is still reached)"; else fail "a global insteadOf is ignored (exit $rc): $(head -n 2 "$tmp/log")"; fi
# The rewrite really bites a plain git ls-remote (so the test above proves something).
rc=0
HOME=$tmp/fakehome GIT_CONFIG_GLOBAL=$tmp/fakehome/gitconfig git ls-remote "$tmp/light" >/dev/null 2>&1 || rc=$?
if [[ $rc -ne 0 ]]; then ok "the insteadOf fixture breaks a plain ls-remote"; else fail "the insteadOf fixture has no effect"; fi

# _ortf_commit must be 40 lowercase hex digits.
for bad in "" abc "${c1^^}" "${c1}0"; do
    printf 'pkgver=9.9.9\n_ortf_commit=%s\n' "$bad" >"$tmp/PKGBUILD.bad"
    rc=0
    ORTF_PKGBUILD=$tmp/PKGBUILD.bad "$tool" --url "$tmp/light" >"$tmp/log" 2>&1 || rc=$?
    if [[ $rc -eq 2 ]] && grep -q "_ortf_commit" "$tmp/log"; then ok "bad _ortf_commit '$bad' exits 2"; else fail "bad _ortf_commit '$bad' (exit $rc)"; fi
done

echo "$((count - fails))/$count passed"
exit $((fails > 0))
