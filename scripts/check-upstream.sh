#!/usr/bin/env bash
# Is github.com/microsoft/onnxruntime back, and is v<pkgver> still our commit?
#
#   scripts/check-upstream.sh [--url URL]
#
# Run it before a release, and from time to time while the mirror (the
# release source-v<pkgver> of this repository) is the default source:
# upstream has returned 404 since 2026-11-17.
#
# It reads pkgver and _ortf_commit from onnxruntime-split/PKGBUILD (the
# top-level lines; never sourced) and runs
# `git ls-remote URL refs/tags/v<pkgver> refs/tags/v<pkgver>^{}` with no
# credentials and no prompts, and isolated from the user's git setup: no
# global or system git config (GIT_CONFIG_GLOBAL=/dev/null,
# GIT_CONFIG_NOSYSTEM=1), so no insteadOf rewrite and no http.extraHeader;
# HOME and XDG_CONFIG_HOME are an empty temporary directory, so no ~/.netrc
# and no ~/.ssh; ssh is refused (protocol.ssh.allow=never, GIT_SSH_COMMAND=false);
# GIT_TERMINAL_PROMPT=0, askpass disabled, no credential helper, timeout 60.
# The repository's own config is not read (ls-remote uses none). The peeled
# ^{} line is used when present, else the plain one, and compared with
# _ortf_commit (40 lowercase hex digits, else exit 2).
#
# --url URL  the repository (default https://github.com/microsoft/onnxruntime.git;
#            it exists for the tests and for other mirrors)
# ORTF_PKGBUILD=FILE in the environment reads another PKGBUILD (for the tests).
#
# Exit status and what it means for the maintainer:
#   0  back and identical: tag v<pkgver> points at _ortf_commit. Switching the
#      default back to ORTF_SOURCE=upstream is possible; it is a maintainer decision.
#   1  back but different: the repository answers, and the tag is missing or
#      points at another commit (it says which). Do NOT switch: investigate;
#      a moved tag is different source.
#   3  still gone: ls-remote failed (404 or removed, or no network; the two are
#      not distinguished, git's error line is printed). Keep the mirror.
#   2  usage error, or the PKGBUILD cannot be read.
set -euo pipefail

usage() { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "$0"; }
die() { # die <status> <message>
    echo "check-upstream: $2" >&2
    exit "$1"
}

url=https://github.com/microsoft/onnxruntime.git
while (($#)); do
    case $1 in
        -h | --help) usage; exit 0 ;;
        --url) (($# >= 2)) || die 2 "--url needs a URL"; url=$2; shift 2 ;;
        --url=*) url=${1#*=}; shift ;;
        *) die 2 "unknown argument: $1 (see --help)" ;;
    esac
done

top=$(cd "$(dirname "$0")/.." && pwd)
pkgbuild=${ORTF_PKGBUILD:-$top/onnxruntime-split/PKGBUILD}
[[ -r $pkgbuild ]] || die 2 "$pkgbuild: cannot read it"
value() { sed -n "s/^$1=//p" "$pkgbuild" | head -n 1 | tr -d "\"'"; }
pkgver=$(value pkgver)
commit=$(value _ortf_commit)
[[ -n $pkgver && -n $commit ]] || die 2 "$pkgbuild: no pkgver= or _ortf_commit= line"
[[ $commit =~ ^[0-9a-f]{40}$ ]] || die 2 "$pkgbuild: no usable _ortf_commit= line (40 hex digits)"
tag=v$pkgver

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/home"
rc=0
env GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 HOME="$tmp/home" XDG_CONFIG_HOME="$tmp/home" \
    GIT_SSH_COMMAND=false GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=true SSH_ASKPASS=true GCM_INTERACTIVE=never \
    timeout 60 git -c protocol.ssh.allow=never -c credential.helper= -c core.askPass=true \
    ls-remote "$url" "refs/tags/$tag" "refs/tags/$tag^{}" >"$tmp/out" 2>"$tmp/err" </dev/null || rc=$?
if ((rc != 0)); then
    msg=$(grep -v '^$' "$tmp/err" | tail -n 1 || true)
    [[ $rc -eq 124 ]] && msg="timed out after 60 s"
    echo "still gone: $url is not reachable (${msg:-git exit $rc}); 404/removed and no network are not distinguished: keep the mirror"
    exit 3
fi

plain=$(awk -v r="refs/tags/$tag" '$2 == r {print $1}' "$tmp/out")
peeled=$(awk -v r="refs/tags/$tag^{}" '$2 == r {print $1}' "$tmp/out")
got=${peeled:-$plain}
if [[ -z $got ]]; then
    echo "back but different: $url answers and has no tag $tag (expected $commit): do not switch, investigate"
    exit 1
fi
if [[ $got != "$commit" ]]; then
    echo "back but different: $url has $tag at $got, _ortf_commit is $commit: do not switch, investigate"
    exit 1
fi
echo "back and identical: $url has $tag at $commit: switching the default to ORTF_SOURCE=upstream is possible (maintainer decision)"
