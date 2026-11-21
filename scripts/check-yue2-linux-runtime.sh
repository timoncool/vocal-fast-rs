#!/usr/bin/env bash
set -euo pipefail

root=${1:?Pass the extracted runtime directory}
test -x "$root/vocal-fast"
for elf in "$root/vocal-fast" "$root/"*.so*; do
    echo "--- ${elf##*/}"
    readelf -h "$elf" | grep -E 'Class:|Machine:'
    readelf -d "$elf" | grep -E 'NEEDED|RPATH|RUNPATH' || true
    if readelf -d "$elf" | grep -E 'RPATH|RUNPATH' | grep -q '/usr\|/tmp\|/work\|/src'; then
        echo "Absolute build rpath in $elf" >&2
        exit 1
    fi
    for symbol in GLIBC GLIBCXX; do
        maximum=$(objdump -T "$elf" | awk '/\*UND\*/ {print}' |
            grep -Eo "${symbol}_[0-9.]+[0-9]" | sort -uV | tail -n1 || true)
        [ -z "$maximum" ] || echo "$maximum"
        case "$symbol" in
            GLIBC) limit=GLIBC_2.35 ;;
            GLIBCXX) limit=GLIBCXX_3.4.30 ;;
        esac
        if [ -n "$maximum" ] && [ "$(printf '%s\n' "$maximum" "$limit" | sort -V | tail -n1)" != "$limit" ]; then
            echo "Incompatible required symbol in $elf: $maximum exceeds $limit" >&2
            exit 1
        fi
    done
done
dependencies=$(LD_LIBRARY_PATH="$root" ldd "$root/vocal-fast")
printf '%s\n' "$dependencies"
if grep -q 'not found' <<< "$dependencies"; then exit 1; fi
LD_LIBRARY_PATH="$root" "$root/vocal-fast" --help
