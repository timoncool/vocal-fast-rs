#!/usr/bin/env bash
set -euo pipefail

mode=${1:?Pass prepare or verify}
root=${2:?Pass the component directory}

rpaths() {
  otool -l "$1" | awk '
    $1 == "cmd" { rpath = $2 == "LC_RPATH" }
    rpath && $1 == "path" {
      sub(/^[[:space:]]*path[[:space:]]+/,"")
      sub(/[[:space:]]+\(offset [0-9]+\)$/,"")
      print
    }'
}

if [[ "$mode" == prepare ]]; then
  for binary in "$root/vocal-fast" "$root"/*.dylib; do
    [[ -L "$binary" ]] && continue
    changed=false
    paths=$(rpaths "$binary")
    while IFS= read -r path; do
      case "$path" in
        /Applications/Xcode*.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/*)
          install_name_tool -delete_rpath "$path" "$binary"
          changed=true
          ;;
      esac
    done <<< "$paths"
    if "$changed"; then
      codesign --force --sign - --timestamp=none "$binary"
    fi
  done
elif [[ "$mode" != verify ]]; then
  exit 2
fi

# Verify the executable and every shipped library. A relative load command must
# resolve to another verified file beside the binary. System libraries resolve
# through macOS, including those held in the dynamic linker cache.
for binary in "$root/vocal-fast" "$root"/*.dylib; do
  test -f "$binary"
  if [[ -L "$binary" ]]; then
    target=$(readlink "$binary")
    [[ "$target" != */* && "$target" == *.dylib ]]
  fi
  file -L "$binary" | grep -E 'Mach-O.*arm64'
  codesign --verify --strict "$binary"
  minimum=$(otool -l "$binary" | awk '$1 == "minos" {print $2}')
  printf '%s minimum macOS %s\n' "$binary" "$minimum"
  [[ "$minimum" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]
  awk -v version="$minimum" 'BEGIN {split(version, v, "."); exit !(v[1] < 13 || (v[1] == 13 && v[2] == 0 && v[3] == 0))}'
  paths=$(rpaths "$binary")
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    case "$path" in
      @executable_path|@loader_path) printf '  RUNPATH %s\n' "$path" ;;
      *) printf 'Invalid RUNPATH in %s: %s\n' "$binary" "$path" >&2; exit 1 ;;
    esac
  done <<< "$paths"
  dependencies=$(otool -L "$binary" | tail -n +2 |
    sed -E 's/^[[:space:]]+//; s/[[:space:]]+\(compatibility version .*$//')
  while IFS= read -r dependency; do
    [[ -z "$dependency" ]] && continue
    printf '  dependency %s\n' "$dependency"
    case "$dependency" in
      /System/Library/*|/usr/lib/*) ;;
      @rpath/*|@loader_path/*|@executable_path/*)
        name=${dependency#*/}
        [[ "$name" != */* && -f "$root/$name" && "$name" == *.dylib ]]
        ;;
      *) printf 'Unresolved dependency in %s: %s\n' "$binary" "$dependency" >&2; exit 1 ;;
    esac
  done <<< "$dependencies"
done
rpaths "$root/vocal-fast" | grep -Fx '@executable_path'
