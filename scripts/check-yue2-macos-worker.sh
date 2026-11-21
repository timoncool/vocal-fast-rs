#!/usr/bin/env bash
set -euo pipefail
worker=${1:?Pass the worker binary}
fixture=${2:?Pass an unused fixture directory}
backend=${3:-cpu-graph}
mkdir -p "$fixture"
acoustics=https://modelscope.cn/api/v1/models/MosRat/vfrm-models/repo?Revision=9527a1bcf861c70717d4aba619ba92f2b5f42396

fetch() {
    curl --fail --location --retry 3 --silent --show-error --max-time 360 "$2" -o "$fixture/$1"
    printf '%s  %s\n' "$3" "$fixture/$1" | shasum -a 256 --check
}

fetch hubert.vfrm "$acoustics&FilePath=shared%2Fhubert.vfrm" a84fed477f49a040d53b44b7c27184d711fb246d5f9119c7c046941aaee65c59 &
hubert=$!
fetch rmvpe.vfrm "$acoustics&FilePath=shared%2Frmvpe.vfrm" 462ee5494525ed1029abfad368abb56ca20285478e24aa846575ed400bff4734 &
rmvpe=$!
wait "$hubert"
wait "$rmvpe"

# Exercise readiness and framing with an invalid request. No conversion runs.
printf '{}\n' | env -u DYLD_LIBRARY_PATH -u DYLD_FALLBACK_LIBRARY_PATH \
    "$worker" yue2-worker "$fixture" "$fixture/cache" > "$fixture/reply.ndjson"
cat "$fixture/reply.ndjson"
node - "$fixture/reply.ndjson" "$backend" <<'NODE'
const fs = require("node:fs");
const replies = fs.readFileSync(process.argv[2], "utf8").trim().split("\n").map(line => JSON.parse(line));
const [hello, rejected] = replies;
if (replies.length !== 2 || hello.protocol !== 1 || hello.backend !== process.argv[3] || hello.version !== "0.1.7" || typeof hello.device_available !== "boolean") {
  throw Error("unexpected macOS worker protocol");
}
if (rejected.ok !== false || !rejected.error?.includes("missing field")) {
  throw Error("worker did not reject the invalid request");
}
NODE
