#!/usr/bin/env bash
set -euo pipefail
worker=${1:?Pass the worker binary}
fixture=${2:?Pass an unused fixture directory}
mkdir -p "$fixture"
voice_repo=https://huggingface.co/licyk/rvc-model/resolve/ad931ef3dc18f2d63ef81ecafddd498a111c8ecd/voices/guanguanV1
acoustics=https://modelscope.cn/api/v1/models/MosRat/vfrm-models/repo?Revision=9527a1bcf861c70717d4aba619ba92f2b5f42396
voice_sha=5a4920ef7b8c28be41dbd3afe2bd8cc852bccc5da82ac0590afb2a160ca81bb2
index_sha=d04454f009988ad7e3d830d3d8870b3d107978738e2e3b10bd415038ed882cac
fetch() {
    curl --fail --location --retry 3 --silent --show-error --max-time 360 "$2" -o "$fixture/$1"
    printf '%s  %s\n' "$3" "$fixture/$1" | shasum -a 256 --check
}

fetch hubert.vfrm "$acoustics&FilePath=shared%2Fhubert.vfrm" a84fed477f49a040d53b44b7c27184d711fb246d5f9119c7c046941aaee65c59 &
hubert=$!
fetch rmvpe.vfrm "$acoustics&FilePath=shared%2Frmvpe.vfrm" 462ee5494525ed1029abfad368abb56ca20285478e24aa846575ed400bff4734 &
rmvpe=$!
fetch voice.pth "$voice_repo/guanguanV1.pth" "$voice_sha" &
voice=$!
fetch retrieval.index "$voice_repo/guanguanV1.index" "$index_sha" &
index=$!
for transfer in "$hubert" "$rmvpe" "$voice" "$index"; do
    wait "$transfer"
done

node - "$fixture" "$voice_sha" "$index_sha" <<'NODE'
const fs = require("node:fs");
const path = require("node:path");
const [root, voiceSha, indexSha] = process.argv.slice(2);
const at = name => path.join(root, name);
const rate = 16_000;
const samples = 4 * rate;
const wav = Buffer.alloc(44 + 2 * samples);
wav.write("RIFF");
wav.writeUInt32LE(wav.length - 8, 4);
wav.write("WAVEfmt ", 8);
wav.writeUInt32LE(16, 16);
wav.writeUInt16LE(1, 20);
wav.writeUInt16LE(1, 22);
wav.writeUInt32LE(rate, 24);
wav.writeUInt32LE(2 * rate, 28);
wav.writeUInt16LE(2, 32);
wav.writeUInt16LE(16, 34);
wav.write("data", 36);
wav.writeUInt32LE(2 * samples, 40);
for (let frame = 0; frame < samples; frame++) {
  const t = frame / rate;
  const phase = (2 * Math.PI * 180) * t + 7 * Math.sin(2 * Math.PI * 0.75 * t);
  const voiced = Math.sin(phase) + 0.25 * Math.sin(2 * phase) + 0.2 * Math.sin(3 * phase);
  const envelope = 0.35 + 0.65 * Math.sin(Math.PI * t) ** 2;
  wav.writeInt16LE(Math.round(voiced * envelope * 6_000), 44 + 2 * frame);
}
fs.writeFileSync(at("input.wav"), wav);
const requests = [0, 1].map(pass => JSON.stringify({
  vocal: at("input.wav"), voice: at("voice.pth"), index: at("retrieval.index"),
  voice_sha256: voiceSha, index_sha256: indexSha, output: at(`output-${pass}.wav`),
  semitone_shift: 0, retrieval_blend: 0.5,
}));
fs.writeFileSync(at("request.ndjson"), requests.join("\n") + "\n");
NODE
env -u DYLD_LIBRARY_PATH -u DYLD_FALLBACK_LIBRARY_PATH \
    "$worker" yue2-worker "$fixture" "$fixture/cache" \
    < "$fixture/request.ndjson" > "$fixture/reply.ndjson"
cat "$fixture/reply.ndjson"
node - "$fixture" <<'NODE'
const fs = require("node:fs");
const path = require("node:path");
const root = process.argv[2];
const [hello, ...replies] = fs.readFileSync(path.join(root, "reply.ndjson"), "utf8")
  .trim().split("\n").map(line => JSON.parse(line));
if (hello.protocol !== 1 || hello.backend !== "cpu-graph" || hello.version !== "0.1.7" || replies.length !== 2) {
  throw Error("unexpected worker protocol");
}
for (const [pass, reply] of replies.entries()) {
  if (!reply.ok || reply.details.retrieval_blend !== 0.5 || reply.details.offline.outputSampleRate !== 40_000) {
    throw Error(JSON.stringify(reply));
  }
  const wav = fs.readFileSync(path.join(root, `output-${pass}.wav`));
  if (wav.toString("ascii", 0, 4) !== "RIFF" || wav.readUInt16LE(20) !== 1 ||
      wav.readUInt16LE(22) !== 1 || wav.readUInt32LE(24) !== 40_000 || wav.length < 310_000) {
    throw Error("invalid native RVC output WAV");
  }
  let peak = 0;
  for (let offset = 44; offset + 1 < wav.length; offset += 2) peak = Math.max(peak, Math.abs(wav.readInt16LE(offset)));
  if (peak < 256) throw Error("silent native RVC output WAV");
  console.log({ pass, bytes: wav.length, peak, preparationMs: reply.details.offline.preparationMs });
}
if (replies[1].details.offline.preparationMs >= 10) throw Error("resident CPU graph was rebuilt");
NODE
