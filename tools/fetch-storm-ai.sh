#!/usr/bin/env bash
# Fetch the STORM-AI evaluation archives from Harvard Dataverse, for `tools/benchmark-splits.py storm-ai-manifest`
# (plans/17 Ф25.E).
#
# STORM-AI's data are CC BY 4.0 (doi:10.7910/DVN/U6K6MJ; `docs/provenance.md`, `storm-ai`). The manifest needs the
# twelve `private_eval__*` archives only: the initial states of all 4 557 evaluation cases and each mission's
# orbit-mean densities, 26 GiB. The file ids, MD5s and sizes are Dataverse's own (read 2026-11-12). A file is kept
# only if its MD5 matches, and each archive is fetched once: a present `NAME.ok` skips it, a partial one resumes.
# A download that would leave less than 30 GiB free stops before it starts.
#
# Usage: tools/fetch-storm-ai.sh [DIR]     (default data/validation/storm-ai)

set -euo pipefail

DEST="${1:-data/validation/storm-ai}"
BASE="https://dataverse.harvard.edu/api/access/datafile"
RESERVE=$((30 * 1024 ** 3))

FILES=(
  13456845:9f6b5b9d08b585fd6dc03dd74947d9fe:380636:private_eval__initial_states.zip
  13456849:e9407ee4ce191e09c39817efec919759:19555270:private_eval__2002-2010_GRACE_1.zip
  13456851:7952d7252cd74bb53d85f9324a245b5b:651187241:private_eval__2002-2010_GRACE_2.zip
  13456848:ae7d58c1f35af8ec41f6685cd78d8896:870140237:private_eval__2000-2010_CHAMP.zip
  13456847:829ea73a9ed806747cb1108526bf6e99:569016377:private_eval__2020_SWARM_A.zip
  13457036:3cb78267c316eb74416817b591f084a9:2374738752:private_eval__2014-2020_SWARM_B.zip
  13457037:f605d858416ed31d5d6ad286e430b285:2370518145:private_eval__2014-2020_SWARM_C.zip
  13456852:d7e853b65dd30547cee793a123112dd9:2724448196:private_eval__2021-2024_SWARM_A.zip
  13456850:ee2f12c1a41e406bf187a92b686fac25:2847489630:private_eval__2021-2024_SWARM_B.zip
  13456846:de94cd302883c8bb9691a70e6e744d11:2767333606:private_eval__2021-2024_SWARM_C.zip
  13457034:f0bdac9ebee6a5b04e3ecc4bbd8950ec:3116906813:private_eval__2018-2020_GRACE-FO_1.zip
  13457035:f8871e64e024c02aed312ed00851fff5:5863208453:private_eval__2021-2024_GRACE-FO_1.zip
)

mkdir -p "$DEST"
for spec in "${FILES[@]}"; do
  IFS=: read -r id md5 size name <<<"$spec"
  out="$DEST/$name"
  if [[ -f "$out.ok" ]]; then
    echo "present: $name"
    continue
  fi
  have="$(stat -c %s "$out" 2>/dev/null || echo 0)"
  free="$(df -B1 --output=avail "$DEST" | tail -1)"
  if ((free - (size - have) < RESERVE)); then
    echo "stopped before $name: $((size - have)) bytes to come would leave less than 30 GiB of $free free" >&2
    exit 3
  fi
  curl -sS -L --fail --retry 3 --retry-delay 30 --max-time 3600 -C - -o "$out" "$BASE/$id"
  got="$(md5sum "$out" | cut -d' ' -f1)"
  if [[ "$got" != "$md5" ]]; then
    echo "$name: MD5 $got, Dataverse says $md5; removed" >&2
    rm -f "$out"
    exit 1
  fi
  touch "$out.ok"
  echo "$name: $(stat -c %s "$out") bytes, MD5 matches"
done
