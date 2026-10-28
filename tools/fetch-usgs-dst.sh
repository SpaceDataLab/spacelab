#!/usr/bin/env bash
# Fetch the USGS hourly Dst series (elements Dst4 and Dst3 of the virtual observatory USGS).
#
# USGS-authored data is U.S. public domain; credit "U.S. Geological Survey". The method is Rigler
# (2017), doi:10.3133/ofr20171037. Both indices start in late September 2015: Dst4 from the four
# observatories of the traditional Dst, Dst3 from USGS observatories only.
#
# `/ws/data/` holds only the 1-minute series; the hourly one is that series boxcar-averaged by the
# service's filter endpoint, one hour per output sample, stamped at the hour's centre (hh:29:30).
# `input_sampling_period=60` has to be stated: without it the service assumes 10 Hz input, refuses
# more than 36 output samples, and for old dates answers with test data (Rigler, 2026-10-27).
# Only `type=variation` carries values; the other types answer all-null.
#
# One request per month, both elements together, strictly one at a time with a pause between —
# the load USGS agreed to. A month already in the cache is not fetched again unless it is the
# current or the previous month, which the service may still be filling in.
#
# Usage: tools/fetch-usgs-dst.sh [FROM_YYYY-MM [TO_YYYY-MM]]
#        tools/fetch-usgs-dst.sh excerpt FIRST_YYYY-MM-DD LAST_YYYY-MM-DD   (stdout, from the cache)

set -euo pipefail

DEST="${DENSITY_DIR:-data/validation/density}"
CACHE="$DEST/usgs-dst"
BASE="https://geomag.usgs.gov/ws/algorithms/filter/"
mkdir -p "$CACHE"

assemble() {
  python3 -I - "$CACHE" "$1" "$2" "$BASE" <<'PY'
import datetime as dt, json, pathlib, sys

cache, first, last, base = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
first, last = dt.date.fromisoformat(first), dt.date.fromisoformat(last)
print("# USGS Geomagnetism Program: hourly Dst, virtual observatory USGS, elements Dst4 and Dst3, nT")
print("# LICENSE: U.S. Public Domain. \"USGS-authored or produced data and information are considered to be")
print("# in the U.S. Public Domain\"; credit: U.S. Geological Survey (usgs.gov, Copyrights and Credits).")
print("# METHOD: Rigler, E. J. (2017). Time-causal decomposition of geomagnetic time series into secular")
print("# variation, solar quiet, and disturbance signals. USGS Open-File Report, doi:10.3133/ofr20171037")
print(f"# SOURCE: {base}?id=USGS&elements=Dst4,Dst3&type=variation&format=json")
print("#   &input_sampling_period=60&output_sampling_period=3600, one request per calendar month")
print("# Each value is the mean of the hour's 1-minute samples, stamped at the hour's centre (hh:29:30).")
print("# Missing hours are written null. Columns: time (UTC), Dst4, Dst3.")
months, retrieved = [], []
y, m = first.year, first.month
while (y, m) <= (last.year, last.month):
    months.append((y, m))
    y, m = (y + 1, 1) if m == 12 else (y, m + 1)
for y, m in months:
    path = cache / f"{y:04d}-{m:02d}.json"
    retrieved.append(f"{y:04d}-{m:02d} {json.loads(path.with_suffix('.json.meta').read_text())['retrieved_utc']}")
print("# RETRIEVED: " + (", ".join(retrieved) if len(retrieved) <= 2 else f"{retrieved[0]} … {retrieved[-1]}"))
print("#time                         Dst4                 Dst3")
for y, m in months:
    j = json.loads((cache / f"{y:04d}-{m:02d}.json").read_text())
    columns = {v["id"]: v["values"] for v in j["values"]}
    for i, t in enumerate(j["times"]):
        day = dt.date(int(t[:4]), int(t[5:7]), int(t[8:10]))
        if first <= day <= last:
            cells = ["null" if columns[e][i] is None else repr(float(columns[e][i])) for e in ("Dst4", "Dst3")]
            print(f"{t[:19]}Z  {cells[0]:>20} {cells[1]:>20}")
PY
}

if [[ "${1:-}" == "excerpt" ]]; then
  for d in "${2:?first day}" "${3:?last day}"; do
    [[ "$d" =~ ^[0-9]{4}-(0[1-9]|1[0-2])-[0-3][0-9]$ ]] || { echo "day must be YYYY-MM-DD, got '$d'" >&2; exit 2; }
  done
  assemble "$2" "$3"
  exit 0
fi

from="${1:-2015-09}"
to="${2:-$(date -u +%Y-%m)}"
for m in "$from" "$to"; do
  if [[ ! "$m" =~ ^[0-9]{4}-(0[1-9]|1[0-2])$ ]]; then
    echo "month must be YYYY-MM with MM in 01..12, got '$m'" >&2
    exit 2
  fi
done
recent="$(date -u -d "$(date -u +%Y-%m-01) -1 month" +%Y-%m)"

month="$from"
fetched=0
while [[ ! "$month" > "$to" ]]; do
  out="$CACHE/$month.json"
  if [[ -s "$out" && "$month" < "$recent" ]]; then
    month="$(date -u -d "$month-01 +1 month" +%Y-%m)"
    continue
  fi
  start="${month}-01T00:00:00Z"
  end="$(date -u -d "$month-01 +1 month -1 second" +%Y-%m-%dT%H:%M:%SZ)"
  url="${BASE}?id=USGS&elements=Dst4,Dst3&type=variation&format=json&input_sampling_period=60&output_sampling_period=3600&starttime=${start}&endtime=${end}"
  (( fetched > 0 )) && sleep 2
  tmp="$(mktemp)"
  # A lone 404 from a host that answers the same request a minute later has been seen
  # (2026-10-28, 2023-01), so a failure is retried twice, 30 s apart, before the run stops.
  for attempt in 1 2 3; do
    curl -sS --fail --max-time 300 -o "$tmp" "$url" && break
    if (( attempt == 3 )); then
      rm -f "$tmp"
      echo "could not fetch $month from $url" >&2
      exit 1
    fi
    sleep 30
  done
  fetched=$((fetched + 1))
  # The answer is checked before it is kept: exactly the month's hours, each at hh:29:30, both
  # elements, hourly variation data, and every value a number or null.
  if ! summary="$(python3 -I - "$tmp" "$month" <<'PY'
import calendar, datetime as dt, json, sys

j = json.load(open(sys.argv[1]))
y, m = map(int, sys.argv[2].split("-"))
hours = calendar.monthrange(y, m)[1] * 24
t0 = dt.datetime(y, m, 1, 0, 29, 30)
want = [(t0 + dt.timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%S.000Z") for h in range(hours)]
meta = j["metadata"]["intermagnet"]
assert j["metadata"]["status"] == 200, j["metadata"]
assert meta["data_type"] == "variation" and meta["sampling_period"] == 3600.0, meta
assert j["times"] == want, "times are not the month's hours at hh:29:30"
assert sorted(v["id"] for v in j["values"]) == ["Dst3", "Dst4"], [v["id"] for v in j["values"]]
for v in j["values"]:
    assert len(v["values"]) == hours
    assert all(x is None or isinstance(x, (int, float)) for x in v["values"]), v["id"]
count = {v["id"]: sum(x is not None for x in v["values"]) for v in j["values"]}
print(f"{hours} hours, Dst4 {count['Dst4']} values, Dst3 {count['Dst3']} values")
PY
)"; then
    rm -f "$tmp"
    echo "unexpected answer for $month from $url" >&2
    exit 1
  fi
  mv "$tmp" "$out"
  cat >"$out.meta" <<EOF
{
  "url": "$url",
  "retrieved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sha256": "$(sha256sum "$out" | cut -d' ' -f1)"
}
EOF
  echo "$month  $summary"
  month="$(date -u -d "$month-01 +1 month" +%Y-%m)"
done

name="usgs_dst_hourly.txt"
assemble "$from-01" "$(date -u -d "$to-01 +1 month -1 day" +%Y-%m-%d)" >"$DEST/$name"
cat >"$DEST/$name.json" <<EOF
{
  "url": "$BASE (one request per month, see the file header)",
  "assembled_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sha256": "$(sha256sum "$DEST/$name" | cut -d' ' -f1)",
  "bytes": $(stat -c%s "$DEST/$name"),
  "rows": $(grep -vc '^#' "$DEST/$name"),
  "contents": "hourly Dst4 and Dst3 from $from to $to, U.S. Geological Survey, public domain"
}
EOF
echo "$name  $(grep -vc '^#' "$DEST/$name") rows"
