#!/usr/bin/env bash
# Fetch everything the SLR range-residual test measures against, for one day.
#
# Five independent products have to line up before a laser range can be modelled, and none of
# them is produced here:
#
#   normal points   ILRS monthly CRD v2, per satellite                 EDC / DGFI-TUM
#   orbit           ilrsa combined LAGEOS orbit, SP3, 2-minute grid    EDC / DGFI-TUM
#   stations        ITRF2020-ILRS-TRF.SNX, positions + velocities      IGN
#   eccentricities  ILRS site logs, section 3                          ILRS / NASA GSFC
#   Earth rotation  finals2000A.all                                    tools/fetch-eop.sh
#
# The eccentricity is the one that needs explaining. ITRF2020 and SLRF2020 both quote the
# *survey monument*, and at three of the twelve pads that tracked LAGEOS on 2025-01-06 the
# telescope stands 2.6-3.2 m above it. The ILRS keeps the official offsets in ecc_une.snx and
# ecc_xyz.snx, but publishes them only through NASA CDDIS, which requires an Earthdata login and
# so cannot be scripted here. The site logs carry the same numbers, are login-free, and are the
# primary source this script reads.
#
# The whole month of normal points is 1.4 MB per satellite and the orbits 780 kB per week, so
# they are fetched, never vendored; excerpts are committed under crates/spacelab/tests/fixtures
# for the always-run tests — one day for the single-pass reality check, and a span of days for
# the arc campaign, which needs a week of passes to resolve anything by frequency.
#
# Usage: tools/fetch-slr-data.sh [YYYY-MM-DD [YYYY-MM-DD]]   (default 2025-01-06, one day)

set -euo pipefail

date_arg="${1:-2025-01-06}"
end_arg="${2:-$date_arg}"
# The dates reach a URL, so they are validated here rather than at the consumer: an allow-list of
# exactly ten digits and two dashes, then a calendar check by `date`.
for d in "$date_arg" "$end_arg"; do
  if [[ ! "$d" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "date must be YYYY-MM-DD, got '$d'" >&2
    exit 2
  fi
  if ! date -u -d "$d" +%s >/dev/null 2>&1; then
    echo "not a calendar date: $d" >&2
    exit 2
  fi
done
start_s="$(date -u -d "$date_arg" +%s)"
end_s="$(date -u -d "$end_arg" +%s)"
if (( end_s < start_s )); then
  echo "the span ends before it starts: $date_arg .. $end_arg" >&2
  exit 2
fi
# A bound rather than a guard: the span decides how many months and weeks are fetched, and there
# is no analysis here that wants more than a month of passes.
span_days=$(( (end_s - start_s) / 86400 + 1 ))
if (( span_days > 31 )); then
  echo "span of $span_days days is longer than this script fetches" >&2
  exit 2
fi

# An ILRS orbit product covers Sunday to Saturday and is named for the Saturday that *ends* it, so
# the directory for a day is the next Saturday on or after it. Naming it for the preceding Saturday
# instead fetches the week before, whose file does not contain the day at all.
months=()
weeks=()
for (( i = 0; i < span_days; i++ )); do
  d="$(date -u -d "$date_arg +$i days" +%Y-%m-%d)"
  m="$(date -u -d "$d" +%Y%m)"
  w="$(date -u -d "$d +$(( (6 - $(date -u -d "$d" +%u) + 7) % 7 )) days" +%y%m%d)"
  [[ " ${months[*]-} " == *" $m "* ]] || months+=("$m")
  [[ " ${weeks[*]-} " == *" $w "* ]] || weeks+=("$w")
done

DEST="${SLR_DIR:-data/slr}"
EDC="${EDC_URL:-https://edc.dgfi.tum.de/pub/slr}"
IGN="${IGN_URL:-https://itrf.ign.fr/ftp/pub/itrf/itrf2020}"
ILRS="${ILRS_URL:-https://ilrs.gsfc.nasa.gov/network/stations/active}"
ASC="${ASC_URL:-https://ilrs.gsfc.nasa.gov/science/dataAnalysisResources/index.html}"

mkdir -p "$DEST"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

record() {
  local name="$1" url="$2" note="$3"
  cat > "$DEST/${name}.json" <<JSON
{
  "url": "$url",
  "retrieved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sha256": "$(sha256sum "$DEST/$name" | cut -d' ' -f1)",
  "bytes": $(stat -c%s "$DEST/$name"),
  "note": "$note"
}
JSON
  echo "  $name  $(stat -c%s "$DEST/$name") bytes"
}

echo "normal points, ${months[*]}"
for ym in "${months[@]}"; do
  for sat in lageos1 lageos2; do
    name="${sat}_${ym}.np2"
    url="$EDC/data/npt_crd_v2/$sat/${ym:0:4}/$name"
    curl -fsSL --retry 3 --retry-delay 2 -o "$DEST/$name" "$url"
    record "$name" "$url" "ILRS monthly normal points, CRD version 2"
  done
done

echo "combined orbits, weeks ${weeks[*]}"
for week in "${weeks[@]}"; do
  for sat in lageos1 lageos2; do
    name="ilrsa.orb.${sat}.${week}.v80.sp3"
    url="$EDC/products/orbits/$sat/$week/$name.gz"
    curl -fsSL --retry 3 --retry-delay 2 -o "$tmp/$name.gz" "$url"
    gunzip -c "$tmp/$name.gz" > "$DEST/$name"
    record "$name" "$url" "ILRS combination centre orbit, SLRF2020, 120 s, centre of mass"
  done
done

echo "terrestrial reference frame"
name="ITRF2020-ILRS-TRF.SNX"
url="$IGN/$name.gz"
curl -fsSL --retry 3 --retry-delay 2 -o "$tmp/$name.gz" "$url"
gunzip -c "$tmp/$name.gz" > "$DEST/$name"
record "$name" "$url" "ITRF2020, SLR technique, positions at the survey monument"

# The eccentricity of each station that observed, keyed on the CDP system and occupation numbers
# the station itself wrote into its CRD h2 record. Matching on those rather than on the pad alone
# matters: a pad that has hosted two systems has an entry for each, and they differ by metres.
echo "station eccentricities, from the CRD headers"
if [[ "$date_arg" == "$end_arg" ]]; then
  name="ilrs_eccentricity.snx"
else
  name="ilrs_eccentricity_${date_arg//-/}_${end_arg//-/}.snx"
fi
python3 - "$DEST" "$ILRS" "$date_arg" "$end_arg" > "$DEST/$name" <<'PY'
import datetime, html, re, sys, urllib.request
from pathlib import Path

dest, base, day, last = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
span = (datetime.date.fromisoformat(last) - datetime.date.fromisoformat(day)).days
want = {
    (d.year, d.month, d.day)
    for d in (
        datetime.date.fromisoformat(day) + datetime.timedelta(days=i)
        for i in range(span + 1)
    )
}

systems = {}
for crd in sorted(dest.glob("lageos?_*.np2")):
    keep = False
    for line in crd.read_text(errors="replace").splitlines():
        t = line.split()
        if not t:
            continue
        if t[0].lower() == "h2" and len(t) >= 5:
            pending = (t[1], int(t[2]), int(t[3]), int(t[4]))
        elif t[0].lower() == "h4" and len(t) >= 5:
            keep = tuple(int(x) for x in t[2:5]) in want
            if keep:
                systems[pending[1]] = pending

BLOCK = re.compile(
    r"System Name\s*:\s*(?P<name>[^\n]*)\n"
    r"\s*4-Character Code\s*:\s*(?P<code>\S+)\s*\n"
    r"\s*CDP System Number\s*:\s*(?P<sys>\d+)\s*\n"
    r"\s*CDP Occupation Number\s*:\s*(?P<occ>\d+)\s*\n"
    r"\s*Eccentricity to SRP[^\n]*\n"
    r"\s*North\s*\[m\]:\s*(?P<n>[-\d.]+)[^\n]*\n"
    r"\s*East\s*\[m\]:\s*(?P<e>[-\d.]+)[^\n]*\n"
    r"\s*Up\s*\[m\]:\s*(?P<u>[-\d.]+)[^\n]*\n"
    r"\s*Date Measured\s*:\s*(?P<meas>[\d-]*)\s*\n"
    r"\s*Date Installed\s*:\s*(?P<inst>[\d-]*)\s*\n"
    r"\s*Date Removed\s*:\s*(?P<rem>[\d-]*)\s*\n"
)


def sinex_epoch(iso):
    d = datetime.date.fromisoformat(iso)
    return f"{d.year % 100:02d}:{d.timetuple().tm_yday:03d}:00000"


lines, absent = [], []
for pad, (code, _, sys_no, occ) in sorted(systems.items()):
    url = f"{base}/{code}_sitelog.html"
    try:
        raw = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")
    except OSError as ex:
        lines.append(f"* {pad} {code}: site log unreachable: {ex}")
        continue
    txt = html.unescape(re.sub(r"<[^>]+>", "", raw))
    hits = [
        m
        for m in BLOCK.finditer(txt)
        if int(m.group("sys")) == sys_no
        and int(m.group("occ")) == occ
        and not m.group("rem").strip()
    ]
    if not hits:
        absent.append(f"{pad} {code}")
        continue
    b = max(hits, key=lambda m: m.group("meas"))
    lines.append(
        f" {pad:>4}  A {occ:>4} L {sinex_epoch(b.group('inst'))} 00:000:00000 UNE"
        f" {float(b.group('u')):8.4f} {float(b.group('n')):8.4f} {float(b.group('e')):8.4f}"
    )
    lines.append(
        f"*    {code}, {b.group('name').strip()}, CDP system {sys_no:02d}"
        f" occupation {occ:02d}, measured {b.group('meas')}"
    )

print("%=SNX 2.02 SDL 00:000:00000 ILR 00:000:00000 00:000:00000 P 00000 0 X")
print("*" + "-" * 79)
tracked = day if day == last else f"{day} .. {last}"
print(f"* Eccentricities of the stations that tracked LAGEOS on {tracked}, generated by")
print("* tools/fetch-slr-data.sh.")
print("*")
print("* The offset is from the Ranging System Reference Point -- the survey monument the")
print("* terrestrial reference frame solves for -- to the invariant point of the telescope,")
print("* which is what the range is measured from. ITRF2020 and SLRF2020 both quote the")
print("* monument, so a station whose telescope does not stand on it needs this added.")
print("*")
print("* Source: the ILRS Site and System Information Forms, section 3, 'Eccentricity to")
print("* SRP (if Not Identical With SRP)', matched on the CDP system and occupation numbers")
print("* each station wrote in its own CRD h2 record that day:")
print(f"*   {base}/<CODE>_sitelog.html")
print("* The ILRS's own databases (ecc_une.snx, ecc_xyz.snx) hold the same numbers but are")
print("* published only through NASA CDDIS, which requires an Earthdata login and so cannot")
print("* be scripted; the site logs are the login-free primary source.")
print("*")
print("* DATA START is the system's Date Installed; the value is the most recent Date")
print("* Measured for that system and occupation, given on the line after it.")
print("*")
print("* These pads left the eccentricity fields blank, which is the statement that their")
print("* monument and their telescope are the same point. They are absent below rather than")
print("* written as zeros, so a station this file has nothing to say about stays visible:")
for chunk in range(0, len(absent), 5):
    print("*   " + ", ".join(absent[chunk : chunk + 5]))
print("*" + "-" * 79)
print("+SITE/ECCENTRICITY")
print("*                                             UP______ NORTH___ EAST____")
print("*CODE PT SBIN T _DATA START_ __DATA_END__ AXE ARP->BENCHMARK(M)_________")
for line in lines:
    print(line)
print("-SITE/ECCENTRICITY")
print("%ENDSNX")
PY
record "$name" "$ILRS/<CODE>_sitelog.html" "monument-to-telescope offsets, ILRS site logs"

# The Data Handling File carries a release date in its own filename, so the URL is read off the
# ILRS analysis-resources page rather than constructed. That page is the ILRS's own pointer to the
# current release; CDDIS also serves the file but behind an Earthdata login, as with the
# eccentricities above.
echo "data handling file"
name="ILRS_Data_Handling_File.snx"
url="$(curl -fsSL "$ASC" \
  | grep -oE '"[^"]*ILRS_Data_Handling_File[^"]*\.snx"' | tr -d '"' | head -1)"
if [[ -z "$url" ]]; then
  echo "the analysis-resources page no longer links a Data Handling File" >&2
  exit 1
fi
[[ "$url" == http* ]] || url="https://ilrs.gsfc.nasa.gov$url"
curl -fsSL --retry 3 --retry-delay 2 -o "$DEST/$name" "$url"
record "$name" "$url" "ILRS/ASC range, time and pressure biases and data to be deleted"

# The excerpts the always-run test reads. Written beside the snapshots rather than into the source
# tree: what makes them reproducible is that `diff` against the committed copies is empty, and a
# script that overwrites its own reference cannot demonstrate that.
echo "excerpts for $date_arg .. $end_arg"
python3 - "$DEST" "$date_arg" "$end_arg" "${EOP_DIR:-data/eop}" <<'PY'
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

dest, day, last = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
eop_dir = Path(sys.argv[4])
first_day = datetime(*(int(x) for x in day.split("-")))
last_day = datetime(*(int(x) for x in last.split("-")))
span = (last_day - first_day).days + 1
days = [first_day + timedelta(days=i) for i in range(span)]
want = {(d.year, d.month, d.day) for d in days}
single = span == 1
out = dest / "fixtures"
out.mkdir(exist_ok=True)
stamp = first_day.strftime("%Y%m%d") if single else (
    f"{first_day:%Y%m%d}_{last_day:%Y%m%d}"
)
pads = set()
# Minutes of orbit kept either side of the day, at the product's 120 s grid.
MARGIN = 20.0


def flush(kept, buf, keep, epochs):
    """A session is kept whole, and only if it holds at least one normal point."""
    if keep and any(l.split() and l.split()[0] == "11" for l in buf):
        kept.extend(buf)
        started = None
        for b in buf:
            t = b.split()
            if not t:
                continue
            if t[0].lower() == "h2" and len(t) > 2:
                pads.add(t[2])
            elif t[0].lower() == "h4" and len(t) > 4:
                started = datetime(int(t[2]), int(t[3]), int(t[4]))
            elif t[0] == "11" and started is not None:
                # Seconds of the session's own day, which is how the model reads them too.
                epochs.append(started + timedelta(seconds=float(t[1])))


def arc_week(d):
    """The ILRS orbit product covering `d`, named for the Saturday that ends its week."""
    return (d + timedelta(days=(5 - d.weekday()) % 7)).strftime("%y%m%d")


months = sorted({f"{d:%Y%m}" for d in days})
for sat in (1, 2):
    lines = []
    for ym in months:
        lines += (dest / f"lageos{sat}_{ym}.np2").read_text(errors="replace").splitlines()
    kept, buf, keep, epochs = [], [], False, []
    for line in lines:
        t = line.split()
        k = t[0].lower() if t else ""
        # `h1` opens a file and `h4` a session; either closes whatever came before, because about
        # an eighth of the sessions in an ILRS monthly file never get their `h8`.
        if k == "h1":
            flush(kept, buf, keep, epochs)
            buf, keep = [line], False
            continue
        if k == "h4" and len(t) > 4:
            keep = tuple(int(x) for x in t[2:5]) in want
        buf.append(line)
        if k == "h8":
            flush(kept, buf, keep, epochs)
            buf, keep = [], False
    flush(kept, buf, keep, epochs)
    p = out / f"slr_lageos{sat}_{stamp}.npt"
    p.write_text("\n".join(kept) + "\n")
    print(f"  {p.name}  {sum(1 for l in kept if l.split()[:1] == ['11'])} normal points")

    # The orbit. One day is trimmed to the day plus MARGIN minutes at each end — enough for a
    # degree-10 Neville window to sit inside the file at midnight rather than straddle its edge.
    #
    # A span crosses arcs, and the weekly products do not overlap: they abut at a two-minute seam.
    # So each arc is written separately, and trimmed to MARGIN minutes around the passes that were
    # actually observed rather than to the whole week, which halves it. The interpolator takes the
    # samples nearest by *index*, so what that trim must guarantee is a contiguous 120 s run around
    # every epoch the model asks about; the reader asserts that rather than trusting it.
    windows = set()
    for e in epochs:
        base = e.replace(second=0, microsecond=0)
        base -= timedelta(minutes=base.minute % 2)
        for k in range(-int(MARGIN // 2) - 1, int(MARGIN // 2) + 2):
            windows.add(base + timedelta(minutes=2 * k))

    for week in sorted({arc_week(d) for d in days}):
        src = dest / f"ilrsa.orb.lageos{sat}.{week}.v80.sp3"
        body = src.read_text(errors="replace").splitlines()
        head = [l for l in body if not l.startswith(("*", "P", "V", "EOF"))]
        kept, take = [], False
        for line in body:
            if line.startswith("*"):
                t = line.split()
                epoch = datetime(int(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[5]))
                if single:
                    offset = (epoch - days[0]).total_seconds() / 60.0
                    take = -MARGIN <= offset <= 1440.0 + MARGIN
                else:
                    take = epoch in windows
            if take and line.startswith(("*", "P", "V")):
                kept.append(line)
        p = out / (
            f"ilrsa_lageos{sat}_{stamp}.sp3"
            if single
            else f"ilrsa_lageos{sat}_arc{week}.sp3"
        )
        p.write_text("\n".join(head + kept) + "\n")
        print(f"  {p.name}  {sum(1 for l in kept if l.startswith('*'))} epochs")

# The station block, cut to the pads that actually observed. The covariance is 99.9% of the file.
keep_blocks = {"SITE/ID", "SOLUTION/EPOCHS", "SOLUTION/ESTIMATE"}
kept, block = [], ""
for line in (dest / "ITRF2020-ILRS-TRF.SNX").read_text(errors="replace").splitlines():
    if line.startswith("%=SNX"):
        kept.append(line)
        continue
    if line.startswith("+"):
        block = line.strip()[1:]
        if block in keep_blocks:
            kept.append(line)
        continue
    if line.startswith("-"):
        if block in keep_blocks:
            kept.append(line)
        block = ""
        continue
    if block not in keep_blocks:
        continue
    if line.startswith("*"):
        kept.append(line)
        continue
    f = line.split()
    if (f[2] if block == "SOLUTION/ESTIMATE" else f[0]) in pads:
        kept.append(line)
p = out / ("itrf2020_ilrs_slr_stations.snx" if single else f"itrf2020_ilrs_slr_stations_{stamp}.snx")
p.write_text("\n".join(kept) + "\n")
print(f"  {p.name}  {len(pads)} pads: {' '.join(sorted(pads))}")

# The Data Handling File, cut the same way. Only the two blocks `slr::dhf` reads are kept: the
# pressure, humidity and timing records correct a different measurement than the range, and a
# fixture that carried them would suggest they were being applied.
import re

dhf = (dest / "ILRS_Data_Handling_File.snx").read_text(errors="replace")
version = re.search(r"^\s*VERSION\s+(\S+)", dhf, re.M).group(1)
release = re.search(r"^\s*RELEASE DATE\s+(\S+)", dhf, re.M).group(1)
lines = [
    dhf.splitlines()[0],
    "*" + "-" * 79,
    f"* ILRS Data Handling File version {version}, released {release}, cut to the pads that",
    f"* tracked LAGEOS on {day if single else f'{day} .. {last}'}."
    " Generated by tools/fetch-slr-data.sh.",
    "*",
    "* Only MODEL/RANGE_BIAS and SOLUTION/DATA_HANDLING are kept. The file's own statement of",
    "* the sign is that values are to be subtracted from the 1-way observations.",
    "*",
    "* Source: the ILRS analysis-resources page's link to the current release.",
    "*" + "-" * 79,
]
for block in ("MODEL/RANGE_BIAS", "SOLUTION/DATA_HANDLING"):
    body = re.search(
        rf"^\+{re.escape(block)}\n(.*?)^-{re.escape(block)}", dhf, re.M | re.S
    ).group(1)
    lines.append(f"+{block}")
    lines.append(
        "*CODE PT SOLN T START_DATE__ END_DATE____ M __E-VALUE___ STD_DEV"
        " _E-RATE__ UNIT   CMNTS"
    )
    lines += [
        l.rstrip()
        for l in body.splitlines()
        if l.strip() and not l.startswith("*") and l.split()[0] in pads
    ]
    lines.append(f"-{block}")
lines.append("%ENDSNX")
p = out / ("ilrs_data_handling.snx" if single else f"ilrs_data_handling_{stamp}.snx")
p.write_text("\n".join(lines) + "\n")
print(f"  {p.name}  version {version}")

# The eccentricity block writes straight into $DEST rather than into the excerpt directory, because
# it is generated from the site logs rather than cut from a bulk product. Copied in so that one
# `diff -r` covers every fixture the tests read.
p = out / f"ilrs_eccentricity{'' if single else '_' + stamp}.snx"
p.write_text((dest / p.name).read_text())
print(f"  {p.name}")

# Earth orientation. The series is fetched by tools/fetch-eop.sh and never vendored whole -- it is
# 3.7 MB and the IERS rewrite it daily -- so the excerpt is cut here, from the snapshot with the
# most recent fetch. Eight days either side is what the four-point Lagrange interpolation wants at
# the ends of the span, with room to spare rather than exactly enough.
manifests = sorted(
    eop_dir.glob("finals2000A.*.json"),
    key=lambda m: json.loads(m.read_text())["fetched_utc"],
)
if not manifests:
    raise SystemExit(f"no EOP snapshot in {eop_dir}: run tools/fetch-eop.sh first")
snapshot = manifests[-1].with_suffix(".txt")
MJD_EPOCH = datetime(1858, 11, 17)
lo = (days[0] - timedelta(days=8) - MJD_EPOCH).days
hi = (days[-1] + timedelta(days=8) - MJD_EPOCH).days
rows = [
    l
    for l in snapshot.read_text(errors="replace").splitlines()
    if l[7:15].strip() and lo <= float(l[7:15]) <= hi
]
p = out / f"finals2000A_{stamp}.txt"
p.write_text("\n".join(rows) + "\n")
print(f"  {p.name}  MJD {lo} .. {hi}, {len(rows)} rows, from {snapshot.name}")
PY

echo
echo "Earth orientation is a separate product: tools/fetch-eop.sh"
echo "snapshots in $DEST"
echo "compare: diff -r $DEST/fixtures crates/spacelab/tests/fixtures"
