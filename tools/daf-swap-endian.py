#!/usr/bin/env python3
"""Rewrite a big-endian DAF (SPK/CK/PCK/DAF) into the little-endian layout ANISE requires.

NAIF's PDS archives are published in IEEE big-endian form — the SPICE toolkit reads either
byte order transparently, but ANISE rejects a file whose `LOCFMT` is not the host's
(`naif/daf/file_record.rs`: `if file_endian != Endian::f64_native() { WrongEndian }`). The
generic kernels under `pub/naif/generic_kernels` are little-endian; anything from
`pub/naif/pds/data` is not.

The conversion is not a blanket byte swap. A DAF is a sequence of 1024-byte records of four
different kinds, and only two of them hold binary numbers:

  record 1                the file record — three character fields and five 32-bit integers
  records 2 .. FWARD-1    the comment area, ASCII
  the summary records     3 leading doubles, then summaries of ND doubles followed by NI
                          32-bit integers packed two to a double slot
  each summary's successor  the name record, ASCII
  everything else         element data, doubles throughout

Swapping a summary record's packed integers as 8-byte words instead of 4 transposes every
descriptor pair (body/centre, frame/type, start/end address) and produces a file that parses
and then lies, so the integer half is swapped in 4-byte groups.

Usage:
    uv run python tools/daf-swap-endian.py <input.bsp> <output.bsp>
"""

import argparse
import struct
import sys
from pathlib import Path

RECORD = 1024
SUMMARY_HEADER_DOUBLES = 3


def swap(block: bytes, width: int) -> bytes:
    """Reverse every `width`-byte group of `block`."""
    if len(block) % width:
        raise ValueError(
            f"{len(block)} bytes is not a whole number of {width}-byte words"
        )
    return b"".join(block[i : i + width][::-1] for i in range(0, len(block), width))


def convert(data: bytes) -> bytes:
    if len(data) < RECORD or len(data) % RECORD:
        raise ValueError(
            f"{len(data)} bytes is not a whole number of {RECORD}-byte records"
        )

    file_record = bytearray(data[:RECORD])
    identifier = bytes(file_record[0:8]).decode("ascii", "replace")
    if not identifier.startswith("DAF/"):
        raise ValueError(f"not a DAF: the file record opens with {identifier!r}")

    layout = bytes(file_record[88:96]).decode("ascii", "replace")
    if layout == "LTL-IEEE":
        raise ValueError("already little-endian; nothing to convert")
    if layout != "BIG-IEEE":
        raise ValueError(f"unrecognised binary format {layout!r}")

    nd, ni = struct.unpack(">ii", file_record[8:16])
    fward, bward, free = struct.unpack(">iii", file_record[76:88])
    summary_doubles = nd + (ni + 1) // 2
    records = len(data) // RECORD
    if not 0 <= nd <= 124 or not 0 <= ni <= 250 or summary_doubles == 0:
        raise ValueError(f"implausible summary shape ND={nd}, NI={ni}")
    if not 2 <= fward <= records:
        raise ValueError(
            f"first summary record {fward} is outside a {records}-record file"
        )

    file_record[8:16] = struct.pack("<ii", nd, ni)
    file_record[76:88] = struct.pack("<iii", fward, bward, free)
    file_record[88:96] = b"LTL-IEEE"

    summaries: dict[int, bytes] = {}
    names: set[int] = set()
    record = fward
    while record != 0:
        if record in summaries:
            raise ValueError(f"the summary record chain loops back to record {record}")
        if not 2 <= record < records:
            raise ValueError(
                f"summary record {record} is outside a {records}-record file"
            )
        raw = data[(record - 1) * RECORD : record * RECORD]
        following, _previous, count = struct.unpack(">ddd", raw[:24])
        capacity = (RECORD // 8 - SUMMARY_HEADER_DOUBLES) // summary_doubles
        if not float(count).is_integer() or not 0 <= count <= capacity:
            raise ValueError(
                f"record {record} claims {count} summaries; it holds at most {capacity}"
            )
        if not float(following).is_integer() or not 0 <= following <= records:
            raise ValueError(f"record {record} points at record {following}")

        out = bytearray(raw)
        out[:24] = swap(raw[:24], 8)
        for index in range(int(count)):
            start = 24 + index * summary_doubles * 8
            split = start + nd * 8
            end = start + summary_doubles * 8
            out[start:split] = swap(raw[start:split], 8)
            out[split:end] = swap(raw[split:end], 4)
        summaries[record] = bytes(out)
        names.add(record + 1)
        record = int(following)

    pieces = [bytes(file_record)]
    for number in range(2, records + 1):
        raw = data[(number - 1) * RECORD : number * RECORD]
        if number < fward or number in names:
            pieces.append(raw)  # comment and name records are ASCII
        elif number in summaries:
            pieces.append(summaries[number])
        else:
            pieces.append(swap(raw, 8))
    return b"".join(pieces)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="big-endian DAF to read")
    parser.add_argument("destination", type=Path, help="little-endian DAF to write")
    arguments = parser.parse_args()

    try:
        converted = convert(arguments.source.read_bytes())
    except (OSError, ValueError) as error:
        print(f"{arguments.source}: {error}", file=sys.stderr)
        return 1
    arguments.destination.write_bytes(converted)
    print(
        f"{arguments.source} -> {arguments.destination} ({len(converted)} bytes, LTL-IEEE)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
