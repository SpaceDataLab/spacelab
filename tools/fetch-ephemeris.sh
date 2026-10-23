#!/usr/bin/env bash
# Fetch the JPL short-span planetary ephemeris kernels (de440s.bsp, de442s.bsp) from NAIF.
#
# The kernels are tens of megabytes each and are NOT committed to the repository;
# `spacelab-ephemeris` validation tests are gated on their presence and skip without them, so CI
# stays network-free. Both cover ~1849-2150 and are machine-precision-equal to SPICE (see
# plans/13-deep-space.md).
#
# Source is NAIF itself, on the same reasoning as tools/fetch-lunar-kernels.sh: provenance for an
# ephemeris is not a detail. It is also the only source that currently works -- the NyxSpace mirror
# this script used to read presents a certificate that does not match its hostname, so `curl -f`
# fails outright and the script has been unable to fetch anything.
#
# **DE440 is what the tests anchor to, and DE442 is fetched to keep that a measurement.** DE442
# (2025-02-06) is the current JPL solution, and `spacelab-ephemeris/tests/de442_delta.rs` reads both
# so that the choice is a published number rather than an assumption: the Moon differs by 0.300 m
# worst over 2000-2030 and the Sun by 19.633 m, which is 0.152 mm on a 32-satellite GPS orbit fit.
# NAIF publishes no DE442 lunar principal-axis orientation, frame kernel or mass file, so the
# selenocentric chain has no DE442 counterpart to move to -- see docs/verification.md 2.5.
#
# The digests are pinned rather than recorded. These are released, versioned products (NAIF's DE440
# copy has not changed since 2020-12-21) so there is a right answer to compare against, and a
# truncated download is otherwise indistinguishable from a good one until a test reads past the end
# of it.
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/ephemeris"
base="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets"

# kernel:sha256
kernels=(
    "de440s.bsp:c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2"
    "de442s.bsp:54d97562a5b094d298b1b8eafa5a2e17e3e010ce85e1a366d07f003ad159323c"
)

verify() {
    local path="$1" want="$2" got
    got="$(sha256sum "$path" | cut -d' ' -f1)"
    if [ "$got" != "$want" ]; then
        echo "sha256 mismatch for $path" >&2
        echo "  expected $want" >&2
        echo "  got      $got" >&2
        return 1
    fi
}

mkdir -p "$dir"
for entry in "${kernels[@]}"; do
    name="${entry%%:*}"
    sha="${entry#*:}"
    dest="$dir/$name"

    if [ -f "$dest" ]; then
        verify "$dest" "$sha"
        echo "already present: $dest ($(du -h "$dest" | cut -f1))"
        continue
    fi

    # Download beside the destination and move only once the digest checks out, so a failed or
    # partial transfer never leaves something at the path the tests read.
    tmp="$dest.partial"
    trap 'rm -f "$tmp"' EXIT
    echo "downloading $name -> $dest"
    curl -fSL "$base/$name" -o "$tmp"
    verify "$tmp" "$sha"
    mv "$tmp" "$dest"
    trap - EXIT
    echo "done: $(du -h "$dest" | cut -f1)"
done
