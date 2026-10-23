#!/usr/bin/env bash
# Fetch the NAIF lunar orientation kernels needed for selenocentric propagation.
#
#   moon_pa_de440_200625.bpc  binary PCK, ~12 MB: the Moon's principal-axis (PA) orientation from
#                             DE440. The gravity field GRGM1200A is expressed in this frame, so a
#                             selenocentric geopotential evaluation needs it. Covers 1550-2650.
#   moon_de440_250416.tf      text frame kernel, ~19 kB: defines MOON_ME as a fixed rotation off
#                             MOON_PA. Only needed to cross-check spacelab's own PA->ME rotation
#                             against ANISE's, which builds Euler parameters from this file.
#   lrorg_2025349_2026074_v01.bsp
#                             SPK, ~7 MB: NASA's reconstructed LRO trajectory, the truth the
#                             selenocentric force model is propagated against in lro_arc.rs.
#
# Neither is committed; the tests that use them skip cleanly when absent, so CI stays network-free.
#
# Source is NAIF itself rather than a mirror: these are the files the DE440 documentation names, and
# provenance for an orientation model is not a detail. (The NyxSpace mirror that
# tools/fetch-ephemeris.sh uses currently presents a certificate that does not match its hostname.)
#
# The LRO archive is a PDS3 volume and its kernels are IEEE big-endian. The SPICE toolkit reads
# either byte order; ANISE rejects anything that is not the host's, so the file is rewritten by
# tools/daf-swap-endian.py after download and the tests read the converted copy.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dir="$root/data/ephemeris"
generic="https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
lro="https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/spk"

# No `|| true` at any call site: that would put the call in a condition context, which switches off
# errexit inside the function and turns a failed download into a silent empty file.
fetch() {
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then
        echo "already present: $dest ($(du -h "$dest" | cut -f1))"
        return 0
    fi
    echo "downloading $(basename "$dest") -> $dest"
    curl -fSL "$url" -o "$dest"
    echo "done: $(du -h "$dest" | cut -f1)"
}

mkdir -p "$dir"
for path in pck/moon_pa_de440_200625.bpc fk/satellites/moon_de440_250416.tf; do
    fetch "$generic/$path" "$dir/$(basename "$path")"
done

# The reconstructed arc the acceptance test uses. Later volumes appear in the same directory as the
# mission continues; the test names this one because its span is what the numbers in
# docs/verification.md were measured on.
archived="$dir/lrorg_2025349_2026074_v01.bsp"
converted="$dir/lro_2025349_2026074.bsp"
fetch "$lro/$(basename "$archived")" "$archived"
if [ ! -f "$converted" ]; then
    uv run python "$root/tools/daf-swap-endian.py" "$archived" "$converted"
else
    echo "already present: $converted ($(du -h "$converted" | cut -f1))"
fi
