#!/usr/bin/env bash
# Build NEPTUNE and libslam from the pinned snapshot tools/fetch-neptune.sh leaves in gitignored
# space. Nothing here enters the Cargo workspace.
#
# NEPTUNE's own build.sh is not used. It starts with `git submodule update --init --recursive`,
# which cannot work on a tarball snapshot, and it builds OPI and pFUnit, which this harness does not
# want: the oracle is driven through NEPTUNE's standalone `neptune-sa` executable, and pFUnit tests
# NEPTUNE against itself rather than against us. What is kept is the build order build.sh encodes --
# libslam first, then NEPTUNE against libslam's installed include/ and lib/.
#
# Release, not build.sh's Debug default: -O0 with -fbounds-check turns a seven-day degree-72 lunar
# propagation into something measured in hours, and the comparison is of answers, not of timings.
#
# Usage: tools/neptune-oracle/build.sh [-j JOBS]

set -euo pipefail

jobs="$(nproc)"
while getopts "j:" opt; do
  case "$opt" in
  j) jobs="$OPTARG" ;;
  *)
    echo "usage: $0 [-j JOBS]" >&2
    exit 2
    ;;
  esac
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dir="${NEPTUNE_DIR:-$root/data/oracles/neptune}"

if [[ ! -f "$dir/CMakeLists.txt" ]]; then
  echo "no NEPTUNE snapshot in $dir -- run tools/fetch-neptune.sh first" >&2
  exit 1
fi
if [[ ! -f "$dir/libslam/CMakeLists.txt" ]]; then
  echo "no libslam in $dir/libslam -- re-run tools/fetch-neptune.sh" >&2
  exit 1
fi

echo "== libslam"
cmake -S "$dir/libslam" -B "$dir/libslam/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_Fortran_COMPILER=gfortran \
  -DENABLE_OpenMP_SUPPORT=OFF \
  -DENABLE_POSTGRESQL_SUPPORT=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >"$dir/libslam-configure.log" 2>&1 ||
  {
    tail -30 "$dir/libslam-configure.log" >&2
    exit 1
  }
cmake --build "$dir/libslam/build" -j "$jobs" >"$dir/libslam-build.log" 2>&1 || {
  tail -40 "$dir/libslam-build.log" >&2
  exit 1
}
cmake --install "$dir/libslam/build" >>"$dir/libslam-build.log" 2>&1 || true

# NEPTUNE's CMake looks for libslam under libslam/include and libslam/lib, which build.sh creates as
# symlinks into the build tree rather than as an install prefix, and then again under NEPTUNE's own
# top-level lib/ and include/.
ln -sfn build/include "$dir/libslam/include"
ln -sfn build/lib "$dir/libslam/lib"
mkdir -p "$dir/lib" "$dir/include"
ln -sfn ../libslam/lib/libslam-Fortran.so "$dir/lib/libslam-Fortran.so"
ln -sfn ../libslam/include/SLAM "$dir/include/SLAM"

# -DLIBSLAM rather than trusting `find_library(LIBSLAM slam-Fortran lib)`: that relative "lib" hint is
# resolved against cmake's working directory, not against the source tree, so it finds the library
# only when cmake is invoked from NEPTUNE's own root the way its build.sh does.
echo "== neptune"
cmake -S "$dir" -B "$dir/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_Fortran_COMPILER=gfortran \
  -DLIBSLAM="$dir/libslam/build/lib/libslam-Fortran.so" \
  -DENABLE_OpenMP_SUPPORT=OFF \
  -DENABLE_OPI_SUPPORT=OFF \
  -DENABLE_PFUNIT=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >"$dir/neptune-configure.log" 2>&1 ||
  {
    tail -30 "$dir/neptune-configure.log" >&2
    exit 1
  }
cmake --build "$dir/build" -j "$jobs" >"$dir/neptune-build.log" 2>&1 || {
  tail -40 "$dir/neptune-build.log" >&2
  exit 1
}
cmake --install "$dir/build" >>"$dir/neptune-build.log" 2>&1 || true

found="$(find "$dir" -name 'neptune-sa' -type f -perm -u+x -print -quit)"
if [[ -z "$found" ]]; then
  echo "build finished but no neptune-sa executable was produced; see $dir/neptune-build.log" >&2
  exit 1
fi
echo "neptune-sa -> $found"
