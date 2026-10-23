# Verification loop for spacelab. `just check` is what CI runs and what must be green before a
# change is claimed done; the `mutants*` recipes drive the #213 mutation campaign.

default:
    @just --list

# ── The gate ──────────────────────────────────────────────────────────────────

# fmt, clippy and the test suite, cheapest first.
check: fmt lint test test-optional-features capi

fmt:
    cargo fmt --all -- --check

lint:
    cargo clippy --workspace --all-targets -- -D warnings

# `--no-fail-fast` is not optional: plain `cargo test` stops at the first failing binary, which
# once hid 44 of 46 failures here.
#
# The whole suite.
test:
    cargo test --workspace --no-fail-fast

# `--workspace` does not mean "every line". `onnx` is off by default, so `spacelab-vision/onnx.rs`
# and `spacelab-earthvision/cloud.rs` are `#[cfg]`-ed out of the run above and their tests are never
# linked -- which is why 32 mutants of `usable_footprints` survived against tests that looked green.
# A feature that is never compiled in any gate is a feature that is not tested.
test-optional-features:
    cargo clippy --workspace --all-targets --features spacelab-earthvision/onnx,spacelab-capi/onnx -- -D warnings
    cargo test --workspace --no-fail-fast --features spacelab-earthvision/onnx,spacelab-capi/onnx

# The C ABI, compiled and linked the way a consumer will. `cargo test -p spacelab-capi` checks that
# the header agrees with the Rust; only this proves the header is a header -- that it parses, that
# the struct layouts a C++ compiler computes are the ones Rust wrote, and that the RAII handle frees.
#
# It is in `check` rather than beside it, which costs a cmake and a C++ compiler on any machine that
# runs the gate. That is the price of the crate existing: an ABI whose only consumer-side test is
# optional is an ABI that breaks silently, and it would break for someone else, downstream, with no
# way back to the commit that did it.
#
# Both smoke tests are compiled once per language standard the compiler admits -- C99 through the
# newest C, C++20 through the newest C++, plus an `-fshort-enums` rung -- because the standard a
# consumer passes is the consumer's choice, not ours. The second leg repeats the whole matrix under
# clang: two front-ends disagree about a header far more readily than two standards do.
capi:
    cargo build -p spacelab-capi
    just _capi-matrix capi-build cc c++
    if command -v clang++ >/dev/null 2>&1; then just _capi-matrix capi-build-clang clang clang++; \
    else echo "capi: clang not installed, the matrix ran on the default compiler only" >&2; fi

_capi-matrix DIR CC CXX:
    cmake -S crates/spacelab-capi -B target/{{DIR}} \
        -DCMAKE_C_COMPILER={{CC}} -DCMAKE_CXX_COMPILER={{CXX}} \
        -DSPACELAB_CAPI_LIB="$PWD/target/debug/libspacelab_capi.a"
    cmake --build target/{{DIR}}
    ctest --test-dir target/{{DIR}} --output-on-failure

# ── Mutation campaign ─────────────────────────────────────────────────────────
#
# Every recipe below runs under two nested memory caps, because a mutated adaptive integrator is
# an unbounded allocator and a wall-clock timeout cannot bound an allocation rate. A `dopri5.rs`
# stage-time mutant reached 30.9 GB of dense segments on a 38 GB machine, the kernel OOM-killer
# fired, and systemd took down the whole editor scope this session runs in.
#
#   MEM_TEST   caps each *test binary* in its own transient scope. A runaway dies alone and
#              cargo-mutants records it as caught -- the correct verdict, the suite did notice.
#   MEM_BATCH  caps the batch and, more importantly, moves it out of the editor's cgroup, so no
#              failure of the batch can fail the editor.
#
# It happened a second time, and the caps were not at fault -- they were bypassed. An ad-hoc
# replayer ran `cargo test` on a mutated tree straight from a shell, outside every scope here; the
# mutant was `insar.rs:365 && -> ||`, which makes the branch-cut flood fill re-queue visited pixels
# forever, and the binary reached 34.8 GB before the kernel killed the editor. Two things follow,
# and both are enforced below rather than left to discipline:
#
#   - Mutated code never runs outside a cgroup. `just replay` drives cargo-mutants' own recorded
#     diffs under these same caps; nothing else may apply a mutation. `subprocess.run(timeout=)`
#     bounds the waiting, not the allocating.
#   - One batch at a time. Two concurrent runs budget 2 x MEM_BATCH, which is more memory than the
#     machine has. `_capped-batch` takes a flock, so the second waits instead of overcommitting.
#
# The pre-flight reads MemAvailable and nothing else. This machine's 26 GiB of swap is zram --
# compressed RAM, not headroom -- so counting SwapFree would license exactly the overcommit that
# the OOM-killer then resolves by picking the largest process on the box.
#
# TMPDIR must stay off /tmp: that is a 19 GB tmpfs, i.e. RAM, and cargo-mutants copies the tree
# into it once per job. Overflow surfaces as `Disk quota exceeded (os error 122)` on a *build* --
# a mutant silently recorded as untested.
#
# A default-features run cannot see a module behind a non-default feature, and does not say so: the
# mutation compiles, the module is never built, every test passes, and cargo-mutants reports the
# whole file as MISSED. `earthvision/cloud.rs` came back with 32 phantom survivors that way --
# including `usable_footprints -> vec![]` against a suite that already asserted the opposite. The
# only two such modules here are `earthvision::cloud` and `vision::onnx`, both gated on `onnx`
# (everything else spells it `cfg(any(feature = "std", test))`, which `cargo test` satisfies), and
# they must be scanned as
#     just mutants-pkg <name> "spacelab-earthvision spacelab-vision" \
#         --features spacelab-earthvision/onnx -f cloud.rs -f onnx.rs
# Adding a non-default feature to any crate adds a blind spot here until it is scanned that way.

# Which batch lock a run competes for. One lane is the default and the safe one; a second lane needs
# its own `MEM_BATCH` chosen so the lanes' caps still fit the machine:
#     just LANE=lane2 MEM_BATCH=14G mutants <name> <files...>
# The lock name has to be passed to `_capped-batch` explicitly, because `just VAR=v recipe` overrides
# the variable for *that* invocation only and the nested `just` inside a recipe body re-evaluates
# every default. Both lanes silently shared `batch.lock` at 20 G until that was fixed.
LANE := "batch"

MEM_TEST := "4G"
# MEM_TEST x JOBS is the worst case, so this has to rise with JOBS. This is a ceiling, not a
# reservation: one lane at `--jobs 4` was measured peaking at 6 141 MiB (`meta/prop_ab.log`). The cap
# must never be the thing that kills a run.
MEM_BATCH := "28G"
# What the pre-flight leaves for everything that is not this campaign. Below this the OOM-killer
# starts choosing, and it chooses by size. It was 6G while the editor and a browser shared the
# machine; on a box dedicated to the campaign that reserve is what stalls the second lane, because
# the guard reads MemAvailable *after* the first lane has taken its share.
MEM_HEADROOM := "3G"
# Comfortably above cargo-mutants' own auto-set timeout (21s on this suite) and the replayer's
# per-mutant budget, so only a genuinely non-terminating mutant ever reaches it.
TEST_MAX_SEC := "1200"
# How long a second batch waits for the first. Long enough to serialise a queue, short enough that
# a wedged holder surfaces the same day.
BATCH_WAIT_SEC := "21600"
MUTANTS_CACHE := env("HOME") / ".cache/spacelab-mutants"

# The files every mutation run skips, and the one place the list lives. Three are generated tables
# whose mutants no test can kill; the fourth is a generated table whose mutants one test kills by
# construction. Each is argued out in `.cargo/mutants.toml` -- they are decisions about what a run
# can measure, never equivalences, which is why none of them is an `exclude_re` entry. Keeping the
# list here rather than in each queue script is what stops one block's verdict from being compared
# against another that skipped a different set.
MUTANTS_SKIP := "crates/spacelab-slr/src/loading_tables.rs " + \
                "crates/spacelab-fa/src/cio_tables.rs " + \
                "crates/spacelab-fa/src/nut_tables.rs " + \
                "crates/spacelab-propagation/src/geopotential_coeffs.rs"

# The machine is 8 physical / 16 logical cores -- an earlier comment here said 8, which is why
# `--jobs 4` (4 x CARGO_BUILD_JOBS=2 = 8 rustc) looked saturating when it was using half the box.
#
# `--test-tool nextest` was measured against libtest and lost, so libtest stays. A-B-A-B over
# round-robin shards of spacelab-eop (the legs are comparable by construction): libtest at
# `--jobs 4` did 397 mutants in 94s = 253/min; nextest at `--jobs 6` with RUST_TEST_THREADS=2 did
# 132 in 99s = 80/min, 3.2x worse. nextest is fail-fast, which should help when 90% of mutants are
# caught, but it spawns a process per test and this workspace has ~900 of them -- the fork cost
# swamps the early exit. Raising JOBS was only ever tested with nextest, so it is not adopted
# either; the measured configuration is the one that ships.
#
# What a mutant actually costs, measured over 36 outcomes of the `prop-atmo3` block: build 19.55 s,
# test 3.06 s. 86 % of the campaign is rustc, and the reason is structural rather than fixable --
# `spacelab-propagation` has 32 integration-test targets, every one of which links against the lib
# and so is rebuilt for every single mutation. So `JOBS x CARGO_BUILD_JOBS` is the number that
# matters, and at 4 x 2 = 8 it was half the box per lane and exactly the box across two. 5 x 2 = 10
# oversubscribes by 25 %, which is the useful direction when a quarter of each job is spent linking
# and waiting rather than computing.
JOBS := "5"
TEST_TOOL := "cargo"

# Per-mutant test timeout, seconds. Left unset, cargo-mutants auto-sets it to ~5x the baseline,
# which for a file-scoped block is 20 s -- short enough to turn a *kill* into a timeout whenever the
# mutant makes an integrator run to its own counted step cap. Measured on `driver.rs:436:13`
# (`while t < t1` -> `<=`): the run ends in `StepLimit { steps: 100000 }` and fails its test after
# 5.0 s, but every adaptive integration in the crate then costs 100 000 steps and the suite passes
# 20 s, so `prop-driver` recorded seven timeouts where seven kills were waiting. Package blocks have
# always passed 180 explicitly; this makes the two agree, so a verdict means the same thing whether
# the block was scoped by file or by package.
MUTANTS_TIMEOUT := "180"
_mutants-env := "CARGO_BUILD_JOBS=2 RUSTFLAGS='-C link-arg=-fuse-ld=mold -C link-arg=-Wl,--threads=2'"

# Run cargo-mutants over FILES (repo-relative, space-separated) into a named output directory.
mutants NAME +FILES:
    #!/usr/bin/env bash
    set -uo pipefail
    command -v systemd-run >/dev/null || { echo "systemd-run absent: refusing to run uncapped"; exit 1; }
    out="{{MUTANTS_CACHE}}/{{NAME}}"
    snap="{{MUTANTS_CACHE}}/snapshot/{{NAME}}"
    mkdir -p "$out" "{{MUTANTS_CACHE}}/tmp" "$snap"
    args=(); for f in {{FILES}}; do args+=(--file "$f"); just mutants-anchors "$f" || exit 1; done
    rm -rf "$snap"; mkdir -p "$snap"
    tar -cf - --exclude=./target --exclude=./.claude . | tar -xf - -C "$snap" || exit 1
    just _tree-stamp "$snap" > "$out/tree.stamp"
    just LANE={{LANE}} MEM_BATCH={{MEM_BATCH}} _capped-batch cargo mutants -d "$snap" --jobs {{JOBS}} --test-tool {{TEST_TOOL}} \
        --timeout {{MUTANTS_TIMEOUT}} "${args[@]}" --output "$out" \
        > "{{MUTANTS_CACHE}}/{{NAME}}.log" 2>&1
    just mutants-verdict {{NAME}}

# Run cargo-mutants over whole PACKAGES rather than files.
#
# One invocation per group instead of one per file, because cargo-mutants pays a cold build *per
# scratch directory per invocation*: twelve small batches pay that tax twelve times, while one big
# batch amortises it over thousands of mutants. `--iterate` is what makes a big batch safe -- it
# skips mutants already caught in this output directory, so a kill costs the tail and not the run.
#
# EXTRA takes the leftover arguments, for `-E`/`--shard`/`--timeout`. Pass them last.
#
# PACKAGES is ONE argument -- quote a list. Unquoted, `mutants-pkg gate-face spacelab spacelab-slr`
# binds `spacelab` to PACKAGES and leaves `spacelab-slr` in EXTRA, which is spliced verbatim into
# the command line: `error: unexpected argument 'spacelab-slr' found`, one second in, no missed.txt.
# Two blocks of a four-block queue died that way and read like a snapshot race until the log was
# opened, so the guard below turns it into a message that names the mistake.
mutants-pkg NAME PACKAGES *EXTRA:
    #!/usr/bin/env bash
    set -uo pipefail
    command -v systemd-run >/dev/null || { echo "systemd-run absent: refusing to run uncapped"; exit 1; }
    # Only the first EXTRA word: everything after it may legitimately be a flag's value ("300").
    extra=({{EXTRA}})
    case "${extra[0]:--}" in
        -*) ;;
        *) echo "mutants-pkg: '${extra[0]}' reached EXTRA as a bare word." >&2
           echo "  PACKAGES is a single argument -- quote the list: mutants-pkg NAME \"a b c\"" >&2
           exit 1 ;;
    esac
    out="{{MUTANTS_CACHE}}/{{NAME}}"
    snap="{{MUTANTS_CACHE}}/snapshot/{{NAME}}"
    mkdir -p "$out" "{{MUTANTS_CACHE}}/tmp"
    args=(); for p in {{PACKAGES}}; do args+=(-p "$p"); done
    for f in {{MUTANTS_SKIP}}; do args+=(--exclude "$f"); done
    rm -rf "$snap"; mkdir -p "$snap"
    tar -cf - --exclude=./target --exclude=./.claude . | tar -xf - -C "$snap" || exit 1
    just _tree-stamp "$snap" > "$out/tree.stamp"
    just LANE={{LANE}} MEM_BATCH={{MEM_BATCH}} _capped-batch cargo mutants -d "$snap" --jobs {{JOBS}} --test-tool {{TEST_TOOL}} \
        --iterate "${args[@]}" {{EXTRA}} --output "$out" \
        > "{{MUTANTS_CACHE}}/{{NAME}}.log" 2>&1
    just mutants-verdict {{NAME}}

# cargo-mutants re-reads the source tree while the run is in flight, but validates it exactly once:
# the baseline, at the start. Edit a test after that and later jobs carry a failure the baseline
# never saw -- so their mutants are recorded as caught, by a test that never looked at the mutated
# code. Spurious catches are the dangerous direction: they hide survivors.
#
# That is not hypothetical, twice. The `atmospheres` block ran 09:19-09:46 while the abm8 step-budget
# test was being written (committed 94bd5f8, 12:22) and 523 of its 596 catches are
# `abm::tests::rejects_bad_arguments` failing on dtm2020, jb2008 and harris_priester mutants. Then
# `nrlmsise` was poisoned the same way by a triage injection into dtm2020, 359 logs of 364. Both were
# thrown away. Every other block's dominant failing test is in the module it mutates.
#
# So a run no longer reads the live tree at all: it works on a snapshot taken at the instant it
# starts. 75 MB and 0.13 s buys back the ability to edit source while a block runs, which is most of
# the campaign's wall clock. The stamp is then a check that nothing wrote into the snapshot itself.
_tree-stamp DIR=".":
    @cd {{DIR}} && find crates -name '*.rs' -type f -print0 | sort -z | xargs -0 sha256sum | \
        cat - .cargo/mutants.toml | sha256sum | cut -d' ' -f1

# `LabOutcome::exit_code` (cargo-mutants 27.1.0, outcome.rs:110) tests `timeout > 0` *before*
# `missed > 0`, so a run with both reports 3 -- and 3 reads as "detected". The `integrators` block
# exited 3 carrying 749 undetected survivors, which is why nothing here reads the exit code.
#
# Tally a finished run and succeed only if missed.txt is empty.
mutants-verdict NAME:
    #!/usr/bin/env bash
    set -uo pipefail
    missed="{{MUTANTS_CACHE}}/{{NAME}}/mutants.out/missed.txt"
    [ -f "$missed" ] || { echo "{{NAME}}: no missed.txt -- the run did not complete"; exit 1; }
    stamp="{{MUTANTS_CACHE}}/{{NAME}}/tree.stamp"
    snap="{{MUTANTS_CACHE}}/snapshot/{{NAME}}"
    if [ -f "$stamp" ] && [ "$(cat "$stamp")" != "$(just _tree-stamp "$snap")" ]; then
        echo "{{NAME}}: the snapshot changed during the run -- every verdict in it is void"
        exit 1
    fi
    n=$(grep -c . "$missed")
    for k in caught timeout unviable; do
        f="{{MUTANTS_CACHE}}/{{NAME}}/mutants.out/$k.txt"
        [ -f "$f" ] && printf '%-9s %s\n' "$k" "$(grep -c . "$f")"
    done
    printf '%-9s %s\n' missed "$n"
    [ "$n" -eq 0 ]

# Which crates' verdicts still speak for the tree that ships, and which have been overtaken.
#
# `mutants-pkg` snapshots the tree *before* it takes the batch lock, so a block reports on the code
# as it stood when it started: `lock.json`'s start_time is the date its verdict is about. A commit
# landing on that crate afterwards is unmeasured, and "0 survivors" then describes a tree that no
# longer exists. spacelab-slr sat in the CI gate that way with 23 survivors in a file added after
# its seal.
#
# A block counts as covering a crate only if it planned at least as many mutants for it as the crate
# generates now -- a `--file`-scoped block seals a file, not a crate, and must not be read as one.
# `>=` rather than `==` because the ledger only grows, and because the command-line `--exclude`s for
# the generated coefficient tables shrink the live side. And a block only counts once it has
# *finished*: `mutants.json` is written when the run starts, so a block still in flight would
# otherwise read as a completed seal with no survivors yet.
mutants-staleness:
    #!/usr/bin/env bash
    set -uo pipefail
    live=$(mktemp -t mutants-live-XXXXXX)
    trap 'rm -f "$live"' EXIT
    # A parse of every crate, no build: minutes, and the only cargo call this recipe makes.
    cargo mutants --list > "$live" 2>/dev/null || exit 1
    uv run python - "$live" "{{MUTANTS_CACHE}}" {{MUTANTS_SKIP}} <<'PY'
    import json, pathlib, subprocess, sys
    from collections import Counter

    def crate(path):
        parts = pathlib.PurePosixPath(path).parts
        return "/".join(parts[:2]) if len(parts) > 1 and parts[0] == "crates" else None

    def tally(lines):
        return Counter(filter(None, (crate(l.split(":", 1)[0]) for l in lines if l.strip())))

    # The live side has to skip exactly what the blocks skipped: a run that honoured MUTANTS_SKIP
    # can never reach an unskipped live count, and would read forever as not covering the crate.
    skip = set(sys.argv[3:])
    live = tally(l for l in pathlib.Path(sys.argv[1]).read_text().splitlines()
                 if l.split(":", 1)[0] not in skip)

    covers = {}
    for lock in pathlib.Path(sys.argv[2]).glob("*/mutants.out/lock.json"):
        out = lock.parent
        got = {k: (out / f"{k}.txt").read_text().splitlines() if (out / f"{k}.txt").exists() else []
               for k in ("caught", "missed", "timeout", "unviable", "previously_caught")}
        # `--iterate` is always passed, and it drops previously-caught mutants from mutants.json.
        # Reading coverage off that alone makes a resumed block look narrower than the fresh one it
        # continues -- which is how gnss-seal lost to a run a day older than it.
        planned = Counter(filter(None, (crate(m["file"])
                                        for m in json.loads((out / "mutants.json").read_text()))))
        planned += tally(got["previously_caught"])
        if sum(len([l for l in v if l.strip()]) for v in got.values()) < sum(planned.values()):
            continue  # still in flight, or killed part-way: it has not answered yet
        missed, slow = tally(got["missed"]), tally(got["timeout"])
        started = json.loads(lock.read_text())["start_time"]
        for c, n in planned.items():
            if n >= live.get(c, 0) and started > covers.get(c, ("",))[0]:
                covers[c] = (started, out.parent.name, missed.get(c, 0), slow.get(c, 0))

    unsealed = 0
    for c in sorted(live):
        started, block, open_, slow = covers.get(c, (None, None, 0, 0))
        # Any commit under the crate's directory expires it, `Cargo.toml` included -- a feature or a
        # dependency bump really can move a mutant. Most cannot, so the commit is named as well as
        # dated: `c493d7c` added a version string to sixteen manifests and expired nine crates whose
        # verdicts were minutes old. The rule stays conservative; the output makes the false positive
        # one `git show` away instead of a re-run.
        log = subprocess.run(["git", "log", "-1", "--format=%cI %h", "--", c],
                             capture_output=True, text=True).stdout.strip()
        touched, _, sha = log.partition(" ")
        if started is None:
            state, note = "NEVER MEASURED", "no finished whole-crate block"
        elif touched and touched > started:
            state = "STALE"
            note = f"{block} started {started[:19]}, last commit {touched[:19]} ({sha})"
            note += f", {open_} survivors then" if open_ else ""
        elif open_:
            state, note = "OPEN", f"{block} left {open_} unresolved"
        else:
            state, note = "clean", f"{block} {started[:19]}"
        # A timeout is a detection -- the suite stopped terminating rather than passing -- so it does
        # not open a crate. It is still owed an entry in the catalogue, so it is never silent either.
        note += f"  [+{slow} timeouts]" if slow else ""
        unsealed += state != "clean"
        print(f"{c:<34} {live[c]:>6}  {state:<14} {note}")
    print(f"\n{len(live)} crates, {len(live) - unsealed} whose verdict speaks for HEAD")
    sys.exit(1 if unsealed else 0)
    PY

# An exclusion in .cargo/mutants.toml is a *regex* carrying a file:line:col anchor, so it fails in
# two silent directions. It goes stale whenever a line is inserted above the code it names -- a
# justified mutant then reappears as a survivor. And an unescaped metacharacter changes what it
# matches: `replace + with -` quantifies the space and matches nothing, while `replace && with ||`
# is an alternation with an empty branch, which matches *every* mutant and would switch the file
# off entirely. Both have happened. This proves neither: it lists FILE's mutants with and without
# the config and requires each entry naming FILE to have removed exactly one.
#
# The listing goes to a private temp file, never a fixed path. A fixed one is a race: the queue
# checks 15 files in a loop while a batch or a hand check runs the same recipe, and `>` truncates
# before it fills. `dsst/gaussian.rs` was read at that instant, came back `2 entries, 497 mutants
# excluded` against its true 2/2, and stopped the whole `confirm_dsst` block. The silent direction
# is worse: the stale-anchor search would then run against another file's mutants and pass.
# The same reconciliation over the whole ledger at once, in two `--list` calls rather than two per
# file. Checking the 137 anchored files one at a time costs ~1.6 h and 274 invocations; this is ~2 min
# and answers a strictly stronger question, because it can see across files: `mutants-anchors` counts
# entries against a per-file total, so an entry that matches *two* mutants and one that matches none
# cancel out and the file still reconciles. Here every entry is required to claim exactly one mutant
# and every mutant to be claimed by exactly one entry.
mutants-ledger:
    uv run python tools/mutants-ledger.py

# Does the §5.3 table still say what the runs said? Every row's counts are copied by hand out of what
# `mutants-verdict` printed above, and nothing checked them until this existed: ten rows of sixty-two
# disagreed on the first pass, eight of them carrying the *unviable* count in the Timeouts column,
# which is the substitution that flatters a seal -- a timeout scores as a detection and an unviable
# mutant does not score at all.
#
# Not a CI step and not part of `mutants`: the runs live in {{MUTANTS_CACHE}}, outside the repo, and
# a runner that cannot see them would pass by checking nothing. Run it after writing a row. With no
# argument it checks every row whose block is still cached; with block names, only those.
mutants-table *BLOCKS:
    uv run python tools/mutants-table.py {{BLOCKS}}

# The path a pattern anchors to is not always enough to say which file it belongs to. `lib.rs` is
# the case that broke this: nine entries anchor spacelab-gpu's and two anchor spacelab-coverage's,
# and every one of them is a path-component suffix of both, so either file came back `11 entries,
# 9 excluded` and `11 entries, 2 excluded`. A stricter path rule cannot fix it -- the two files have
# the same name and the ledger anchors basenames -- so the tie is broken on content instead: an
# entry belongs to FILE when it also matches one of FILE's own mutants. Line, column and function
# name make that exact.
#
# Candidates that suffix-match but match nothing here are reported as a trailing count, not dropped
# in silence, because that count is also what a genuinely dead anchor looks like from inside one
# file. Which of the two it is takes the whole ledger to answer, so the verdict on it belongs to
# `just mutants-ledger` and this recipe only points at it.
mutants-anchors FILE:
    #!/usr/bin/env bash
    set -uo pipefail
    live=$(mktemp -t mutants-anchors-XXXXXX); all=$(mktemp -t mutants-anchors-all-XXXXXX)
    trap 'rm -f "$live" "$all"' EXIT
    cargo mutants --list --file {{FILE}} > "$live" 2>/dev/null
    cargo mutants --no-config --list --file {{FILE}} > "$all" 2>/dev/null
    uv run python - {{FILE}} "$all" "$live" <<'PY'
    import pathlib, re, sys
    allm = pathlib.Path(sys.argv[2]).read_text().splitlines()
    live = pathlib.Path(sys.argv[3]).read_text().splitlines()
    toml = pathlib.Path(".cargo/mutants.toml").read_text()
    # The path test is the cheap pre-filter, and it compares at a path-component boundary. A plain
    # string suffix is not enough and the difference is not academic: 56 entries anchored
    # `short_period.rs` also suffix-match `gaussian_short_period.rs`, `tesseral_short_period.rs` and
    # `third_body_short_period.rs`, so the count came out 59/3, 56/0 and 63/7 and stopped the queue
    # on three files whose ledger is in fact correct. Matching on the basename alone would be worse
    # still -- every other `mod.rs` in the ledger.
    # Whether the anchor names a path or only a basename decides what a miss means. An anchor that
    # carries its crate path names one file and no other, so failing to find its mutant is drift --
    # `spacelab-capi`'s two entries went stale exactly that way and were reported as belonging
    # elsewhere for as long as they were wrong. A bare `lib.rs:` cannot tell the two apart, because
    # every crate has one, so there a miss is only ever "not this file".
    candidates = []
    for raw in re.findall(r'^\s*"((?:[^"\\]|\\.)*)"\s*,\s*$', toml, re.M):
        pat = raw.replace("\\\\", "\\")
        anchored = pat.split(":", 1)[0].replace("\\", "")
        if sys.argv[1] == anchored:
            candidates.append((pat, True))
        elif "/" not in anchored and sys.argv[1].endswith("/" + anchored):
            candidates.append((pat, False))
    entries, elsewhere, bad = [], 0, 0
    for pat, names_this_file in candidates:
        if re.search(r"(?<!\\)[|+*?(){}\[\]^$]", pat):
            print(f"  unescaped metacharacter: {pat}"); bad += 1
        elif [l for l in live if re.search(pat, l)]:
            print(f"  stale anchor, mutant still live: {pat}"); bad += 1
            entries.append(pat)
        elif [l for l in allm if re.search(pat, l)]:
            entries.append(pat)
        elif names_this_file:
            print(f"  drifted anchor, nothing at that line:col: {pat}"); bad += 1
        else:
            elsewhere += 1
    excluded = len(allm) - len(live)
    print(f"{len(entries)} entries, {excluded} mutants excluded, {bad} broken"
          + (f", {elsewhere} same-name anchors elsewhere" if elsewhere else ""))
    sys.exit(1 if bad or len(entries) != excluded else 0)
    PY

# Re-inject a run's survivors one at a time and report which the suite now catches.
replay OUT PACKAGE *ARGS:
    just _capped-batch uv run python tools/mutant_triage.py --out {{OUT}} --package {{PACKAGE}} {{ARGS}}

# Refuse to start unless MemAvailable covers NEED plus a desktop's worth of room. Reads
# /proc/meminfo directly: `free`'s swap column is zram here, which is RAM and not headroom.
_mem-guard NEED:
    #!/usr/bin/env bash
    set -uo pipefail
    avail=$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) * 1024 ))
    need=$(( $(numfmt --from=iec {{NEED}}) + $(numfmt --from=iec {{MEM_HEADROOM}}) ))
    if [ "$avail" -lt "$need" ]; then
        echo "refusing to start: MemAvailable $(numfmt --to=iec "$avail") <" \
             "{{NEED}} + {{MEM_HEADROOM}} headroom = $(numfmt --to=iec "$need")" >&2
        exit 1
    fi

# Run CMD capped and detached from the editor's scope, with each test binary capped in turn.
#
# CPUWeight keeps the desktop responsive; the per-test runner is exported here and never in
# .cargo/config.toml, where it would break CI -- a GitHub runner has no user session bus.
#
# RuntimeMaxSec is what stops a non-terminating mutant, and it has to live on the *scope* rather
# than in the caller. cargo-mutants and the replayer both bound a job with a wall clock they own --
# `subprocess.run(timeout=)`, `--timeout` -- and on expiry each kills the process it spawned. The
# test binary is a grandchild in its own transient scope, so it survives, is reparented to
# `systemd --user`, and spins. Seven of them were found doing exactly that, 98% of a core each for
# 18-21 hours, on a machine whose queue had been idle since 15:31. A timeout bounds the waiting;
# only the cgroup bounds the work.
#
# MemorySwapMax=0 on the per-test scope is the difference between a runaway that dies at MEM_TEST
# and one that spills into zram and takes the machine's real memory with it on the way.
#
# The lock is taken before the memory check, not after, so that a queued batch is measured at the
# moment it actually starts rather than while its predecessor still holds 20G -- checking first
# would turn every queued batch into a refusal. It is held on fd 9, which survives the exec into
# systemd-run, so the lock's lifetime is the batch's and not this shell's.
_capped-batch +CMD:
    #!/usr/bin/env bash
    set -uo pipefail
    lock="{{MUTANTS_CACHE}}/{{LANE}}.lock"
    mkdir -p "{{MUTANTS_CACHE}}" || exit 1
    exec 9>"$lock" || exit 1
    if ! flock -n 9; then
        echo "another batch holds $lock -- waiting up to {{BATCH_WAIT_SEC}}s" >&2
        flock -w {{BATCH_WAIT_SEC}} 9 || { echo "gave up waiting for $lock" >&2; exit 1; }
    fi
    just _mem-guard {{MEM_BATCH}} || exit 1
    triple=$(rustc -vV | sed -n 's/^host: //p')
    runner_var="CARGO_TARGET_$(echo "$triple" | tr 'a-z-' 'A-Z_')_RUNNER"
    export TMPDIR="{{MUTANTS_CACHE}}/tmp"
    export "$runner_var=systemd-run --user --scope -q --collect -p MemoryMax={{MEM_TEST}} -p MemorySwapMax=0 -p RuntimeMaxSec={{TEST_MAX_SEC}} --"
    export {{_mutants-env}}
    exec systemd-run --user --scope -q --collect \
        -p MemoryMax={{MEM_BATCH}} -p MemorySwapMax=2G -p CPUWeight=20 \
        -- {{CMD}}

# The tree must be free of live mutations before a batch snapshots it, or the batch bakes one into
# its own baseline and every result in it is meaningless.
#
# The gate cannot be `git diff` -- the campaign legitimately carries thousands of uncommitted test
# lines under crates/*/src, and a diff cannot tell those from a one-operator mutation. It cannot be
# cargo-mutants' `changed by cargo-mutants` marker either: the replayer writes the span replacement
# without it, which is how `gj.rs:80` and `ark.rs:145` sat live in production files unnoticed.
#
# So the replayer declares what it holds. It writes the lock before mutating and removes it after
# restoring; a SIGKILLed replayer leaves the lock *and* its pristine backup, which blocks the queue
# and leaves the tree recoverable instead of silently wrong.
#
# Block until no replayer holds a mutation.
wait-clean:
    #!/usr/bin/env bash
    set -uo pipefail
    lock="{{MUTANTS_CACHE}}/replay.lock"
    while [ -f "$lock" ]; do
        # The replayer removes the lock the instant it finishes, so it can vanish between the test
        # above and the read below. Treat that as released, not as a dead pid -- reporting a stale
        # lock there would stop the queue on a tree that is in fact clean.
        pid=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["pid"])' "$lock" 2>/dev/null)
        if [ -z "$pid" ]; then
            # Either it has just been removed -- the `while` test settles that on the next pass -- or
            # it is a half-written file the replayer is still creating. One second either way.
            sleep 1
            continue
        fi
        if kill -0 "$pid" 2>/dev/null; then
            echo "$(date -Is) waiting: replayer $pid holds a mutation"
            sleep 30
        else
            echo "stale lock from dead pid $pid -- restore the backup it names, then remove it:"
            cat "$lock"
            exit 1
        fi
    done

# ── Orphaned test binaries ────────────────────────────────────────────────────

# A test binary outlives its supervisor only when the supervisor cannot run its own cleanup.
# Measured on this machine: `timeout`, with its default TERM and with `-s KILL` alike, reaps the
# worker, because it puts the child in its own process group and signals the group; Ctrl-C and the
# agent harness do the same. Only a SIGKILL *of cargo itself* orphans the worker -- SIGKILL cannot
# be trapped, so nothing propagates it, and the worker is reparented to the user manager (measured:
# worker 213088 -> ppid 3889) and runs to completion at 100% of a core with nobody to read its
# output. The OOM killer takes exactly that path, which is how a batch under `-p MemoryMax` leaves
# spinners behind.
#
# So the test is ancestry, not the immediate parent: `systemd-run --scope` sits between cargo and
# the binary in the `mutants` recipes above, and a killed cargo leaves that scope process orphaned
# too. A binary under /deps/ with no cargo anywhere above it is running for nobody.
#
# One false positive is possible by design: a test binary you started by hand from a shell, to pass
# it `--nocapture` or a filter. That is why the default only lists.
#
# List orphaned test binaries; `just reap kill` ends them.
reap ACTION="list":
    #!/usr/bin/env bash
    set -uo pipefail
    # Matching on /proc/<pid>/exe rather than on the command line keeps this recipe from finding
    # itself: its own processes are bash and awk, whose exe paths hold no /deps/. `pgrep -f` does
    # not have that property and has twice matched its own invocation here.
    orphans=()
    for entry in /proc/[0-9]*; do
        pid=${entry#/proc/}
        exe=$(readlink "$entry/exe" 2>/dev/null) || continue
        case "$exe" in */deps/*) ;; *) continue ;; esac
        # Walk to pid 1 looking for a live supervisor.
        supervised=0 ancestor=$pid
        while [ "$ancestor" -gt 1 ] 2>/dev/null; do
            ancestor=$(awk '/^PPid:/{print $2}' "/proc/$ancestor/status" 2>/dev/null) || break
            [ -n "$ancestor" ] || break
            case "$(cat "/proc/$ancestor/comm" 2>/dev/null)" in
                cargo|cargo-mutants|cargo-nextest|rustc) supervised=1; break ;;
            esac
        done
        [ "$supervised" -eq 1 ] && continue
        orphans+=("$pid")
        ps -o pid=,etime=,pcpu=,rss=,args= -p "$pid" 2>/dev/null
    done
    if [ ${#orphans[@]} -eq 0 ]; then
        echo "no orphaned test binaries"
        exit 0
    fi
    if [ "{{ACTION}}" != "kill" ]; then
        echo "orphaned test binaries: ${#orphans[@]} -- \`just reap kill\` ends them"
        exit 0
    fi
    # TERM first: a test binary installs no handler, so it dies on the first signal, and a process
    # that did install one gets to run it. KILL only for what is still there a second later.
    kill -TERM "${orphans[@]}" 2>/dev/null
    sleep 1
    survivors=()
    for pid in "${orphans[@]}"; do
        kill -0 "$pid" 2>/dev/null && survivors+=("$pid")
    done
    [ ${#survivors[@]} -gt 0 ] && kill -KILL "${survivors[@]}" 2>/dev/null
    echo "reaped ${#orphans[@]} (${#survivors[@]} needed SIGKILL)"
