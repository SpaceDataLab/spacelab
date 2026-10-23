//! SP3-d against SP3-c, on the three things the version actually changed.
//!
//! SP3-d's own introduction states the whole difference: "the maximum number of satellites is
//! increased from 85 to 999, there are now an unlimited number of comment records allowed in the
//! header, and the maximum length of each comment record has been increased from 60 characters to
//! 80 characters." Every one of those is a header property. The `*`, `P`, `V`, `EP` and `EV`
//! records are byte-for-byte what SP3-c defines.
//!
//! That makes "does this reader handle SP3-d?" a question with a measurable answer rather than an
//! assumption, and this file measures it two ways: a differential test that applies the three
//! changes to a real SP3-c product and demands the parse be unchanged, and a 140-satellite file
//! that a reader written to SP3-c's 85-satellite ceiling could not hold.
//!
//! **No openly served SP3-d product with more than 85 satellites was reachable.** BKG's MGEX
//! directory is empty, AIUB and Wuhan do not answer from here, GFZ's ISDC hosts do not resolve,
//! and CDDIS is behind an Earthdata login — a registration whose terms attach at registration,
//! which is a user's decision and not a fetch script's, the same position as the AVISO grids in
//! `docs/provenance.md`. The specification's own Example 1 declares 140 satellites but elides its
//! data section. So the 140-satellite file here is built by **relabelling** the real product's
//! records across five constellations: every coordinate, clock and epoch in it is IGS data, and
//! only the satellite identifiers are this test's.

#![allow(clippy::expect_used, clippy::float_cmp)]

use spacelab_gnss::sp3::{SatId, Sp3, Version};

/// Three epochs of `IGS0OPSRAP_20262240000_01D_15M_ORB.SP3`, header verbatim. IGS data, used under
/// the IGS Terms of Use recorded in `docs/provenance.md` §6.
const SP3C: &str = include_str!("fixtures/igs_sp3c_2026224.sp3");

/// Everything a caller can observe about a parsed file, so two parses can be compared whole.
fn observable(sp3: &Sp3) -> Vec<String> {
    let mut out = vec![
        format!("{:?}", sp3.reference_frame()),
        format!("{:?}", sp3.time_system()),
        format!("{:?}", sp3.start_epoch()),
        format!("{:?}", sp3.epochs()),
    ];
    // Clocks and velocities have no direct accessor -- they are reached through the interpolators,
    // which at a sample epoch return the sample.
    for sat in sp3.satellites() {
        out.push(format!("{sat:?} {:?}", sp3.positions(sat)));
        for &t in sp3.epochs() {
            out.push(format!(
                "{sat:?} {t} {:?} {:?}",
                sp3.interpolate_clock(sat, t, 2),
                sp3.interpolate_velocity(sat, t, 2)
            ));
        }
    }
    out
}

/// Rewrite the fixture as SP3-d by applying exactly the three changes the version defines.
fn as_sp3d(sp3c: &str) -> String {
    let mut lines: Vec<String> = sp3c.lines().map(str::to_owned).collect();
    // 1. The version character.
    assert!(lines[0].starts_with("#c"), "{}", lines[0]);
    lines[0].replace_range(0..2, "#d");
    // 2 and 3. Comment records: more than SP3-c's four, and longer than SP3-c's sixty characters.
    let last_comment = lines
        .iter()
        .rposition(|l| l.starts_with("/*"))
        .expect("the product's own comment block");
    let extra: Vec<String> = (0..5)
        .map(|i| {
            format!(
                "/* {:<77}",
                format!("comment record {i} of SP3-d's unlimited block")
            )
        })
        .collect();
    for c in &extra {
        assert_eq!(c.len(), 80, "{c}");
    }
    let after_comments = last_comment + 1;
    lines.splice(after_comments..after_comments, extra);
    lines.join("\n")
}

/// The three changes SP3-d makes are header changes, and none of them changes a single value read.
///
/// This is the claim the version bump is worth: a reader that walks records rather than trusting
/// the header's satellite block reads SP3-d for free. It was an assumption until this test; the
/// assertion is over *every* observable, not a spot check, so a change that perturbs one satellite
/// at one epoch fails it.
#[test]
fn the_three_changes_sp3d_makes_do_not_change_a_value_that_is_read() {
    let c = Sp3::parse(SP3C);
    let d = Sp3::parse(&as_sp3d(SP3C));

    assert_eq!(c.version(), Version::C);
    assert_eq!(d.version(), Version::D);
    // The version is the only difference a caller can see.
    assert_eq!(observable(&d), observable(&c));
    assert_eq!(c.epoch_count(), 3);
    assert_eq!(c.satellites().count(), 32);
}

/// A file with more satellites than SP3-c may hold is read in full.
///
/// 140 satellites, which is SP3-c's ceiling plus 55 and a plausible MGEX product's count. The
/// header's satellite block grows from five `+` lines to nine, which is the shape the format's own
/// Example 1 prints; a reader that sized anything off that block, or that stopped at 85, fails
/// here. The coordinates are the real product's, carried over under new identifiers -- so what is
/// synthetic is the labelling and nothing else, and the test says which satellite it expects each
/// real orbit to come back under.
#[test]
fn a_file_with_more_satellites_than_sp3c_allows_is_read_in_full() {
    let source = Sp3::parse(SP3C);
    let real: Vec<SatId> = source.satellites().collect();
    assert_eq!(real.len(), 32);

    // 140 slots across five constellations, in the order SP3 sorts them.
    let slots: Vec<SatId> = [('G', 32u8), ('R', 24), ('E', 36), ('C', 41), ('J', 7)]
        .into_iter()
        .flat_map(|(system, n)| (1..=n).map(move |prn| SatId { system, prn }))
        .collect();
    assert_eq!(slots.len(), 140);

    let mut out: Vec<String> = Vec::new();
    for line in SP3C.lines() {
        if line.starts_with('#') && !line.starts_with("##") {
            out.push(format!("#d{}", &line[2..]));
        } else if line.starts_with('P') || line.starts_with('V') {
            // Each real record is emitted once per constellation, under that constellation's
            // matching slot, so all 140 carry real coordinates and none is invented.
            let id = SatId {
                system: line.as_bytes()[1] as char,
                prn: line[2..4].trim().parse().expect("a PRN"),
            };
            let index = real
                .iter()
                .position(|s| *s == id)
                .expect("a known satellite");
            for slot in slots
                .iter()
                .filter(|s| usize::from(s.prn - 1) % 32 == index)
            {
                out.push(format!(
                    "{}{}{:02}{}",
                    &line[0..1],
                    slot.system,
                    slot.prn,
                    &line[4..]
                ));
            }
        } else if !line.starts_with('+') {
            // The `+` and `++` satellite blocks are rebuilt below, in one piece.
            out.push(line.to_owned());
        }
    }

    // The satellite block, nine `+` lines and nine `++` lines of seventeen entries each.
    let mut block: Vec<String> = Vec::new();
    for (row, chunk) in slots.chunks(17).enumerate() {
        let count = if row == 0 {
            format!("+  {:>3}   ", slots.len())
        } else {
            "+        ".to_owned()
        };
        let mut ids = String::new();
        for s in chunk {
            use std::fmt::Write as _;
            let _ = write!(ids, "{}{:02}", s.system, s.prn);
        }
        while ids.len() < 51 {
            ids.push_str("  0");
        }
        block.push(format!("{count}{ids}"));
    }
    for _ in 0..block.len() {
        let mut accuracy = "++       ".to_owned();
        accuracy.push_str(&"  7".repeat(17));
        block.push(accuracy);
    }
    assert_eq!(block.len(), 18, "nine `+` lines and nine `++` lines");
    // The blocks sit where the original's did: after the `##` line.
    let at = out
        .iter()
        .position(|l| l.starts_with("##"))
        .expect("the GPS-week line")
        + 1;
    out.splice(at..at, block);

    let sp3 = Sp3::parse(&out.join("\n"));
    assert_eq!(sp3.version(), Version::D);
    assert_eq!(
        sp3.satellites().count(),
        140,
        "a reader that stopped at SP3-c's 85 would not get here"
    );
    assert_eq!(sp3.epoch_count(), source.epoch_count());

    // Every slot carries the real orbit it was relabelled from, at every epoch.
    for (i, slot) in slots.iter().enumerate() {
        let original = real[usize::from(slot.prn - 1) % 32];
        assert_eq!(
            sp3.positions(*slot),
            source.positions(original),
            "slot {i} {slot:?} does not carry {original:?}"
        );
        for &t in source.epochs() {
            assert_eq!(
                sp3.interpolate_clock(*slot, t, 2),
                source.interpolate_clock(original, t, 2),
                "{slot:?} clock at {t}"
            );
        }
    }
}

/// The version character is reported, because it is what says which limits the file was written to.
#[test]
fn the_version_character_is_reported_rather_than_assumed() {
    let head = |v: &str| format!("{v}P2026  8 12  0  0  0.00000000      96 ORBIT IGc20 HLM  IGS\n");
    assert_eq!(Sp3::parse(&head("#a")).version(), Version::A);
    assert_eq!(Sp3::parse(&head("#b")).version(), Version::B);
    assert_eq!(Sp3::parse(&head("#c")).version(), Version::C);
    assert_eq!(Sp3::parse(&head("#d")).version(), Version::D);
    // A version from the future is kept as broadcast, not rounded to the newest known one, and
    // neither is a version spelled in a case the format never uses.
    assert_eq!(
        Sp3::parse(&head("#e")).version(),
        Version::Unrecognised('e')
    );
    assert_eq!(
        Sp3::parse(&head("#D")).version(),
        Version::Unrecognised('D')
    );
    // No header line at all is an absence, not SP3-a.
    assert_eq!(
        Sp3::parse("*  2026  8 12  0  0  0.00000000\n").version(),
        Version::Unstated
    );
    assert_eq!(Sp3::parse("#\n").version(), Version::Unstated);
}
