//! The cut of the two EOP excerpts, pinned (plans/17 Ф20.M; `docs/provenance.md`, row `iers-eop`).
//!
//! `finals2000A_excerpt.txt` was cut on 2026-07-10 from `finals.all.iau2000.txt` as
//! `tools/fetch-eop.sh` fetches it from datacenter.iers.org. `eopc04_excerpt.txt` was cut on
//! 2026-10-24 from the `eopc04.1962-now` that `astropy-iers-data` 0.2026.10.19 ships. Three IERS
//! notices bear on them:
//!
//! - Message 546: the rapid solution's x-pole carried a +30 µas bias from 2025-04-02 until
//!   2025-11-12. Both cuts post-date the correction, and neither holds a row inside the episode.
//! - Message 547: the fourth update of EOP 20 C04 took effect on 2026-02-05, and IERS keeps the
//!   earlier series as `eopc04_20_v3`. Both cuts post-date it, so the C04 rows are the fourth update.
//! - Message 566: Bulletin A moved to a USNO Kalman smoother and a weighted cubic spline on
//!   2026-10-15; past values may move by less than 0.05 mas and 20 µs. The `finals2000A` cut
//!   pre-dates it, so its Bulletin A digits are the earlier processing's.
//!
//! A re-cut changes these numbers and fails the test, so that a new cut is a diff with a reason.

#![allow(clippy::expect_used, clippy::unwrap_used)]

const FINALS: &str = include_str!("fixtures/finals2000A_excerpt.txt");
const C04: &str = include_str!("fixtures/eopc04_excerpt.txt");

/// Message 546's bias episode, 2025-04-02 to 2025-11-12, as MJDs.
const BIAS_EPISODE: core::ops::RangeInclusive<u32> = 60_767..=60_991;

/// A `finals2000A` row's MJD, columns 8–15 and a whole day, and its Bulletin A flag for polar
/// motion, column 17: `I` measured, `P` predicted, blank for a day reserved but not yet predicted.
fn finals_rows() -> Vec<(u32, char)> {
    FINALS
        .lines()
        .map(|line| {
            let mjd = line[7..15].trim().strip_suffix(".00").unwrap();
            (mjd.parse().unwrap(), line[16..17].chars().next().unwrap())
        })
        .collect()
}

/// A C04 data row's MJD, its fifth field and a whole day.
fn c04_rows() -> Vec<u32> {
    C04.lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let mjd = line.split_whitespace().nth(4).unwrap();
            mjd.strip_suffix(".00").unwrap().parse().unwrap()
        })
        .collect()
}

#[test]
fn the_finals_excerpt_is_the_cut_of_2026_07_10() {
    let rows = finals_rows();
    let mjds: Vec<u32> = rows.iter().map(|&(mjd, _)| mjd).collect();
    // 1973-01-02 to 2027-09-05.
    assert_eq!(
        (mjds.len(), mjds[0], mjds[mjds.len() - 1]),
        (38, 41_684, 61_653)
    );
    let count = |flag: char| rows.iter().filter(|&&(_, f)| f == flag).count();
    assert_eq!((count('I'), count('P'), count(' ')), (28, 6, 4));
    // The last measured day, 2026-07-09, is the day before the fetch.
    let last_measured = rows.iter().rev().find(|&&(_, f)| f == 'I').unwrap().0;
    assert_eq!(last_measured, 61_230);
    assert!(!mjds.iter().any(|mjd| BIAS_EPISODE.contains(mjd)));
}

#[test]
fn the_c04_excerpt_is_the_cut_of_2026_10_24() {
    let header = C04.lines().take_while(|line| line.starts_with('#')).count();
    let mjds = c04_rows();
    // Six header lines kept verbatim; 1973-01-02 to 2026-07-15.
    assert_eq!(
        (header, mjds.len(), mjds[0], mjds[mjds.len() - 1]),
        (6, 45, 41_684, 61_236)
    );
    assert!(C04.contains("EOP (IERS) 20 C04 TIME SERIES  consistent with ITRF 2020"));
    assert!(!mjds.iter().any(|mjd| BIAS_EPISODE.contains(mjd)));
}
