//! `CITATION.cff` states the same version, licence and repository the workspace manifest does.
//!
//! A citation file is a transcription of `Cargo.toml` into a format nothing compiles, so it drifts
//! exactly the way the README's crate table did: the release bumps the manifest and the file that
//! tells people how to cite the release keeps naming the one before it.
//!
//! **The intra-workspace version pins are deliberately not checked here, because `cargo` already
//! does it.** Each of the 100-odd edges spells `spacelab-core = { path = "...", version = "0.0.1" }`
//! and leaving one behind after a bump looks like the same class of defect. It is not: measured by
//! bumping `workspace.package.version` to `0.1.0` and leaving the pins at `0.0.1`, `cargo metadata`
//! refuses outright -- "failed to select a version for the requirement `spacelab-core = "^0.0.1"`,
//! candidate versions found which didn't match: 0.1.0" -- because a path dependency whose version
//! does not satisfy the requirement sends the resolver to a registry that does not have it. A test
//! that repeats the resolver is a test that can only ever agree with it.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};

/// The repository root: two levels up from `crates/spacelab`.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the workspace root is reachable from the crate directory")
}

fn read(name: &str) -> String {
    let path = root().join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The `[workspace.package]` table, which is where every crate's version, licence and repository
/// come from.
fn workspace_package() -> String {
    let manifest = read("Cargo.toml");
    let (_, package) = manifest
        .split_once("[workspace.package]")
        .expect("the workspace manifest still has a [workspace.package] table");
    package
        .split_once("\n[")
        .map_or_else(|| package.to_owned(), |(table, _)| table.to_owned())
}

/// The value of a `key = "value"` line, taking the first match.
fn toml_string(text: &str, key: &str) -> String {
    text.lines()
        .map(str::trim)
        .find_map(|line| {
            let rest = line.strip_prefix(key)?.trim().strip_prefix('=')?;
            Some(rest.trim().trim_matches('"').to_owned())
        })
        .unwrap_or_else(|| panic!("no `{key} = ...` line where one was expected"))
}

/// The value of a top-level `key: value` line of the citation file, quotes stripped.
fn cff_string(text: &str, key: &str) -> String {
    text.lines()
        .find_map(|line| {
            let rest = line.strip_prefix(key)?.strip_prefix(':')?;
            Some(rest.trim().trim_matches('"').to_owned())
        })
        .unwrap_or_else(|| panic!("CITATION.cff has no top-level `{key}:`"))
}

/// The items of a `key:` block that is a YAML list of plain scalars.
fn cff_list(text: &str, key: &str) -> Vec<String> {
    let mut items = Vec::new();
    let mut inside = false;
    for line in text.lines() {
        if line.trim_end() == format!("{key}:") {
            inside = true;
            continue;
        }
        if inside {
            match line.strip_prefix("  - ") {
                Some(item) => items.push(item.trim().trim_matches('"').to_owned()),
                None => break,
            }
        }
    }
    assert!(!items.is_empty(), "CITATION.cff has no `{key}:` list");
    items
}

#[test]
fn the_citation_file_states_the_version_the_workspace_does() {
    assert_eq!(
        cff_string(&read("CITATION.cff"), "version"),
        toml_string(&workspace_package(), "version"),
        "CITATION.cff names a different version from Cargo.toml"
    );
}

#[test]
fn the_citation_file_states_the_licence_the_workspace_does() {
    // SPDX in Cargo, a list in CFF: the citation format has no expression syntax, and a list of
    // licences means "any of", which is what `OR` means. `AND` cannot be expressed there at all and
    // is not used here, so splitting on ` OR ` is exact rather than an approximation.
    let spdx = toml_string(&workspace_package(), "license");
    let expected: Vec<&str> = spdx.split(" OR ").collect();
    assert_eq!(
        cff_list(&read("CITATION.cff"), "license"),
        expected,
        "CITATION.cff's licence list is not the workspace's SPDX expression"
    );
}

#[test]
fn the_citation_file_points_at_the_repository_the_workspace_does() {
    assert_eq!(
        cff_string(&read("CITATION.cff"), "repository-code"),
        toml_string(&workspace_package(), "repository"),
        "CITATION.cff points somewhere other than the workspace's repository"
    );
}
