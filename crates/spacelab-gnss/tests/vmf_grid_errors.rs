//! The VMF grid parser's error messages and its two early malformed-input refusals.

use spacelab_gnss::{VmfGrid, VmfGridError};

#[test]
fn every_vmf_grid_error_prints_a_distinct_non_empty_message() {
    let errors = [
        VmfGridError::MissingDataType,
        VmfGridError::MalformedHeader,
        VmfGridError::MalformedRow(7),
        VmfGridError::RowCountMismatch {
            expected: 100,
            found: 99,
        },
    ];
    let messages: Vec<String> = errors.iter().map(ToString::to_string).collect();
    assert!(messages.iter().all(|m| !m.is_empty()));
    let distinct: std::collections::HashSet<&String> = messages.iter().collect();
    assert_eq!(
        distinct.len(),
        errors.len(),
        "duplicate message: {messages:?}"
    );
}

#[test]
fn a_short_range_header_is_rejected() {
    // The Range/resolution line must carry six values.
    assert!(matches!(
        VmfGrid::parse("! Data_types: VMF3\n! Range/resolution: 1 2 3\n"),
        Err(VmfGridError::MalformedHeader),
    ));
}

#[test]
fn a_short_data_row_is_rejected() {
    // A data row must carry six columns (lat, lon, ah, aw, zhd, zwd).
    let text = "! Data_types: VMF3\n! Range/resolution: 0 90 0 360 5 5\n1 2 3\n";
    assert!(matches!(
        VmfGrid::parse(text),
        Err(VmfGridError::MalformedRow(_)),
    ));
}
