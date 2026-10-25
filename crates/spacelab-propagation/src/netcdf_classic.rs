//! A reader for NetCDF classic files — the CDF-1 and CDF-2 formats — over a byte slice.
//!
//! The format is Unidata's "NetCDF Classic and 64-bit Offset File Formats" (NASA ESDS-RFC-011): a
//! big-endian header of dimensions, attributes and variables, then each fixed-size variable's data
//! contiguously, then the records, each holding one slice of every record variable in header order.
//! CDF-2 differs from CDF-1 only in writing variable offsets in 8 bytes rather than 4. CDF-5 (64-bit
//! counts, unsigned types) and netCDF-4 (an HDF5 container) are different formats and are refused,
//! as is a file still being streamed, whose record count is not yet written.
//!
//! The input is untrusted: every count and offset is checked against the bytes actually present
//! before it is used, and nothing is allocated from a count that the file could not hold.
//!
//! The unit tests read files this module's own test writer made, so they share its reading of the
//! format; `tests/wam_ipe_product.rs` reads a file NOAA wrote and compares with SciPy's reader.

// `NetCDF` is the format's name, not an identifier.
#![allow(clippy::doc_markdown)]

use alloc::string::String;
use alloc::vec::Vec;

const NC_DIMENSION: u32 = 0x0A;
const NC_VARIABLE: u32 = 0x0B;
const NC_ATTRIBUTE: u32 = 0x0C;
/// The record count a writer leaves while it is still appending records.
const STREAMING: u32 = 0xFFFF_FFFF;

/// Why a byte slice could not be read as a NetCDF classic file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NetcdfError {
    /// The first four bytes are not `CDF\x01` or `CDF\x02`.
    NotClassic([u8; 4]),
    /// The file ends inside a structure that starts at this byte offset.
    Truncated(usize),
    /// The header is not well formed; the reason names the field.
    Malformed(&'static str),
    /// The variable holds characters, not numbers.
    NotNumeric(String),
}

impl core::fmt::Display for NetcdfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotClassic(magic) => {
                write!(
                    f,
                    "magic {magic:02x?} is not NetCDF classic (CDF-1 or CDF-2)"
                )
            }
            Self::Truncated(at) => write!(f, "file ends inside the structure at byte {at}"),
            Self::Malformed(what) => write!(f, "malformed header: {what}"),
            Self::NotNumeric(name) => write!(f, "variable {name} holds characters, not numbers"),
        }
    }
}

impl core::error::Error for NetcdfError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NcType {
    Byte,
    Char,
    Short,
    Int,
    Float,
    Double,
}

impl NcType {
    const fn from_tag(tag: u32) -> Option<Self> {
        match tag {
            1 => Some(Self::Byte),
            2 => Some(Self::Char),
            3 => Some(Self::Short),
            4 => Some(Self::Int),
            5 => Some(Self::Float),
            6 => Some(Self::Double),
            _ => None,
        }
    }

    const fn size(self) -> usize {
        match self {
            Self::Byte | Self::Char => 1,
            Self::Short => 2,
            Self::Int | Self::Float => 4,
            Self::Double => 8,
        }
    }

    /// Big-endian values as `f64`, which every classic numeric type converts to exactly.
    fn decode(self, raw: &[u8]) -> Vec<f64> {
        match self {
            Self::Byte | Self::Char => raw
                .iter()
                .map(|&b| f64::from(i8::from_be_bytes([b])))
                .collect(),
            Self::Short => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| f64::from(i16::from_be_bytes(c)))
                .collect(),
            Self::Int => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f64::from(i32::from_be_bytes(c)))
                .collect(),
            Self::Float => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f64::from(f32::from_be_bytes(c)))
                .collect(),
            Self::Double => raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&c| f64::from_be_bytes(c))
                .collect(),
        }
    }
}

const fn padding(n: usize) -> usize {
    (4 - n % 4) % 4
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], NetcdfError> {
        let start = self.at;
        let end = start
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len())
            .ok_or(NetcdfError::Truncated(start))?;
        self.at = end;
        Ok(&self.bytes[start..end])
    }

    fn u32(&mut self) -> Result<u32, NetcdfError> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// A `NON_NEG` count: the format stores it in a signed 32-bit field.
    fn count(&mut self) -> Result<usize, NetcdfError> {
        let n = self.u32()?;
        if n > i32::MAX.unsigned_abs() {
            return Err(NetcdfError::Malformed("count is negative"));
        }
        usize::try_from(n).map_err(|_| NetcdfError::Malformed("count exceeds the address space"))
    }

    fn offset(&mut self, wide: bool) -> Result<usize, NetcdfError> {
        let value = if wide {
            let b = self.take(8)?;
            u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        } else {
            u64::from(self.u32()?)
        };
        usize::try_from(value)
            .map_err(|_| NetcdfError::Malformed("offset exceeds the address space"))
    }

    /// `n` bytes and the padding that rounds them to four.
    fn padded(&mut self, n: usize) -> Result<&'a [u8], NetcdfError> {
        let raw = self.take(n)?;
        self.take(padding(n))?;
        Ok(raw)
    }

    fn name(&mut self) -> Result<String, NetcdfError> {
        let n = self.count()?;
        let raw = self.padded(n)?;
        core::str::from_utf8(raw)
            .map(String::from)
            .map_err(|_| NetcdfError::Malformed("name is not UTF-8"))
    }

    /// The element count of a list tagged `tag`, or zero for the `ABSENT` marker.
    fn list(&mut self, tag: u32) -> Result<usize, NetcdfError> {
        let found = self.u32()?;
        let n = self.count()?;
        match (found, n) {
            (0, 0) => Ok(0),
            (t, n) if t == tag => {
                // Every element occupies at least eight bytes, so a count the remaining bytes could
                // not hold is refused before anything is allocated for it.
                if n > (self.bytes.len() - self.at) / 8 {
                    Err(NetcdfError::Truncated(self.at))
                } else {
                    Ok(n)
                }
            }
            _ => Err(NetcdfError::Malformed("list tag")),
        }
    }

    fn attributes(&mut self) -> Result<Vec<Attribute>, NetcdfError> {
        let n = self.list(NC_ATTRIBUTE)?;
        let mut out = Vec::new();
        for _ in 0..n {
            let name = self.name()?;
            let nc_type =
                NcType::from_tag(self.u32()?).ok_or(NetcdfError::Malformed("attribute type"))?;
            let count = self.count()?;
            let len = count
                .checked_mul(nc_type.size())
                .ok_or(NetcdfError::Malformed("attribute length"))?;
            let raw = self.padded(len)?;
            let value = if nc_type == NcType::Char {
                AttributeValue::Text(String::from_utf8_lossy(raw).into_owned())
            } else {
                AttributeValue::Numbers(nc_type.decode(raw))
            };
            out.push(Attribute { name, value });
        }
        Ok(out)
    }

    fn variables(
        &mut self,
        dimensions: &[usize],
        wide: bool,
    ) -> Result<Vec<Variable>, NetcdfError> {
        let mut variables = Vec::new();
        for _ in 0..self.list(NC_VARIABLE)? {
            let name = self.name()?;
            let rank = self.count()?;
            let mut ids = Vec::new();
            for _ in 0..rank {
                let id = self.count()?;
                if id >= dimensions.len() {
                    return Err(NetcdfError::Malformed("dimension id"));
                }
                ids.push(id);
            }
            let attributes = self.attributes()?;
            let nc_type =
                NcType::from_tag(self.u32()?).ok_or(NetcdfError::Malformed("variable type"))?;
            // The header's own `vsize` is redundant and saturates for large variables, so the
            // size is recomputed from the shape and the field is only stepped over.
            self.u32()?;
            let begin = self.offset(wide)?;
            let is_record = ids.first().is_some_and(|&id| dimensions[id] == 0);
            if ids.iter().skip(1).any(|&id| dimensions[id] == 0) {
                return Err(NetcdfError::Malformed(
                    "record dimension is not a variable's first",
                ));
            }
            let slice_bytes = ids
                .iter()
                .skip(usize::from(is_record))
                .try_fold(nc_type.size(), |acc, &id| acc.checked_mul(dimensions[id]))
                .ok_or(NetcdfError::Malformed("variable size"))?;
            variables.push(Variable {
                name,
                dimensions: ids,
                attributes,
                nc_type,
                begin,
                slice_bytes,
                is_record,
            });
        }
        Ok(variables)
    }
}

/// The value of an attribute: text for `NC_CHAR`, numbers for every other type.
#[derive(Clone, Debug, PartialEq)]
pub enum AttributeValue {
    /// An `NC_CHAR` attribute.
    Text(String),
    /// A numeric attribute, converted to `f64`.
    Numbers(Vec<f64>),
}

#[derive(Clone, Debug, PartialEq)]
struct Attribute {
    name: String,
    value: AttributeValue,
}

/// A variable's header entry.
#[derive(Clone, Debug, PartialEq)]
pub struct Variable {
    name: String,
    dimensions: Vec<usize>,
    attributes: Vec<Attribute>,
    nc_type: NcType,
    begin: usize,
    /// Bytes of one record's slice, or of the whole variable when it has no record dimension.
    slice_bytes: usize,
    is_record: bool,
}

impl Variable {
    /// The variable's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The attribute called `name`.
    #[must_use]
    pub fn attribute(&self, name: &str) -> Option<&AttributeValue> {
        self.attributes
            .iter()
            .find(|a| a.name == name)
            .map(|a| &a.value)
    }

    /// The text of the attribute called `name`, if it is a text attribute.
    #[must_use]
    pub fn text_attribute(&self, name: &str) -> Option<&str> {
        match self.attribute(name)? {
            AttributeValue::Text(text) => Some(text),
            AttributeValue::Numbers(_) => None,
        }
    }
}

/// A parsed NetCDF classic file, borrowing the bytes its data are read from.
#[derive(Clone, Debug)]
pub struct Dataset<'a> {
    bytes: &'a [u8],
    numrecs: usize,
    /// Each dimension's length; zero marks the record dimension.
    dimensions: Vec<usize>,
    variables: Vec<Variable>,
    record_stride: usize,
}

impl<'a> Dataset<'a> {
    /// Parse the header of `bytes` and check that every variable's data lies inside them.
    ///
    /// # Errors
    /// [`NetcdfError`] names the first thing that is not a well-formed CDF-1 or CDF-2 file.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, NetcdfError> {
        let mut c = Cursor { bytes, at: 0 };
        let magic = c.take(4).map_err(|_| {
            let mut m = [0; 4];
            m[..bytes.len()].copy_from_slice(bytes);
            NetcdfError::NotClassic(m)
        })?;
        let wide = match magic {
            b"CDF\x01" => false,
            b"CDF\x02" => true,
            _ => {
                return Err(NetcdfError::NotClassic([
                    magic[0], magic[1], magic[2], magic[3],
                ]));
            }
        };
        let raw_numrecs = c.u32()?;
        if raw_numrecs == STREAMING {
            return Err(NetcdfError::Malformed("record count is still streaming"));
        }
        let numrecs = usize::try_from(raw_numrecs)
            .map_err(|_| NetcdfError::Malformed("record count exceeds the address space"))?;

        let mut dimensions = Vec::new();
        for _ in 0..c.list(NC_DIMENSION)? {
            c.name()?;
            dimensions.push(c.count()?);
        }
        if dimensions.iter().filter(|&&length| length == 0).count() > 1 {
            return Err(NetcdfError::Malformed("more than one record dimension"));
        }
        // Global attributes are parsed to be stepped over; nothing here reads them.
        c.attributes()?;

        let variables = c.variables(&dimensions, wide)?;

        // Each record holds every record variable's slice padded to four bytes — except when there
        // is exactly one record variable, which the format stores unpadded.
        let records: Vec<&Variable> = variables.iter().filter(|v| v.is_record).collect();
        let record_stride = if let [only] = records.as_slice() {
            only.slice_bytes
        } else {
            records
                .iter()
                .try_fold(0usize, |acc, v| {
                    acc.checked_add(v.slice_bytes + padding(v.slice_bytes))
                })
                .ok_or(NetcdfError::Malformed("record size"))?
        };

        let dataset = Self {
            bytes,
            numrecs,
            dimensions,
            variables,
            record_stride,
        };
        for v in &dataset.variables {
            let end = if v.is_record && numrecs > 0 {
                (numrecs - 1)
                    .checked_mul(record_stride)
                    .and_then(|x| x.checked_add(v.begin))
                    .and_then(|x| x.checked_add(v.slice_bytes))
            } else if v.is_record {
                Some(v.begin)
            } else {
                v.begin.checked_add(v.slice_bytes)
            };
            if end.is_none_or(|end| end > bytes.len()) {
                return Err(NetcdfError::Truncated(v.begin));
            }
        }
        Ok(dataset)
    }

    /// The variable called `name`.
    #[must_use]
    pub fn variable(&self, name: &str) -> Option<&Variable> {
        self.variables.iter().find(|v| v.name == name)
    }

    /// The variable's shape, outermost first, with the record dimension at the file's record count.
    #[must_use]
    pub fn shape(&self, variable: &Variable) -> Vec<usize> {
        variable
            .dimensions
            .iter()
            .map(|&id| match self.dimensions[id] {
                0 => self.numrecs,
                n => n,
            })
            .collect()
    }

    /// Every value of `variable` as `f64`, in the file's row-major order across all records.
    ///
    /// # Errors
    /// [`NetcdfError::NotNumeric`] for an `NC_CHAR` variable.
    pub fn values(&self, variable: &Variable) -> Result<Vec<f64>, NetcdfError> {
        if variable.nc_type == NcType::Char {
            return Err(NetcdfError::NotNumeric(variable.name.clone()));
        }
        let slice = |start: usize| &self.bytes[start..start + variable.slice_bytes];
        if !variable.is_record {
            return Ok(variable.nc_type.decode(slice(variable.begin)));
        }
        let mut out = Vec::new();
        for r in 0..self.numrecs {
            out.extend(
                variable
                    .nc_type
                    .decode(slice(variable.begin + r * self.record_stride)),
            );
        }
        Ok(out)
    }
}

/// A NetCDF classic writer for this crate's tests: the inverse of [`Dataset::parse`], so that a
/// reader of the product can be tested on a file whose every value is known.
#[cfg(test)]
pub(crate) mod writer {
    #![allow(clippy::unwrap_used)]

    use super::{NC_ATTRIBUTE, NC_DIMENSION, NC_VARIABLE, padding};
    use alloc::vec::Vec;

    /// A variable's values; a record variable's hold every record, outermost first.
    pub(crate) enum Values<'a> {
        Byte(&'a [i8]),
        Short(&'a [i16]),
        Int(&'a [i32]),
        Float(&'a [f32]),
        Double(&'a [f64]),
    }

    impl Values<'_> {
        fn tag(&self) -> u32 {
            match self {
                Self::Byte(_) => 1,
                Self::Short(_) => 3,
                Self::Int(_) => 4,
                Self::Float(_) => 5,
                Self::Double(_) => 6,
            }
        }

        fn bytes(&self) -> Vec<u8> {
            match self {
                Self::Byte(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
                Self::Short(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
                Self::Int(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
                Self::Float(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
                Self::Double(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
            }
        }
    }

    pub(crate) struct Var<'a> {
        pub(crate) name: &'a str,
        pub(crate) dims: &'a [u32],
        pub(crate) units: Option<&'a str>,
        pub(crate) values: Values<'a>,
    }

    fn put(out: &mut Vec<u8>, v: u32) {
        out.extend_from_slice(&v.to_be_bytes());
    }

    fn name(out: &mut Vec<u8>, s: &str) {
        put(out, u32::try_from(s.len()).unwrap());
        out.extend_from_slice(s.as_bytes());
        out.resize(out.len() + padding(s.len()), 0);
    }

    /// The file holding `vars` over `dims`, where a dimension of length zero is the record one.
    pub(crate) fn write(
        wide: bool,
        numrecs: u32,
        dims: &[(&str, u32)],
        vars: &[Var<'_>],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(if wide { b"CDF\x02" } else { b"CDF\x01" });
        put(&mut out, numrecs);
        put(&mut out, NC_DIMENSION);
        put(&mut out, u32::try_from(dims.len()).unwrap());
        for &(n, len) in dims {
            name(&mut out, n);
            put(&mut out, len);
        }
        put(&mut out, 0);
        put(&mut out, 0);
        put(&mut out, NC_VARIABLE);
        put(&mut out, u32::try_from(vars.len()).unwrap());
        let mut slots = Vec::new();
        for v in vars {
            name(&mut out, v.name);
            put(&mut out, u32::try_from(v.dims.len()).unwrap());
            for &d in v.dims {
                put(&mut out, d);
            }
            if let Some(units) = v.units {
                put(&mut out, NC_ATTRIBUTE);
                put(&mut out, 1);
                name(&mut out, "units");
                put(&mut out, 2);
                put(&mut out, u32::try_from(units.len()).unwrap());
                out.extend_from_slice(units.as_bytes());
                out.resize(out.len() + padding(units.len()), 0);
            } else {
                put(&mut out, 0);
                put(&mut out, 0);
            }
            put(&mut out, v.values.tag());
            put(&mut out, 0);
            slots.push(out.len());
            out.resize(out.len() + if wide { 8 } else { 4 }, 0);
        }
        let is_record = |v: &Var<'_>| v.dims.first().is_some_and(|&d| dims[d as usize].1 == 0);
        let mut begins = alloc::vec![0usize; vars.len()];
        for (n, v) in vars.iter().enumerate().filter(|(_, v)| !is_record(v)) {
            begins[n] = out.len();
            let bytes = v.values.bytes();
            out.extend_from_slice(&bytes);
            out.resize(out.len() + padding(bytes.len()), 0);
        }
        let records: Vec<(usize, Vec<u8>)> = vars
            .iter()
            .enumerate()
            .filter(|(_, v)| is_record(v))
            .map(|(n, v)| (n, v.values.bytes()))
            .collect();
        let pad = records.len() > 1;
        for r in 0..numrecs as usize {
            for (n, bytes) in &records {
                let slice = bytes.len() / numrecs as usize;
                if r == 0 {
                    begins[*n] = out.len();
                }
                out.extend_from_slice(&bytes[r * slice..(r + 1) * slice]);
                if pad {
                    out.resize(out.len() + padding(slice), 0);
                }
            }
        }
        for (slot, begin) in slots.into_iter().zip(begins) {
            if wide {
                out[slot..slot + 8].copy_from_slice(&(begin as u64).to_be_bytes());
            } else {
                out[slot..slot + 4].copy_from_slice(&u32::try_from(begin).unwrap().to_be_bytes());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]

    use super::writer::{Values, Var};
    use super::*;

    /// `x(3)` float, `t(rec)` double, `v(rec, x)` short with a `units` attribute, two records.
    fn sample(wide: bool) -> Vec<u8> {
        writer::write(
            wide,
            2,
            &[("x", 3), ("rec", 0)],
            &[
                Var {
                    name: "x",
                    dims: &[0],
                    units: None,
                    values: Values::Float(&[1.5, -2.0, 4.25]),
                },
                Var {
                    name: "t",
                    dims: &[1],
                    units: None,
                    values: Values::Double(&[10.0, 20.0]),
                },
                Var {
                    name: "v",
                    dims: &[1, 0],
                    units: Some("K"),
                    values: Values::Short(&[1, -2, 3, -4, 5, -6]),
                },
            ],
        )
    }

    #[test]
    fn fixed_and_record_variables_read_back_in_both_offset_widths() {
        for wide in [false, true] {
            let bytes = sample(wide);
            let d = Dataset::parse(&bytes).unwrap();
            let x = d.variable("x").unwrap();
            assert_eq!(x.name(), "x");
            assert_eq!(d.shape(x), [3]);
            assert_eq!(d.values(x).unwrap(), [1.5, -2.0, 4.25]);
            let t = d.variable("t").unwrap();
            assert_eq!(d.values(t).unwrap(), [10.0, 20.0]);
            let v = d.variable("v").unwrap();
            assert_eq!(d.shape(v), [2, 3]);
            assert_eq!(v.text_attribute("units"), Some("K"));
            assert_eq!(d.values(v).unwrap(), [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
            assert!(d.variable("w").is_none());
        }
    }

    #[test]
    fn every_numeric_type_reads_back_signed_and_exact() {
        let bytes = writer::write(
            false,
            0,
            &[("n", 3)],
            &[
                Var {
                    name: "b",
                    dims: &[0],
                    units: None,
                    values: Values::Byte(&[-128, 0, 127]),
                },
                Var {
                    name: "i",
                    dims: &[0],
                    units: None,
                    values: Values::Int(&[i32::MIN, -1, i32::MAX]),
                },
            ],
        );
        let d = Dataset::parse(&bytes).unwrap();
        let read = |name| d.values(d.variable(name).unwrap()).unwrap();
        assert_eq!(read("b"), [-128.0, 0.0, 127.0]);
        assert_eq!(read("i"), [f64::from(i32::MIN), -1.0, f64::from(i32::MAX)]);
    }

    #[test]
    fn each_error_names_what_went_wrong() {
        let cases = [
            (
                NetcdfError::NotClassic(*b"\x89HDF"),
                "magic [89, 48, 44, 46] is not NetCDF classic (CDF-1 or CDF-2)",
            ),
            (
                NetcdfError::Truncated(16),
                "file ends inside the structure at byte 16",
            ),
            (
                NetcdfError::Malformed("list tag"),
                "malformed header: list tag",
            ),
            (
                NetcdfError::NotNumeric(String::from("title")),
                "variable title holds characters, not numbers",
            ),
        ];
        for (e, text) in cases {
            assert_eq!(alloc::format!("{e}"), text);
        }
    }

    #[test]
    fn a_file_cut_short_is_refused_at_every_length_that_cuts_data() {
        let bytes = sample(false);
        // The last two bytes pad the final record and hold no value, so the prefix without them
        // is still the whole file.
        assert!(Dataset::parse(&bytes[..bytes.len() - 2]).is_ok());
        for n in 0..bytes.len() - 2 {
            assert!(
                Dataset::parse(&bytes[..n]).is_err(),
                "a {n}-byte prefix of a {}-byte file parsed",
                bytes.len()
            );
        }
    }

    #[test]
    fn a_cut_names_the_first_variable_whose_data_it_reaches() {
        let bytes = sample(false);
        let d = Dataset::parse(&bytes).unwrap();
        let t = d.variable("t").unwrap().begin;
        // The fixed `x` ends where the records start; the cut leaves the first record's `t` and
        // nothing after it, so `t`'s second record is the first data the cut reaches.
        assert_eq!(
            Dataset::parse(&bytes[..t + 8]).unwrap_err(),
            NetcdfError::Truncated(t)
        );
    }

    #[test]
    fn a_record_variable_with_no_records_is_empty() {
        let bytes = writer::write(
            false,
            0,
            &[("x", 3), ("rec", 0)],
            &[Var {
                name: "v",
                dims: &[1, 0],
                units: None,
                values: Values::Short(&[]),
            }],
        );
        let d = Dataset::parse(&bytes).unwrap();
        let v = d.variable("v").unwrap();
        assert_eq!(d.shape(v), [0, 3]);
        assert_eq!(d.values(v).unwrap(), [] as [f64; 0]);
    }

    #[test]
    fn other_formats_are_named_rather_than_misread() {
        assert_eq!(
            Dataset::parse(b"\x89HDF\r\n\x1a\n").unwrap_err(),
            NetcdfError::NotClassic(*b"\x89HDF")
        );
        let mut cdf5 = sample(false);
        cdf5[3] = 5;
        assert_eq!(
            Dataset::parse(&cdf5).unwrap_err(),
            NetcdfError::NotClassic(*b"CDF\x05")
        );
        let mut streaming = sample(false);
        streaming[4..8].copy_from_slice(&STREAMING.to_be_bytes());
        assert!(matches!(
            Dataset::parse(&streaming).unwrap_err(),
            NetcdfError::Malformed(_)
        ));
    }

    /// A file laid out by hand from the classic-format specification, not by this module's writer,
    /// so that a reading the writer shares cannot pass: a one-letter dimension name padded by three
    /// bytes, a three-letter variable name padded by one, and three `NC_SHORT` values padded by two.
    fn by_hand() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"CDF\x01");
        for word in [0u32, NC_DIMENSION, 1, 1] {
            b.extend_from_slice(&word.to_be_bytes());
        }
        b.extend_from_slice(b"n\0\0\0");
        for word in [3u32, 0, 0, NC_VARIABLE, 1, 3] {
            b.extend_from_slice(&word.to_be_bytes());
        }
        b.extend_from_slice(b"abc\0");
        // rank 1, dimension 0, no attributes, NC_SHORT, vsize 8, data at byte 80.
        for word in [1u32, 0, 0, 0, 3, 8, 80] {
            b.extend_from_slice(&word.to_be_bytes());
        }
        assert_eq!(b.len(), 80);
        for v in [1i16, -2, 3, 0] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        b
    }

    #[test]
    fn a_file_written_by_hand_from_the_specification_reads_back() {
        let bytes = by_hand();
        let d = Dataset::parse(&bytes).unwrap();
        let v = d.variable("abc").unwrap();
        assert_eq!(d.shape(v), [3]);
        assert_eq!(d.values(v).unwrap(), [1.0, -2.0, 3.0]);
    }

    #[test]
    fn a_list_under_another_lists_tag_is_refused() {
        let mut bytes = by_hand();
        bytes[8..12].copy_from_slice(&NC_VARIABLE.to_be_bytes());
        assert_eq!(
            Dataset::parse(&bytes).unwrap_err(),
            NetcdfError::Malformed("list tag")
        );
    }

    #[test]
    fn the_count_bound_is_the_bytes_left_over_eight() {
        // `n` dimensions with empty names and length 1 — eight bytes each — and nothing after them.
        let dims = |n: u32, present: usize| {
            let mut b = b"CDF\x01".to_vec();
            for word in [0u32, NC_DIMENSION, n] {
                b.extend_from_slice(&word.to_be_bytes());
            }
            for _ in 0..present {
                b.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);
            }
            b
        };
        // Two dimensions in sixteen bytes fit the bound, and the file ends where the attribute
        // list should start; three do not, and are refused at the count.
        assert_eq!(
            Dataset::parse(&dims(2, 2)).unwrap_err(),
            NetcdfError::Truncated(32)
        );
        assert_eq!(
            Dataset::parse(&dims(3, 2)).unwrap_err(),
            NetcdfError::Truncated(16)
        );
    }

    #[test]
    fn a_count_larger_than_the_file_is_refused_before_allocating() {
        let mut bytes = sample(false);
        // The dimension count, right after the NC_DIMENSION tag.
        bytes[12..16].copy_from_slice(&0x7FFF_FFFFu32.to_be_bytes());
        assert_eq!(
            Dataset::parse(&bytes).unwrap_err(),
            NetcdfError::Truncated(16)
        );
    }
}
