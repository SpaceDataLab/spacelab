//! NTRIP — the HTTP-shaped protocol that carries [`crate::rtcm`] frames from a caster to a rover.
//!
//! NTRIP is a thin thing: an HTTP request that never ends, whose response body is a raw RTCM byte
//! stream. Almost all of it is text handling, and text handling is where the bugs and the security
//! problems are, so this module is built the other way round from a socket library — the **protocol
//! is pure and fully tested offline**, and the socket is a dozen lines at the edge that hands its
//! bytes to the same functions the tests use.
//!
//! Three things it takes seriously:
//!
//! * **Credentials are not printable.** [`Credentials`] has a hand-written [`Debug`] that shows the
//!   username and never the password, and no error type in this module carries either. A caster
//!   rejection is `Unauthorized`, full stop: an error message that quoted the request would put the
//!   password in every log that caught it.
//! * **The mountpoint and the credentials are untrusted input.** They land in an HTTP request line
//!   and in headers, so a value containing CR or LF is header injection — a mountpoint of
//!   `X\r\nAuthorization: Basic …` would append a header of the caller's choosing. [`NtripClient`]
//!   rejects them at construction rather than escaping them at use, because there is no legitimate
//!   mountpoint that needs a newline.
//! * **NTRIP 1.0 and 2.0 answer differently.** 1.0 replies `ICY 200 OK` with no headers; 2.0 replies
//!   `HTTP/1.1 200 OK` with them. A client that only understands one silently fails against half the
//!   casters in the world, so [`Response::parse`] accepts both and distinguishes them.
//!
//! # What this is not
//!
//! There is no reconnection policy, no NMEA GGA uplink for a VRS caster, and no TLS. Each is a
//! decision that belongs to the application — how long to back off, what position to claim, which
//! roots to trust — and baking one in here would make it the library's opinion instead. The pieces
//! an application needs to build them ([`NtripClient::request`], [`Response::parse`],
//! [`SourceTable`]) are all public.

use std::fmt;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// Caster credentials.
///
/// The password is never printed, never compared in a public API, and never reaches an error value.
#[derive(Clone, PartialEq, Eq)]
pub struct Credentials {
    username: String,
    password: String,
}

impl Credentials {
    /// New credentials.
    ///
    /// # Errors
    ///
    /// [`NtripError::InvalidField`] if either field contains a control character, which cannot
    /// survive base64 into a header but can survive a naive concatenation into one.
    pub fn new(username: &str, password: &str) -> Result<Self, NtripError> {
        check_field("username", username)?;
        check_field("password", password)?;
        Ok(Self {
            username: username.to_owned(),
            password: password.to_owned(),
        })
    }

    /// The username. There is deliberately no accessor for the password: nothing outside this module
    /// needs it, and every use of one is a chance to log it.
    #[must_use]
    pub fn username(&self) -> &str {
        &self.username
    }

    fn basic(&self) -> String {
        base64(format!("{}:{}", self.username, self.password).as_bytes())
    }
}

impl fmt::Debug for Credentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The username is useful in a log and the password never is.
        f.debug_struct("Credentials")
            .field("username", &self.username)
            .field("password", &"<redacted>")
            .finish()
    }
}

/// Standard base64, no line breaks — the encoding HTTP basic authentication specifies.
fn base64(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[((n >> (18 - 6 * i)) & 0x3F) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn check_field(what: &'static str, value: &str) -> Result<(), NtripError> {
    if value.chars().any(char::is_control) {
        return Err(NtripError::InvalidField { field: what });
    }
    Ok(())
}

/// What a caster answered, or why it could not be asked.
///
/// No variant carries a credential or a request body — an error that is logged must not be a leak.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NtripError {
    /// A field that goes into the request line or a header holds a control character.
    InvalidField {
        /// Which field.
        field: &'static str,
    },
    /// The caster's first line is neither an NTRIP nor an HTTP status line.
    MalformedStatus,
    /// The caster refused the credentials (401), or demanded some (407).
    Unauthorized,
    /// The caster does not serve that mountpoint (404), or served a source table instead of the
    /// stream, which is how NTRIP 1.0 says the same thing.
    UnknownMountpoint,
    /// The caster answered with a status this client does not handle.
    Status(u16),
}

impl fmt::Display for NtripError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidField { field } => write!(f, "{field} contains a control character"),
            Self::MalformedStatus => f.write_str("the caster's status line is not NTRIP or HTTP"),
            Self::Unauthorized => f.write_str("the caster refused the credentials"),
            Self::UnknownMountpoint => f.write_str("the caster does not serve that mountpoint"),
            Self::Status(code) => write!(f, "the caster answered {code}"),
        }
    }
}

impl std::error::Error for NtripError {}

/// The most a caster's response head may occupy before it stops being a header block.
///
/// The head is a status line and a handful of headers; a caster that has sent more than this
/// without terminating either has stopped speaking the protocol. The cap is what keeps
/// [`NtripClient::connect`]'s accumulating buffer bounded: without it a hostile or broken caster
/// could stream headers for ever and the client would keep asking for more, because "the head is not
/// complete yet" and "the head will never be complete" look identical one read at a time.
pub const MAX_RESPONSE_HEAD: usize = 8192;

/// Which protocol revision a caster answered with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Revision {
    /// `ICY 200 OK`, no headers, body follows immediately.
    Ntrip1,
    /// `HTTP/1.1 200 OK` with headers.
    Ntrip2,
}

/// A caster's answer to a stream request, with the offset at which the RTCM body begins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    /// Which revision the caster spoke.
    pub revision: Revision,
    /// Byte offset of the first body byte within the buffer that was parsed.
    pub body_offset: usize,
    /// True when the caster sent a source table rather than a stream, which is what both revisions
    /// do when the mountpoint is `/` — and what 1.0 does when it is simply unknown.
    pub is_source_table: bool,
}

impl Response {
    /// Parse a caster's response head.
    ///
    /// Returns `Ok(None)` when the head is not yet complete, so a socket loop can keep reading; this
    /// is the same distinction [`crate::rtcm::RtcmDecoder`] draws between a partial frame and a bad
    /// one, and for the same reason.
    ///
    /// # Errors
    ///
    /// [`NtripError`] when the status line is malformed or the status is a refusal.
    pub fn parse(head: &[u8]) -> Result<Option<Self>, NtripError> {
        // Every "not complete yet" answer in this function goes through `incomplete`, so the cap
        // applies to the header block as well as to the status line. A caster that oversteps it is
        // reported rather than waited on.
        let incomplete = || {
            if head.len() > MAX_RESPONSE_HEAD {
                Err(NtripError::MalformedStatus)
            } else {
                Ok(None)
            }
        };
        let Some(line_end) = find(head, b"\r\n") else {
            return incomplete();
        };
        let status =
            core::str::from_utf8(&head[..line_end]).map_err(|_| NtripError::MalformedStatus)?;

        let (revision, code) = if let Some(rest) = status.strip_prefix("ICY ") {
            (Revision::Ntrip1, parse_code(rest)?)
        } else if let Some(rest) = status.strip_prefix("SOURCETABLE ") {
            // Both revisions announce a source table this way.
            let code = parse_code(rest)?;
            if code != 200 {
                return Err(status_error(code));
            }
            return match head_end(head) {
                Some(body_offset) => Ok(Some(Self {
                    revision: Revision::Ntrip2,
                    body_offset,
                    is_source_table: true,
                })),
                None => incomplete(),
            };
        } else if let Some(rest) = status
            .strip_prefix("HTTP/1.1 ")
            .or_else(|| status.strip_prefix("HTTP/1.0 "))
        {
            (Revision::Ntrip2, parse_code(rest)?)
        } else {
            return Err(NtripError::MalformedStatus);
        };

        if code != 200 {
            return Err(status_error(code));
        }

        // NTRIP 1.0 sends no headers at all: the body starts right after the status line.
        let body_offset = match revision {
            Revision::Ntrip1 => line_end + 2,
            Revision::Ntrip2 => match head_end(head) {
                Some(offset) => offset,
                None => return incomplete(),
            },
        };
        Ok(Some(Self {
            revision,
            body_offset,
            is_source_table: false,
        }))
    }
}

fn parse_code(rest: &str) -> Result<u16, NtripError> {
    rest.split_whitespace()
        .next()
        .and_then(|token| token.parse().ok())
        .ok_or(NtripError::MalformedStatus)
}

fn status_error(code: u16) -> NtripError {
    match code {
        401 | 407 => NtripError::Unauthorized,
        404 => NtripError::UnknownMountpoint,
        other => NtripError::Status(other),
    }
}

fn head_end(head: &[u8]) -> Option<usize> {
    find(head, b"\r\n\r\n").map(|at| at + 4)
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// One `STR` record of a caster's source table: a mountpoint and what it carries.
///
/// Only the fields a client acts on are lifted out. The record is kept whole in
/// [`SourceTableEntry::raw`] so an application can read the rest without this module having to model
/// every optional column the standard defines.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceTableEntry {
    /// The mountpoint to request.
    pub mountpoint: String,
    /// Free-text identifier, usually the station or city.
    pub identifier: String,
    /// Data format, e.g. `RTCM 3.3`.
    pub format: String,
    /// Approximate latitude, degrees north.
    pub latitude: f64,
    /// Approximate longitude, degrees east.
    pub longitude: f64,
    /// Whether the caster expects a periodic NMEA GGA uplink — the marker of a VRS mountpoint, whose
    /// corrections are computed for the position the rover reports.
    pub requires_nmea: bool,
    /// Whether the mountpoint requires authentication.
    pub requires_authentication: bool,
    /// The record as it arrived, semicolon-separated.
    pub raw: String,
}

/// A parsed source table.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SourceTable {
    /// The `STR` records, in the order the caster listed them.
    pub streams: Vec<SourceTableEntry>,
}

impl SourceTable {
    /// Parse a source-table body.
    ///
    /// Malformed `STR` records are skipped rather than failing the parse: a caster with one bad row
    /// out of four hundred is a caster with three hundred and ninety-nine usable mountpoints, and
    /// refusing all of them would be the wrong trade. Rows that are not `STR` — the `CAS` and `NET`
    /// records — are not modelled and are skipped too.
    #[must_use]
    pub fn parse(body: &str) -> Self {
        let streams = body
            .lines()
            .filter_map(|line| {
                let line = line.trim_end_matches('\r');
                let fields: Vec<&str> = line.split(';').collect();
                // mountpoint..bitrate is columns 1..=18; a shorter row cannot name a mountpoint's
                // format and coordinates, which is the minimum a client needs to choose one.
                if fields.first() != Some(&"STR") || fields.len() < 13 {
                    return None;
                }
                Some(SourceTableEntry {
                    mountpoint: fields[1].to_owned(),
                    identifier: fields[2].to_owned(),
                    format: fields[3].to_owned(),
                    latitude: fields[9].trim().parse().ok()?,
                    longitude: fields[10].trim().parse().ok()?,
                    requires_nmea: fields[11].trim() == "1",
                    requires_authentication: !matches!(fields.get(15).map(|f| f.trim()), Some("N")),
                    raw: line.to_owned(),
                })
            })
            .collect();
        Self { streams }
    }

    /// The entry for `mountpoint`, if the caster lists it.
    #[must_use]
    pub fn find(&self, mountpoint: &str) -> Option<&SourceTableEntry> {
        self.streams.iter().find(|s| s.mountpoint == mountpoint)
    }
}

/// A caster connection's parameters.
///
/// Building one validates every field that reaches the wire, so [`NtripClient::request`] is
/// infallible and cannot be handed a value that injects a header.
#[derive(Debug, Clone)]
pub struct NtripClient {
    host: String,
    port: u16,
    mountpoint: String,
    credentials: Option<Credentials>,
    user_agent: String,
}

/// The user agent this client announces when the caller does not choose one.
pub const DEFAULT_USER_AGENT: &str = "NTRIP spacelab-gnss";

impl NtripClient {
    /// A client for `mountpoint` on `host:port`.
    ///
    /// # Errors
    ///
    /// [`NtripError::InvalidField`] when the host or the mountpoint holds a control character —
    /// they land in the request line and in the `Host` header, where a newline would let the caller
    /// append headers of their choosing.
    pub fn new(
        host: &str,
        port: u16,
        mountpoint: &str,
        credentials: Option<Credentials>,
    ) -> Result<Self, NtripError> {
        check_field("host", host)?;
        check_field("mountpoint", mountpoint)?;
        Ok(Self {
            host: host.to_owned(),
            port,
            mountpoint: mountpoint.to_owned(),
            credentials,
            user_agent: DEFAULT_USER_AGENT.to_owned(),
        })
    }

    /// Announce a different user agent.
    ///
    /// # Errors
    ///
    /// [`NtripError::InvalidField`] when it holds a control character.
    pub fn with_user_agent(mut self, user_agent: &str) -> Result<Self, NtripError> {
        check_field("user agent", user_agent)?;
        user_agent.clone_into(&mut self.user_agent);
        Ok(self)
    }

    /// The request that asks for the stream.
    ///
    /// `Ntrip-Version: Ntrip/2.0` is what makes a 2.0-capable caster answer with headers; a 1.0-only
    /// caster ignores it and answers `ICY 200 OK`, which [`Response::parse`] also accepts. Sending
    /// the header is therefore strictly better than not.
    #[must_use]
    pub fn request(&self) -> String {
        let mut request = format!(
            "GET /{} HTTP/1.1\r\nHost: {}:{}\r\nNtrip-Version: Ntrip/2.0\r\nUser-Agent: {}\r\n",
            self.mountpoint, self.host, self.port, self.user_agent
        );
        if let Some(credentials) = &self.credentials {
            request.push_str("Authorization: Basic ");
            request.push_str(&credentials.basic());
            request.push_str("\r\n");
        }
        request.push_str("Connection: close\r\n\r\n");
        request
    }

    /// The request that asks for the caster's source table instead of a stream.
    #[must_use]
    pub fn source_table_request(&self) -> String {
        format!(
            "GET / HTTP/1.1\r\nHost: {}:{}\r\nNtrip-Version: Ntrip/2.0\r\nUser-Agent: {}\r\nConnection: close\r\n\r\n",
            self.host, self.port, self.user_agent
        )
    }

    /// Open the stream: connect, send [`NtripClient::request`], read until the response head is
    /// complete, and return the socket positioned at the first RTCM byte together with whatever
    /// body bytes arrived in the same read.
    ///
    /// The leftover matters. A caster commonly delivers the head and the first RTCM frame in one
    /// segment, and a client that discarded the tail of its own first read would drop that frame and
    /// resynchronise a second later — a fault that only shows on a real network.
    ///
    /// # Errors
    ///
    /// [`ConnectError::Io`] for anything the socket reports, or [`ConnectError::Ntrip`] for a caster
    /// that refuses. Neither carries a credential.
    pub fn connect(&self, timeout: Duration) -> Result<(TcpStream, Vec<u8>), ConnectError> {
        let mut stream = TcpStream::connect((self.host.as_str(), self.port))?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;
        stream.write_all(self.request().as_bytes())?;

        let mut head = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let read = stream.read(&mut buffer)?;
            if read == 0 {
                return Err(ConnectError::Ntrip(NtripError::MalformedStatus));
            }
            head.extend_from_slice(&buffer[..read]);
            if let Some(response) = Response::parse(&head)? {
                if response.is_source_table {
                    // The caster answered the mountpoint request with its catalogue, which is how
                    // NTRIP 1.0 says the mountpoint does not exist.
                    return Err(ConnectError::Ntrip(NtripError::UnknownMountpoint));
                }
                let leftover = head.split_off(response.body_offset);
                return Ok((stream, leftover));
            }
        }
    }
}

/// A connection failure: the socket's, or the caster's.
#[derive(Debug)]
pub enum ConnectError {
    /// The socket failed.
    Io(std::io::Error),
    /// The caster refused.
    Ntrip(NtripError),
}

impl fmt::Display for ConnectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "ntrip socket: {error}"),
            Self::Ntrip(error) => write!(f, "ntrip caster: {error}"),
        }
    }
}

impl std::error::Error for ConnectError {}

impl From<std::io::Error> for ConnectError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<NtripError> for ConnectError {
    fn from(error: NtripError) -> Self {
        Self::Ntrip(error)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;

    fn client() -> NtripClient {
        NtripClient::new(
            "caster.example",
            2101,
            "RTCM3EPH",
            Some(Credentials::new("rover", "hunter2").expect("plain credentials")),
        )
        .expect("a plain client")
    }

    #[test]
    fn the_base64_encoder_reproduces_the_rfc_4648_vectors() {
        // The padding cases are the ones an off-by-one in the chunk length gets wrong, and a broken
        // encoder here is a client that never authenticates against any caster.
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
        // The bytes that exercise the top of the alphabet, where a transposed `+`/`/` hides.
        assert_eq!(base64(&[0xFB, 0xFF, 0xBF]), "+/+/");
    }

    #[test]
    fn the_request_is_well_formed_and_carries_basic_authentication() {
        let request = client().request();
        assert!(
            request.starts_with("GET /RTCM3EPH HTTP/1.1\r\n"),
            "{request}"
        );
        assert!(
            request.contains("Host: caster.example:2101\r\n"),
            "{request}"
        );
        assert!(
            request.contains("Ntrip-Version: Ntrip/2.0\r\n"),
            "{request}"
        );
        // "rover:hunter2" in base64.
        assert!(
            request.contains("Authorization: Basic cm92ZXI6aHVudGVyMg==\r\n"),
            "{request}"
        );
        assert!(request.ends_with("\r\n\r\n"), "{request}");
        // Exactly one blank line: a second would end the head early and put headers in the body.
        assert_eq!(request.matches("\r\n\r\n").count(), 1, "{request}");
    }

    #[test]
    fn a_mountpoint_or_credential_that_would_inject_a_header_is_refused_at_construction() {
        // The whole point of validating at the boundary. Each of these, concatenated naively, adds a
        // header the caller chose.
        for injection in ["EPH\r\nX-Evil: 1", "EPH\nX-Evil: 1", "EPH\rX", "EPH\u{0}"] {
            assert_eq!(
                NtripClient::new("caster.example", 2101, injection, None).err(),
                Some(NtripError::InvalidField {
                    field: "mountpoint"
                }),
                "{injection:?} was accepted as a mountpoint"
            );
        }
        assert_eq!(
            NtripClient::new("host\r\nX: 1", 2101, "EPH", None).err(),
            Some(NtripError::InvalidField { field: "host" })
        );
        assert_eq!(
            Credentials::new("rover", "pass\r\nX: 1").err(),
            Some(NtripError::InvalidField { field: "password" })
        );
        assert_eq!(
            Credentials::new("ro\nver", "pass").err(),
            Some(NtripError::InvalidField { field: "username" })
        );
    }

    #[test]
    fn the_password_never_reaches_a_debug_rendering_or_an_error() {
        let credentials = Credentials::new("rover", "hunter2").expect("plain credentials");
        let rendered = format!("{credentials:?}");
        assert!(!rendered.contains("hunter2"), "{rendered}");
        assert!(rendered.contains("rover"), "{rendered}");
        assert!(rendered.contains("redacted"), "{rendered}");
        // The accessor that does exist returns the username rather than a placeholder: a caller
        // logging which account it connected under needs the real one.
        assert_eq!(credentials.username(), "rover");

        // And the whole client, which holds them.
        let rendered = format!("{:?}", client());
        assert!(!rendered.contains("hunter2"), "{rendered}");

        // A refusal names the failure and nothing else. The base64 of the credentials would be just
        // as much of a leak as the plaintext, so check for that too.
        let refusal = format!("{}", NtripError::Unauthorized);
        assert!(
            !refusal.contains("hunter2") && !refusal.contains("cm92ZXI"),
            "{refusal}"
        );
    }

    #[test]
    fn both_protocol_revisions_are_recognised_and_their_bodies_start_in_the_right_place() {
        // NTRIP 1.0: no headers at all, so the body begins immediately after the status line. A
        // client that looked for a blank line here would wait forever.
        let one = b"ICY 200 OK\r\n\xD3\x00\x04";
        let parsed = Response::parse(one)
            .expect("a status")
            .expect("a complete head");
        assert_eq!(parsed.revision, Revision::Ntrip1);
        assert_eq!(parsed.body_offset, 12);
        assert_eq!(&one[parsed.body_offset..], &[0xD3, 0x00, 0x04]);

        // NTRIP 2.0: headers, then a blank line.
        let two = b"HTTP/1.1 200 OK\r\nNtrip-Version: Ntrip/2.0\r\nContent-Type: gnss/data\r\n\r\n\xD3\x00";
        let parsed = Response::parse(two)
            .expect("a status")
            .expect("a complete head");
        assert_eq!(parsed.revision, Revision::Ntrip2);
        assert_eq!(&two[parsed.body_offset..], &[0xD3, 0x00]);
        assert!(!parsed.is_source_table);
    }

    #[test]
    fn an_incomplete_head_is_not_an_error_and_a_refusal_is() {
        // The distinction a socket loop turns on: keep reading, or give up.
        assert_eq!(Response::parse(b"HTTP/1.1 200 OK\r\nSome"), Ok(None));
        assert_eq!(Response::parse(b"ICY 20"), Ok(None));
        assert_eq!(Response::parse(b""), Ok(None));

        assert_eq!(
            Response::parse(b"HTTP/1.1 401 Unauthorized\r\n\r\n"),
            Err(NtripError::Unauthorized)
        );
        assert_eq!(
            Response::parse(b"HTTP/1.1 407 Proxy Authentication Required\r\n\r\n"),
            Err(NtripError::Unauthorized)
        );
        assert_eq!(
            Response::parse(b"HTTP/1.1 404 Not Found\r\n\r\n"),
            Err(NtripError::UnknownMountpoint)
        );
        assert_eq!(
            Response::parse(b"HTTP/1.1 503 Service Unavailable\r\n\r\n"),
            Err(NtripError::Status(503))
        );
        assert_eq!(
            Response::parse(b"GARBAGE\r\n\r\n"),
            Err(NtripError::MalformedStatus)
        );
        // A head that never ends must not grow without bound, and there are two ways for it not to
        // end. An unterminated *status line* past the cap is the obvious one; an unterminated
        // *header block* is the one a naive parser waits on for ever, because it has a valid status
        // line and is merely missing its blank line.
        let flood = vec![b'A'; MAX_RESPONSE_HEAD + 1];
        assert_eq!(Response::parse(&flood), Err(NtripError::MalformedStatus));

        let mut headers = b"HTTP/1.1 200 OK\r\n".to_vec();
        while headers.len() <= MAX_RESPONSE_HEAD {
            headers.extend_from_slice(b"X-Filler: 0123456789\r\n");
        }
        assert_eq!(Response::parse(&headers), Err(NtripError::MalformedStatus));
        // And one byte under the cap is still a partial read, not a fault.
        assert_eq!(
            Response::parse(&headers[..MAX_RESPONSE_HEAD]),
            Ok(None),
            "the cap must bind above the head, not at it"
        );
        // A source table whose head never ends goes the same way.
        let mut table = b"SOURCETABLE 200 OK\r\n".to_vec();
        table.resize(MAX_RESPONSE_HEAD + 1, b'X');
        assert_eq!(Response::parse(&table), Err(NtripError::MalformedStatus));
    }

    #[test]
    fn a_source_table_answer_is_recognised_rather_than_taken_for_a_stream() {
        // What a caster sends when the mountpoint is `/`, and what NTRIP 1.0 sends when it does not
        // know the mountpoint at all. Feeding it to an RTCM decoder would produce nothing but
        // rejected preambles.
        let head =
            b"SOURCETABLE 200 OK\r\nServer: NTRIP Caster\r\n\r\nSTR;A;;;;;;;;0;0;0;0;;;N;;;\r\n";
        let parsed = Response::parse(head)
            .expect("a status")
            .expect("a complete head");
        assert!(parsed.is_source_table);
        assert_eq!(&head[parsed.body_offset..parsed.body_offset + 3], b"STR");
    }

    #[test]
    fn the_source_table_yields_the_fields_a_client_chooses_a_mountpoint_by() {
        let body = "\
CAS;caster.example;2101;EXAMPLE;Example;0;DEU;50.09;8.66;0.0.0.0;0;http://example\r
NET;EUREF;EUREF;B;N;http://example;none;none;none\r
STR;FRAN0;Frankfurt;RTCM 3.3;1005(10),1077(1),1087(1);2;GPS+GLO;EUREF;DEU;50.09;8.66;0;0;sNTRIP;none;B;N;9600;\r
STR;VRS_NRW;Nordrhein;RTCM 3.2;1004(1);2;GPS;NW;DEU;51.50;7.50;1;1;GNSMART;none;B;N;4800;\r
STR;BROKEN;No coordinates;RTCM 3.3;;2;GPS;;DEU;north;east;0;0;;;N;;;\r
STR;OPEN0;Open mountpoint;RTCM 3.3;1004(1);2;GPS;EUREF;DEU;52.52;13.40;0;0;none;none;N;N;4800;\r
malformed line without semicolons\r
";
        let table = SourceTable::parse(body);
        // The CAS and NET rows are not streams, the coordinate-less STR row cannot be chosen by
        // position, and the junk line is junk. Three usable mountpoints remain.
        assert_eq!(table.streams.len(), 3, "{:?}", table.streams);

        let frankfurt = table.find("FRAN0").expect("FRAN0 is listed");
        assert_eq!(frankfurt.identifier, "Frankfurt");
        assert_eq!(frankfurt.format, "RTCM 3.3");
        assert!((frankfurt.latitude - 50.09).abs() < 1e-9);
        assert!((frankfurt.longitude - 8.66).abs() < 1e-9);
        assert!(!frankfurt.requires_nmea);
        assert!(frankfurt.raw.starts_with("STR;FRAN0;"));

        // The VRS mountpoint is the one that will stall without a GGA uplink, which is exactly what
        // a client needs to know before it opens the socket.
        let vrs = table.find("VRS_NRW").expect("VRS_NRW is listed");
        assert!(vrs.requires_nmea);
        assert!((vrs.latitude - 51.50).abs() < 1e-9);

        assert!(table.find("ABSENT").is_none());
        assert!(table.find("BROKEN").is_none());

        // Field 15 is the authentication mode: `N` is the only value that means none, and `B`,
        // `D` or an absent field all mean the client needs credentials. Getting the sense of this
        // backwards sends a plain request to a caster that will answer 401, or worse, sends
        // credentials to a mountpoint that never asked for them.
        assert!(frankfurt.requires_authentication, "FRAN0 publishes `B`");
        let open = table.find("OPEN0").expect("OPEN0 is listed");
        assert!(!open.requires_authentication, "OPEN0 publishes `N`");
    }

    #[test]
    fn a_row_too_short_to_name_a_mountpoint_is_skipped_and_the_shortest_usable_row_is_not() {
        // Thirteen fields is the minimum: mountpoint through solution. The guard is what stands
        // between a truncated row and an index panic in the middle of a four-hundred-row
        // catalogue, and it has to admit the boundary as well as reject below it.
        let minimal = "STR;MIN13;Minimal;RTCM 3.3;1004(1);2;GPS;NET;DEU;12.25;-3.75;0;0";
        assert_eq!(minimal.split(';').count(), 13);
        let table = SourceTable::parse(minimal);
        let entry = table.find("MIN13").expect("thirteen fields is enough");
        assert!((entry.latitude - 12.25).abs() < 1e-9);
        assert!((entry.longitude + 3.75).abs() < 1e-9);
        // With no authentication field at all, the safe reading is that one is required.
        assert!(entry.requires_authentication);

        // One field short, and there is nothing to read columns 9 and 10 from.
        let truncated = "STR;SHORT;Truncated;RTCM 3.3;1004(1);2;GPS;NET;DEU;12.25;-3.75;0";
        assert_eq!(truncated.split(';').count(), 12);
        assert_eq!(SourceTable::parse(truncated).streams.len(), 0);
        assert_eq!(SourceTable::parse("STR;ONLY").streams.len(), 0);
    }

    #[test]
    fn the_source_table_request_asks_for_the_root_and_carries_no_credentials() {
        // The catalogue is public on every caster, and sending credentials to fetch it would put
        // them on the wire for no reason.
        let request = client().source_table_request();
        assert!(request.starts_with("GET / HTTP/1.1\r\n"), "{request}");
        assert!(!request.contains("Authorization"), "{request}");
    }

    #[test]
    fn a_client_without_credentials_sends_no_authorization_header() {
        let request = NtripClient::new("caster.example", 2101, "OPEN", None)
            .expect("a plain client")
            .with_user_agent("NTRIP test/1")
            .expect("a plain user agent")
            .request();
        assert!(!request.contains("Authorization"), "{request}");
        assert!(
            request.contains("User-Agent: NTRIP test/1\r\n"),
            "{request}"
        );
    }

    #[test]
    fn every_failure_renders_as_the_failure_it_is() {
        // These strings are what an operator reads at 3 a.m. when a base station stops feeding a
        // network, and an empty or interchangeable rendering is the difference between "the
        // caster does not serve that mountpoint" and no diagnosis at all.
        assert_eq!(
            NtripError::InvalidField { field: "host" }.to_string(),
            "host contains a control character"
        );
        assert_eq!(
            NtripError::MalformedStatus.to_string(),
            "the caster's status line is not NTRIP or HTTP"
        );
        assert_eq!(
            NtripError::Unauthorized.to_string(),
            "the caster refused the credentials"
        );
        assert_eq!(
            NtripError::UnknownMountpoint.to_string(),
            "the caster does not serve that mountpoint"
        );
        assert_eq!(
            NtripError::Status(503).to_string(),
            "the caster answered 503"
        );

        // And the wrapper says which side failed, which is the first thing to know.
        assert_eq!(
            ConnectError::Ntrip(NtripError::Unauthorized).to_string(),
            "ntrip caster: the caster refused the credentials"
        );
        let io = ConnectError::Io(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "read timed out",
        ))
        .to_string();
        assert_eq!(io, "ntrip socket: read timed out");
    }

    #[test]
    fn the_client_keeps_the_body_bytes_that_arrived_with_the_head() {
        // A caster commonly delivers the head and the first RTCM frame in one segment, and a
        // client that discarded the tail of its own first read would drop that frame and
        // resynchronise a second later. This is the only test that runs `connect` at all.
        use std::io::{Read as _, Write as _};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let port = listener.local_addr().expect("a bound address").port();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("the client connects");
            let mut request = [0_u8; 1024];
            let read = socket.read(&mut request).expect("the request arrives");
            let text = String::from_utf8_lossy(&request[..read]).into_owned();
            assert!(text.starts_with("GET /FRAN0 HTTP/1.1\r\n"), "{text}");
            socket
                .write_all(b"ICY 200 OK\r\n\xD3\x00\x04")
                .expect("the head and a body fragment go out");
            // Stay open until the client is done, so the client never sees a premature close.
            let _ = socket.read(&mut request);
        });

        let client = NtripClient::new("127.0.0.1", port, "FRAN0", None).expect("a client");
        let (stream, leftover) = client
            .connect(Duration::from_secs(5))
            .expect("the caster answers");
        assert_eq!(leftover, vec![0xD3, 0x00, 0x04]);
        drop(stream);
        server.join().expect("the server thread finishes");
    }

    #[test]
    fn a_caster_that_stops_mid_head_is_a_fault_rather_than_an_empty_stream() {
        // End of stream with an incomplete head is not "the head is still coming". Without that
        // branch the read loop would spin on a closed socket for ever.
        use std::io::{Read as _, Write as _};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let port = listener.local_addr().expect("a bound address").port();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("the client connects");
            let mut request = [0_u8; 1024];
            let _ = socket.read(&mut request);
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nNtrip-Version: ")
                .expect("half a head goes out");
        });

        let client = NtripClient::new("127.0.0.1", port, "FRAN0", None).expect("a client");
        let error = client
            .connect(Duration::from_secs(5))
            .expect_err("a truncated head is not an answer");
        assert!(
            matches!(error, ConnectError::Ntrip(NtripError::MalformedStatus)),
            "{error:?}"
        );
        server.join().expect("the server thread finishes");
    }
}
