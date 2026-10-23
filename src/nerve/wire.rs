//! A small HTTP/1.1 client over Tokio sockets, for the outbound side of the nervous system
//! (webhooks, upstream event feeds): plain `http://` with keep-alive, `Content-Length` and
//! chunked bodies, nothing else. It keeps the crate free of an HTTP client dependency, and
//! the hop to a webhook as short as a socket write.
//!
//! Crate-internal: the effectors and receptors that call out use it. The nerve bench reaches
//! it through `furnace::nerve::bench_wire`, which is not part of the API.

use std::collections::HashMap;
use std::io;
use std::sync::Mutex;
use std::time::Duration;

use bytes::Bytes;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Largest response head (status line and headers) accepted
const MAX_HEAD: usize = 64 * 1024;
/// Largest response body collected by [`Client::request`]
const MAX_BODY: usize = 16 << 20;
/// Most trailer bytes read after a chunked body
const MAX_TRAILERS: usize = MAX_HEAD;

/// Why a call through the client failed
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The URL is not `http://host[:port]/path` (an `https://` URL among them)
    Url {
        /// The URL as given
        url: String,
        /// What is wrong with it
        reason: String,
    },
    /// The call did not finish within the client's timeout
    Timeout(&'static str),
    /// Connecting, writing or reading failed: refused, reset, closed early
    Io(io::Error),
    /// The server's answer is not HTTP/1.1 the client reads
    Protocol(String),
    /// The response body is larger than the client collects
    TooLarge,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Url { url, reason } => write!(f, "{url}: {reason}"),
            Self::Timeout(what) => write!(f, "{what} timed out"),
            Self::Io(e) => std::fmt::Display::fmt(e, f),
            Self::Protocol(m) => f.write_str(m),
            Self::TooLarge => f.write_str("response body too large"),
        }
    }
}

impl std::error::Error for Error {
    // `Io` is the I/O error itself, its source included
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => e.source(),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// What the client's calls return
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A parsed `http://host[:port]/path?query` URL
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Url {
    /// Host name or address, IPv6 without its brackets
    pub host: String,
    /// 80 unless the URL gives one
    pub port: u16,
    /// Path and query, starting with `/`
    pub path: String,
}

impl Url {
    /// The error names the URL and what is wrong with it (an `https://` URL among them)
    pub fn parse(url: &str) -> Result<Self> {
        let invalid = |reason: String| Error::Url {
            url: url.to_string(),
            reason,
        };
        let rest = url.strip_prefix("http://").ok_or_else(|| {
            invalid("only http:// URLs are supported by the nerve client".to_string())
        })?;
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        let (host, port) = match authority.rsplit_once(':') {
            Some((host, port)) if !host.contains(']') || host.ends_with(']') => (
                host,
                port.parse::<u16>()
                    .map_err(|_| invalid(format!("bad port '{port}'")))?,
            ),
            _ => (authority, 80),
        };
        if host.is_empty() {
            return Err(invalid("no host".to_string()));
        }
        Ok(Self {
            host: host.trim_matches(['[', ']']).to_string(),
            port,
            path: path.to_string(),
        })
    }

    fn key(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    fn host_header(&self) -> String {
        if self.port == 80 {
            self.host.clone()
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }
}

/// A response with its body collected
#[derive(Debug)]
pub struct Response {
    /// The HTTP status code
    pub status: u16,
    /// Name and value, in the order received
    pub headers: Vec<(String, String)>,
    /// The whole body, de-chunked
    pub body: Bytes,
}

impl Response {
    /// The first header named `name`, ignoring case
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// How the body ends
enum Framing {
    Length(usize),
    Chunked,
    /// Until the connection closes
    Close,
}

/// A response whose body is still being read, chunk by chunk (an event stream), from a
/// connection (or, in the tests, any byte source)
pub struct Streaming<S = TcpStream> {
    /// The HTTP status code
    pub status: u16,
    /// Name and value, in the order received
    pub headers: Vec<(String, String)>,
    stream: S,
    framing: Framing,
    /// Bytes read past the head, not yet handed out
    buffered: Vec<u8>,
    /// Chunked: bytes left in the current chunk, or `None` before a chunk's size line
    chunk_left: Option<usize>,
    done: bool,
}

impl<S: AsyncRead + Unpin> Streaming<S> {
    /// The next piece of the body; `None` at its end
    pub async fn next(&mut self) -> Result<Option<Bytes>> {
        loop {
            if self.done {
                return Ok(None);
            }
            match self.framing {
                Framing::Length(left) => {
                    if left == 0 {
                        self.done = true;
                        return Ok(None);
                    }
                    if !self.buffered.is_empty() {
                        let n = self.buffered.len().min(left);
                        let out: Vec<u8> = self.buffered.drain(..n).collect();
                        self.framing = Framing::Length(left - n);
                        return Ok(Some(Bytes::from(out)));
                    }
                    let mut buf = vec![0u8; left.min(16 * 1024)];
                    let n = self.stream.read(&mut buf).await?;
                    if n == 0 {
                        return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
                    }
                    buf.truncate(n);
                    self.framing = Framing::Length(left - n);
                    return Ok(Some(Bytes::from(buf)));
                }
                Framing::Close => {
                    if !self.buffered.is_empty() {
                        return Ok(Some(Bytes::from(std::mem::take(&mut self.buffered))));
                    }
                    let mut buf = vec![0u8; 16 * 1024];
                    let n = self.stream.read(&mut buf).await?;
                    if n == 0 {
                        self.done = true;
                        return Ok(None);
                    }
                    buf.truncate(n);
                    return Ok(Some(Bytes::from(buf)));
                }
                Framing::Chunked => {
                    if let Some(piece) = self.next_chunk_piece().await? {
                        if piece.is_empty() {
                            continue;
                        }
                        return Ok(Some(piece));
                    }
                    self.done = true;
                    return Ok(None);
                }
            }
        }
    }

    /// Part of the current chunk, an empty piece between chunks, or `None` after the last one
    async fn next_chunk_piece(&mut self) -> Result<Option<Bytes>> {
        match self.chunk_left {
            None => {
                let line = self.read_line().await?;
                let size = chunk_size(line.as_bytes())
                    .ok_or_else(|| Error::Protocol(format!("bad chunk size line '{line}'")))?;
                if size == 0 {
                    // Trailers up to the empty line, within a bound
                    let mut trailers = 0;
                    loop {
                        let line = self.read_line().await?;
                        if line.is_empty() {
                            return Ok(None);
                        }
                        trailers += line.len();
                        if trailers > MAX_TRAILERS {
                            return Err(Error::Protocol("trailers too large".to_string()));
                        }
                    }
                }
                self.chunk_left = Some(size);
                Ok(Some(Bytes::new()))
            }
            Some(0) => {
                // The CRLF after a chunk: anything else means the size was wrong
                if !self.read_line().await?.is_empty() {
                    return Err(Error::Protocol(
                        "chunk data longer than its size".to_string(),
                    ));
                }
                self.chunk_left = None;
                Ok(Some(Bytes::new()))
            }
            Some(left) => {
                if self.buffered.is_empty() {
                    let mut buf = vec![0u8; left.min(16 * 1024)];
                    let n = self.stream.read(&mut buf).await?;
                    if n == 0 {
                        return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
                    }
                    buf.truncate(n);
                    self.buffered = buf;
                }
                let n = self.buffered.len().min(left);
                let out: Vec<u8> = self.buffered.drain(..n).collect();
                self.chunk_left = Some(left - n);
                Ok(Some(Bytes::from(out)))
            }
        }
    }

    async fn read_line(&mut self) -> Result<String> {
        loop {
            if let Some(i) = self.buffered.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = self.buffered.drain(..=i).collect();
                let line = String::from_utf8_lossy(&line).trim_end().to_string();
                return Ok(line);
            }
            if self.buffered.len() > MAX_HEAD {
                return Err(Error::Protocol("line too long".to_string()));
            }
            let mut buf = [0u8; 4096];
            let n = self.stream.read(&mut buf).await?;
            if n == 0 {
                return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
            }
            self.buffered.extend_from_slice(&buf[..n]);
        }
    }

    /// Whether the connection can carry another request once the body is read
    fn keep_alive(&self) -> bool {
        !matches!(self.framing, Framing::Close)
            && !self.headers.iter().any(|(n, v)| {
                n.eq_ignore_ascii_case("connection") && v.eq_ignore_ascii_case("close")
            })
    }

    /// The whole body, refused past [`MAX_BODY`]
    async fn collect(&mut self) -> Result<Bytes> {
        if let Framing::Length(length) = self.framing
            && length > MAX_BODY
        {
            return Err(Error::TooLarge);
        }
        let mut collected = Vec::new();
        while let Some(piece) = self.next().await? {
            collected.extend_from_slice(&piece);
            if collected.len() > MAX_BODY {
                return Err(Error::TooLarge);
            }
        }
        Ok(Bytes::from(collected))
    }
}

/// Connections kept alive per host, and a timeout per request: the whole exchange for
/// [`request`](Self::request), up to the response head for [`send`](Self::send)
pub struct Client {
    pool: Mutex<HashMap<String, Vec<TcpStream>>>,
    timeout: Duration,
}

impl Default for Client {
    fn default() -> Self {
        Self::new(Duration::from_secs(5))
    }
}

impl Client {
    /// An empty pool, with `timeout` per request
    pub fn new(timeout: Duration) -> Self {
        Self {
            pool: Mutex::new(HashMap::new()),
            timeout,
        }
    }

    /// Open a connection to `url`'s host and keep it for the next request
    pub async fn warm(&self, url: &Url) -> Result<()> {
        let stream = self.connect(url).await?;
        self.put_back(url, stream);
        Ok(())
    }

    async fn connect(&self, url: &Url) -> Result<TcpStream> {
        let stream = tokio::time::timeout(
            self.timeout,
            TcpStream::connect((url.host.as_str(), url.port)),
        )
        .await
        .map_err(|_| Error::Timeout("connect"))??;
        stream.set_nodelay(true)?;
        Ok(stream)
    }

    fn take(&self, url: &Url) -> Option<TcpStream> {
        self.pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(&url.key())?
            .pop()
    }

    fn put_back(&self, url: &Url, stream: TcpStream) {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        let idle = pool.entry(url.key()).or_default();
        if idle.len() < 8 {
            idle.push(stream);
        }
    }

    /// Send a request and collect the whole response, all within the timeout: a server that
    /// sends its head and then stalls fails with [`Error::Timeout`]
    pub async fn request(
        &self,
        method: &str,
        url: &Url,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<Response> {
        tokio::time::timeout(self.timeout, self.request_inner(method, url, headers, body))
            .await
            .map_err(|_| Error::Timeout("request"))?
    }

    async fn request_inner(
        &self,
        method: &str,
        url: &Url,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<Response> {
        let mut streaming = self.send_inner(method, url, headers, body).await?;
        let body = streaming.collect().await?;
        if streaming.keep_alive() {
            self.put_back(url, streaming.stream);
        }
        Ok(Response {
            status: streaming.status,
            headers: streaming.headers,
            body,
        })
    }

    /// Send a request and return the response with its body still streaming. The timeout
    /// covers the request and the response head; reading the body is the caller's to bound.
    pub async fn send(
        &self,
        method: &str,
        url: &Url,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<Streaming> {
        tokio::time::timeout(self.timeout, self.send_inner(method, url, headers, body))
            .await
            .map_err(|_| Error::Timeout("request"))?
    }

    async fn send_inner(
        &self,
        method: &str,
        url: &Url,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<Streaming> {
        use std::fmt::Write as _;
        let mut head = format!(
            "{method} {} HTTP/1.1\r\nHost: {}\r\nUser-Agent: furnace-nerve\r\n",
            url.path,
            url.host_header()
        );
        for (name, value) in headers {
            head.push_str(name);
            head.push_str(": ");
            head.push_str(value);
            head.push_str("\r\n");
        }
        if !body.is_empty() || method == "POST" || method == "PUT" {
            let _ = write!(head, "Content-Length: {}\r\n", body.len());
        }
        // Room for the body too, so it is appended without growing the buffer again
        head.reserve_exact(2 + body.len());
        head.push_str("\r\n");
        let mut request = head.into_bytes();
        request.extend_from_slice(body);

        // A kept-alive connection may have been closed by the server: try it once, then a
        // fresh one
        if let Some(stream) = self.take(url)
            && let Ok(streaming) = write_and_read_head(stream, &request).await
        {
            return Ok(streaming);
        }
        let stream = self.connect(url).await?;
        write_and_read_head(stream, &request).await
    }
}

async fn write_and_read_head(mut stream: TcpStream, request: &[u8]) -> Result<Streaming> {
    stream.write_all(request).await?;
    read_head(stream).await
}

/// Read the response head from `stream`; the body is left to [`Streaming::next`]
async fn read_head<S: AsyncRead + Unpin>(mut stream: S) -> Result<Streaming<S>> {
    let mut buffered = Vec::with_capacity(4096);
    let head_end = loop {
        if let Some(i) = find_head_end(&buffered) {
            break i;
        }
        if buffered.len() > MAX_HEAD {
            return Err(Error::Protocol("response head too large".to_string()));
        }
        let mut buf = [0u8; 4096];
        let n = stream.read(&mut buf).await?;
        if n == 0 {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
        }
        buffered.extend_from_slice(&buf[..n]);
    };
    if head_end > MAX_HEAD {
        return Err(Error::Protocol("response head too large".to_string()));
    }
    let rest = buffered.split_off(head_end + 4);
    parse_head(
        &String::from_utf8_lossy(&buffered[..head_end]),
        stream,
        rest,
    )
}

/// The response whose head (without its blank line) is `head`, its body to come from `rest`
/// and then `stream`
fn parse_head<S>(head: &str, stream: S, rest: Vec<u8>) -> Result<Streaming<S>> {
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let status = status_code(status_line.as_bytes())
        .ok_or_else(|| Error::Protocol(format!("bad status line '{status_line}'")))?;
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((name.trim().to_string(), value.trim().to_string()))
        })
        .collect();
    let all = |name: &'static str| {
        headers
            .iter()
            .filter(move |(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    };
    let framing = if all("transfer-encoding").any(|v| v.to_ascii_lowercase().contains("chunked")) {
        Framing::Chunked
    } else if let Some(first) = all("content-length").next() {
        // Digits only, and the same in every copy of the header: a body whose length is in
        // doubt is not read at all
        let length = content_length(first.as_bytes())
            .filter(|_| all("content-length").all(|v| v == first))
            .ok_or_else(|| Error::Protocol(format!("bad content-length '{first}'")))?;
        Framing::Length(length)
    } else if status == 204 || status == 304 || (100..200).contains(&status) {
        Framing::Length(0)
    } else {
        Framing::Close
    };
    Ok(Streaming {
        status,
        headers,
        stream,
        framing,
        buffered: rest,
        chunk_left: None,
        done: false,
    })
}

/// The code of an `HTTP/1.x` status line: the version, then three digits, 100 to 599
fn status_code(line: &[u8]) -> Option<u16> {
    let mut parts = line
        .split(u8::is_ascii_whitespace)
        .filter(|part| !part.is_empty());
    if !parts.next()?.starts_with(b"HTTP/1.") {
        return None;
    }
    let &[a, b, c] = parts.next()? else {
        return None;
    };
    let code = [a, b, c].iter().try_fold(0u16, |n, &digit| {
        digit
            .is_ascii_digit()
            .then(|| n * 10 + u16::from(digit - b'0'))
    })?;
    (100..600).contains(&code).then_some(code)
}

/// A `Content-Length` value: decimal digits only, no sign or space, within a `usize`
fn content_length(value: &[u8]) -> Option<usize> {
    if value.is_empty() {
        return None;
    }
    value.iter().try_fold(0usize, |n, &digit| {
        digit.is_ascii_digit().then_some(())?;
        n.checked_mul(10)?.checked_add(usize::from(digit - b'0'))
    })
}

/// The size on a chunk's size line, before any `;` extension, spaces around it allowed: hex
/// digits only, no sign, within a `usize`
fn chunk_size(line: &[u8]) -> Option<usize> {
    let size = match line.iter().position(|&b| b == b';') {
        Some(end) => &line[..end],
        None => line,
    }
    .trim_ascii();
    if size.is_empty() {
        return None;
    }
    size.iter().try_fold(0usize, |n, &digit| {
        let digit = char::from(digit).to_digit(16)?;
        n.checked_mul(16)?.checked_add(digit as usize)
    })
}

fn find_head_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|w| w == b"\r\n\r\n")
}

/// Kani proofs (`cargo kani`): the parts of a response head, from any bytes
#[cfg(kani)]
mod verification {
    use super::*;

    /// Any bytes, at most `N` of them
    fn any_bytes<const N: usize>(bytes: &[u8; N]) -> &[u8] {
        &bytes[..kani::any_where(|&len: &usize| len <= N)]
    }

    /// The value of `digits` in `radix`, in 128 bits (no overflow for up to 20 hex digits);
    /// `None` for a byte that is not a digit
    fn value(digits: &[u8], radix: u32) -> Option<u128> {
        let mut n = 0u128;
        for &d in digits {
            n = n * u128::from(radix) + u128::from(char::from(d).to_digit(radix)?);
        }
        Some(n)
    }

    /// The value of a non-empty `digits`, when it fits in a `usize`
    fn fitting(digits: &[u8], radix: u32) -> Option<usize> {
        value(digits, radix)
            .filter(|_| !digits.is_empty())
            .and_then(|n| usize::try_from(n).ok())
    }

    /// The head ends at the first blank line, wherever it is
    #[kani::proof]
    #[kani::unwind(12)]
    fn the_head_ends_at_the_first_blank_line() {
        let bytes: [u8; 10] = kani::any();
        let buffer = any_bytes(&bytes);
        let blank = |i: usize| buffer[i..].starts_with(b"\r\n\r\n");
        match find_head_end(buffer) {
            Some(i) => {
                assert!(blank(i));
                assert!((0..i).all(|j| !blank(j)));
            }
            None => {
                assert!((0..buffer.len()).all(|j| !blank(j)));
            }
        }
    }

    /// Any status line of up to 14 bytes: a code exactly when, past any whitespace, a word
    /// starting `HTTP/1.` comes before a word of three digits from 100 to 599. The expected
    /// answer is found by index, apart from the iterators `status_code` uses.
    #[kani::proof]
    #[kani::unwind(16)]
    fn status_codes_are_three_digits_in_range() {
        let bytes: [u8; 14] = kani::any();
        let line = any_bytes(&bytes);
        let space = |i: usize| line[i].is_ascii_whitespace();
        let mut i = 0;
        while i < line.len() && space(i) {
            i += 1;
        }
        let version = i;
        while i < line.len() && !space(i) {
            i += 1;
        }
        let version_ok = line[version..i].starts_with(b"HTTP/1.");
        while i < line.len() && space(i) {
            i += 1;
        }
        let code = i;
        while i < line.len() && !space(i) {
            i += 1;
        }
        let expected = (version_ok && i - code == 3)
            .then(|| value(&line[code..i], 10))
            .flatten()
            .filter(|c| (100..600).contains(c));
        assert_eq!(status_code(line).map(u128::from), expected);
    }

    /// Any `Content-Length` of up to 21 bytes: exactly the decimal numbers that fit in a
    /// `usize`, with no sign or space (21 digits reach past `usize::MAX`)
    #[kani::proof]
    #[kani::unwind(23)]
    fn content_lengths_are_digits_only() {
        let bytes: [u8; 21] = kani::any();
        let v = any_bytes(&bytes);
        assert_eq!(content_length(v), fitting(v, 10));
    }

    /// Any chunk size line of up to 20 bytes: exactly the hex numbers that fit in a `usize`,
    /// spaces around them and a `;` extension after them allowed, no sign (17 digits reach past
    /// `usize::MAX`)
    #[kani::proof]
    #[kani::unwind(22)]
    fn chunk_sizes_are_hex_only() {
        let bytes: [u8; 20] = kani::any();
        let line = any_bytes(&bytes);
        let mut end = 0;
        while end < line.len() && line[end] != b';' {
            end += 1;
        }
        assert_eq!(chunk_size(line), fitting(line[..end].trim_ascii(), 16));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::body::Body;
    use axum::routing::{get, post};

    #[test]
    fn urls_parse() {
        assert_eq!(
            Url::parse("http://localhost:9000/hook?x=1").unwrap(),
            Url {
                host: "localhost".into(),
                port: 9000,
                path: "/hook?x=1".into()
            }
        );
        assert_eq!(Url::parse("http://example.com").unwrap().path, "/");
        assert_eq!(Url::parse("http://example.com").unwrap().port, 80);
        assert!(Url::parse("https://example.com").is_err());
        assert!(Url::parse("http://:80/").is_err());
    }

    async fn serve(app: Router) -> Url {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Url::parse(&format!("http://127.0.0.1:{port}/")).unwrap()
    }

    #[tokio::test]
    async fn posts_and_reuses_the_connection() {
        let base =
            serve(Router::new().route("/echo", post(|body: Bytes| async move { body }))).await;
        let url = Url {
            path: "/echo".into(),
            ..base
        };
        let client = Client::default();
        for i in 0..3 {
            let body = format!("hello {i}");
            let response = client
                .request(
                    "POST",
                    &url,
                    &[("content-type", "text/plain")],
                    body.as_bytes(),
                )
                .await
                .unwrap();
            assert_eq!(response.status, 200);
            assert_eq!(response.body, Bytes::from(body));
        }
        assert_eq!(client.pool.lock().unwrap()[&url.key()].len(), 1);
    }

    #[tokio::test]
    async fn reads_chunked_streams() {
        let base = serve(Router::new().route(
            "/stream",
            get(|| async {
                let stream = futures_util_free_stream();
                axum::response::Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }),
        ))
        .await;
        let url = Url {
            path: "/stream".into(),
            ..base
        };
        let client = Client::default();
        let mut streaming = client.send("GET", &url, &[], b"").await.unwrap();
        assert_eq!(streaming.status, 200);
        let mut all = Vec::new();
        while let Some(piece) = streaming.next().await.unwrap() {
            all.extend_from_slice(&piece);
        }
        assert_eq!(
            String::from_utf8(all).unwrap(),
            "data: 0\n\ndata: 1\n\ndata: 2\n\n"
        );
    }

    /// A server that answers every connection with `response`, then holds it open, or closes
    /// it if `close`
    async fn raw(response: &'static [u8], close: bool) -> Url {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                tokio::spawn(async move {
                    let mut request = [0u8; 4096];
                    let _ = stream.read(&mut request).await;
                    let _ = stream.write_all(response).await;
                    if !close {
                        tokio::time::sleep(Duration::from_secs(60)).await;
                    }
                });
            }
        });
        Url::parse(&format!("http://127.0.0.1:{port}/")).unwrap()
    }

    #[tokio::test]
    async fn a_body_that_stalls_times_out() {
        let url = raw(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nhalf", false).await;
        let client = Client::new(Duration::from_millis(100));
        let started = std::time::Instant::now();
        let err = client.request("POST", &url, &[], b"x").await.unwrap_err();
        assert!(matches!(err, Error::Timeout(_)), "{err}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[tokio::test]
    async fn a_body_without_length_runs_until_the_connection_closes() {
        let url = raw(b"HTTP/1.1 200 OK\r\n\r\nuntil the end", true).await;
        let response = Client::default()
            .request("GET", &url, &[], b"")
            .await
            .unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(&response.body[..], b"until the end");
    }

    #[tokio::test]
    async fn a_truncated_head_is_an_error() {
        let url = raw(b"HTTP/1.1 200 OK\r\nContent-Le", true).await;
        let err = Client::default()
            .request("GET", &url, &[], b"")
            .await
            .unwrap_err();
        assert!(
            matches!(&err, Error::Io(e) if e.kind() == io::ErrorKind::UnexpectedEof),
            "{err}"
        );
    }

    #[tokio::test]
    async fn a_body_over_the_limit_is_refused() {
        let base =
            serve(Router::new().route("/big", get(|| async { vec![0u8; MAX_BODY + 1] }))).await;
        let url = Url {
            path: "/big".into(),
            ..base
        };
        let err = Client::default()
            .request("GET", &url, &[], b"")
            .await
            .unwrap_err();
        assert!(matches!(err, Error::TooLarge), "{err}");
    }

    /// Three SSE events, as a stream of results
    fn futures_util_free_stream()
    -> impl tokio_stream::Stream<Item = Result<Bytes, std::convert::Infallible>> {
        tokio_stream::iter((0..3).map(|i| Ok(Bytes::from(format!("data: {i}\n\n")))))
    }

    /// Read a whole response from `bytes` as [`Client::request`] does after sending
    fn read(bytes: &[u8]) -> Result<(u16, Bytes)> {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                let mut streaming = read_head(bytes).await?;
                let body = streaming.collect().await?;
                Ok((streaming.status, body))
            })
    }

    /// Whether the response ended before its body did
    fn ended_early(result: Result<(u16, Bytes)>) -> bool {
        matches!(result, Err(Error::Io(e)) if e.kind() == io::ErrorKind::UnexpectedEof)
    }

    fn protocol_error(bytes: &[u8]) -> String {
        match read(bytes) {
            Err(Error::Protocol(message)) => message,
            other => panic!("{:?}", other.map(|(status, body)| (status, body.len()))),
        }
    }

    #[test]
    fn a_length_in_doubt_is_refused() {
        for length in ["-1", "+5", "5 5", "0x10", "", "99999999999999999999999"] {
            let response = format!("HTTP/1.1 200 OK\r\nContent-Length: {length}\r\n\r\nhello");
            assert!(
                protocol_error(response.as_bytes()).contains("content-length"),
                "{length}"
            );
        }
        // Two copies that disagree
        let two = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\ncontent-length: 4\r\n\r\nhello";
        assert!(protocol_error(two).contains("content-length"));
        // Two that agree are one length
        let same = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\ncontent-length: 5\r\n\r\nhello";
        assert_eq!(read(same).unwrap().1, Bytes::from_static(b"hello"));
    }

    #[test]
    fn a_length_past_the_limit_is_refused_before_reading() {
        let response = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", usize::MAX);
        assert!(matches!(read(response.as_bytes()), Err(Error::TooLarge)));
        // A length the body never reaches is an early end
        let short = b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nhalf";
        assert!(ended_early(read(short)));
    }

    #[test]
    fn broken_chunked_bodies_are_refused() {
        let chunked =
            |body: &str| format!("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{body}");
        for size in ["+5", "-5", "", "g", "ffffffffffffffffffffffff", "5 5"] {
            let response = chunked(&format!("{size}\r\nhello\r\n0\r\n\r\n"));
            assert!(
                protocol_error(response.as_bytes()).contains("chunk size"),
                "{size}"
            );
        }
        // More data than the size says
        let long = chunked("3\r\nhello\r\n0\r\n\r\n");
        assert!(protocol_error(long.as_bytes()).contains("longer than its size"));
        // Less data than the size says, then the end
        let short = chunked("10\r\nhello");
        assert!(ended_early(read(short.as_bytes())));
        // A size near the limit of usize, and the data runs out
        let huge = chunked(&format!("{:x}\r\nhello", usize::MAX / 2));
        assert!(ended_early(read(huge.as_bytes())));
        // Trailers without end
        let trailers = chunked(&format!(
            "0\r\n{}",
            "x-trailer: 1\r\n".repeat(MAX_TRAILERS / 8)
        ));
        assert!(protocol_error(trailers.as_bytes()).contains("trailers"));
    }

    #[test]
    fn chunks_past_the_limit_are_refused() {
        let mut response = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        let chunk = vec![b'x'; 1 << 20];
        for _ in 0..=MAX_BODY >> 20 {
            response.extend_from_slice(format!("{:x}\r\n", chunk.len()).as_bytes());
            response.extend_from_slice(&chunk);
            response.extend_from_slice(b"\r\n");
        }
        response.extend_from_slice(b"0\r\n\r\n");
        assert!(matches!(read(&response), Err(Error::TooLarge)));
    }

    #[test]
    fn heads_without_end_or_status_are_refused() {
        let endless = format!("HTTP/1.1 200 OK\r\nx: {}", "a".repeat(MAX_HEAD + 10));
        assert!(protocol_error(endless.as_bytes()).contains("head too large"));
        // One that ends, but past the limit
        let long = format!(
            "HTTP/1.1 200 OK\r\nx: {}\r\n\r\n",
            "a".repeat(MAX_HEAD + 10)
        );
        assert!(protocol_error(long.as_bytes()).contains("head too large"));
        for status_line in [
            "",
            "HTTP/1.1",
            "HTTP/1.1 abc OK",
            "HTTP/1.1 2000 OK",
            "SSH-2.0 200",
            "HTTP/1.1 099",
        ] {
            let response = format!("{status_line}\r\nContent-Length: 0\r\n\r\n");
            assert!(
                protocol_error(response.as_bytes()).contains("status line"),
                "{status_line}"
            );
        }
    }

    mod properties {
        use super::*;
        use proptest::prelude::*;

        /// How a generated response frames its body
        #[derive(Clone, Debug)]
        enum Frame {
            Length,
            /// Chunk sizes (the last one takes the rest), upper-case hex, an extension on
            /// each size line, trailers
            Chunked(Vec<usize>, bool, bool, bool),
            Close,
        }

        fn frame() -> impl Strategy<Value = Frame> {
            prop_oneof![
                Just(Frame::Length),
                (
                    prop::collection::vec(1usize..300, 0..8),
                    any::<bool>(),
                    any::<bool>(),
                    any::<bool>()
                )
                    .prop_map(|(sizes, upper, extension, trailers)| {
                        Frame::Chunked(sizes, upper, extension, trailers)
                    }),
                Just(Frame::Close),
            ]
        }

        /// A well-formed response carrying `body`
        fn response(
            status: u16,
            body: &[u8],
            frame: &Frame,
            extra: &[(String, String)],
        ) -> Vec<u8> {
            let mut out = format!("HTTP/1.1 {status} Whatever\r\n").into_bytes();
            for (name, value) in extra {
                out.extend_from_slice(format!("{name}: {value}\r\n").as_bytes());
            }
            match frame {
                Frame::Length => {
                    out.extend_from_slice(
                        format!("content-LENGTH: {}\r\n\r\n", body.len()).as_bytes(),
                    );
                    out.extend_from_slice(body);
                }
                Frame::Chunked(sizes, upper, extension, trailers) => {
                    out.extend_from_slice(b"Transfer-Encoding: chunked\r\n\r\n");
                    let mut rest = body;
                    let mut sizes = sizes.iter();
                    while !rest.is_empty() {
                        let size = sizes.next().map_or(rest.len(), |&n| n.min(rest.len()));
                        let (chunk, after) = rest.split_at(size);
                        let mut line = if *upper {
                            format!("{size:X}")
                        } else {
                            format!("{size:x}")
                        };
                        if *extension {
                            line.push_str(";name=value");
                        }
                        out.extend_from_slice(format!("{line}\r\n").as_bytes());
                        out.extend_from_slice(chunk);
                        out.extend_from_slice(b"\r\n");
                        rest = after;
                    }
                    out.extend_from_slice(b"0\r\n");
                    if *trailers {
                        out.extend_from_slice(b"x-checksum: 1\r\n");
                    }
                    out.extend_from_slice(b"\r\n");
                }
                Frame::Close => {
                    out.extend_from_slice(b"\r\n");
                    out.extend_from_slice(body);
                }
            }
            out
        }

        fn header() -> impl Strategy<Value = (String, String)> {
            ("x-[a-z]{1,8}", "[ -~]{0,20}")
                .prop_map(|(name, value)| (name, value.trim().to_string()))
        }

        fn generated() -> impl Strategy<Value = (u16, Vec<u8>, Frame, Vec<(String, String)>)> {
            (
                prop::sample::select(vec![200u16, 201, 400, 404, 500, 503]),
                prop::collection::vec(any::<u8>(), 0..1500),
                frame(),
                prop::collection::vec(header(), 0..4),
            )
        }

        proptest! {
            #[test]
            fn well_formed_responses_are_read_whole(
                (status, body, frame, extra) in generated()
            ) {
                let bytes = response(status, &body, &frame, &extra);
                let (read_status, read_body) = read(&bytes).unwrap();
                prop_assert_eq!(read_status, status);
                prop_assert_eq!(&read_body[..], &body[..]);
            }

            #[test]
            fn broken_responses_never_panic(
                (status, body, frame, extra) in generated(),
                edits in prop::collection::vec(
                    (
                        any::<prop::sample::Index>(),
                        0u8..3,
                        prop::sample::select(b"\r\n:;0123456789abcdefxX -".to_vec()),
                    ),
                    1..5,
                ),
                cut in any::<Option<prop::sample::Index>>(),
            ) {
                let mut bytes = response(status, &body, &frame, &extra);
                for (at, kind, byte) in edits {
                    let i = at.index(bytes.len() + 1);
                    match kind {
                        0 if i < bytes.len() => {
                            bytes.remove(i);
                        }
                        1 if i < bytes.len() => bytes[i] = byte,
                        _ => bytes.insert(i, byte),
                    }
                }
                if let Some(cut) = cut {
                    bytes.truncate(cut.index(bytes.len() + 1));
                }
                // An answer, or an error: never a panic, never more than the limit
                if let Ok((_, read_body)) = read(&bytes) {
                    prop_assert!(read_body.len() <= MAX_BODY);
                    prop_assert!(read_body.len() <= bytes.len());
                }
            }

            #[test]
            fn any_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
                let mut response = b"HTTP/1.1 200 OK\r\n".to_vec();
                response.extend_from_slice(&bytes);
                let _ = read(&bytes);
                let _ = read(&response);
            }
        }
    }
}
