//! A small HTTP/1.1 client over Tokio sockets, for the outbound side of the nervous system
//! (webhooks, upstream event feeds): plain `http://` with keep-alive, `Content-Length` and
//! chunked bodies, nothing else. It keeps the crate free of an HTTP client dependency, and
//! the hop to a webhook as short as a socket write.

use std::collections::HashMap;
use std::io;
use std::sync::Mutex;
use std::time::Duration;

use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Largest response head (status line and headers) accepted
const MAX_HEAD: usize = 64 * 1024;
/// Largest response body collected by [`Client::request`]
const MAX_BODY: usize = 16 << 20;

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
    pub fn parse(url: &str) -> Result<Self, String> {
        let rest = url
            .strip_prefix("http://")
            .ok_or_else(|| format!("{url}: only http:// URLs are supported by the nerve client"))?;
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        let (host, port) = match authority.rsplit_once(':') {
            Some((host, port)) if !host.contains(']') || host.ends_with(']') => (
                host,
                port.parse::<u16>()
                    .map_err(|_| format!("{url}: bad port '{port}'"))?,
            ),
            _ => (authority, 80),
        };
        if host.is_empty() {
            return Err(format!("{url}: no host"));
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

/// A response whose body is still being read, chunk by chunk (an event stream)
pub struct Streaming {
    /// The HTTP status code
    pub status: u16,
    /// Name and value, in the order received
    pub headers: Vec<(String, String)>,
    stream: TcpStream,
    framing: Framing,
    /// Bytes read past the head, not yet handed out
    buffered: Vec<u8>,
    /// Chunked: bytes left in the current chunk, or `None` before a chunk's size line
    chunk_left: Option<usize>,
    done: bool,
}

impl Streaming {
    /// The next piece of the body; `None` at its end
    pub async fn next(&mut self) -> io::Result<Option<Bytes>> {
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
                        return Err(io::ErrorKind::UnexpectedEof.into());
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
    async fn next_chunk_piece(&mut self) -> io::Result<Option<Bytes>> {
        match self.chunk_left {
            None => {
                let line = self.read_line().await?;
                let size = line.split(';').next().unwrap_or("").trim();
                let size = usize::from_str_radix(size, 16)
                    .map_err(|_| io::Error::other(format!("bad chunk size '{size}'")))?;
                if size == 0 {
                    // Trailers up to the empty line
                    while !self.read_line().await?.is_empty() {}
                    return Ok(None);
                }
                self.chunk_left = Some(size);
                Ok(Some(Bytes::new()))
            }
            Some(0) => {
                // The CRLF after a chunk
                self.read_line().await?;
                self.chunk_left = None;
                Ok(Some(Bytes::new()))
            }
            Some(left) => {
                if self.buffered.is_empty() {
                    let mut buf = vec![0u8; left.min(16 * 1024)];
                    let n = self.stream.read(&mut buf).await?;
                    if n == 0 {
                        return Err(io::ErrorKind::UnexpectedEof.into());
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

    async fn read_line(&mut self) -> io::Result<String> {
        loop {
            if let Some(i) = self.buffered.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = self.buffered.drain(..=i).collect();
                let line = String::from_utf8_lossy(&line).trim_end().to_string();
                return Ok(line);
            }
            if self.buffered.len() > MAX_HEAD {
                return Err(io::Error::other("line too long"));
            }
            let mut buf = [0u8; 4096];
            let n = self.stream.read(&mut buf).await?;
            if n == 0 {
                return Err(io::ErrorKind::UnexpectedEof.into());
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
    pub async fn warm(&self, url: &Url) -> io::Result<()> {
        let stream = self.connect(url).await?;
        self.put_back(url, stream);
        Ok(())
    }

    async fn connect(&self, url: &Url) -> io::Result<TcpStream> {
        let stream = tokio::time::timeout(
            self.timeout,
            TcpStream::connect((url.host.as_str(), url.port)),
        )
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "connect timed out"))??;
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
    /// sends its head and then stalls fails with [`io::ErrorKind::TimedOut`]
    pub async fn request(
        &self,
        method: &str,
        url: &Url,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> io::Result<Response> {
        tokio::time::timeout(self.timeout, self.request_inner(method, url, headers, body))
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "request timed out"))?
    }

    async fn request_inner(
        &self,
        method: &str,
        url: &Url,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> io::Result<Response> {
        let mut streaming = self.send_inner(method, url, headers, body).await?;
        let mut collected = Vec::new();
        while let Some(piece) = streaming.next().await? {
            collected.extend_from_slice(&piece);
            if collected.len() > MAX_BODY {
                return Err(io::Error::other("response body too large"));
            }
        }
        if streaming.keep_alive() {
            self.put_back(url, streaming.stream);
        }
        Ok(Response {
            status: streaming.status,
            headers: streaming.headers,
            body: Bytes::from(collected),
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
    ) -> io::Result<Streaming> {
        tokio::time::timeout(self.timeout, self.send_inner(method, url, headers, body))
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "request timed out"))?
    }

    async fn send_inner(
        &self,
        method: &str,
        url: &Url,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> io::Result<Streaming> {
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
            head.push_str(&format!("Content-Length: {}\r\n", body.len()));
        }
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

async fn write_and_read_head(mut stream: TcpStream, request: &[u8]) -> io::Result<Streaming> {
    stream.write_all(request).await?;
    let mut buffered = Vec::with_capacity(4096);
    let head_end = loop {
        if let Some(i) = find_head_end(&buffered) {
            break i;
        }
        if buffered.len() > MAX_HEAD {
            return Err(io::Error::other("response head too large"));
        }
        let mut buf = [0u8; 4096];
        let n = stream.read(&mut buf).await?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        buffered.extend_from_slice(&buf[..n]);
    };
    let head = String::from_utf8_lossy(&buffered[..head_end]).to_string();
    let rest = buffered.split_off(head_end + 4);
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or_else(|| io::Error::other(format!("bad status line '{status_line}'")))?;
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((name.trim().to_string(), value.trim().to_string()))
        })
        .collect();
    let header = |name: &str| {
        headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    };
    let framing = if header("transfer-encoding")
        .is_some_and(|v| v.to_ascii_lowercase().contains("chunked"))
    {
        Framing::Chunked
    } else if let Some(length) = header("content-length").and_then(|v| v.parse::<usize>().ok()) {
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

fn find_head_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|w| w == b"\r\n\r\n")
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
        assert_eq!(err.kind(), io::ErrorKind::TimedOut, "{err}");
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
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof, "{err}");
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
        assert!(err.to_string().contains("too large"), "{err}");
    }

    /// Three SSE events, as a stream of results
    fn futures_util_free_stream()
    -> impl tokio_stream::Stream<Item = Result<Bytes, std::convert::Infallible>> {
        tokio_stream::iter((0..3).map(|i| Ok(Bytes::from(format!("data: {i}\n\n")))))
    }
}
