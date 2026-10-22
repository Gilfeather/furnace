//! TLS for the HTTP and gRPC servers (feature `tls`): `--tls-cert` and `--tls-key`, and with
//! `--tls-client-ca` mutual TLS, where clients must present a certificate signed by that CA.
//!
//! Certificates are reloaded without a restart: the files are checked every
//! [`RELOAD_INTERVAL`], and new connections use the new certificate once it loads (connections
//! already open keep theirs). A file that fails to load is logged and the previous
//! certificate stays in use, so a half-written renewal does not take the server down.
//!
//! rustls runs the protocol, with the pure-Rust crypto provider of the rustls-pure crate
//! (RustCrypto primitives, TLS 1.3, ECDSA and Ed25519 keys).

use std::io;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};
use std::task::{Context, Poll};
use std::time::{Duration, SystemTime};

use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::server::WebPkiClientVerifier;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_rustls::TlsAcceptor;
use tokio_rustls::server::TlsStream;
use tracing::{debug, info, warn};

pub use crate::api::TlsSettings;

/// How often the certificate files are checked for changes
pub const RELOAD_INTERVAL: Duration = Duration::from_secs(10);
/// A client that has not finished its handshake by then is dropped
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// Handshaken connections waiting for the server to take them
const ACCEPT_QUEUE: usize = 256;
/// Handshakes in flight at once, per listener. Past it the listener stops accepting until one
/// ends, and further connections wait in the operating system's backlog, so clients that open
/// connections and never send a ClientHello cannot hold more than this many tasks and sockets
/// (each for up to 10 seconds, the handshake timeout). `--max-concurrent-requests` bounds requests, which
/// come after the handshake.
pub const MAX_HANDSHAKES: usize = 1024;

/// Why the certificate, key or client CA cannot be used. The underlying error is the
/// [`source`](std::error::Error::source), boxed so that furnace's API does not depend on the
/// TLS library's error types.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A certificate file (`--tls-cert` or `--tls-client-ca`) cannot be read as PEM
    Certificate {
        /// The file
        path: PathBuf,
        /// What went wrong
        source: BoxError,
    },
    /// A certificate file holds no certificate
    NoCertificate {
        /// The file
        path: PathBuf,
    },
    /// `--tls-key` holds no PEM private key
    PrivateKey {
        /// The file
        path: PathBuf,
        /// What went wrong
        source: BoxError,
    },
    /// A client CA certificate cannot be used to verify clients
    ClientCa {
        /// The `--tls-client-ca` file
        path: PathBuf,
        /// What went wrong
        source: BoxError,
    },
    /// The certificate and key do not make a server configuration: the key is not one the
    /// server takes (RSA, an unsupported curve) or does not match the certificate
    CertificateAndKey {
        /// `--tls-cert`
        cert: PathBuf,
        /// `--tls-key`
        key: PathBuf,
        /// What went wrong
        source: BoxError,
    },
    /// The TLS library refused the protocol versions; a bug rather than a configuration error
    Setup(BoxError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Certificate { path, source } | Self::ClientCa { path, source } => {
                write!(f, "{}: {source}", path.display())
            }
            Self::NoCertificate { path } => write!(f, "{}: no PEM certificate", path.display()),
            Self::PrivateKey { path, source } => {
                write!(f, "{}: no PEM private key ({source})", path.display())
            }
            Self::CertificateAndKey { cert, key, source } => {
                write!(f, "{} and {}: {source}", cert.display(), key.display())
            }
            Self::Setup(source) => write!(f, "TLS setup: {source}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Certificate { source, .. }
            | Self::PrivateKey { source, .. }
            | Self::ClientCa { source, .. }
            | Self::CertificateAndKey { source, .. }
            | Self::Setup(source) => Some(&**source),
            Self::NoCertificate { .. } => None,
        }
    }
}

/// The error inside an [`Error`]
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Which protocols a listener offers through ALPN
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Protocols {
    /// HTTP/2 and HTTP/1.1, for the HTTP API
    Http,
    /// HTTP/2 only, for gRPC
    Grpc,
}

/// The current rustls configurations, replaced when the files change
pub struct TlsConfigs {
    settings: TlsSettings,
    http: RwLock<Arc<rustls::ServerConfig>>,
    grpc: RwLock<Arc<rustls::ServerConfig>>,
    /// The files' newest modification time at the last load or reload attempt
    loaded: Mutex<Option<SystemTime>>,
}

impl TlsConfigs {
    /// Load the certificate, key and client CA, or say what is wrong with them
    pub fn load(settings: TlsSettings) -> Result<Arc<Self>, Error> {
        let (http, grpc) = build(&settings)?;
        let loaded = newest_change(&settings);
        Ok(Arc::new(Self {
            settings,
            http: RwLock::new(http),
            grpc: RwLock::new(grpc),
            loaded: Mutex::new(loaded),
        }))
    }

    /// Whether clients must present a certificate (`--tls-client-ca`)
    pub fn mutual(&self) -> bool {
        self.settings.client_ca.is_some()
    }

    fn current(&self, protocols: Protocols) -> Arc<rustls::ServerConfig> {
        let config = match protocols {
            Protocols::Http => &self.http,
            Protocols::Grpc => &self.grpc,
        };
        config.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Reload if a file changed since the last load; returns whether it did. On error the
    /// previous configuration stays.
    pub fn reload_if_changed(&self) -> Result<bool, Error> {
        // One lock for the check and the update, held through the reload, so two callers
        // cannot both see the change and load it twice
        let mut loaded = self.loaded.lock().unwrap_or_else(|e| e.into_inner());
        let changed = newest_change(&self.settings);
        if changed == *loaded {
            return Ok(false);
        }
        // Record the attempt either way, so a bad file is reported once, not every interval
        *loaded = changed;
        let (http, grpc) = build(&self.settings)?;
        *self.http.write().unwrap_or_else(|e| e.into_inner()) = http;
        *self.grpc.write().unwrap_or_else(|e| e.into_inner()) = grpc;
        Ok(true)
    }

    /// Check the files every [`RELOAD_INTERVAL`] for as long as the server runs
    pub fn watch(self: &Arc<Self>) {
        let configs = Arc::downgrade(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(RELOAD_INTERVAL);
            interval.tick().await;
            loop {
                interval.tick().await;
                let Some(configs) = configs.upgrade() else {
                    return;
                };
                match configs.reload_if_changed() {
                    Ok(true) => info!("🔐 Reloaded the TLS certificate"),
                    Ok(false) => {}
                    Err(e) => warn!("Keeping the previous TLS certificate: {e}"),
                }
            }
        });
    }
}

/// The latest modification time among the files, `None` if one cannot be read
fn newest_change(settings: &TlsSettings) -> Option<SystemTime> {
    let mut newest = None;
    for path in [&settings.cert, &settings.key]
        .into_iter()
        .chain(settings.client_ca.as_ref())
    {
        let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok()?;
        newest = newest.max(Some(modified));
    }
    newest
}

fn build(
    settings: &TlsSettings,
) -> Result<(Arc<rustls::ServerConfig>, Arc<rustls::ServerConfig>), Error> {
    let certs = read_certs(&settings.cert)?;
    let key = PrivateKeyDer::from_pem_file(&settings.key).map_err(|e| Error::PrivateKey {
        path: settings.key.clone(),
        source: e.into(),
    })?;
    let provider = Arc::new(rustls_pure::provider());
    let builder = rustls::ServerConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|e| Error::Setup(e.into()))?;
    let builder = match &settings.client_ca {
        None => builder.with_no_client_auth(),
        Some(ca) => {
            let client_ca = |e: BoxError| Error::ClientCa {
                path: ca.clone(),
                source: e,
            };
            let mut roots = rustls::RootCertStore::empty();
            for cert in read_certs(ca)? {
                roots.add(cert).map_err(|e| client_ca(e.into()))?;
            }
            let verifier = WebPkiClientVerifier::builder_with_provider(Arc::new(roots), provider)
                .build()
                .map_err(|e| client_ca(e.into()))?;
            builder.with_client_cert_verifier(verifier)
        }
    };
    let config = builder
        .with_single_cert(certs, key)
        .map_err(|e| Error::CertificateAndKey {
            cert: settings.cert.clone(),
            key: settings.key.clone(),
            source: e.into(),
        })?;
    let mut http = config.clone();
    http.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    let mut grpc = config;
    grpc.alpn_protocols = vec![b"h2".to_vec()];
    Ok((Arc::new(http), Arc::new(grpc)))
}

fn read_certs(path: &Path) -> Result<Vec<CertificateDer<'static>>, Error> {
    let certs = CertificateDer::pem_file_iter(path)
        .and_then(|certs| certs.collect::<Result<Vec<_>, _>>())
        .map_err(|e| Error::Certificate {
            path: path.to_path_buf(),
            source: e.into(),
        })?;
    if certs.is_empty() {
        return Err(Error::NoCertificate {
            path: path.to_path_buf(),
        });
    }
    Ok(certs)
}

/// A TLS connection, with the peer's address
pub struct TlsConnection {
    stream: TlsStream<TcpStream>,
    local: Option<SocketAddr>,
    peer: SocketAddr,
}

impl TlsConnection {
    /// The client's address
    pub fn peer(&self) -> SocketAddr {
        self.peer
    }

    /// The address the client connected to
    pub fn local(&self) -> Option<SocketAddr> {
        self.local
    }
}

impl AsyncRead for TlsConnection {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_read(cx, buf)
    }
}

impl AsyncWrite for TlsConnection {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write(cx, buf)
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.stream.is_write_vectored()
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(cx)
    }
}

#[cfg(feature = "grpc")]
impl tonic::transport::server::Connected for TlsConnection {
    type ConnectInfo = tonic::transport::server::TcpConnectInfo;

    fn connect_info(&self) -> Self::ConnectInfo {
        tonic::transport::server::TcpConnectInfo {
            local_addr: self.local(),
            remote_addr: Some(self.peer()),
        }
    }
}

/// Accept TCP connections on `listener` and run their TLS handshakes concurrently, at most
/// [`MAX_HANDSHAKES`] at once and each under a timeout; completed connections arrive on the
/// returned channel. Stops when the receiver is dropped.
pub fn accept(
    listener: TcpListener,
    configs: Arc<TlsConfigs>,
    protocols: Protocols,
) -> mpsc::Receiver<TlsConnection> {
    accept_bounded(listener, configs, protocols, MAX_HANDSHAKES)
}

fn accept_bounded(
    listener: TcpListener,
    configs: Arc<TlsConfigs>,
    protocols: Protocols,
    max_handshakes: usize,
) -> mpsc::Receiver<TlsConnection> {
    let (tx, rx) = mpsc::channel(ACCEPT_QUEUE);
    let local = listener.local_addr().ok();
    let handshakes = Arc::new(tokio::sync::Semaphore::new(max_handshakes));
    tokio::spawn(async move {
        loop {
            // A slot first, so a full set of handshakes stops accepting
            let handshake = tokio::select! {
                permit = handshakes.clone().acquire_owned() => match permit {
                    Ok(permit) => permit,
                    Err(_) => return,
                },
                () = tx.closed() => return,
            };
            let (tcp, peer) = tokio::select! {
                accepted = listener.accept() => match accepted {
                    Ok(accepted) => accepted,
                    Err(e) => {
                        // E.g. out of file descriptors: wait instead of spinning
                        warn!("Accepting a connection failed: {e}");
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                },
                () = tx.closed() => return,
            };
            let _ = tcp.set_nodelay(true);
            let acceptor = TlsAcceptor::from(configs.current(protocols));
            let tx = tx.clone();
            tokio::spawn(async move {
                let handshaken =
                    tokio::time::timeout(HANDSHAKE_TIMEOUT, acceptor.accept(tcp)).await;
                drop(handshake);
                match handshaken {
                    Ok(Ok(stream)) => {
                        let _ = tx
                            .send(TlsConnection {
                                stream,
                                local,
                                peer,
                            })
                            .await;
                    }
                    Ok(Err(e)) => debug!("TLS handshake with {peer} failed: {e}"),
                    Err(_) => debug!("TLS handshake with {peer} timed out"),
                }
            });
        }
    });
    rx
}

/// [`accept`] as a listener for `axum::serve`
pub struct TlsListener {
    connections: mpsc::Receiver<TlsConnection>,
    local: SocketAddr,
}

impl TlsListener {
    /// Accept TLS connections on `listener` with the HTTP protocols, as [`accept`] does
    pub fn new(listener: TcpListener, configs: Arc<TlsConfigs>) -> io::Result<Self> {
        let local = listener.local_addr()?;
        Ok(Self {
            connections: accept(listener, configs, Protocols::Http),
            local,
        })
    }
}

impl axum::serve::Listener for TlsListener {
    type Io = TlsConnection;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self.connections.recv().await {
            Some(connection) => {
                let peer = connection.peer;
                (connection, peer)
            }
            // The accept loop only ends when this receiver is gone
            None => std::future::pending().await,
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        Ok(self.local)
    }
}

/// Paths given for TLS, for messages
pub fn describe(settings: &TlsSettings) -> String {
    let mut text = format!("{}", settings.cert.display());
    if let Some(ca) = &settings.client_ca {
        text.push_str(&format!(", clients verified against {}", ca.display()));
    }
    text
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::api::tests::router;
    use crate::api::{ClientAddr, ServerConfig};
    use crate::model::{DummyModel, Model};
    use std::path::PathBuf;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tls");

    fn fixture(name: &str) -> PathBuf {
        Path::new(FIXTURES).join(name)
    }

    /// The first certificate in fixture `name`. A failure names the file only: no certificate,
    /// nor anything read from one, goes into test output
    fn fixture_cert(name: &str) -> CertificateDer<'static> {
        match read_certs(&fixture(name)) {
            Ok(mut certs) if !certs.is_empty() => certs.remove(0),
            _ => panic!("no certificate in the fixture {name}"),
        }
    }

    pub(crate) fn settings(server: &str, client_ca: bool) -> TlsSettings {
        TlsSettings {
            cert: fixture(&format!("{server}.crt")),
            key: fixture(&format!("{server}.key")),
            client_ca: client_ca.then(|| fixture("ca.crt")),
        }
    }

    /// An HTTPS server for a dummy model on a free port
    async fn serve(configs: Arc<TlsConfigs>) -> SocketAddr {
        let model = Model::new(
            Box::new(DummyModel::new("test_model".into(), vec![4], vec![2])),
            "dummy",
            0,
        );
        let app = router(model, &ServerConfig::default()).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let listener = TlsListener::new(listener, configs).unwrap();
        tokio::spawn(async move {
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<ClientAddr>(),
            )
            .await
        });
        addr
    }

    /// A client trusting the test CA, with the named client certificate if any
    fn client(certificate: Option<&str>) -> tokio_rustls::TlsConnector {
        tokio_rustls::TlsConnector::from(Arc::new(client_config(certificate, b"http/1.1")))
    }

    /// A client configuration trusting the test CA, offering `alpn`
    pub(crate) fn client_config(certificate: Option<&str>, alpn: &[u8]) -> rustls::ClientConfig {
        let mut roots = rustls::RootCertStore::empty();
        roots.add(fixture_cert("ca.crt")).unwrap();
        let builder =
            rustls::ClientConfig::builder_with_provider(Arc::new(rustls_pure::provider()))
                .with_protocol_versions(&[&rustls::version::TLS13])
                .unwrap()
                .with_root_certificates(roots);
        let mut config = match certificate {
            None => builder.with_no_client_auth(),
            Some(name) => builder
                .with_client_auth_cert(
                    read_certs(&fixture(&format!("{name}.crt"))).unwrap(),
                    PrivateKeyDer::from_pem_file(fixture(&format!("{name}.key"))).unwrap(),
                )
                .unwrap(),
        };
        config.alpn_protocols = vec![alpn.to_vec()];
        config
    }

    /// `GET /healthz` over TLS: the status line and the server's certificate, or the error
    async fn get_healthz(
        addr: SocketAddr,
        connector: &tokio_rustls::TlsConnector,
    ) -> io::Result<(String, CertificateDer<'static>)> {
        let tcp = TcpStream::connect(addr).await?;
        let name = rustls::pki_types::ServerName::try_from("localhost").unwrap();
        let mut tls = connector.connect(name, tcp).await?;
        let certificate = tls
            .get_ref()
            .1
            .peer_certificates()
            .and_then(|chain| chain.first())
            .cloned()
            .ok_or_else(|| io::Error::other("the server sent no certificate"))?;
        tls.write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await?;
        let mut response = String::new();
        tls.read_to_string(&mut response).await?;
        let status = response.lines().next().unwrap_or_default().to_string();
        Ok((status, certificate))
    }

    #[tokio::test]
    async fn https_with_each_key_type() {
        for server in ["server-p256", "server-p384", "server-ed25519"] {
            let addr = serve(TlsConfigs::load(settings(server, false)).unwrap()).await;
            let (status, certificate) = get_healthz(addr, &client(None)).await.unwrap();
            assert_eq!(status, "HTTP/1.1 200 OK", "{server}");
            let expected = fixture_cert(&format!("{server}.crt"));
            assert!(
                certificate == expected,
                "{server}: another certificate was served"
            );
        }
    }

    #[tokio::test]
    async fn handshakes_in_flight_are_bounded() {
        let configs = TlsConfigs::load(settings("server-p256", false)).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mut connections = accept_bounded(listener, configs, Protocols::Http, 1);
        // A client that connects and never says hello holds the only handshake...
        let silent = TcpStream::connect(addr).await.unwrap();
        let handshake = tokio::spawn(async move {
            let tcp = TcpStream::connect(addr).await?;
            let name = rustls::pki_types::ServerName::try_from("localhost").unwrap();
            client(None).connect(name, tcp).await
        });
        // ...so the next one waits (a slow machine only makes this wait longer)
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!handshake.is_finished());
        // and goes through once the silent client leaves
        drop(silent);
        let handshaken = tokio::time::timeout(Duration::from_secs(10), handshake)
            .await
            .expect("the handshake goes through")
            .unwrap();
        assert!(handshaken.is_ok(), "{handshaken:?}");
        assert!(connections.recv().await.is_some());
    }

    #[tokio::test]
    async fn plain_http_gets_no_answer() {
        let addr = serve(TlsConfigs::load(settings("server-p256", false)).unwrap()).await;
        let mut tcp = TcpStream::connect(addr).await.unwrap();
        tcp.write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut response = Vec::new();
        let _ = tcp.read_to_end(&mut response).await;
        assert!(!String::from_utf8_lossy(&response).contains("200 OK"));
    }

    #[tokio::test]
    async fn mutual_tls_needs_a_certificate_from_the_ca() {
        let configs = TlsConfigs::load(settings("server-p256", true)).unwrap();
        assert!(configs.mutual());
        let addr = serve(configs).await;
        // One client certificate per key type the provider verifies
        for client_name in ["client", "client-p384", "client-ed25519"] {
            let (status, _) = get_healthz(addr, &client(Some(client_name))).await.unwrap();
            assert_eq!(status, "HTTP/1.1 200 OK", "{client_name}");
        }
        // In TLS 1.3 the server checks the client's certificate after the client has finished
        // its side of the handshake, so the refusal shows when the request goes unanswered.
        // Only the status line is reported, not the certificate the server sent
        for (client_name, presented) in [("no certificate", None), ("stranger", Some("stranger"))] {
            let status = get_healthz(addr, &client(presented))
                .await
                .map(|(status, _)| status);
            assert!(
                !matches!(&status, Ok(status) if status.contains("200")),
                "{client_name}: {status:?}"
            );
        }
    }

    #[tokio::test]
    async fn certificates_are_reloaded_and_bad_files_keep_the_old_one() {
        let dir = tempfile::tempdir().unwrap();
        let (cert, key) = (dir.path().join("tls.crt"), dir.path().join("tls.key"));
        // Each install gets a later timestamp, even on file systems with coarse ones
        let mut stamp = SystemTime::now();
        let mut touch = |path: &Path| {
            stamp += Duration::from_secs(60);
            let file = std::fs::File::options().write(true).open(path).unwrap();
            file.set_modified(stamp).unwrap();
        };
        let mut install = |server: &str| {
            std::fs::copy(fixture(&format!("{server}.crt")), &cert).unwrap();
            std::fs::copy(fixture(&format!("{server}.key")), &key).unwrap();
            touch(&cert);
            touch(&key);
        };
        install("server-p256");
        let configs = TlsConfigs::load(TlsSettings {
            cert: cert.clone(),
            key: key.clone(),
            client_ca: None,
        })
        .unwrap();
        let addr = serve(configs.clone()).await;
        let p256 = fixture_cert("server-p256.crt");
        let p384 = fixture_cert("server-p384.crt");
        let served = || async { get_healthz(addr, &client(None)).await.unwrap().1 };
        assert!(served().await == p256, "the first certificate is served");

        assert!(matches!(configs.reload_if_changed(), Ok(false)));
        install("server-p384");
        assert!(matches!(configs.reload_if_changed(), Ok(true)));
        assert!(served().await == p384, "the renewed certificate is served");

        // A half-written renewal: the error is reported once and the certificate stays
        std::fs::write(&key, "not a key").unwrap();
        let file = std::fs::File::options().write(true).open(&key).unwrap();
        file.set_modified(SystemTime::now() + Duration::from_secs(3600))
            .unwrap();
        assert!(matches!(
            configs.reload_if_changed(),
            Err(Error::PrivateKey { .. })
        ));
        assert!(matches!(configs.reload_if_changed(), Ok(false)));
        assert!(served().await == p384, "the renewed certificate stays");
    }

    #[test]
    fn key_formats() {
        let load = |key: &str| {
            build(&TlsSettings {
                cert: fixture("server-p256.crt"),
                key: fixture(key),
                client_ca: None,
            })
        };
        assert!(load("server-p256.key").is_ok());
        assert!(load("server-p256-sec1.key").is_ok());
        let error = build(&settings("server-rsa", false)).err().unwrap();
        assert!(matches!(error, Error::CertificateAndKey { .. }), "{error}");
        let error = error.to_string();
        assert!(error.contains("RSA keys are not supported"), "{error}");
        assert!(error.contains("ECDSA"), "{error}");
        let error = load("server-p521.key").err().unwrap().to_string();
        // P-521's object identifier
        assert!(
            error.contains("unsupported EC key curve 1.3.132.0.35"),
            "{error}"
        );

        let missing = load("nope.key").err().unwrap().to_string();
        assert!(missing.contains("nope.key"), "{missing}");
    }
}
