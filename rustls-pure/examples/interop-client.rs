//! A TLS 1.3 client on rustls-pure, for the interoperability tests (`interop/run.sh`). It
//! connects to 127.0.0.1 as `localhost`, sends `GET /`, and prints the cipher suite and key
//! exchange group it agreed on, then the response. It offers one suite or group only when told
//! to, so that each is negotiated in turn.
//!
//! ```text
//! interop-client <ca.pem> <port> [--suite <name>] [--group <name>] [--cert <pem> --key <pem>]
//! ```

use std::error::Error;
use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use rustls::{ClientConfig, ClientConnection, RootCertStore, StreamOwned};

const USAGE: &str = "usage: interop-client <ca.pem> <port> [--suite <name>] [--group <name>] \
                     [--cert <pem> --key <pem>]";

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(ca), Some(port)) = (args.next(), args.next()) else {
        return Err(USAGE.into());
    };
    let (mut suite, mut group, mut cert, mut key) = (None, None, None, None);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(USAGE)?;
        match flag.as_str() {
            "--suite" => suite = Some(value),
            "--group" => group = Some(value),
            "--cert" => cert = Some(value),
            "--key" => key = Some(value),
            _ => return Err(USAGE.into()),
        }
    }

    let mut provider = rustls_pure::provider();
    if let Some(suite) = &suite {
        provider
            .cipher_suites
            .retain(|offered| format!("{:?}", offered.suite()) == *suite);
    }
    if let Some(group) = &group {
        provider
            .kx_groups
            .retain(|offered| format!("{:?}", offered.name()) == *group);
    }
    if provider.cipher_suites.is_empty() || provider.kx_groups.is_empty() {
        return Err(format!("rustls-pure offers no suite {suite:?} or group {group:?}").into());
    }
    let mut roots = RootCertStore::empty();
    for cert in CertificateDer::pem_file_iter(&ca)? {
        roots.add(cert?)?;
    }
    let builder = ClientConfig::builder_with_provider(Arc::new(provider))
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .with_root_certificates(roots);
    let config = match (cert, key) {
        (None, None) => builder.with_no_client_auth(),
        (Some(cert), Some(key)) => builder.with_client_auth_cert(
            CertificateDer::pem_file_iter(cert)?.collect::<Result<Vec<_>, _>>()?,
            PrivateKeyDer::from_pem_file(key)?,
        )?,
        _ => return Err(USAGE.into()),
    };

    let tcp = TcpStream::connect(("127.0.0.1", port.parse::<u16>()?))?;
    tcp.set_read_timeout(Some(Duration::from_secs(10)))?;
    let conn = ClientConnection::new(Arc::new(config), ServerName::try_from("localhost")?)?;
    let mut tls = StreamOwned::new(conn, tcp);
    // Writing completes the handshake first
    tls.write_all(b"GET / HTTP/1.0\r\nHost: localhost\r\n\r\n")?;
    tls.flush()?;
    let mut response = Vec::new();
    match tls.read_to_end(&mut response) {
        Ok(_) => {}
        // Some servers close without a close_notify once the response is out
        Err(e) if e.kind() == ErrorKind::UnexpectedEof && !response.is_empty() => {}
        Err(e) => return Err(e.into()),
    }
    let suite = tls
        .conn
        .negotiated_cipher_suite()
        .map(|suite| suite.suite());
    let group = tls
        .conn
        .negotiated_key_exchange_group()
        .map(|group| group.name());
    println!(
        "negotiated {} {}",
        suite.map(|suite| format!("{suite:?}")).unwrap_or_default(),
        group.map(|group| format!("{group:?}")).unwrap_or_default(),
    );
    print!("{}", String::from_utf8_lossy(&response));
    Ok(())
}
