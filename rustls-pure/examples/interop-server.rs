//! A TLS 1.3 server on rustls-pure, for the interoperability tests (`interop/run.sh`). It
//! answers every connection with an HTTP response naming the cipher suite and the key exchange
//! group it agreed on, and with a CA given, requires client certificates signed by it.
//!
//! ```text
//! interop-server <cert.pem> <key.pem> <port> [<client-ca.pem>]
//! ```

use std::error::Error;
use std::io::{self, BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::server::WebPkiClientVerifier;
use rustls::{RootCertStore, ServerConfig, ServerConnection, StreamOwned};

const USAGE: &str = "usage: interop-server <cert.pem> <key.pem> <port> [<client-ca.pem>]";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [cert, key, port, client_ca @ ..] = &args[..] else {
        return Err(USAGE.into());
    };
    let provider = Arc::new(rustls_pure::provider());
    let builder = ServerConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])?;
    let builder = match client_ca {
        [] => builder.with_no_client_auth(),
        [ca] => {
            let mut roots = RootCertStore::empty();
            for cert in CertificateDer::pem_file_iter(ca)? {
                roots.add(cert?)?;
            }
            let verifier =
                WebPkiClientVerifier::builder_with_provider(Arc::new(roots), provider).build()?;
            builder.with_client_cert_verifier(verifier)
        }
        _ => return Err(USAGE.into()),
    };
    let certs = CertificateDer::pem_file_iter(cert)?.collect::<Result<Vec<_>, _>>()?;
    let config = Arc::new(builder.with_single_cert(certs, PrivateKeyDer::from_pem_file(key)?)?);

    let listener = TcpListener::bind(("127.0.0.1", port.parse::<u16>()?))?;
    for tcp in listener.incoming() {
        let tcp = tcp?;
        tcp.set_read_timeout(Some(Duration::from_secs(10)))?;
        let mut tls = StreamOwned::new(ServerConnection::new(config.clone())?, tcp);
        // A peer the test expects to be refused fails here; the server goes on to the next
        if let Err(e) = answer(&mut tls) {
            eprintln!("interop-server: {e}");
        }
    }
    Ok(())
}

fn answer(tls: &mut StreamOwned<ServerConnection, TcpStream>) -> io::Result<()> {
    // Reading the request line completes the handshake
    let mut request = String::new();
    BufReader::new(&mut *tls).read_line(&mut request)?;
    let suite = tls
        .conn
        .negotiated_cipher_suite()
        .map(|suite| suite.suite());
    let group = tls
        .conn
        .negotiated_key_exchange_group()
        .map(|group| group.name());
    write!(
        tls,
        "HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n\
         rustls-pure {} {}\n",
        suite.map(|suite| format!("{suite:?}")).unwrap_or_default(),
        group.map(|group| format!("{group:?}")).unwrap_or_default(),
    )?;
    tls.conn.send_close_notify();
    tls.flush()
}
