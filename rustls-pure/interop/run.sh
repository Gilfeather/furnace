#!/usr/bin/env bash
# Interoperability tests of rustls-pure against other TLS 1.3 implementations, all on this
# machine: OpenSSL always, GnuTLS and curl when installed. Each other implementation connects as
# a client to rustls-pure's server and serves rustls-pure's client, for every cipher suite, key
# exchange group and key type the provider offers, and with client certificates. Servers with
# an RSA certificate or an unknown CA must be refused.
#
# Usage, from the repository root: bash rustls-pure/interop/run.sh
# OPENSSL picks the openssl binary (it needs TLS 1.3 and Ed25519: OpenSSL 1.1.1 or later).
set -euo pipefail

# Git Bash on Windows would rewrite arguments such as /CN=localhost into Windows paths
export MSYS2_ARG_CONV_EXCL='*'

openssl=${OPENSSL:-openssl}
echo "OpenSSL: $("$openssl" version)"
cargo build --quiet -p rustls-pure --examples
examples=${CARGO_TARGET_DIR:-target}/debug/examples
exe=""
[ -f "$examples/interop-server.exe" ] && exe=.exe
server=$examples/interop-server$exe
client=$examples/interop-client$exe

work=$(mktemp -d)
# Git Bash: the native programs take Windows paths
if command -v cygpath >/dev/null; then
  work=$(cygpath -m "$work")
fi
pids=()
cleanup() {
  for pid in ${pids[@]+"${pids[@]}"}; do
    kill "$pid" 2>/dev/null || true
  done
  rm -rf "$work"
}
trap cleanup EXIT

failures=0
shopt -s nocasematch
check() { # check <output> <expected text, any case> <description>
  if [[ $1 == *"$2"* ]]; then
    echo "ok      $3"
  else
    echo "FAILED  $3"
    printf '%s\n' "$1" | sed 's/^/        /' | head -20
    failures=$((failures + 1))
  fi
}
refused() { # refused <output> <text that must not be there, any case> <description>
  if [[ $1 == *"$2"* ]]; then
    echo "FAILED  $3 (it was accepted)"
    failures=$((failures + 1))
  else
    echo "ok      $3"
  fi
}

port=$((20000 + RANDOM % 20000))
start() { # start <command...>: runs a server on $port in the background, once it listens
  "$@" >"$work/server-$port.log" 2>&1 &
  pids+=($!)
  for _ in $(seq 100); do
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
      return 0
    fi
    sleep 0.1
  done
  echo "nothing listens on $port:"
  cat "$work/server-$port.log"
  exit 1
}
stop() { # stops the last server; the next one gets a fresh port
  kill "${pids[${#pids[@]} - 1]}" 2>/dev/null || true
  port=$((port + 1))
}

# A CA, a server and a client certificate per key type; RSA only to be refused
keygen() { # keygen <kind> <file>
  case $1 in
    p256) "$openssl" genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out "$2" ;;
    p384) "$openssl" genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-384 -out "$2" ;;
    ed25519) "$openssl" genpkey -algorithm ED25519 -out "$2" ;;
    rsa) "$openssl" genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$2" ;;
  esac 2>/dev/null
}
printf '%s\n' 'basicConstraints=critical,CA:FALSE' 'keyUsage=critical,digitalSignature' \
  'extendedKeyUsage=serverAuth,clientAuth' 'subjectAltName=DNS:localhost' >"$work/leaf.ext"
for kind in p256 p384 ed25519 rsa; do
  case $kind in
    p384) digest=(-sha384) ;;
    ed25519) digest=() ;; # Ed25519 hashes internally
    *) digest=(-sha256) ;;
  esac
  keygen "$kind" "$work/$kind-ca.key"
  "$openssl" req -x509 -new -key "$work/$kind-ca.key" -subj "/CN=interop $kind CA" -days 2 \
    ${digest[@]+"${digest[@]}"} -addext 'basicConstraints=critical,CA:TRUE' \
    -addext 'keyUsage=critical,keyCertSign' -out "$work/$kind-ca.crt"
  for role in server client; do
    keygen "$kind" "$work/$kind-$role.key"
    "$openssl" req -new -key "$work/$kind-$role.key" -subj "/CN=localhost" \
      -out "$work/$kind-$role.csr"
    "$openssl" x509 -req -in "$work/$kind-$role.csr" -CA "$work/$kind-ca.crt" \
      -CAkey "$work/$kind-ca.key" -CAcreateserial -CAserial "$work/serial" -days 2 \
      ${digest[@]+"${digest[@]}"} -extfile "$work/leaf.ext" -out "$work/$kind-$role.crt" 2>/dev/null
  done
done

# The provider's suites and groups, in rustls's names, OpenSSL's and GnuTLS's
suites=(TLS13_AES_128_GCM_SHA256 TLS13_AES_256_GCM_SHA384 TLS13_CHACHA20_POLY1305_SHA256)
openssl_suites=(TLS_AES_128_GCM_SHA256 TLS_AES_256_GCM_SHA384 TLS_CHACHA20_POLY1305_SHA256)
gnutls_suites=(AES-128-GCM AES-256-GCM CHACHA20-POLY1305)
kx=(X25519 secp256r1 secp384r1)
openssl_kx=(X25519 P-256 P-384)
gnutls_kx=(GROUP-X25519 GROUP-SECP256R1 GROUP-SECP384R1)
request=$'GET / HTTP/1.0\r\nHost: localhost\r\n\r\n'

openssl_client() { # openssl_client <ca> <extra s_client flags...>
  local ca=$1
  shift
  printf '%s' "$request" | "$openssl" s_client -connect "127.0.0.1:$port" -servername localhost \
    -verify_hostname localhost -CAfile "$ca" -verify_return_error -tls1_3 -quiet "$@" \
    2>/dev/null || true
}

echo "== OpenSSL client, rustls-pure server"
for kind in p256 p384 ed25519; do
  start "$server" "$work/$kind-server.crt" "$work/$kind-server.key" "$port"
  for i in 0 1 2; do
    for j in 0 1 2; do
      out=$(openssl_client "$work/$kind-ca.crt" -ciphersuites "${openssl_suites[i]}" \
        -groups "${openssl_kx[j]}")
      check "$out" "rustls-pure ${suites[i]} ${kx[j]}" "$kind certificate, ${suites[i]}, ${kx[j]}"
    done
  done
  other=p384
  [ "$kind" = p384 ] && other=p256
  out=$(openssl_client "$work/$other-ca.crt")
  refused "$out" "rustls-pure" "$kind certificate, refused by a client trusting another CA"
  stop
done

echo "== rustls-pure client, OpenSSL server"
for kind in p256 p384 ed25519; do
  start "$openssl" s_server -accept "$port" -cert "$work/$kind-server.crt" \
    -key "$work/$kind-server.key" -www -tls1_3 -groups X25519:P-256:P-384
  for i in 0 1 2; do
    for j in 0 1 2; do
      out=$("$client" "$work/$kind-ca.crt" "$port" --suite "${suites[i]}" --group "${kx[j]}" \
        2>&1 || true)
      check "$out" "negotiated ${suites[i]} ${kx[j]}" "$kind certificate, ${suites[i]}, ${kx[j]}"
      check "$out" "200 ok" "$kind certificate, ${suites[i]}, ${kx[j]}: response"
    done
  done
  stop
done
start "$openssl" s_server -accept "$port" -cert "$work/p256-server.crt" \
  -key "$work/p256-server.key" -www -tls1_3
out=$("$client" "$work/p384-ca.crt" "$port" 2>&1 || true)
refused "$out" "200 ok" "a server certificate from an unknown CA is refused"
stop
start "$openssl" s_server -accept "$port" -cert "$work/rsa-server.crt" \
  -key "$work/rsa-server.key" -www -tls1_3
out=$("$client" "$work/rsa-ca.crt" "$port" 2>&1 || true)
refused "$out" "200 ok" "a server with an RSA certificate is refused (no RSA)"
stop

echo "== client certificates (mutual TLS)"
for kind in p256 p384 ed25519; do
  start "$server" "$work/p256-server.crt" "$work/p256-server.key" "$port" "$work/$kind-ca.crt"
  out=$(openssl_client "$work/p256-ca.crt" -cert "$work/$kind-client.crt" \
    -key "$work/$kind-client.key")
  check "$out" "rustls-pure" "rustls-pure server accepts an OpenSSL client's $kind certificate"
  out=$(openssl_client "$work/p256-ca.crt")
  refused "$out" "rustls-pure" "rustls-pure server refuses an OpenSSL client without a certificate"
  stop
  start "$openssl" s_server -accept "$port" -cert "$work/p256-server.crt" \
    -key "$work/p256-server.key" -www -tls1_3 -Verify 1 -verify_return_error \
    -CAfile "$work/$kind-ca.crt"
  out=$("$client" "$work/p256-ca.crt" "$port" --cert "$work/$kind-client.crt" \
    --key "$work/$kind-client.key" 2>&1 || true)
  check "$out" "200 ok" "OpenSSL server accepts rustls-pure's $kind client certificate"
  stop
done

if command -v gnutls-cli >/dev/null && command -v gnutls-serv >/dev/null; then
  echo "== GnuTLS client, rustls-pure server"
  for kind in p256 p384 ed25519; do
    start "$server" "$work/$kind-server.crt" "$work/$kind-server.key" "$port"
    for i in 0 1 2; do
      for j in 0 1 2; do
        # gnutls-cli stops at the end of its input: keep it open until the answer is in
        out=$({ printf '%s' "$request"; sleep 1; } | gnutls-cli 127.0.0.1 --port "$port" \
          --sni-hostname localhost --verify-hostname localhost \
          --x509cafile "$work/$kind-ca.crt" \
          --priority "NORMAL:-VERS-ALL:+VERS-TLS1.3:-CIPHER-ALL:+${gnutls_suites[i]}:-GROUP-ALL:+${gnutls_kx[j]}" \
          2>&1 || true)
        check "$out" "rustls-pure ${suites[i]} ${kx[j]}" "$kind certificate, ${suites[i]}, ${kx[j]}"
      done
    done
    stop
  done

  echo "== rustls-pure client, GnuTLS server"
  for kind in p256 p384 ed25519; do
    start gnutls-serv --http --port "$port" --x509certfile "$work/$kind-server.crt" \
      --x509keyfile "$work/$kind-server.key" --priority "NORMAL:-VERS-ALL:+VERS-TLS1.3"
    for i in 0 1 2; do
      for j in 0 1 2; do
        out=$("$client" "$work/$kind-ca.crt" "$port" --suite "${suites[i]}" \
          --group "${kx[j]}" 2>&1 || true)
        check "$out" "negotiated ${suites[i]} ${kx[j]}" "$kind certificate, ${suites[i]}, ${kx[j]}"
        check "$out" "200 ok" "$kind certificate, ${suites[i]}, ${kx[j]}: response"
      done
    done
    stop
  done
else
  echo "== GnuTLS: gnutls-cli and gnutls-serv are not installed, skipped"
fi

if command -v curl >/dev/null; then
  echo "== curl ($(curl --version | head -1)), rustls-pure server"
  # ECDSA only: some curl builds (Schannel, LibreSSL) take no Ed25519 certificates
  for kind in p256 p384; do
    start "$server" "$work/$kind-server.crt" "$work/$kind-server.key" "$port"
    # --ssl-no-revoke: Schannel would look for revocation lists the test CA does not have
    out=$(curl -sS --tlsv1.3 --ssl-no-revoke --cacert "$work/$kind-ca.crt" \
      --resolve "localhost:$port:127.0.0.1" "https://localhost:$port/" 2>&1 || true)
    check "$out" "rustls-pure" "$kind certificate"
    stop
  done
else
  echo "== curl: not installed, skipped"
fi

if [ "$failures" -gt 0 ]; then
  echo "$failures checks failed"
  exit 1
fi
echo "all checks passed"
