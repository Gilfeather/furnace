#!/bin/sh
# Test-only certificates and keys for the TLS tests (src/tls). Never use them for anything else.
# Regenerate with: sh tests/fixtures/tls/generate.sh (needs OpenSSL 3)
set -eu
cd "$(dirname "$0")"
days=36500

cat > server.ext <<'X'
subjectAltName = DNS:localhost, IP:127.0.0.1
basicConstraints = CA:FALSE
keyUsage = digitalSignature
extendedKeyUsage = serverAuth
X
cat > client.ext <<'X'
basicConstraints = CA:FALSE
keyUsage = digitalSignature
extendedKeyUsage = clientAuth
X

# Certificate authorities: one trusted by the tests, one not
for ca in ca other-ca; do
    openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out $ca.key
    openssl req -x509 -new -key $ca.key -sha256 -days $days -subj "/CN=furnace test $ca" -out $ca.crt
done

# Server certificates with keys in each format the server takes, and one it refuses (RSA)
leaf() { # name key-algorithm-options ext ca
    openssl genpkey $2 -out $1.key
    openssl req -new -key $1.key -subj "/CN=$1" -out $1.csr
    openssl x509 -req -in $1.csr -CA $4.crt -CAkey $4.key -days $days -sha256 -extfile $3 -out $1.crt
    rm $1.csr
}
leaf server-p256 "-algorithm EC -pkeyopt ec_paramgen_curve:P-256" server.ext ca
leaf server-p384 "-algorithm EC -pkeyopt ec_paramgen_curve:P-384" server.ext ca
leaf server-ed25519 "-algorithm ED25519" server.ext ca
leaf server-rsa "-algorithm RSA -pkeyopt rsa_keygen_bits:2048" server.ext ca
# The P-256 key again in SEC1 ("BEGIN EC PRIVATE KEY") form
openssl ec -in server-p256.key -out server-p256-sec1.key
# A key on a curve the server does not take (P-521), for the error message
openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-521 -out server-p521.key
# Client certificates for mutual TLS, one per key type the server verifies ("client" is P-256)
leaf client "-algorithm EC -pkeyopt ec_paramgen_curve:P-256" client.ext ca
leaf client-p384 "-algorithm EC -pkeyopt ec_paramgen_curve:P-384" client.ext ca
leaf client-ed25519 "-algorithm ED25519" client.ext ca
leaf stranger "-algorithm EC -pkeyopt ec_paramgen_curve:P-256" client.ext other-ca

rm -f server.ext client.ext ca.srl other-ca.srl ca.key other-ca.key
