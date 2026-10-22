#!/bin/sh
# Test-only certificates and keys for the handshake tests (tests/handshake.rs). Never use them
# for anything else. Regenerate with: sh rustls-pure/testdata/generate.sh (needs OpenSSL 3)
set -eu
cd "$(dirname "$0")"
# Git Bash on Windows would rewrite /CN=... into a path
export MSYS2_ARG_CONV_EXCL='*'
days=36500

printf '%s\n' 'basicConstraints = critical, CA:FALSE' 'keyUsage = critical, digitalSignature' \
    'extendedKeyUsage = serverAuth' 'subjectAltName = DNS:localhost' > server.ext
printf '%s\n' 'basicConstraints = critical, CA:FALSE' 'keyUsage = critical, digitalSignature' \
    'extendedKeyUsage = clientAuth' > client.ext

# A CA per key type, signing a server and a client certificate of its own type
for kind in p256 p384 ed25519; do
    case $kind in
        p256) algorithm="-algorithm EC -pkeyopt ec_paramgen_curve:P-256" digest=-sha256 ;;
        p384) algorithm="-algorithm EC -pkeyopt ec_paramgen_curve:P-384" digest=-sha384 ;;
        ed25519) algorithm="-algorithm ED25519" digest= ;;
    esac
    openssl genpkey $algorithm -out $kind-ca.key
    openssl req -x509 -new -key $kind-ca.key $digest -days $days -subj "/CN=rustls-pure test $kind CA" \
        -addext 'basicConstraints = critical, CA:TRUE' -addext 'keyUsage = critical, keyCertSign' \
        -out $kind-ca.crt
    for role in server client; do
        openssl genpkey $algorithm -out $kind-$role.key
        openssl req -new -key $kind-$role.key -subj "/CN=localhost" -out $kind-$role.csr
        openssl x509 -req -in $kind-$role.csr -CA $kind-ca.crt -CAkey $kind-ca.key \
            -CAcreateserial -CAserial ca.srl $digest -days $days -extfile $role.ext \
            -out $kind-$role.crt
        rm $kind-$role.csr
    done
    rm $kind-ca.key
done

rm -f server.ext client.ext ca.srl
