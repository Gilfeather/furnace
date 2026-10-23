#!/usr/bin/env bash
# Checks the machine code of rustls-pure's own functions that handle secrets, as the release
# build compiles them for this machine: the X25519 all-zero check (`all_zero`) and how the
# X25519MLKEM768 secret is put together (`put_together`). Neither may contain a conditional
# branch or jump, a call (whose code would not be checked here) or a load whose address is
# computed from a register other than the base, which a table lookup by a secret would need.
# Every other secret-dependent operation is in the RustCrypto crates (CONSTANT_TIME.md).
#
# bash rustls-pure/ct-asm.sh, from the repository root, on x86_64 or aarch64 (Linux or macOS)
set -euo pipefail

target_dir="${CARGO_TARGET_DIR:-target}/ct-asm"
CARGO_TARGET_DIR="$target_dir" cargo rustc --quiet --release -p rustls-pure --lib -- --emit asm
asm=$(ls -t "$target_dir"/release/deps/rustls_pure-*.s | head -1)

case "$(uname -m)" in
  x86_64)
    # Any jump, a tail call (jmp) included; two registers inside an address: (%base,%index)
    branch='^\s+(j[a-z]+|loop[a-z]*)\s'
    call='^\s+call'
    indexed='\(%[a-z0-9]+,\s*%'
    ;;
  aarch64 | arm64)
    branch='^\s+(b|br|b\.[a-z]+|cbn?z|tbn?z)\s'
    call='^\s+(bl|blr)\s'
    indexed='\[[xw][0-9]+,\s*[xw][0-9]+'
    ;;
  *)
    echo "no check for $(uname -m)" >&2
    exit 1
    ;;
esac

failed=0
for function in all_zero put_together; do
  # v0 mangling: the name is preceded by its length; macOS adds an underscore in front
  symbol=$(grep -oE "^_?_R[A-Za-z0-9_]*rustls_pure[0-9]+${function}:" "$asm" | head -1 || true)
  if [ -z "$symbol" ]; then
    echo "$function: not found in $asm" >&2
    failed=1
    continue
  fi
  body=$(awk -v symbol="$symbol" '$0 == symbol {p = 1; next} p && (/func_end|seh_endproc/ || /^_?_R[A-Za-z0-9_]*:/) {exit} p' "$asm" \
    | grep -E '^\s+[a-z]' | grep -vE '^\s+\.' || true)
  echo "== $function ($(uname -m)): $(echo "$body" | wc -l) instructions"
  echo "$body"
  problems=$(echo "$body" | grep -E "$branch|$call|$indexed" || true)
  if [ -n "$problems" ]; then
    echo "::error::$function has branches, calls or indexed loads:" >&2
    echo "$problems" >&2
    failed=1
  fi
done
exit "$failed"
