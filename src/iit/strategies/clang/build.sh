#!/usr/bin/env bash
# Compila el kernel Zeta de QSW a __cache__/libqsw.so
#
# Elige el primer compilador disponible: $CC, clang, gcc. Se usa `command -v` a
# propósito: en shells interactivos `cc` puede estar aliaseado a otra cosa, y los
# alias no se heredan acá.
set -euo pipefail

raiz="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fuente="$raiz/qsw/code.c"
destino="$raiz/__cache__/libqsw.so"

for cand in "${CC:-}" clang gcc cc; do
    [ -n "$cand" ] || continue
    if command -v "$cand" >/dev/null 2>&1; then compilador="$cand"; break; fi
done
if [ -z "${compilador:-}" ]; then
    echo "✗ No se encontró un compilador C (probé \$CC, clang, gcc, cc)." >&2
    exit 1
fi

# -march=native no existe en todos los targets (Apple silicon lo acepta; algunos
# cross-compilers no). Se prueba y si falla se compila sin él.
mkdir -p "$raiz/__cache__"
if "$compilador" -O3 -march=native -shared -fPIC -o "$destino" "$fuente" 2>/dev/null; then
    echo "✓ $destino  ($compilador -O3 -march=native)"
else
    "$compilador" -O3 -shared -fPIC -o "$destino" "$fuente"
    echo "✓ $destino  ($compilador -O3, sin -march=native)"
fi
