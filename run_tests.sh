#!/bin/bash
# Script helper para ejecutar tests

set -e

echo "🧪 Ejecutando tests..."
echo ""

# Verificar si pytest está instalado
if ! uv run pytest --version > /dev/null 2>&1; then
    echo "⚠️  pytest no encontrado. Instalando..."
    uv add --dev pytest pytest-cov
    echo "✅ pytest instalado"
    echo ""
fi

# Ejecutar tests
echo "📊 Ejecutando todos los tests..."
uv run pytest tests/ -v "$@"
