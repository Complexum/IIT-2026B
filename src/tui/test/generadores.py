"""Registro de funciones generadoras de combinaciones binarias.

Cada generador recibe `n` (dimensiones) y retorna un string binario de largo n.
Ejemplo: pares(6) → '101010'

Estos generadores se usan en los patrones de prueba para producir
combinaciones de (estado, condición, alcance, mecanismo).
"""

from typing import Callable

# Registro global: nombre → función(n) → str binario
GENERADORES: dict[str, Callable[[int], str]] = {}


def __registrar(nombre: str):
    """Decorador que registra una función generadora."""

    def envolver(fn: Callable[[int], str]) -> Callable[[int], str]:
        GENERADORES[nombre] = fn
        return fn

    return envolver


# ── Generadores built-in ─────────────────────────────────


@__registrar("todos")
def __todos(n: int) -> str:
    """Todas las dimensiones activas: '111...1'"""
    return "1" * n


@__registrar("inicial")
def __inicial(n: int) -> str:
    """Solo la primera dimensión: '100...0'"""
    return "1" + "0" * (n - 1) if n > 0 else ""


@__registrar("no_inicial")
def __no_inicial(n: int) -> str:
    """Solo la primera dimensión: '100...0'"""
    return "0" + "1" * (n - 1) if n > 0 else ""


@__registrar("final")
def __final(n: int) -> str:
    """Solo la última dimensión: '000...1'"""
    return "0" * (n - 1) + "1" if n > 0 else ""


@__registrar("no_final")
def __no_final(n: int) -> str:
    """Solo la última dimensión: '000...1'"""
    return "1" * (n - 1) + "0" if n > 0 else ""


@__registrar("pares")
def __pares(n: int) -> str:
    """Posiciones pares activas: '101010...'"""
    return "".join("1" if i % 2 == 0 else "0" for i in range(n))


@__registrar("impares")
def __impares(n: int) -> str:
    """Posiciones impares activas: '010101...'"""
    return "".join("0" if i % 2 == 0 else "1" for i in range(n))


@__registrar("mult_3")
def __mult_3(n: int) -> str:
    """Múltiplos de 3: posiciones 0, 3, 6, ..."""
    return "".join("1" if i % 3 == 0 else "0" for i in range(n))


@__registrar("no_mult_3")
def __no_mult_3(n: int) -> str:
    """Múltiplos de 3: posiciones 0, 3, 6, ..."""
    return "".join("0" if i % 3 == 0 else "1" for i in range(n))


@__registrar("mult_4")
def __mult_4(n: int) -> str:
    """Múltiplos de 3: posiciones 0, 3, 6, ..."""
    return "".join("1" if i % 4 == 0 else "0" for i in range(n))


@__registrar("no_mult_4")
def __no_mult_4(n: int) -> str:
    """Múltiplos de 3: posiciones 0, 3, 6, ..."""
    return "".join("0" if i % 4 == 0 else "1" for i in range(n))


@__registrar("ninguno")
def __ninguno(n: int) -> str:
    """Ninguna dimensión activa: '000...0'"""
    return "0" * n


# ── API pública ──────────────────────────────────────────


def listar_generadores() -> list[str]:
    """Retorna nombres de generadores disponibles, ordenados."""
    return sorted(GENERADORES.keys())


def generar(nombre: str, n: int) -> str:
    """Ejecuta un generador por nombre. Lanza ValueError si no existe."""
    if nombre not in GENERADORES:
        raise ValueError(f"Generador '{nombre}' no encontrado")
    return GENERADORES[nombre](n)


def etiquetas(binario: str) -> str:
    """Convierte un string binario a etiquetas de letra.

    '101' → 'AC'   (dimensión 0=A, 2=C)
    '110' → 'AB'
    '000111' → 'DEF'
    """
    return "".join(chr(65 + i) for i, c in enumerate(binario) if c == "1") or "∅"
