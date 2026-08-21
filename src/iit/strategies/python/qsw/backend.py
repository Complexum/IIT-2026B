"""Selección y carga del backend de QSW: ruta numpy (Python) o kernel C.

Única responsabilidad de este módulo: decidir qué backend corre y, si es el C,
cargar ``libqsw.so`` y exponer ``qsw_solve`` vía ctypes. El algoritmo vive en
``core.py`` y la estrategia SIA en ``code.py``.

El backend es un **atributo** de la estrategia (``QSW.backend``), configurable con
``ejecutar(..., opciones={"backend": "c"})``. Ver ``SIA.opciones``.
"""

import ctypes
from pathlib import Path

import numpy as np

CLANG_DIR = Path("src/iit/strategies/clang")
FUENTE_C = CLANG_DIR / "qsw" / "code.c"
LIB_PATH = CLANG_DIR / "__cache__" / "libqsw.so"
CMD_COMPILAR = (
    f"cc -O3 -march=native -shared -fPIC -o {LIB_PATH} {FUENTE_C}"
)

_ARGTYPES = [
    ctypes.POINTER(ctypes.c_float),  # sumas (N, 2^D) row-major
    ctypes.c_int,                    # N
    ctypes.c_int,                    # D
    ctypes.c_int,                    # V
    ctypes.POINTER(ctypes.c_int),    # vert_kind  (0 = ACTUAL, 1 = EFFECT)
    ctypes.POINTER(ctypes.c_int),    # vert_slot  (posición de bit / fila de ncubo)
    ctypes.c_int,                    # modo (0 = estatico, 1 = exacto)
    ctypes.POINTER(ctypes.c_uint64),  # out_candidatos [V]
    ctypes.POINTER(ctypes.c_double),  # out_valores    [V]
    ctypes.POINTER(ctypes.c_int),     # out_n
]


def cargar_libqsw() -> ctypes.CDLL | None:
    """Carga ``libqsw.so`` si existe y expone ``qsw_solve``. ``None`` si no."""
    if not LIB_PATH.exists():
        return None
    try:
        lib = ctypes.CDLL(str(LIB_PATH))
        lib.qsw_solve.argtypes = _ARGTYPES
        lib.qsw_solve.restype = ctypes.c_int
    except (OSError, AttributeError):
        return None
    return lib


def resolver_backend(backend: str) -> str:
    """Resuelve ``backend`` a ``"python"`` o ``"c"``.

    - ``"python"`` (default de `qsw`): ruta numpy.
    - ``"c"``: exige el ``.so``. Si falta, **lanza** con el comando de compilación.
      Nunca degrada a Python en silencio: el nombre de la estrategia acaba en el
      nombre del CSV, así que correr Python bajo la etiqueta ``backend=c`` haría que
      el resultado mienta y la comparación Python-vs-C quede inservible.
    - ``"auto"``: C si el ``.so`` está disponible, Python si no.
    """
    if backend == "python":
        return "python"
    if backend == "auto":
        return "c" if cargar_libqsw() is not None else "python"
    if backend == "c":
        if cargar_libqsw() is None:
            estado = (
                "el fuente está vacío" if FUENTE_C.stat().st_size == 0
                else "la librería no está compilada"
            ) if FUENTE_C.exists() else "no existe el fuente"
            raise RuntimeError(
                f"backend 'c' no disponible: {estado} ({LIB_PATH}).\n"
                f"Compilar con:\n    {CMD_COMPILAR}\n"
                f"O usar la estrategia 'qsw' (backend Python)."
            )
        return "c"
    raise ValueError(f"backend desconocido: {backend!r} (usar 'python' | 'c' | 'auto')")


def candidatas_c(
    sumas: np.ndarray,
    N: int,
    D: int,
    V: int,
    vert_kind: np.ndarray,
    vert_slot: np.ndarray,
    modo: str,
) -> list[int]:
    """Corre el MAO + contracciones en el kernel C y devuelve las candidatas.

    El kernel sólo **busca**: Python re-puntúa las candidatas con la ``f`` exacta y
    reconstruye el ganador, igual que en la ruta Python.
    """
    lib = cargar_libqsw()
    if lib is None:
        raise RuntimeError(f"libqsw.so no disponible. Compilar:\n    {CMD_COMPILAR}")
    if V > 64:
        raise ValueError(f"kernel C limitado a V <= 64 (máscaras uint64); V={V}")

    sumas_c = np.ascontiguousarray(sumas, dtype=np.float32)
    kind_c = np.ascontiguousarray(vert_kind, dtype=np.int32)
    slot_c = np.ascontiguousarray(vert_slot, dtype=np.int32)
    out_masks = np.zeros(V, dtype=np.uint64)
    out_vals = np.zeros(V, dtype=np.float64)
    out_n = ctypes.c_int(0)

    rc = lib.qsw_solve(
        sumas_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(N), ctypes.c_int(D), ctypes.c_int(V),
        kind_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        slot_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(1 if modo == "exacto" else 0),
        out_masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        out_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(out_n),
    )
    if rc != 0:
        raise RuntimeError(f"qsw_solve falló con código {rc}")
    return [int(m) for m in out_masks[: out_n.value]]
