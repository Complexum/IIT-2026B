"""Selección y carga del backend de QSW: ruta numpy (Python) o kernel C.

Única responsabilidad de este módulo: decidir qué backend corre y, si es el C,
cargar ``libqsw.so`` y exponer ``qsw_zeta`` vía ctypes. El algoritmo vive en
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
SCRIPT_BUILD = CLANG_DIR / "build.sh"
CMD_COMPILAR = str(SCRIPT_BUILD)

# El kernel hace el **Zeta**, no el MAO. Medido: la búsqueda son ~3 ms mientras el
# precómputo Zeta es el resto del tiempo (a n=22, 0.96 s de 0.95 s totales), así
# que portar el MAO no cambiaría nada. El Zeta en C da 5–10× sobre numpy y es
# bit-exacto (mismo orden de sumas).
_ARGTYPES = [
    ctypes.POINTER(ctypes.c_float),  # flat (N, 2^D) float32 contiguo, in-place
    ctypes.c_int,                    # N
    ctypes.c_int,                    # D
    ctypes.c_uint64,                 # pivot_flat
]


def cargar_libqsw() -> ctypes.CDLL | None:
    """Carga ``libqsw.so`` si existe y expone ``qsw_zeta``. ``None`` si no."""
    if not LIB_PATH.exists():
        return None
    try:
        lib = ctypes.CDLL(str(LIB_PATH))
        lib.qsw_zeta.argtypes = _ARGTYPES
        lib.qsw_zeta.restype = ctypes.c_int
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


def zeta_c(flat, N: int, D: int, pivot_flat: int):
    """Butterfly Zeta in-place en el kernel C. Mismo resultado que `zeta_inplace`.

    `flat` debe ser (N, 2^D) float32 contiguo; se modifica en el lugar.
    """
    lib = cargar_libqsw()
    if lib is None:
        raise RuntimeError(f"libqsw.so no disponible. Compilar:\n    {CMD_COMPILAR}")
    if flat.dtype != np.float32 or not flat.flags["C_CONTIGUOUS"]:
        raise ValueError("qsw_zeta requiere float32 contiguo")
    rc = lib.qsw_zeta(
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(N),
        ctypes.c_int(D),
        ctypes.c_uint64(pivot_flat),
    )
    if rc != 0:
        raise RuntimeError(f"qsw_zeta falló con código {rc}")
    return flat
