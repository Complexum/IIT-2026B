"""Capa IO: carga y persistencia de redes (TPMs) y resultados."""

import re
from pathlib import Path

import numpy as np

from src.iit.core.ncube import NCube
from src.iit.core.params import Params
from src.iit.core.system import System

REDES_DIR = Path("data/input/networks")
RESULTADOS_DIR = Path("data/output")


def _natural_key(s: str) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


# ── Redes ────────────────────────────────────────────────


def listar_redes() -> list[str]:
    """Listar nombres de redes disponibles en data/input/networks/."""
    if not REDES_DIR.exists():
        return []
    return sorted((f.stem for f in REDES_DIR.glob("*.csv")), key=_natural_key)


def dims_de_red(nombre: str) -> int:
    """n_dims de una red sin cargar el TPM.

    1) Del nombre (N25B → 25). 2) Fallback: primera línea del CSV.
    """
    m = re.match(r"[Nn](\d+)", nombre)
    if m:
        return int(m.group(1))
    ruta = REDES_DIR / f"{nombre}.csv"
    try:
        with ruta.open() as f:
            return len(f.readline().split(","))
    except OSError:
        return 0


def _ruta_csv(nombre: str) -> Path:
    return REDES_DIR / f"{nombre}.csv"


def _ruta_npy(nombre: str) -> Path:
    return REDES_DIR / f"{nombre}.npy"


def red_optimizada(nombre: str) -> bool:
    """True si existe el sidecar .npy (caché binaria) de la red."""
    return _ruta_npy(nombre).exists()


def cargar_mpt(nombre: str) -> np.ndarray:
    """Cargar una MPT por nombre de red.

    El CSV es el formato canónico (cross-language). Si existe un sidecar
    `.npy` (generado con `optimizar_red`), se mapea en memoria con mmap para
    no cargar todo el TPM en RAM ni reparsear texto.

    Args:
        nombre: Nombre sin extensión (ej: 'N3A')

    Returns:
        numpy array de forma (2^n, n)
    """
    npy = _ruta_npy(nombre)
    if npy.exists():
        return np.atleast_2d(np.load(npy, mmap_mode="r"))

    ruta = _ruta_csv(nombre)
    if not ruta.exists():
        raise FileNotFoundError(f"Red '{nombre}' no encontrada en {ruta}")
    tpm = np.loadtxt(ruta, delimiter=",", dtype=np.float64)
    return np.atleast_2d(tpm)


def optimizar_red(nombre: str) -> bool:
    """Generar el sidecar .npy desde el CSV para acelerar cargas futuras.

    No toca el CSV (sigue siendo la fuente canónica). Retorna True si quedó
    el .npy en disco.
    """
    ruta = _ruta_csv(nombre)
    if not ruta.exists():
        raise FileNotFoundError(f"Red '{nombre}' no encontrada en {ruta}")
    tpm = np.atleast_2d(np.loadtxt(ruta, delimiter=",", dtype=np.float64))
    np.save(_ruta_npy(nombre), tpm)
    return red_optimizada(nombre)


def desoptimizar_red(nombre: str) -> bool:
    """Eliminar el sidecar .npy (vuelve a usarse el CSV). True si se borró."""
    npy = _ruta_npy(nombre)
    if npy.exists():
        npy.unlink()
        return True
    return False


def crear_sistema(tpm: np.ndarray, estado_inicial: tuple[int, ...]) -> System:
    """Construir un System a partir de una MPT y estado inicial.

    Cada columna de la MPT se convierte en un NCube.
    """
    n_dims = tpm.shape[1]
    dims = tuple(range(n_dims))
    ncubos = tuple(
        NCube(indice=i, dims=dims, data=tpm[:, i].copy()) for i in range(n_dims)
    )
    return System(estado_inicial=estado_inicial, ncubos=ncubos)


def reducir_a_subsistema(tpm: np.ndarray, params: Params) -> System:
    """Sistema completo → candidato (condicionar) → subsistema (substraer).

    La preparación queda fuera de las estrategias: quien ejecuta llama aquí
    una vez por (tpm, params) y pasa el subsistema resultante a la estrategia.
    Así se puede reutilizar la misma estrategia para muchos params sin
    re-leer TPM ni repetir lógica.
    """
    estado = params.estado_tuple()
    completo = crear_sistema(tpm, estado)
    dims_cond = params.dims_condicionar()
    candidato = completo.condicionar(dims_cond)
    alcance = params.dims_alcance()
    mecanismo = params.dims_mecanismo()
    return candidato.substraer(alcance, mecanismo)


def eliminar_red(nombre: str) -> bool:
    """Eliminar el CSV de una red (y su sidecar .npy). True si había CSV."""
    npy = _ruta_npy(nombre)
    if npy.exists():
        npy.unlink()
    ruta = _ruta_csv(nombre)
    if ruta.exists():
        ruta.unlink()
        return True
    return False


# ── Resultados ───────────────────────────────────────────


def listar_resultados() -> list[str]:
    """Listar archivos de resultados en data/output/, como rutas relativas sin extensión."""
    if not RESULTADOS_DIR.exists():
        return []
    return sorted(
        (str(f.relative_to(RESULTADOS_DIR).with_suffix("")) for f in RESULTADOS_DIR.glob("*/*.csv")),
        key=_natural_key,
        reverse=True,
    )


def cargar_resultados(nombre: str) -> str:
    """Cargar contenido de un CSV de resultados como texto."""
    ruta = RESULTADOS_DIR / f"{nombre}.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"Resultado '{nombre}' no encontrado en {ruta}")
    return ruta.read_text(encoding="utf-8")
