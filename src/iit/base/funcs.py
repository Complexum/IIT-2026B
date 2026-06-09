import numpy as np
from numpy.typing import NDArray


def emd_efecto(u: NDArray[np.float32], v: NDArray[np.float32]) -> np.float32:
    """
    Solución analítica de la Earth Mover's Distance basada en variables independientes condicionalmente y la EMD como distribuciones marginales de probabilidad.
    Sean `X_1`, `X_2` dos variables aleatorias con su correspondiente espacio de estados. Si `X_1` y `X_2` son independientes y `u_1`, `v_1` dos distribuciones de probabilidad en `X_1` y `u_2`, `v_2` dos distribuciones de probabilidad en `X_2`.

    Args:
        `u` (NDArray[np.float32]): Histograma/distribución/vector/serie donde cada indice asocia un valor de pobabilidad de tener el nodo en estado ON/OFF.
        `v` (NDArray[np.float32]): Histograma/distribución/vector/serie donde cada indice asocia un valor de pobabilidad de tener el nodo en estado ON/OFF.

    Returns:
        float: La EMD entre los repertorios efecto es igual a la suma entre las EMD de las distribuciones marginales de cada nodo y, la EMD entre las distribuciones marginales para un nodo es la diferencia absoluta entre las probabilidades con el nodo ON/OFF.
    """
    return np.sum(np.abs(u - v))
