from enum import Enum


class MetricDistance(str, Enum):
    """La clase notaciones recopila diferentes notaciones binarias."""

    HAMMING = "distancia-hamming"
    MANHATTAN = "distancia-manhattan"
    EUCLIDIANA = "distancia-euclidiana"
