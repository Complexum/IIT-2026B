from enum import Enum


class TimeEMD(str, Enum):
    """La clase temporal emd recopila diferentes tipos de emd según su tiempo."""

    EMD_EFECTO = "emd-effect"
    EMD_CAUSA = "emd-cause"
    EMD_INTEGRADA = "emd-cause-effect"
