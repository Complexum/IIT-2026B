from enum import Enum


class Notation(str, Enum):
    """La clase notaciones recopila diferentes notaciones binarias."""

    LIL_ENDIAN = "little-endian"
    BIG_ENDIAN = "big-endian"
    GRAY_CODE = "gray-code"
    SIGN_MAGNITUDE = "sign-magnitude"
    TWOS_COMPLEMENT = "two's-complement"
