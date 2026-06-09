import logging

from colorama import Fore, Style, init


class ColorFormatter(logging.Formatter):
    """Formatter personalizado para consola con colores usando colorama."""

    COLORS = {
        logging.DEBUG: Fore.LIGHTBLACK_EX,  # gris
        logging.INFO: Fore.BLUE,  # azul
        logging.WARNING: Fore.YELLOW,  # amarillo
        logging.ERROR: Fore.RED,  # rojo
        logging.CRITICAL: Fore.MAGENTA,  # magenta
        logging.FATAL: Fore.RED + Style.BRIGHT,  # rojo brillante
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init(autoreset=True)  # Inicializa colorama

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        # Guarda el nombre del nivel original
        original_levelname = record.levelname
        # Aplica el color al nombre del nivel
        record.levelname = f"{color}{original_levelname}{Style.RESET_ALL}"

        # Formato del mensaje
        formatted = super().format(record)
        # Restaura el nombre del nivel original
        record.levelname = original_levelname
        return formatted
