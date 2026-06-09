"""Auto-reloader para la TUI — reinicia el proceso ante cambios en código fuente."""

import os
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


def _get_watched_paths() -> list[Path]:
    """Devuelve los directorios a observar: src/tui/ y raíz del proyecto."""
    tui_dir = Path(__file__).parent
    project_root = tui_dir.parent.parent
    return [tui_dir, project_root / "src"]


def _should_restart(event: FileSystemEvent) -> bool:
    """Determina si un cambio en archivo requiere reinicio."""
    if event.is_directory:
        return False
    path = Path(event.src_path)
    # Solo reiniciar ante cambios en .py (no cache, no pyc, no tests)
    if path.suffix != ".py":
        return False
    if "__pycache__" in path.parts:
        return False
    return True


class _ReloadHandler(FileSystemEventHandler):
    """Handler que reinicia el proceso cuando cambia código fuente."""

    def __init__(self) -> None:
        self._restart_pending = False

    def on_any_event(self, event: FileSystemEvent) -> None:
        if not self._restart_pending and _should_restart(event):
            self._restart_pending = True
            print(f"\n[reload] Detectado cambio en {event.src_path}")
            print("[reload] Reiniciando TUI...\n")
            time.sleep(0.3)  # Pequeña espera para evitar reinicios múltiples
            # Reiniciar el proceso con los mismos argumentos
            os.execv(sys.executable, [sys.executable, *sys.argv])


def run_with_reloader(target) -> None:
    """Ejecuta `target` con un watcher que reinicia ante cambios en .py.

    Args:
        target: Callable sin argumentos que inicia la aplicación TUI.
    """
    observer = Observer()
    handler = _ReloadHandler()

    for path in _get_watched_paths():
        if path.exists():
            observer.schedule(handler, str(path), recursive=True)

    observer.start()
    try:
        target()
    finally:
        observer.stop()
        observer.join()
