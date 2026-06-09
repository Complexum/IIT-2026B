import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

from pyinstrument import Profiler
from pyinstrument.renderers import HTMLRenderer

from src.iit.base.app import aplicacion
from src.iit.base.consts import HTML_EXTENSION, PATH_PROFILING


class ProfilingManager:
    """
    Gestor central de profiling que mantiene configuración y estado
    """

    def __init__(
        self,
        habilitado: bool = aplicacion.profiler_habilitado,
        dir_salida: Path = Path(PATH_PROFILING),
        intervalo: float = 0.001,
    ):
        self.enabled = habilitado
        self.output_dir = dir_salida
        self.interval = intervalo
        self.current_session: Optional[str] = None
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Prepara estructura de directorios para resultados"""
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def start_session(self, session_name: str) -> None:
        if self.enabled:
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp = datetime.now().strftime("%d_%m_%Y/%Hhrs")
            session_path = self.output_dir / session_name / timestamp
            session_path.mkdir(parents=True, exist_ok=True)
            self.current_session = str(session_path.relative_to(self.output_dir))

    def get_output_path(self, name: str, format: str) -> Path:
        """Genera ruta de salida para un perfil específico"""
        session_dir = self.current_session or "default"
        return self.output_dir / session_dir / f"{name}.{format}"


class ProfilerContext:
    """
    Contexto para medición de una función específica
    """

    def __init__(
        self,
        manager: ProfilingManager,
        name: str,
        context: dict,
        unique: bool = False,
    ):
        self.manager = manager
        self.name = name
        self.context = context
        self.unique = unique
        self.start_time = None
        self.profiler = (
            None
            if not manager.enabled
            else Profiler(interval=manager.interval, async_mode="disabled")
        )

    def __enter__(self):
        if self.manager.enabled:
            self.start_time = time.perf_counter()
            self.profiler.start()
        return self

    def _get_unique_path(self, base_path: Path) -> Path:
        if not self.unique or not base_path.exists():
            return base_path
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.manager.enabled:
            return

        self.profiler.stop()

        # Generar reporte HTML detalladito
        html_path = self.manager.get_output_path(f"{self.name}", HTML_EXTENSION)
        html_path = self._get_unique_path(html_path)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(
                self.profiler.output(
                    renderer=HTMLRenderer(show_all=True, timeline=True)
                )
            )


# Instancia global del gestor
gestor_perfilado = ProfilingManager()


# ── Decorador de clase ────────────────────────────────────────────────────────


def perfilar(cls: type) -> type:
    """Decorador de clase: envuelve resolver() con pyinstrument.

    Guarda el HTML dentro del directorio de la propia estrategia:
        src/iit/strategies/python/{nombre}/profiling/{DD_MM_YYYY}/{HH}hrs/resolver.html

    Uso:
        @perfilar
        class Basic(SIA, nombre="basic"):
            def resolver(self) -> Solution: ...

    Solo actúa si gestor_perfilado.enabled es True.
    """
    original = cls.resolver

    @wraps(original)
    def _perfilamiento(self):
        if not gestor_perfilado.enabled:
            return original(self)

        nombre = getattr(self, "nombre", cls.__name__.lower())
        timestamp = datetime.now().strftime("%d_%m_%Y/%Hhrs")
        out_dir = Path("src/iit/strategies/python") / nombre / "profiling" / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)

        p = Profiler(interval=0.001, async_mode="disabled")
        p.start()
        try:
            result = original(self)
        finally:
            p.stop()
            html = p.output(renderer=HTMLRenderer())
            (out_dir / "resolver.html").write_text(html, encoding="utf-8")

        return result

    cls.resolver = _perfilamiento
    return cls


def profile(
    name: Optional[str] = None,
    context: Optional[dict] = None,
    unique: bool = False,
) -> Callable:
    """
    Decorador para perfilar funciones a nivel de llamados y ejecuciones.

    Args:
        name: Nombre personalizado para el perfil
        context: Información adicional de contexto
        unique: Si es True, evita sobrescribir agregando sufijo numérico
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not gestor_perfilado.enabled:
                return func(*args, **kwargs)

            profile_name = name or func.__name__
            profile_context = {
                **(context or {}),
                "args": str(args),
                "kwargs": str(kwargs),
            }

            with ProfilerContext(
                gestor_perfilado,
                profile_name,
                profile_context,
                unique=unique,
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator
