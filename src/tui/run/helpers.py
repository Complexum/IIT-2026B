"""Persistencia y lógica de programas de ejecución.

Un programa vincula: dataset + patrón + estrategia.
Al ejecutarse, produce resultados que se guardan en data/output/.
"""

import datetime
import json
import logging
import re
import shlex
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PROGRAMAS_DIR = Path("data/input/programas")
_META_DIR = Path("data/output")

DEFAULT_MPI_TEMPLATE = (
    "mpiexec -n {n_procs} python -m mpi4py.futures -m src.cli run execution {nombre}"
)


def _natural_key(s: str) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


@dataclass
class Programa:
    """Configuración de un programa de ejecución."""

    nombre: str
    dataset: str = ""
    patron: str = ""
    estrategia: str = ""
    estado: str = "pendiente"  # pendiente | ejecutando | completado | error
    inicio: float = 0.0
    progreso: float = 0.0
    n_procs: int = 4
    exec_template: str = DEFAULT_MPI_TEMPLATE
    #: Opciones de la estrategia, ``{atributo: valor}`` (ver ``SIA.opciones``).
    #: Se validan al ejecutar. Los JSON viejos sin la clave usan {} por defecto.
    opciones: dict[str, str] = field(default_factory=dict)


# ── Sidecar metadata (checkpoint invalidation) ───────────


def etiqueta_estrategia(estrategia: str, opciones: dict[str, str] | None) -> str:
    """``'qsw'`` + ``{'modo': 'estatico'}`` → ``'qsw+modo=estatico'``.

    Sólo se anexan las opciones que difieren del default, así que las corridas
    normales conservan el nombre de siempre. La etiqueta entra en el nombre del
    CSV vía `build_output_stem`, de modo que dos corridas de la misma estrategia
    con opciones distintas producen archivos distintos y `parse_result_key` las
    trata como series separadas en las comparaciones y gráficas.
    """
    if not opciones:
        return estrategia
    try:
        from src.iit.strategies.python.sia import SIA
        from src.iit.strategies.runner import importar_estrategias

        importar_estrategias()
        defaults = SIA.registry[estrategia].defaults()
    except Exception:
        defaults = {}
    extra = [f"{k}={v}" for k, v in sorted(opciones.items()) if defaults.get(k) != v]
    return estrategia + ("+" + "+".join(extra) if extra else "")


def build_output_stem(nombre: str, dataset: str, patron: str, estrategia: str) -> str:
    """Construye la ruta relativa del resultado dentro de data/output/.

    Formato: nombre/dataset--estrategia--patron
    Ejemplo: program-01/N20A--analytic_bb--patron-1
    """
    return f"{nombre}/{dataset}--{estrategia}--{patron}"


def extract_program_name(stem: str) -> str:
    """Extrae el nombre del programa desde la ruta relativa del resultado.

    Formato: nombre/dataset--estrategia--patron → nombre
    Ejemplo: program-01/N20A--analytic_bb--patron-1 → program-01
    """
    return stem.split("/", 1)[0]


def cargar_meta_programa(stem: str) -> dict | None:
    """Lee el sidecar .meta.json; None si no existe o está corrupto."""
    path = _META_DIR / f"{stem}.meta.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def guardar_meta_programa(stem: str, prog: "Programa") -> None:
    """Escribe el sidecar .meta.json con los params actuales del programa."""
    path = _META_DIR / f"{stem}.meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset": prog.dataset,
        "patron": prog.patron,
        "estrategia": prog.estrategia,
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def guardar_meta_completo(
    stem: str,
    prog: "Programa",
    n_total: int,
    cancelado: bool,
    output_path: Path,
) -> None:
    """Reescribe el sidecar .meta.json con analíticas completas al terminar."""
    import polars as pl

    data: dict = {
        "dataset": prog.dataset,
        "patron": prog.patron,
        "estrategia": prog.estrategia,
        "n_total": n_total,
        "cancelado": cancelado,
        "timestamp_fin": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    try:
        df = (
            pl.read_csv(output_path, infer_schema_length=0)
            .select(["perdida", "tiempo_wall_s", "tiempo_cpu_s", "cpu_user_s", "cpu_sys_s", "mem_rss_mb", "gpu_mem_mb", "plataforma"])
            .with_columns(
                pl.col("perdida").cast(pl.Float64),
                pl.col("tiempo_wall_s").cast(pl.Float64),
                pl.col("tiempo_cpu_s").cast(pl.Float64),
                pl.col("cpu_user_s").cast(pl.Float64),
                pl.col("cpu_sys_s").cast(pl.Float64),
                pl.col("mem_rss_mb").cast(pl.Float64),
                pl.col("gpu_mem_mb").cast(pl.Float64),
            )
        )
        numeric = df.select(["perdida", "tiempo_wall_s", "tiempo_cpu_s", "cpu_user_s", "cpu_sys_s", "mem_rss_mb", "gpu_mem_mb"]).drop_nulls()
        n_comp = len(numeric)
        data["n_completados"] = n_comp
        if n_comp > 0:
            data["tiempo_wall_total_s"] = round(float(numeric["tiempo_wall_s"].sum()), 4)
            data["tiempo_wall_medio_s"] = round(float(numeric["tiempo_wall_s"].mean()), 4)
            data["tiempo_wall_min_s"] = round(float(numeric["tiempo_wall_s"].min()), 4)
            data["tiempo_wall_max_s"] = round(float(numeric["tiempo_wall_s"].max()), 4)
            data["tiempo_cpu_total_s"] = round(float(numeric["tiempo_cpu_s"].sum()), 4)
            data["tiempo_cpu_medio_s"] = round(float(numeric["tiempo_cpu_s"].mean()), 4)
            data["tiempo_cpu_min_s"] = round(float(numeric["tiempo_cpu_s"].min()), 4)
            data["tiempo_cpu_max_s"] = round(float(numeric["tiempo_cpu_s"].max()), 4)
            data["cpu_user_total_s"] = round(float(numeric["cpu_user_s"].sum()), 4)
            data["cpu_sys_total_s"] = round(float(numeric["cpu_sys_s"].sum()), 4)
            data["mem_rss_max_mb"] = round(float(numeric["mem_rss_mb"].max()), 3)
            data["gpu_mem_max_mb"] = round(float(numeric["gpu_mem_mb"].max()), 3)
            data["perdida_media"] = round(float(numeric["perdida"].mean()), 6)
            data["perdida_min"] = round(float(numeric["perdida"].min()), 6)
            data["perdida_max"] = round(float(numeric["perdida"].max()), 6)
        plat_col = df["plataforma"].drop_nulls()
        if len(plat_col) > 0:
            data["plataforma"] = plat_col[0]
    except Exception as exc:
        logging.warning(f"meta_completo: no se pudieron calcular stats: {exc}")
        data["n_completados"] = -1

    path = _META_DIR / f"{stem}.meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── CRUD ─────────────────────────────────────────────────


def guardar_programa(programa: Programa) -> None:
    """Guardar programa como JSON."""
    PROGRAMAS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = PROGRAMAS_DIR / f"{programa.nombre}.json"
    ruta.write_text(json.dumps(asdict(programa), indent=2, ensure_ascii=False))


def cargar_programa(nombre: str) -> Programa:
    """Cargar programa desde JSON con validación de estrategia."""
    ruta = PROGRAMAS_DIR / f"{nombre}.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))

    # Ignorar claves obsoletas/renombradas de versiones previas del esquema
    campos_validos = {f.name for f in fields(Programa)}
    datos = {k: v for k, v in datos.items() if k in campos_validos}

    programa = Programa(**datos)

    # Validar y migrar estrategia
    estrategia_original = programa.estrategia
    estrategias_validas = listar_estrategias()
    programa.estrategia = validar_estrategia(programa.estrategia, estrategias_validas)

    # Si hubo cambios, guardar y registrar
    if programa.estrategia != estrategia_original:
        if programa.estrategia:
            # Migración exitosa
            logging.info(
                f"Migrated strategy in program '{nombre}': '{estrategia_original}' → '{programa.estrategia}'"
            )
        else:
            # Estrategia inválida sin migración disponible
            logging.warning(
                f"Invalid strategy in program '{nombre}': '{estrategia_original}' reset to empty"
            )
        cambio = True
    else:
        cambio = False

    # Podar opciones que ya no aplican (típicamente porque cambió la estrategia).
    if programa.opciones:
        validas, descartadas = podar_opciones(programa.estrategia, programa.opciones)
        if descartadas:
            logging.warning(
                f"Program '{nombre}': descartadas opciones que "
                f"'{programa.estrategia or 'sin estrategia'}' no admite: "
                + ", ".join(f"{k}={v}" for k, v in sorted(descartadas.items()))
            )
            programa.opciones = validas
            cambio = True

    if cambio:
        guardar_programa(programa)

    return programa


def listar_programas() -> list[str]:
    """Listar nombres de programas disponibles."""
    if not PROGRAMAS_DIR.exists():
        return []
    return sorted(
        (f.stem for f in PROGRAMAS_DIR.glob("*.json")), key=_natural_key, reverse=True
    )


def eliminar_programa(nombre: str) -> bool:
    """Eliminar programa. Retorna True si existía."""
    ruta = PROGRAMAS_DIR / f"{nombre}.json"
    if ruta.exists():
        ruta.unlink()
        return True
    return False


def siguiente_nombre_programa() -> str:
    """Genera prog-01, prog-02, ..."""
    existentes = set(listar_programas())
    i = 0
    while f"program-{i:02d}" in existentes:
        i += 1
    return f"program-{i:02d}"


# ── Estrategias ──────────────────────────────────────────

# Mapeo de valores obsoletos a valores actuales
STRATEGY_MIGRATIONS = {
    "base": "basic",
}


def validar_estrategia(estrategia: str, estrategias_disponibles: list[str]) -> str:
    """Valida y migra valores de estrategia.

    Args:
        estrategia: Valor de estrategia del programa
        estrategias_disponibles: Lista de estrategias válidas

    Returns:
        Estrategia válida o string vacío si no se puede migrar
    """
    # Si el valor ya es válido, retornarlo tal cual
    if estrategia in estrategias_disponibles:
        return estrategia

    # Si está en el mapa de migraciones, retornar el valor migrado
    if estrategia in STRATEGY_MIGRATIONS:
        return STRATEGY_MIGRATIONS[estrategia]

    # Si no es válido ni migrable, retornar string vacío
    return ""


def podar_opciones(
    estrategia: str, opciones: dict[str, str] | None
) -> tuple[dict[str, str], dict[str, str]]:
    """Descarta las opciones que la estrategia actual no declara.

    Las opciones son **por estrategia** (`SIA.opciones`), pero se guardan en el
    programa. Al cambiarle la estrategia a una ejecución, las de la anterior
    quedaban pegadas y hacían fallar la nueva: un programa que fue `qsw` con
    `backend=c` reventaba con *"'queyranne' no admite la opción 'backend'"* al
    cambiarlo a cualquier estrategia que no tenga esa opción.

    Returns:
        `(validas, descartadas)`.
    """
    opciones = dict(opciones or {})
    if not opciones:
        return {}, {}

    from src.iit.strategies.python.sia import SIA
    from src.iit.strategies.runner import importar_estrategias

    importar_estrategias()
    cls = SIA.registry.get(estrategia)
    if cls is None:
        return opciones, {}  # estrategia desconocida: que se queje quien ejecute

    admisibles = cls.opciones
    validas, descartadas = {}, {}
    for attr, valor in opciones.items():
        destino = validas if (attr in admisibles and valor in admisibles[attr]) else descartadas
        destino[attr] = valor
    return validas, descartadas


def es_estrategia_mpi(estrategia: str) -> bool:
    """Estrategias que requieren mpiexec multi-proceso (no corren en hilo)."""
    return estrategia.endswith("_mpi")


def build_mpi_command(prog: "Programa") -> list[str]:
    """Construye el argv para subprocess.Popen a partir de prog.exec_template."""
    template = prog.exec_template or DEFAULT_MPI_TEMPLATE
    cmd_str = template.format(n_procs=prog.n_procs, nombre=prog.nombre)
    return shlex.split(cmd_str)


def listar_estrategias() -> list[str]:
    """Estrategias **ejecutables**, leídas del registro de `SIA`.

    Antes esto listaba nombres de carpeta, lo que dejaba entrar cosas que no se
    pueden correr: un `code.py` vacío (placeholder), un import roto, o una carpeta
    cuyo `nombre=` registrado no coincide con ella. Aparecían en el dropdown de la
    TUI y en `cli run`, y fallaban al ejecutar con "Estrategia desconocida".

    El registro es la fuente de verdad: `importar_estrategias()` recorre las mismas
    carpetas y `SIA.__init_subclass__` registra sólo lo que efectivamente se importó
    y declaró un nombre.
    """
    from src.iit.strategies.python.sia import SIA
    from src.iit.strategies.runner import importar_estrategias

    importar_estrategias()
    registradas = set(SIA.registry)

    ruta = Path("src/iit/strategies/python")
    if ruta.exists():
        for d in sorted(ruta.iterdir()):
            if not d.is_dir() or d.name.startswith("__"):
                continue
            if not (d / "code.py").exists():
                logging.warning(f"Strategy directory '{d.name}' excluded: missing code.py")
            elif d.name not in registradas:
                logging.warning(
                    f"Strategy directory '{d.name}' excluded: no registra ninguna "
                    f"estrategia (¿code.py vacío o import roto?)"
                )

    return sorted(registradas)


# ── Estimación de tiempo ─────────────────────────────────


def estimar_tiempo(n_dims: int) -> float:
    """Estima tiempo de ejecución en segundos.

    Aproximación exponencial basada en datos empíricos:
      n=10 → ~10s, n=20 → ~100s, n=25 → ~10^6s
    Fórmula: t = 10^(1 + (n-10)/3)
    """
    if n_dims < 1:
        return 0.0
    return 10 ** (1 + (n_dims - 10) / 3)


def formatear_duracion(segundos: float) -> str:
    """Formatea segundos a dd:hh:mm:ss legible."""
    if segundos <= 0:
        return "--:--:--"
    dias = int(segundos // 86400)
    horas = int((segundos % 86400) // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = int(segundos % 60)
    if dias > 0:
        return f"{dias}d {horas:02d}:{minutos:02d}:{segs:02d}"
    return f"{horas:02d}:{minutos:02d}:{segs:02d}"
