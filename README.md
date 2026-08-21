# IIT-2026B

Base de proyecto para el cálculo de la **Mínima Partición Informativa (MIP)** bajo la Teoría de la Información Integrada. Incluye TUI, CLI, persistencia de datasets/ejecuciones y un conjunto de estrategias intercambiables.

Cada equipo trabaja sus propias estrategias sobre esta base sin modificar el núcleo: una estrategia nueva es una carpeta con un `code.py` que se auto-registra.

---

## Requisitos

- Python **3.12** o superior.
- [`uv`](https://docs.astral.sh/uv/) como gestor de entorno y dependencias.
- Terminal con soporte de colores (PowerShell, Bash o Zsh).

## Instalación

```bash
git clone https://github.com/Complexum/IIT-2026B.git
cd IIT-2026B
uv sync
```

`uv sync` crea el `.venv`, resuelve `pyproject.toml` y deja disponibles los comandos `tui`, `cli` y `strat`.

Extras opcionales para ejecución en clúster:

```bash
uv sync --extra mpi     # backends qn_mpi / analytic_mpi (mpi4py)
uv sync --extra cuda    # backends qn_cuda / analytic_cuda (cupy)
uv sync --extra dev     # pytest y pytest-cov
```

> En VS Code el `.venv` se activa solo en la terminal integrada (`.vscode/settings.json`), así que los comandos se invocan sin el prefijo `uv run`.

---

## Ejecución

```bash
uv run tui              # interfaz Textual (recomendado)
uv run cli list datasets
uv run strat mi_algoritmo
```

Equivalentes sin los scripts instalados:

```bash
python -m src.tui.app
python -m src.cli list datasets
```

### TUI

Atajos de pestaña: `d` Dataset · `t` Testing · `e` Execution · `r` Results · `a` Analysis · `q` Quit.

La TUI arranca con **auto-reload**: vigila los `.py` bajo `src/` y reinicia el proceso al guardar. Los estilos de `src/tui/styles.scss` se recargan sin reiniciar.

```bash
uv run tui --no-watch   # desactivar auto-reload
```

### Demo mínima

`main.py` carga una TPM, arma el subsistema desde unos `Params` fijos y ejecuta una estrategia. Sirve como ejemplo del flujo `tpm + params → subsistema → Solution`; los valores se editan dentro del archivo.

```bash
uv run python main.py
```

---

## CLI

Opera todo el flujo desde la terminal, sin TUI.

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| `list` | Listar recursos (`datasets`, `patterns`, `executions`, `strategies`) | `cli list strategies` |
| `show` | Detalle de un recurso | `cli show dataset N15A` |
| `new` | Crear dataset, execution o patrón | `cli new dataset 5 --discretos` |
| `edit` | Modificar un execution | `cli edit execution program-01 --estrategia phi` |
| `run` | Ejecutar un execution | `cli run execution program-01` |
| `results` | Consultar resultados con SQL | `cli results program-01 "SELECT estado, perdida FROM self LIMIT 10"` |
| `compare` | Comparar resultados de varias estrategias (`--plot` genera las gráficas) | `cli compare program-01 program-02 --plot` |
| `delete` | Eliminar un recurso | `cli delete execution program-01` |

Flag global: `-v, --verbose` (en `run` muestra cada combinación resuelta).

`new`, `edit` y `run` aceptan `--opcion ATTR=VALOR` (repetible) para configurar los atributos
que la estrategia declare en `SIA.opciones` — ver [Opciones de estrategia](#opciones-de-estrategia).

Flujo completo:

```bash
# 1. Crear dataset y execution
cli new dataset 4 --discretos
cli new execution exec-x --dataset N4A --patron patron-2 --estrategia analytic

# 2. Verificar y ejecutar
cli show execution exec-x
cli run execution exec-x               # reanuda desde el checkpoint si existe
cli run execution exec-x --no-resume   # reinicia desde cero

# 3. Consultar y comparar resultados
cli results exec-x
cli results exec-x "SELECT estado, perdida, tiempo FROM self WHERE perdida > 0.5 ORDER BY tiempo DESC LIMIT 10"
cli results exec-x "SELECT estado, COUNT(*) AS total, AVG(perdida) FROM self GROUP BY estado"

# 4. Comparar contra otra estrategia y graficar
cli compare exec-x exec-y                     # tabla: n_ok/n, %, max_diff, mean_diff
cli compare exec-x exec-y --plot --ref analytic  # + página HTML interactiva (plotly)
cli compare --all --dataset N15A --plot       # agrupa por (dataset, patrón)
cli compare exec-x exec-y --paper             # figuras matplotlib en src/pics/

# 5. Limpiar
cli list executions
cli delete execution exec-x
```

`compare` sólo enfrenta resultados del mismo `(dataset, patrón)`; los agrupa solo. Flags:
`--tol` (default `1e-4`), `--ref <estrategia>`, `--plot`, `--paper`, `--no-open`.

### Qué mide el CSV

`tiempo_wall_s` y las columnas de recursos (`tiempo_cpu_s`, `cpu_user_s`, `cpu_sys_s`,
`mem_rss_mb`) miden **sólo el algoritmo**: el monitor arranca después de reducir el sistema al
subsistema. Esa preparación va aparte, en `tiempo_preparacion_s`.

La razón es que la preparación cuesta lo mismo para todas las estrategias, así que incluirla actúa
como una constante compartida que comprime todos los speedups hacia 1. Medido sobre N20A +
`patron-2`: `qsw+backend=c` aparecía **1.55×** más rápido que `analytic` cuando el algoritmo va
**3.25×** — más de la mitad de la ventaja quedaba escondida.

> Los CSV generados antes de este cambio no traen `tiempo_preparacion_s`, y en ellos `tiempo_wall_s`
> sí incluye la preparación. `cli compare` avisa cuando se mezclan: los `perdida` siguen siendo
> comparables, los tiempos no. Re-ejecutar con `--no-resume` homogeneiza.

CLI y TUI comparten la misma capa de persistencia (JSON en `data/input/`, CSV en `data/output/`), así que se pueden alternar libremente.

---

## Estrategias

Las estrategias viven en `src/iit/strategies/python/<nombre>/code.py` y se registran solas al heredar de `SIA`. Disponibles hoy:

| Estrategia | Notas |
|------------|-------|
| `analytic`, `analytic_concurrent`, `analytic_mul`, `analytic_mpi`, `analytic_cuda` | Solución analítica y sus backends paralelos (threads, multiprocessing, MPI, CUDA) |
| `analytical`, `analytical_concurrent` | Variantes previas de la analítica |
| `qn`, `qn_mul`, `qn_mpi`, `qn_cuda`, `queyranne` | Familia Queyranne y sus backends paralelos |
| `qsw` | Queyranne × Stoer-Wagner: MAO con keys incrementales sobre el oráculo Zeta. Opciones `modo` (`exacto`/`estatico`/`estocastico`), `backend` (`python`/`c`) y `k` |
| `qsw_mul`, `qsw_cuda` | Mismo algoritmo con el precómputo Zeta paralelizado (multiprocessing sobre las filas / kernel CUDA). Resultado idéntico a `qsw`; `qsw_mul` agrega la opción `workers` |
| `force` | Fuerza bruta |
| `phi` | pyphi como referencia (requiere la TPM completa) |

`cli list strategies` imprime el listado vigente junto con cuáles necesitan la TPM completa.

### Opciones de estrategia

Una estrategia puede declarar atributos configurables sin crear una carpeta por variante:

```python
class MiEstrategia(SIA, nombre="mi_algoritmo"):
    opciones = {"modo": ("rapido", "preciso"), "backend": ("python", "c")}
    modo: str = "rapido"       # el primer valor de cada tupla es el default
    backend: str = "python"
```

Se setean por CLI, y `SIA.validar_opciones` las rechaza **antes de arrancar** si el atributo no
está declarado o el valor no es admisible:

```bash
cli edit execution exec-x --opcion modo=preciso
cli run  execution exec-x --opcion backend=c     # override puntual
cli edit execution exec-x --opcion modo=          # borrar una opción
```

Las opciones que difieren del default entran en el nombre del CSV
(`N15A--qsw+modo=estatico--patron-2.csv`), así que dos corridas de la misma estrategia con
opciones distintas quedan como series separadas en `cli compare` y en las gráficas.

Antes de usar `--opcion backend=c` hay que compilar el kernel:

```bash
./src/iit/strategies/clang/build.sh
```

Si falta la librería, la ejecución **aborta antes de empezar** con el comando a correr, en vez de
fallar una vez por combinación.

> Selección desde el tab Execution de la TUI: pendiente. Hoy las opciones se configuran por CLI y
> quedan persistidas en el JSON del execution, listas para que la UI las lea.

### Crear una estrategia

```bash
uv run strat mi_algoritmo          # estrategia simple
uv run strat mi_fuerza_bruta --tpm # recibe además la TPM completa (como phi)
```

Genera:

```
src/iit/strategies/python/mi_algoritmo/
├── __init__.py
└── code.py        ← listo para implementar resolver()
```

El `code.py` trae la clase con `@perfilar`, la herencia de `SIA` y el esqueleto de `resolver()`. **La estrategia aparece sola en el dropdown de la TUI y en `ejecutar()`; no hay que registrar nada más.** Solo queda el algoritmo:

```python
def resolver(self) -> Solution:
    dm_original = self.distribucion   # distribución marginal del subsistema
    sistema = self.sistema            # System listo para bipartir

    # Calcular la bipartición de mínima pérdida...
    mejor_perdida = ...
    mejor_dist = ...
    particion_str = fmt_particion(Parte(...), Parte(...))

    return Solution(
        estrategia=self.nombre.capitalize(),
        perdida=mejor_perdida,
        distribucion_subsistema=dm_original,
        distribucion_particion=mejor_dist,
        particion=particion_str.strip(),
        tiempo_total=...,
        quiere_hablar=False,
    )
```

---

## Estructura

```
src/
├── cli/          comandos de terminal (list, show, new, edit, run, results, delete)
├── tui/          interfaz Textual: dataset, test, run, results, analysis
├── iit/          núcleo: core (params, solution), strategies (python/, runner, scaffolder)
├── io/           carga de TPMs, subsistemas y persistencia
├── infra/        middlewares (perfilado) y utilidades
└── paper/        generación de figuras y tablas
data/
├── input/        networks (CSV), patrones y programas (JSON)
└── output/       resultados por programa (CSV)
iit/              puerto experimental del núcleo en Rust
benchmarks/       mediciones de rendimiento por estrategia
tests/            suite pytest
```

## Tests

```bash
uv run pytest tests/     # o ./run_tests.sh
```

## Documentación

- [`.docs/architecture/ARCHITECTURE.md`](.docs/architecture/ARCHITECTURE.md) — arquitectura del sistema.
- [`.docs/architecture/Parallelization.md`](.docs/architecture/Parallelization.md) y `Parallelization_Results.md` — backends paralelos y mediciones.
- [`.docs/planning/EXECUTION.md`](.docs/planning/EXECUTION.md) — flujo de ejecución.

---

## Trabajo por equipos

Cada grupo desarrolla sobre su propio **fork**, desmarcando *"Copy the `main` branch only"* para conservar las demás ramas. Tras clonar el fork, se asocia el repositorio original como `upstream` para recibir actualizaciones:

```bash
git clone https://github.com/<grupo-usuario>/<fork>.git
cd <fork>
git remote add upstream https://github.com/Complexum/IIT-2026B.git
git fetch upstream
```

El desarrollo del equipo va en su rama (`dev`), y `git pull upstream main` trae los cambios de la base sin perder el trabajo propio.
