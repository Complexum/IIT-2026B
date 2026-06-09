# Arquitectura — MIP-IIT Solver

Sistema para calcular la Partición Mínima de Información (MIP) sobre redes dinámicas, implementando Teoría de Información Integrada (IIT 4.0). El resultado central es φ (phi): la pérdida mínima al bipartir un subsistema.

---

## Pipeline central

```
CSV (TPM)
  └─→ System (NCubes)
        └─→ condicionar(dims) → Candidato
              └─→ substraer(alcance, mecanismo) → Subsistema
                    └─→ SIA.resolver() → Solution (φ, partición)
                          └─→ CSV de resultados
```

---

## Capa de datos — `src/iit/core/`

### `NCube`
Hipercubo N-dimensional de probabilidades, almacenado flat como array `(2^k,)`.
- `condicionar(indices, estado)` — fija dimensiones a valores del estado (reduce dims)
- `marginalizar(ejes)` — promedia sobre ejes (colapsa dims)
- `__getitem__(estado)` — probabilidad en un estado binario concreto

### `System`
Colección de NCubes representando el sistema completo, candidato o subsistema.
- `condicionar(dims)` → Candidato (elimina nodos)
- `substraer(alcance, mecanismo)` → Subsistema (marginaliza en alcance y mecanismo)
- `bipartir(alcance, mecanismo)` → Sistema con la bipartición aplicada
- `distribucion_marginal()` → distribución de probabilidad del sistema

### `Params`
Dataclass frozen con cuatro strings binarios de igual longitud:

| Campo | Descripción |
|---|---|
| `estado` | Estado inicial del sistema (e.g. `"100"`) |
| `condicion` | `0` = condicionar ese nodo, `1` = mantener |
| `alcance` | `0` = sacar del alcance (substraer), `1` = mantener |
| `mecanismo` | `0` = marginalizar en mecanismo, `1` = mantener |

### `Solution`
Resultado de una estrategia:
- `perdida` — valor φ (small-phi)
- `particion` — string formateado con brackets `⎛ A ⎞⎛ B,C ⎞\n⎝ ∅ ⎠⎝ B,C ⎠`
- `distribucion_subsistema`, `distribucion_particion` — arrays numpy
- `tiempo_ejecucion` — segundos del algoritmo (sin preparación del subsistema)

---

## Sistema de estrategias — `src/iit/strategies/python/`

### Auto-registro vía `SIA.__init_subclass__`

```python
class SIA(ABC):
    registry:      ClassVar[dict[str, type["SIA"]]] = {}
    necesita_mpt:  ClassVar[dict[str, bool]]         = {}

    def __init_subclass__(cls, nombre="", necesita_mpt=False, **kw):
        if nombre:
            cls.nombre = nombre
            SIA.registry[nombre]     = cls
            SIA.necesita_mpt[nombre] = necesita_mpt
```

**Crear una nueva estrategia** — solo un archivo:

```python
# src/iit/strategies/python/mi_estrategia/code.py
from src.iit.strategies.python.sia import SIA
from src.iit.core.solution import Solution

class MiEstrategia(SIA, nombre="mi_estrategia"):
    def resolver(self) -> Solution:
        dm = self.distribucion          # distribución marginal del subsistema
        sistema = self.sistema          # System listo para bipartir
        ...
        return Solution(
            estrategia=self.nombre.capitalize(),
            perdida=...,
            distribucion_subsistema=dm,
            distribucion_particion=...,
            particion=...,
            tiempo_total=...,
            quiere_hablar=False,
        )
```

Aparece automáticamente en el dropdown de la TUI y en `ejecutar()` — sin tocar `runner.py` ni ningún registry manual.

### Estrategias implementadas

| Nombre | Clase | Descripción |
|---|---|---|
| `basic` | `Basic` | Greedy: prueba biparticiones de 1 nodo, 1 dim, y pares |
| `phi` | `Phi` | Wrapper de pyphi (fuerza bruta); requiere `needs_tpm=True` |

### Profiling — `@perfilar`

```python
from src.infra.profile import perfilar

@perfilar
class MiEstrategia(SIA, nombre="mi_estrategia"):
    ...
```

Genera `review/profiling/{nombre}/{DD_MM_YYYY}/{HH}hrs/resolver.html` al ejecutar. Controlado por `aplicacion.profiler_habilitado` (default: `True`).

### Formateo — `fmt_particion(*Parte)` en `func.py`

Formato compartido para k-particiones. Cada `Parte(numerador, denominador)` produce una columna de brackets:

```python
from src.iit.strategies.python.func import Parte, fmt_particion

fmt_particion(
    Parte("A", "∅"),
    Parte("B,C", "A,B,C"),
)
# → "⎛ A ⎞⎛  B,C  ⎞\n⎝ ∅ ⎠⎝ A,B,C ⎠"
```

---

## I/O — `src/io/`

| Directorio | Formato | Función |
|---|---|---|
| `data/input/networks/` | CSV `(2^n, n)` | TPMs — una columna por nodo |
| `data/input/patrones/` | JSON | Generadores por categoría (estado, condicion, alcance, mecanismo) |
| `data/input/programas/` | JSON | Config de programa (dataset + patrón + estrategia) |
| `data/output/` | CSV `(9 cols)` | Resultados: indice, estado, condicion, alcance, mecanismo, perdida, tiempo, particion, plataforma |

Funciones clave:
```python
from src.io.manager import cargar_mpt, listar_redes, reducir_a_subsistema
from src.io.generator import generar_red, peso_estimado
```

---

## TUI — `src/tui/`

Aplicación Textual con 4 tabs, acceso por teclado `d / t / e / r`.

### Dataset (`d`)
- Lista redes disponibles (orden natural: N2A, N3A, … N15A)
- Seleccionar red muestra en tiempo real: Sistema Completo → Candidato → Subsistema
- Advertencia modal para redes ≥ 10 dims (carga puede tardar minutos)
- Crear nueva red con N dims (discreta o continua)

### Testing (`t`)
- Lista patrones (más reciente primero)
- Crear patrón: 4 categorías con generadores seleccionables (`todos`, `inicial`, `pares`, `impares`, `mult_3`, …)
- Preview de todas las combinaciones para N dims dado

### Execution (`e`)
- Lista programas (más reciente primero)
- Cada programa vincula: dataset + patrón + estrategia
- Click "Empezar" → worker en hilo de fondo
  - Progreso real por combinación
  - Label muestra combinación actual: `Test 3/48: (100 | 111 | 110 | 011)  →  (A | B,C)`
  - Checkpoint/resume automático (ver abajo)
- Al terminar: Results tab se refresca automáticamente

### Results (`r`)
- Lista resultados (más reciente primero)
- Tabla Rich sin líneas, zebra stripes, scroll horizontal
- Título muestra `Resultado: program-01  [basic]`
- Columna `tiempo` formateada a 4 decimales en segundos

---

## Checkpoint/Resume — ejecuciones largas

Redes de ≥10 dims pueden tardar horas. El sistema persiste el progreso por fila:

```
Al calcular cada combinación:
  1. Calcular Solution
  2. writer.writerow([...])   ← escribe al buffer
  3. f.flush()                ← vacía buffer al OS (sin cerrar el handle)

Al re-ejecutar (click "Empezar" de nuevo):
  1. Leer CSV existente → set{indices completados}  ← una sola lectura O(n)
  2. Abrir CSV en modo append
  3. Loop: si i in completados → skip (O(1))
  4. Continuar desde donde se quedó
```

Filas fallidas (excepción) no se escriben → se reintentarán en la siguiente ejecución.

---

## Logging

| Destino | Contenido |
|---|---|
| `.logs/{DD_MM_YYYY}/{HH}hrs/execution.log` | Errores del worker de ejecución (SafeLogger) |
| `pyphi.log` | Logs de la librería pyphi |
| `review/profiling/{estrategia}/{fecha}/resolver.html` | Perfiles de pyinstrument |

---

## Paralelismo en estrategias

`resolver()` es una caja negra para el TUI. Cualquier técnica de paralelismo funciona:

| Técnica | Nota |
|---|---|
| `ThreadPoolExecutor` | numpy libera GIL — útil para operaciones matriciales |
| `ProcessPoolExecutor` | Procesos separados, sin conflicto con Textual |
| `multiprocessing` | En macOS ≥ 3.12: usar `mp.get_context("spawn")` (no `fork`) |
| numba / CUDA | Transparente |

---

## Tests — `tests/`

```
tests/
├── core/
│   ├── test_ncube.py        # NCube: condicionar, marginalizar, indexación
│   └── test_system.py       # System: condicionar, substraer, bipartir, distribucion
├── dataset/
│   └── test_io.py           # I/O de redes
├── execution/
│   └── test_ejecutar.py     # ejecutar() end-to-end sin TUI; generar_combinaciones; Params
└── tui/
    └── test_program_card_safety.py  # ProgramCard con estrategias inválidas/vacías
```

```bash
pytest tests/ -v
```