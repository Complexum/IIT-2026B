# Guía de Ejecución — MIP-IIT Solver

## Instalación y Setup

```bash
# Asegúrate de estar en el directorio raíz del proyecto
cd /path/to/IIT-2026A

# Instalar dependencias con uv
uv sync

# O con venv directamente
source .venv/bin/activate
```

---

## Ejecutar la TUI (Interfaz principal)

```bash
# Recomendado
uv run tui

# O directamente
uv run python -m src.tui.app
```

**Tabs y atajos de teclado:**

| Tecla | Tab | Función |
|---|---|---|
| `d` | Dataset | Cargar redes, previsualizar sistema |
| `t` | Testing | Crear y gestionar patrones |
| `e` | Execution | Crear y ejecutar programas |
| `r` | Results | Ver resultados en tabla |
| `q` | — | Salir |

---

## Ejecutar demo básico

```bash
uv run python main.py
```

Carga N5A con la estrategia `basic` y muestra la solución en terminal.

---

## Ejecutar Tests

```bash
# Todos los tests
uv run pytest tests/ -v

# Solo ejecución end-to-end
uv run pytest tests/execution/ -v

# Solo core (NCube, System)
uv run pytest tests/core/ -v
```

**Estructura de tests:**
```
tests/
├── core/
│   ├── test_ncube.py        # Operaciones NCube: condicionar, marginalizar, indexación
│   └── test_system.py       # Operaciones System: condicionar, substraer, bipartir
├── dataset/
│   └── test_io.py           # I/O de redes
├── execution/
│   └── test_ejecutar.py     # ejecutar() sin TUI; generar_combinaciones; Params
└── tui/
    └── test_program_card_safety.py  # ProgramCard con estrategias inválidas
```

---

## Crear una nueva estrategia

Solo un archivo necesario:

```python
# src/iit/strategies/python/mi_estrategia/code.py
from src.iit.strategies.python.sia import SIA
from src.iit.core.solution import Solution

class MiEstrategia(SIA, nombre="mi_estrategia"):
    def resolver(self) -> Solution:
        dm = self.distribucion       # distribución marginal del subsistema
        sistema = self.sistema       # System listo para bipartir
        # ... lógica del algoritmo
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

Aparece automáticamente en el dropdown de la TUI y en `ejecutar()`.

Para habilitar profiling en la estrategia:

```python
from src.infra.profile import perfilar

@perfilar
class MiEstrategia(SIA, nombre="mi_estrategia"):
    ...
```

---

## Generar una nueva red

```bash
uv run python -c "
from src.io.generator import generar_red, peso_estimado
dims = 5
print(f'Tamaño estimado: {peso_estimado(dims) * 1024:.2f} MB')
nombre = generar_red(dims, datos_deterministas=True)
print(f'Red generada: {nombre}')
"
```

---

## Uso programático

```python
from src.io.manager import cargar_mpt, listar_redes, reducir_a_subsistema
from src.iit.core.params import Params
from src.iit.strategies.runner import ejecutar

# Listar redes disponibles (orden natural)
redes = listar_redes()  # ['N2A', 'N3A', 'N4A', 'N5A', 'N7A', 'N15A']

# Cargar TPM
tpm = cargar_mpt("N3A")  # numpy array (2^n, n)

# Definir parámetros (strings binarios, misma longitud = n dims)
params = Params(
    estado="100",      # estado inicial
    condicion="111",   # 1=mantener nodo, 0=condicionar
    alcance="110",     # 1=mantener en alcance, 0=substraer
    mecanismo="011",   # 1=mantener en mecanismo, 0=marginalizar
)

# Ejecutar estrategia
sol = ejecutar(tpm, params, "basic")
print(f"φ = {sol.perdida:.4f}")
print(f"Partición:\n{sol.particion}")
```

---

## Profiling

Los perfiles HTML se generan automáticamente si la estrategia tiene `@perfilar`:

```bash
# Ver perfiles generados
ls review/profiling/

# Abrir en browser
open review/profiling/basic/15_05_2026/01hrs/resolver.html
```

Controlar desde código:
```python
from src.iit.base.app import aplicacion
aplicacion.desactivar_profiling()  # deshabilitar
aplicacion.activar_profiling()     # habilitar (default)
```

---

## Logs

Los errores del worker de ejecución se guardan en:
```
.logs/{DD_MM_YYYY}/{HH}hrs/execution.log
```

```bash
# Ver log más reciente
ls -t .logs/**/**/*.log | head -1 | xargs cat
```

---

## Troubleshooting

### Error: `ModuleNotFoundError`
```bash
uv sync
```

### No aparecen redes en la TUI
```bash
ls data/input/networks/
# Si está vacío:
uv run python -c "from src.io.generator import generar_red; generar_red(3)"
```

### Estrategia no aparece en dropdown
Verificar que existe `src/iit/strategies/python/{nombre}/code.py` y que la clase declara `nombre=`:
```python
class MiEstrategia(SIA, nombre="mi_estrategia"):  # ← requerido
```

### Ejecución lenta / quiero reanudar
Si la TUI se cierra a mitad de una ejecución, el CSV parcial se preserva.
Click "Empezar" de nuevo — el sistema detecta los índices ya calculados y continúa desde donde se quedó.

### Ver estructura del proyecto
```bash
tree -I '.venv|__pycache__|target|*.pyc|.git' src/ --dirsfirst
```