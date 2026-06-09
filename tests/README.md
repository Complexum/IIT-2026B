# Tests

## Instalación

```bash
# Añadir pytest como dependencia de desarrollo
uv add --dev pytest pytest-cov

# O si prefieres instalarlo manualmente
uv pip install pytest pytest-cov
```

## Ejecutar Tests

### Todos los tests
```bash
uv run pytest tests/ -v
```

### Tests específicos por categoría
```bash
# Solo tests de IO (dataset)
uv run pytest tests/dataset/ -v

# Solo tests de core (ncube, system)
uv run pytest tests/core/ -v
```

### Test específico
```bash
uv run pytest tests/dataset/test_io.py::TestLoadTPM::test_load_n3a_has_correct_shape -v
```

### Con cobertura
```bash
uv run pytest tests/ --cov=src --cov-report=term-missing
```

### Modo watch (si tienes pytest-watch)
```bash
uv run pytest-watch tests/
```

## Estructura

```
tests/
├── dataset/
│   └── test_io.py          # Tests de IO: list, load, build, delete, generate
├── core/
│   ├── test_ncube.py       # Tests de NCube: condicionar, marginalizar, value_at
│   └── test_system.py      # Tests de System: condicionar, substraer, bipartir
└── README.md               # Este archivo
```

## Tests Disponibles

### `tests/dataset/test_io.py`
- `TestListNetworks`: Verifica que `list_networks()` devuelve lista con N1A, N2A, N3A
- `TestLoadTPM`: Verifica que `load_tpm()` carga CSVs con shape correcto
- `TestBuildSystem`: Verifica que `build_system()` crea System con NCubes correctos
- `TestGenerator`: Verifica que `generar_red()` crea archivos CSV
- `TestDeleteNetwork`: Verifica que `delete_network()` elimina archivos
- `TestResults`: Verifica carga de resultados desde `data/output/`

### `tests/core/test_ncube.py`
- `TestNCubeOperations`: Verifica operaciones básicas de NCube
  - Creación
  - `pos_of_dim()`
  - `value_at()`
  - `condicionar()`
  - `marginalizar()`

### `tests/core/test_system.py`
- `TestSystemOperations`: Verifica operaciones de System
  - Creación desde TPM
  - `condicionar()`
  - `substraer()`
  - `distribucion_marginal()`
  - `bipartir()`
