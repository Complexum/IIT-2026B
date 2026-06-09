# Design Document

## Overview

Este diseño aborda la desincronización entre la GUI (TUI) y el sistema de estrategias de IIT. El problema principal es que los programas guardados pueden contener referencias a estrategias obsoletas o inválidas (como "base"), causando que el Select widget de Textual falle al inicializar.

La solución implementa validación y migración de valores de estrategia en el momento de carga, asegurando que la GUI siempre trabaje con valores válidos que correspondan a estrategias realmente disponibles en el sistema.

## Architecture

El sistema tiene tres capas principales que interactúan:

1. **Capa de Persistencia** (`src/tui/run/helpers.py`): Maneja la carga/guardado de programas en JSON
2. **Capa de Descubrimiento** (`src/tui/run/helpers.py::listar_estrategias`): Descubre estrategias disponibles escaneando el filesystem
3. **Capa de Presentación** (`src/tui/run/widgets.py::ProgramCard`): Renderiza los selectores en la GUI

El flujo actual es:
```
Cargar JSON → Crear ProgramCard → Inicializar Select → ERROR (valor inválido)
```

El flujo corregido será:
```
Cargar JSON → Validar/Migrar estrategia → Crear ProgramCard → Inicializar Select → OK
```

## Components and Interfaces

### 1. Strategy Validator

Nuevo componente que valida y migra valores de estrategia.

```python
def validar_estrategia(estrategia: str, estrategias_disponibles: list[str]) -> str:
    """Valida y migra valores de estrategia.
    
    Args:
        estrategia: Valor de estrategia del programa
        estrategias_disponibles: Lista de estrategias válidas
        
    Returns:
        Estrategia válida o string vacío si no se puede migrar
    """
```

### 2. Migration Map

Mapeo de valores obsoletos a valores actuales:

```python
STRATEGY_MIGRATIONS = {
    "base": "basic",
    # Agregar más migraciones según sea necesario
}
```

### 3. Modified Program Loading

La función `cargar_programa` se modificará para aplicar validación:

```python
def cargar_programa(nombre: str) -> Programa:
    """Cargar programa desde JSON con validación de estrategia."""
    ruta = PROGRAMAS_DIR / f"{nombre}.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    programa = Programa(**datos)
    
    # Validar y migrar estrategia
    estrategias_validas = listar_estrategias()
    programa.estrategia = validar_estrategia(programa.estrategia, estrategias_validas)
    
    # Si hubo cambios, guardar
    if programa.estrategia != datos.get("estrategia"):
        guardar_programa(programa)
    
    return programa
```

### 4. Safe Widget Initialization

El ProgramCard se modificará para manejar valores vacíos de forma segura:

```python
# En ProgramCard.compose()
yield Select(
    options=[(n, n) for n in listar_estrategias()],
    prompt="Estrategia",
    value=p.estrategia if p.estrategia and p.estrategia in [e for e in listar_estrategias()] else Select.BLANK,
    classes="card-select",
    id=f"sel-str-{p.nombre}",
)
```

## Data Models

### Programa (existente, sin cambios)

```python
@dataclass
class Programa:
    nombre: str
    dataset: str = ""
    patron: str = ""
    estrategia: str = ""  # Ahora siempre será válida o vacía
    estado: str = "pendiente"
    inicio: float = 0.0
    progreso: float = 0.0
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Widget initialization is safe for all strategy values

*For any* program with any strategy value (valid, invalid, or empty), creating a ProgramCard widget should complete without raising an exception, and invalid strategies should display as blank in the selector.

**Validates: Requirements 1.1, 1.3**

### Property 2: Invalid strategies are handled gracefully

*For any* program with an unrecognized strategy value that is not in the migration map, the system should reset the strategy to empty string and continue loading without crashing.

**Validates: Requirements 1.2, 2.2**

### Property 3: Data integrity is preserved during validation

*For any* program with an invalid strategy, all other configuration values (dataset, patron, estado, progreso) should remain unchanged after validation.

**Validates: Requirements 1.4**

### Property 4: Strategy migration is persistent and idempotent

*For any* program that undergoes migration, the migrated value should be saved to disk immediately, and applying migration again should produce the same result.

**Validates: Requirements 2.3, 2.4**

### Property 5: Strategy discovery returns only valid strategies

*For any* strategy returned by `listar_estrategias()`, the path `src/iit/strategies/python/{strategy}/code.py` must exist, and the returned list must be sorted.

**Validates: Requirements 3.1, 3.2**

### Property 6: Time estimation format is consistent

*For any* dataset with N dimensions, the displayed time estimation should match the format "Est: HH:MM:SS (N dims)" where HH:MM:SS is properly formatted.

**Validates: Requirements 4.2**

### Property 7: Error handling never crashes the system

*For any* error during program loading (malformed JSON, missing files, invalid data), the system should log the error and continue, never raising an unhandled exception that crashes the TUI.

**Validates: Requirements 4.3, 5.1, 5.2, 5.3**

## Error Handling

### Strategy Validation Errors

- **Invalid strategy value**: Log warning, set to empty string, continue
- **Migration applied**: Log info message indicating old → new value
- **Strategy directory missing**: Exclude from available list, log warning

### Program Loading Errors

- **Malformed JSON**: Log error with filename, skip program, continue loading others
- **Missing required fields**: Use dataclass defaults, log warning
- **File read error**: Log error, skip program

### Widget Initialization Errors

- **Strategy not in options**: Use Select.BLANK instead of crashing
- **Time estimation fails**: Silently continue, don't update display
- **Dataset load fails**: Catch exception, log, continue

## Testing Strategy

### Unit Tests

Unit tests will verify specific examples and edge cases:

- Test migration of "base" → "basic"
- Test handling of completely invalid strategy names
- Test empty strategy values
- Test program loading with missing files
- Test widget initialization with various strategy states

### Property-Based Tests

Property-based tests will verify universal properties using Hypothesis:

- Generate random strategy names and verify validation behavior
- Generate random program configurations and verify no crashes
- Generate random filesystem states and verify discovery consistency
- Test idempotence of migration by applying it multiple times

The property-based testing library for Python will be **Hypothesis**. Each property test should run a minimum of 100 iterations to ensure thorough coverage of the input space.

Each property-based test will be tagged with a comment explicitly referencing the correctness property in this design document using the format: **Feature: strategy-gui-sync, Property {number}: {property_text}**
