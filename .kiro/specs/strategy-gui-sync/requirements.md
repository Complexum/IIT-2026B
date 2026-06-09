# Requirements Document

## Introduction

Este documento especifica los requisitos para corregir la desincronización entre la interfaz gráfica (TUI) y el sistema de estrategias de IIT. El problema actual es que los programas guardados pueden contener referencias a estrategias que ya no existen o que nunca fueron válidas, causando errores al cargar la GUI.

## Glossary

- **TUI**: Terminal User Interface, la interfaz de usuario basada en terminal usando Textual
- **Programa**: Configuración que vincula un dataset, patrón y estrategia para ejecutar análisis IIT
- **Estrategia**: Implementación específica de algoritmos IIT ubicada en `src/iit/strategies/python/`
- **Select Widget**: Componente de Textual que permite seleccionar valores de una lista
- **Strategy Discovery**: Proceso de descubrir dinámicamente las estrategias disponibles escaneando el sistema de archivos

## Requirements

### Requirement 1

**User Story:** Como usuario del sistema, quiero que la TUI se cargue correctamente incluso cuando los programas guardados contienen estrategias inválidas, para poder continuar trabajando sin errores.

#### Acceptance Criteria

1. WHEN the TUI loads a program with an invalid strategy value THEN the system SHALL set the strategy selector to blank state instead of crashing
2. WHEN a program file contains a strategy that no longer exists THEN the system SHALL log a warning and continue loading
3. WHEN the strategy selector is initialized THEN the system SHALL validate that the stored value exists in the available options before setting it
4. WHEN a program is loaded with an invalid strategy THEN the system SHALL preserve all other valid configuration values (dataset, patron, estado, progreso)

### Requirement 2

**User Story:** Como usuario, quiero que el sistema valide y migre automáticamente valores de estrategia obsoletos, para que mis programas guardados sigan funcionando después de cambios en el código.

#### Acceptance Criteria

1. WHEN a program contains the legacy strategy value "base" THEN the system SHALL automatically migrate it to "basic"
2. WHEN a program contains an unrecognized strategy value THEN the system SHALL reset it to blank and notify the user
3. WHEN the system performs a migration THEN it SHALL save the updated program file immediately
4. WHEN multiple programs need migration THEN the system SHALL process all of them during startup

### Requirement 3

**User Story:** Como desarrollador, quiero que el sistema de descubrimiento de estrategias sea robusto y consistente, para evitar desincronizaciones futuras.

#### Acceptance Criteria

1. WHEN the system discovers available strategies THEN it SHALL return a sorted list of valid strategy names
2. WHEN a strategy directory lacks a required `code.py` file THEN the system SHALL exclude it from the available strategies
3. WHEN the strategy enum (Strats) is updated THEN the system SHALL ensure consistency with the filesystem-based discovery
4. WHEN displaying strategy options in the GUI THEN the system SHALL use the same discovery mechanism as the execution engine

### Requirement 4

**User Story:** Como usuario, quiero que las estimaciones de tiempo se actualicen correctamente cuando selecciono un dataset, para planificar mis ejecuciones.

#### Acceptance Criteria

1. WHEN a user selects a dataset in a program card THEN the system SHALL calculate and display the estimated execution time
2. WHEN the dataset has N dimensions THEN the system SHALL display "Est: HH:MM:SS (N dims)"
3. WHEN the estimation calculation fails THEN the system SHALL silently continue without updating the time display
4. WHEN a program card is initialized with a dataset THEN the system SHALL immediately show the time estimation

### Requirement 5

**User Story:** Como usuario, quiero que el sistema maneje errores de carga de programas de forma elegante, para no perder mi trabajo por archivos corruptos.

#### Acceptance Criteria

1. WHEN a program JSON file is malformed THEN the system SHALL log the error and skip that program
2. WHEN loading programs fails THEN the system SHALL display an empty program list instead of crashing
3. WHEN a program file cannot be parsed THEN the system SHALL provide a clear error message indicating which file is problematic
4. WHEN the programas directory does not exist THEN the system SHALL create it automatically on first save
