# Implementation Plan

- [x] 1. Implement strategy validation and migration system
  - [x] 1.1 Create `STRATEGY_MIGRATIONS` dictionary mapping obsolete values to current ones
    - Add "base" → "basic" migration
    - Place in `src/tui/run/helpers.py`
    - _Requirements: 2.1_

  - [x] 1.2 Implement `validar_estrategia()` function
    - Accept strategy value and list of available strategies
    - Check if value is in available strategies (return as-is)
    - Check if value is in migration map (return migrated value)
    - Otherwise return empty string
    - _Requirements: 1.3, 2.2_

  - [ ]* 1.3 Write property test for strategy validation
    - **Property 2: Invalid strategies are handled gracefully**
    - **Validates: Requirements 1.2, 2.2**

  - [ ]* 1.4 Write property test for migration idempotence
    - **Property 4: Strategy migration is persistent and idempotent**
    - **Validates: Requirements 2.3, 2.4**

- [x] 2. Modify program loading to apply validation
  - [x] 2.1 Update `cargar_programa()` to validate strategy on load
    - Call `validar_estrategia()` after loading JSON
    - Compare original vs validated value
    - If different, call `guardar_programa()` to persist migration
    - Add logging for migrations and invalid values using Python's logging module
    - _Requirements: 1.2, 2.3_

  - [ ]* 2.2 Write property test for data integrity during validation
    - **Property 3: Data integrity is preserved during validation**
    - **Validates: Requirements 1.4**

  - [ ]* 2.3 Write unit test for "base" → "basic" migration
    - Create program with "base" strategy
    - Load program
    - Verify strategy is now "basic"
    - Verify file was updated
    - _Requirements: 2.1_

- [-] 3. Update ProgramCard widget initialization
  - [ ] 3.1 Add safety check in strategy Select initialization
    - Get list of available strategies using `listar_estrategias()`
    - Verify `p.estrategia` is in the available strategies list before using as value
    - Use `Select.BLANK` if strategy is empty or invalid
    - _Requirements: 1.1, 1.3_

  - [ ]* 3.2 Write property test for widget initialization safety
    - **Property 1: Widget initialization is safe for all strategy values**
    - **Validates: Requirements 1.1, 1.3**

- [x] 4. Improve error handling in ExecutionScreen
  - [x] 4.1 Add proper logging to program loading error handling
    - Import Python's logging module
    - In `__refrescar_programas()`, log specific error details when `cargar_programa()` fails
    - Include program name and error type in log message
    - _Requirements: 5.1, 5.2, 5.3_

  - [ ]* 4.2 Write property test for error handling
    - **Property 7: Error handling never crashes the system**
    - **Validates: Requirements 4.3, 5.1, 5.2, 5.3**

- [ ] 5. Enhance strategy discovery with logging
  - [ ] 5.1 Add logging to `listar_estrategias()` for excluded directories
    - Import Python's logging module
    - Log warning when a directory is found but lacks code.py
    - Include directory name in warning message
    - _Requirements: 3.2_

  - [ ]* 5.2 Write property test for strategy discovery
    - **Property 5: Strategy discovery returns only valid strategies**
    - **Validates: Requirements 3.1, 3.2**

- [ ] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
