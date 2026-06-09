# 🪟 Guía: Configurar CUDA en Windows — IIT-2026

> Para el alumnado que usa Windows: esta guía documenta los problemas más comunes al intentar
> correr la TUI y el backend CUDA del proyecto, y cómo resolverlos paso a paso.

---

## Índice

1. [Segmentation fault al importar numpy](#1-segmentation-fault-al-importar-numpy)
2. [uv run ignora el venv y vuelve a Python 3.13](#2-uv-run-ignora-el-venv-y-vuelve-a-python-313)
3. [UnicodeDecodeError al cargar estilos CSS](#3-unicodedecodeerror-al-cargar-estilos-css)
4. [CuPy: nvrtc64_120_0.dll not found](#4-cupy-nvrtc64_120_0dll-not-found)
5. [CuPy: múltiples versiones instaladas](#5-cupy-múltiples-versiones-instaladas)
6. [CuPy: cudaErrorInsufficientDriver](#6-cupy-cudaerrorinsufficientdriver)
7. [Resumen: stack validado en Windows](#resumen-stack-validado-en-windows)

---

## 1. Segmentation fault al importar numpy

### Síntoma

```
Warning: Numpy built with MINGW-W64 on Windows 64 bits is experimental...
CRASHES ARE TO BE EXPECTED - PLEASE REPORT THEM TO NUMPY DEVELOPERS
RuntimeWarning: invalid value encountered in exp2
Segmentation fault
```

### Causa

`numpy==1.26.4` **no tiene wheel oficial para Python 3.13 en Windows**. pip/uv cae al
build MINGW-W64, que es experimental e inestable. El crash ocurre en la inicialización
de tipos flotantes, antes de ejecutar cualquier código del proyecto.

### Fix

Bajar a **Python 3.12** y actualizar `numpy` a `>=2.0.0` en el `pyproject.toml`:

```toml
# pyproject.toml
requires-python = ">=3.12"

dependencies = [
    "numpy>=2.0.0",
    "scipy>=1.13.0",
    ...
]
```

```powershell
# Instalar Python 3.12 con uv
uv python install 3.12

# Recrear el venv
uv venv --python 3.12

# Reinstalar dependencias
uv pip install -e ".[cuda]"

# Verificar que numpy NO menciona MINGW
uv run python -c "import numpy; print(numpy.__version__)"
```

---

## 2. uv run ignora el venv y vuelve a Python 3.13

### Síntoma

Creaste el venv con Python 3.12, pero al ejecutar `uv run tui` aparece:

```
Using CPython 3.13.1
Removed virtual environment at: .venv
Creating virtual environment at: .venv
```

O bien:

```
error: No interpreter found for executable name `3.12`
```

### Causa

Dos problemas encadenados:

- El `pyproject.toml` tenía `requires-python = ">=3.13"`, que hace que `uv run`
  descarte cualquier venv de 3.12 y recree uno con 3.13.
- El archivo `.python-version` fue creado con `echo` de PowerShell, que escribe
  **UTF-16 con BOM** — formato que uv no puede leer.

### Fix

**Paso 1:** Corregir `pyproject.toml` (ver sección anterior).

**Paso 2:** Reescribir `.python-version` en UTF-8 puro:

```powershell
# NO usar echo (escribe UTF-16). Usar esto:
[System.IO.File]::WriteAllText(
    (Join-Path (Get-Location) ".python-version"),
    "3.12.8",
    [System.Text.Encoding]::UTF8
)
```

**Paso 3:** Recrear el venv y verificar:

```powershell
uv venv --python 3.12
uv pip install -e ".[cuda]"
uv run tui   # debe usar 3.12 sin recrear el venv
```

---

## 3. UnicodeDecodeError al cargar estilos CSS

### Síntoma

```
UnicodeDecodeError: 'charmap' codec can't decode byte 0x90 in position 5
File "...\src\tui\shared\consts.py", line 15, in estilizar
    return ruta.read_text()
```

### Causa

`Path.read_text()` sin argumento de encoding usa la codificación del sistema operativo,
que en Windows es `cp1252`. Los archivos `.scss` del proyecto están en UTF-8 y contienen
caracteres que cp1252 no puede decodificar.

### Fix

En `src/tui/shared/consts.py`, agregar `encoding="utf-8"`:

```python
# Antes
return ruta.read_text() if ruta.exists() else ""

# Después
return ruta.read_text(encoding="utf-8") if ruta.exists() else ""
```

> **Nota para el equipo:** este fix ya está aplicado en `main`. Si hiciste fork antes
> de este cambio, recuerda hacer `git pull upstream main`.

---

## 4. CuPy: nvrtc64_120_0.dll not found

### Síntoma

```
RuntimeError: CuPy failed to load nvrtc64_120_0.dll:
FileNotFoundError: Could not find module 'nvrtc64_120_0.dll'
```

### Causa

`cupy-cuda12x` busca las DLLs de **CUDA 12**, pero tienes instalado **CUDA 13.x**.
Son incompatibles — cada versión de cupy-cudaXXX está compilada contra una versión
específica del toolkit.

### Fix

```powershell
# Desinstalar la versión incorrecta
uv pip uninstall cupy-cuda12x

# Instalar la versión que corresponde a tu toolkit
uv pip install cupy-cuda13x
```

Y actualizar `pyproject.toml`:

```toml
[project.optional-dependencies]
cuda = ["cupy-cuda13x>=14"]
```

> **Regla general:** el sufijo de cupy (`cuda12x`, `cuda13x`) debe coincidir con
> la versión del CUDA Toolkit instalado. Verificar con `nvcc --version`.

---

## 5. CuPy: múltiples versiones instaladas

### Síntoma

```
UserWarning: CuPy may not function correctly because multiple CuPy packages
are installed in your environment:
cupy-cuda12x, cupy-cuda13x
```

Seguido de `ImportError: numpy.core.multiarray failed to import`.

### Causa

Al cambiar de `cupy-cuda12x` a `cupy-cuda13x`, la versión anterior quedó instalada.
Tener dos versiones de CuPy en el mismo entorno las hace colisionar.

### Fix

```powershell
# Eliminar ambas
uv pip uninstall cupy-cuda12x cupy-cuda13x

# Reinstalar solo la correcta
uv pip install cupy-cuda13x

# Verificar que importa limpio
uv run python -c "import cupy as cp; print('OK, GPUs:', cp.cuda.runtime.getDeviceCount())"
```

---

## 6. CuPy: cudaErrorInsufficientDriver

### Síntoma

```
cupy_backends.cuda.api.runtime.CUDARuntimeError:
cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version
```

### Causa

El **driver de NVIDIA** instalado es demasiado viejo para el CUDA Toolkit que bajaste.
Cada versión del toolkit requiere una versión mínima del driver:

| CUDA Toolkit | Driver mínimo (Windows) |
|:---:|:---:|
| 12.x | 527.41 |
| 13.x | 576.02 |

### Diagnóstico

```powershell
# Ver GPU y versión de driver actual
Get-WmiObject Win32_VideoController | Select-Object Name, DriverVersion
```

La versión de Windows (`32.0.15.XXXX`) se convierte a versión NVIDIA tomando
los últimos 4 dígitos: `32.0.15.5599` → driver **555.99**.

### Fix

Ir a la página oficial de NVIDIA y descargar el driver más reciente para tu GPU:

**https://www.nvidia.com/en-us/drivers/**

Seleccionar:
- **Product Type:** GeForce
- **Product Series:** GeForce RTX 30 Series (Notebooks) ← o la que corresponda
- **Product:** tu modelo exacto
- **OS:** Windows 10/11 64-bit

Instalar el `.exe` descargado y **reiniciar el PC**.

```powershell
# Después del reinicio, verificar
uv run python -c "import cupy as cp; print('GPUs:', cp.cuda.runtime.getDeviceCount())"
# Esperado: GPUs: 1
```

---

## Resumen: stack validado en Windows

Este es el stack que funciona correctamente en Windows con RTX 30 Series:

| Componente | Versión |
|---|---|
| Python | 3.12.x |
| numpy | >=2.0.0 |
| scipy | >=1.13.0 |
| cupy | cupy-cuda13x >= 14 |
| CUDA Toolkit | 13.x |
| Driver NVIDIA | >= 576.02 |

### Comandos de verificación rápida

```powershell
# Python correcto
uv run python --version
# → Python 3.12.x

# numpy sin MINGW
uv run python -c "import numpy as np; print(np.__version__)"
# → 2.x.x  (sin warnings de MINGW)

# CuPy detecta la GPU
uv run python -c "import cupy as cp; print('GPUs:', cp.cuda.runtime.getDeviceCount())"
# → GPUs: 1

# TUI arranca
uv run tui
```

---

> 📝 **¿Encontraste un problema nuevo?** Abre un issue en el repo o avísale al monitor.