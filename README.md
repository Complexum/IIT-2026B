# Proyecto-2026A

<!--

TODO: Leer de nuevo porque pues actualizar a sólo UV y tasks:

[ ] Scream Architecture
    [ ] Clases bien organizadsa
[ ] Diseñado para pruebas
[ ] TUI para mejor usabilidad
    - Si le voy a meter TUI la idea es que no tengan complicaciones creando una estrategia, que lo máximo que deban hacer es crearla y subscribirla al listado de ellas. O inclusive aplicar la de Angular y que con un comando se genere el boilerplate.

[ ] Buen diseño al encapsular parametros.
    - Por un lado está mal enviar muchos parámetros, para eso se construye un objeto y es en este el enviado.

    Bueno, si es con tui se redefinen los pasos porque ya es declarativo, no imperativo

    1. usr:     Se carga la TUI
    2. sys:     Se recopiló del usuario su plataforma operativa
    3. sys:     Se definen las métricas y configs precargadas en csv para persistencia cross-time.
    4. usr:     Va al apartado
        4.1.    Dataset: Puede generar o eliminar uno (quizás editar pero muy loco) (lo que sí estaría épico es que se vea cómo queda el sistema en cada fase!)
        4.2.    Strats: Que pueda [crear] (este apartado a lo mejor ni existe)
        4.3.    Testing: Selecciona si precarga o da los parametros 
            4.3.1.  Selecciona Precargar y aparecen las opciones de CSV para ejecutar y listo.
            4.3.2.  Selecciona Parametrizar y puede alterar en cada prueba la marginalización aplicada. Debería hacer un Excalidraw.

    Vale y pasa que puede pasar que haga también el back con rust, pero entonces cómo haría para usar la TUI? Lo que importa por así decirlo es que aunque sean lenguajes distintos, si quiero ejecutar con rust pueda hacerlo sin problema, si quiero con python tampoco.


[ ] Logica con principios funcionales
    - El flujo de datos debe de ser claro, cada objeto debe tener sentido
    - Entonces:
        1. Se carga la TPM como arreglo numpy
            Quien la debe cargar? La clase aplicación? No, debe pasar que... asociemoslo a un sistema crud, stateless.

            Los objetos son:
                sistem 1:N candidate
                candidates 1:N subsystem
                sistem 1:1 tpm
                tpm 1:N initial-state

            Se requiere que el sistema pueda encontrarse su MIP de forma paralelizable con multiples subsistemas

            >> candidato = completo.condicionar(prueba_i)
                >> completo.resolver()
            >> completo.condicionar(i)

            En la interfaz que puedan generar la combinación.

            [estrategia-x]
            [sistema][estado-inicial][alcance][mecanismo]
            >> [N25A][10000][alcance][mecanismo]
                - Automaticamente aparecen las opciones o se ingresan `input-text`

                A lo mejor lo mejor es que se pueda hacer N pruebas y luego mirar cómo hacerlo escalable con CSV.


        Hay que definir a nivel de:
            - aplicacion:
                - emd(metric-distance), notation
                - paralelism?
                - pyphi config?
            - usuario:
                - plataforma:
                    - ram, ssd, nucleos, os, ...
                - sesion?
                - theme
                - seed
            
        



[ ] Pensado para memoización, depende de la foto esa. Depende de si garcía encuentra el doble-loop de Basic strat
[ ] Carga de datos listo para forma geométrica
[ ] Solución por fuerza bruta

-->









> Base del proyecto para dar desarrollo a estrategias más elaboradas.

Para el correcto uso del aplicativo se buscará lo siguiente:
El alumnado se conformará por grupos de desarrollo de forma que puedan usar el aplicativo base para desarrollar sus estrategias de forma independiente con su información segura en una rama propia para el desarrollo (`dev`). A su vez, podrán recibir actualizaciones del proyecto principal (`main`) mediante `git pull origin main` mientras sea necesario.

Para lograr esto, primero vamos a realizar un **Fork** desmarcando la casilla de "Copy the `main` branch only." para que podamos tener acceso a las demás ramas del repositorio, asignaremos un nombre de preferencia según el equipo de desarrollo. Procederemos a clonar dicho fork en nuestro ordenador mediante `git clone https://github.com/<grupo-usuario>/<Fork-Proyecto-2025A> .` usando GIT, tras esto podremos asociar este repo **local** del equipo con el original para recibir actualizaciones, se logras mediante el comando 
```bash
git remote add upstream https://github.com/Complexum/Proyecto-2025A.git
```
 De forma tal que siempre que estés sobre la rama **`dev`** al aplicar el comando `git pull` o `git fetch upstream` recibirás las actualizaciones ocurridas en `dev`, y a su vez podrás subir código al fork para trabajar en colaborativo.

---

## Instalación

Guía de Configuración del Entorno con VSCode

### ⚙️ Instalación - Configuración

#### 📋 **Requisitos Mínimos**
- ![PowerShell](https://img.shields.io/badge/-PowerShell-blue?style=flat-square) Terminal PowerShell/Bash.
- ![VSCode](https://img.shields.io/badge/-VSCode-007ACC?logo=visualstudiocode&style=flat-square) Visual Studio Code instalado.
- ![Python](https://img.shields.io/badge/-Python%203.9.13-3776AB?logo=python&style=flat-square) Versión python 3.9.13 (o similar).

---

#### 🚀 **Configuración**

1. **🔥 Crear Entorno Virtual**  
   - Abre VSCode y presiona `Ctrl + Shift + P`.
   - Busca y selecciona:  
     `Python: Create Environment` → `Venv` → `Python 3.9.13 64-bit` y si es el de la `(Microsoft Store)` mejor. En este paso, es usualmente recomendable el hacer instalación del Virtual Environment mediante el archivo de requerimientos, no obstante si deseas jugartela a una instalación más eficiente y controlada _(no aplica a todos)_, puedes usar UV. También es importante aclarar lo siguiente, si eres fan de los antivirus, habrás de desactivar cada uno de ellos, uno por uno en su análisis de tiempo real, permitiendo así la generación de los ficheros necesarios para el virtual-environment.
   - ![Wait](https://img.shields.io/badge/-ESPERA_5_segundos-important) Hasta que aparezca la carpeta `.venv`

2. **🔄 Reinicio**
   - Cierra y vuelve a abrir VSCode (obligado ✨).
   - Verifica que en la terminal veas `(.venv)` al principio  
     *(Si no: Ejecuta `.\.venv\Scripts\activate` manualmente, pon `activate.bat` si estás en Bash)*


> **💣 (Opcional) Instalación de librerías con UV**
>   En la terminal PowerShell (.venv activado): 
>   Primero instalamos `uv` con 
>   ```powershell
>   pip install uv
>   ```
>   Procedemos a instalar las librerías con
>   ```powershell
>   python -m uv pip install -e .
>   ```

> **Este comando:**
> Instala dependencias de pyproject.toml
> Configura el proyecto en modo desarrollo (-e)
> Genera proyecto_2025a.egg-info con metadatos

> 1. ✅ Verificación Exitosa
   ✔️ Sin errores en terminal
   ✔️ Carpeta proyecto_2025a.egg-info creada
   ✔️ Posibilidad de importar dependencias desde Python

> 🔥 Notas Críticas
   - Procura usar la PowerShell como terminal predeterminada (o Bash).
   - Activar entorno virtual antes de cualquier operación.
   - Si usaste UV la carpeta `proyecto_2025a.egg-info` es esencial.

---

## 🚀 Ejecución Rápida

### Con UV (Recomendado)

```bash
# Demo básico (carga N3A y muestra operaciones)
uv run python main.py

# CLI
uv run mip

# TUI (Interfaz Textual)
uv run mip-tui
```

### Con venv activado

```bash
# Activar venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Ejecutar
python main.py
python -m src.cli
python -m src.tui.app
```

> **💡 Tip VS Code:** Si abres el proyecto en VS Code, la configuración de `.vscode/settings.json` activa automáticamente el entorno virtual en la terminal integrada. Esto significa que puedes usar los comandos directamente (`tui`, `cli`, `strat`) sin escribir `uv run` cada vez.
>
> ```bash
> # Desde la terminal integrada de VS Code (venv auto-activado)
> tui
> cli list datasets
> strat mi_algoritmo
> ```

**Atajos TUI:** `d` (Dataset), `t` (Testing), `e` (Execution), `r` (Results), `a` (Analysis), `q` (Quit)

### Desarrollo con Auto-Reload

La TUI incluye **auto-reload** por defecto: detecta cambios en archivos `.py` bajo `src/` y reinicia automáticamente la aplicación. No es necesario cerrar y volver a abrir la TUI durante el desarrollo.

```bash
# Ejecutar con auto-reload (comportamiento por defecto)
uv run tui

# Desactivar el auto-reload
uv run tui --no-watch
```

> **Nota:** Los estilos CSS (`styles.scss`) se recargan automáticamente sin necesidad de reiniciar la aplicación.

📖 **Guía completa:** Ver [`.docs/EXECUTION.md`](.docs/EXECUTION.md) y [`.docs/ARCHITECTURE.md`](.docs/ARCHITECTURE.md).

---

## 🖥️ CLI (sin TUI)

El proyecto incluye un CLI modular para operar completamente desde terminal, útil para automatización o LLMs.

```bash
# Invocar el CLI
python -m src.cli <comando> [args]
# o si está instalado:
uv run iit <comando> [args]
```

**Comandos disponibles:**

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| `list` | Listar recursos | `iit list datasets` |
| `show` | Mostrar detalles | `iit show dataset N5A` |
| `new` | Crear recurso | `iit new dataset 5 --discretos` |
| `edit` | Editar execution | `iit edit execution exec-01 --estrategia phi` |
| `run` | Ejecutar execution | `iit run execution exec-01` |
| `results` | Consultar resultados con SQL | `iit results exec-01 "SELECT estado, perdida WHERE perdida > 0.5 ORDER BY tiempo DESC LIMIT 10"` |
| `delete` | Eliminar recurso | `iit delete execution exec-01` |

**Flags globales:**
- `-v, --verbose` — salida detallada (útil en `run` para ver cada combinación).

**Ejemplos de flujo completo:**

```bash
# 1. Crear dataset y execution
iit new dataset 4 --discretos
iit new execution exec-x --dataset N4A --patron patron-1 --estrategia basic

# 2. Verificar y ejecutar
iit show execution exec-x
iit run execution exec-x          # reanuda automáticamente si hay checkpoint
iit run execution exec-x --no-resume  # fuerza reinicio desde cero

# 3. Consultar resultados con SQL
iit results exec-x                              # mostrar todo
iit results exec-x "SELECT estado, perdida, tiempo FROM self WHERE perdida > 0.5 ORDER BY tiempo DESC LIMIT 10"
iit results exec-x "SELECT AVG(perdida), MAX(tiempo), COUNT(*) FROM self"
iit results exec-x "SELECT estado, COUNT(*) as total, AVG(perdida) FROM self GROUP BY estado"

# 4. Listar executions y limpiar
iit list executions
iit delete execution exec-x
```

> **💡 Tip VS Code:** Si trabajas dentro de VS Code, el entorno virtual se activa automáticamente en la terminal integrada (gracias a `.vscode/settings.json`). Puedes usar los comandos directamente (`tui`, `cli`, `strat`) sin escribir `uv run` cada vez.
>
> ```bash
> # Desde la terminal integrada de VS Code (venv auto-activado)
> iit list datasets
> tui --no-watch
> strat mi_algo
> ```

> Los comandos reutilizan la misma lógica de persistencia que la TUI (JSON en `data/input/`, CSV en `data/output/`), por lo que puedes alternar entre TUI y CLI sin problemas.

---

## Crear una nueva estrategia

El proyecto incluye un comando de scaffolding que genera el boilerplate necesario para implementar una nueva estrategia:

```bash
# Estrategia simple
uv run strat mi_algoritmo

# Estrategia que requiere la TPM completa (como pyphi)
uv run strat mi_fuerza_bruta --tpm
```

Esto crea automáticamente:

```
src/iit/strategies/python/mi_algoritmo/
├── __init__.py
└── code.py        ← listo para implementar resolver()
```

El `code.py` generado incluye la clase con `@perfilar`, la herencia de `SIA`, y el esqueleto de `resolver()` con comentarios de referencia. **La estrategia aparece automáticamente en el dropdown de la TUI y en `ejecutar()` sin tocar nada más.**

Solo hay que implementar el algoritmo dentro de `resolver()`:

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

*Para más detalles sobre la arquitectura del sistema, ver [`.docs/ARCHITECTURE.md`](.docs/ARCHITECTURE.md).*