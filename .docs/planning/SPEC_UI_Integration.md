# Plan — Especificaciones como entrada de primera clase en la TUI

> **ESTADO: completo.** Partes 1–4 implementadas y verificadas. El resultado de la
> validación por CLI (Punto 4 de la sección Verificación), sobre `N10C` que tiene
> spec **y** CSV:
>
> ```
> N10C | patron-2
>   par                   n_ok / n   %        max_diff   mean_diff
>   analytic vs qsw_fst   96/96      100.0%   0.000062   0.000001
> ```
>
> Y sobre `N25B`, que tiene spec y **no** CSV: `qsw_fst` completa 96/96, mientras
> `analytic` falla con un mensaje accionable. Esa es la capacidad que la ruta nueva
> agrega — no velocidad a n chico, donde de hecho `qsw_fst` es 2.25× más lento
> porque el overhead de la búsqueda QSW no se amortiza con `D ≤ 10`.
>
> El riesgo señalado en la Parte 3b se resolvió sin tocar `SIA` ni el runner:
> resultó que la clase base sólo usa `distribucion_marginal()`, `dims` e `indices`,
> así que `SubEspec` los implementa y duck-typea como subsistema. Los equivalentes
> nativos de `bipartir` y `distribucion_marginal` dan `Δ = 0` exacto contra el
> `System` real sobre 65 subsistemas.

## Context

El oráculo por Zeta local ya funciona punta a punta: lee la **especificación** (padres +
fórmula por nodo) en vez de la TPM materializada, hace la reducción a subsistema sobre
las tablas locales (`src/io/reduccion.py`), y da resultados idénticos al pipeline
tradicional — `Δ = 0` en binario, `~9e-08` (piso float32 de la referencia) en
probabilístico, sobre 195 subsistemas y ~11 800 cortes.

Pero **nada de eso es alcanzable desde la interfaz**. Hoy la TUI asume que la entrada es
siempre un CSV de `2^n × n`: el tab Dataset genera TPMs al azar, y Execution elige un
dataset de una lista que sale de globear `*.csv`. Una spec de n=60, cuyo CSV pesaría
2.8×10¹¹ GB y jamás va a existir, no aparece en ninguna lista.

Además hay una deuda documentada y nunca implementada. `SIA.opciones` (`sia.py:38`) dice:

> *"La TUI renderiza un selector por opción y el CLI las acepta con `--opcion attr=valor`"*

El CLI sí; la TUI no. `Programa.opciones` ya existe, se persiste, se valida y alimenta
`etiqueta_estrategia` — falta puro cableado visual.

### Decisión de diseño: un solo namespace de redes

En disco una red **ya es un nombre** que puede tener `.csv`, `.npy` y/o `.spec.json`;
`cargar_espec(nombre)` devuelve `None` si no hay spec. Se mantiene así, con **un solo
selector anotado** en Execution, en vez de separar "specs" y "datasets" en dos objetos.

Razón concreta: `build_output_stem` / `parse_result_key` aparean resultados por nombre de
red. Con dos namespaces, una corrida de `analytic` (que usa CSV) y una del oráculo local
(que usa spec) sobre la misma red quedarían como series distintas y **`cli compare` no
podría compararlas** — que es exactamente el objetivo del trabajo.

---

## Orden de entrega

Las tres partes son independientes y se pueden mergear por separado. Ese es el orden
recomendado, de menor a mayor riesgo.

---

## Parte 1 — Fila de opciones en `ProgramCard` (independiente, sin bloqueos)

Lo más barato y ya estaba diseñado.

**`src/tui/run/widgets.py`** — `ProgramCard.compose`, tras la fila `card-selectores`:

- Un contenedor `card-opciones` con un `Select` por cada entrada de
  `SIA.registry[estrategia].opciones`, con el valor actual de `prog.opciones` o el default
  (`SIA.defaults()`).
- Vacío ⇒ el contenedor no se muestra (la mayoría de las estrategias no tienen opciones).
- Al cambiar la estrategia, recomponer la fila y llamar a **`podar_opciones`**
  (`run/helpers.py:293`) — ya existe y su docstring documenta el bug que previene:
  opciones de la estrategia anterior que quedaban pegadas y reventaban la nueva.
- El evento `Configurado` debe llevar también `opciones`; hoy sólo lleva
  `(nombre, dataset, patron, estrategia)`.

**`src/tui/run/screen.py`** — `on_program_card_configurado` guarda `prog.opciones`.

Verificación visual: elegir `qsw` debe mostrar `modo`, `backend` y `k`; elegir `analytic`
no debe mostrar nada; cambiar de `qsw` a `queyranne` debe descartar `backend`.

---

## Parte 2 — Tab Specification

Nuevo tab **antes** de Dataset. Crear e inspeccionar specs; no ejecuta nada.

**`src/tui/app.py`** — `TabPane("Specification", id="spec")` como `initial`, binding `s`.

**`src/tui/spec/`** — módulo nuevo, espejando la estructura de `src/tui/dataset/`
(`screen.py`, `widgets.py`, `helpers.py`).

Toolbar de creación:

| control | fuente |
|---|---|
| `n` (dims) | — |
| `k` (fan-in) | — |
| modelo: binario / PBN L=… / ruido-ε | `fst.generar_espec`, `fst.generar_espec_pbn` |
| topología: aleatoria / anillo | `topologia.MODOS` |
| compuerta: aleatoria / and / or / xor / mayoría | `fst.COMPUERTAS` |

Panel de inspección de la spec seleccionada:

- Fórmulas CNF legibles por nodo — **`cnf.formatear`**, que ya usa la notación `⟨A + B⟩`.
- Matriz de adyacencia y fan-in por nodo — derivados de `espec.padres`.
- **`topologia.es_separable(espec.padres)`** con aviso destacado cuando da `True`: es lo
  que anticipa que `φ = 0` va a ser trivial, y es la información más útil del panel.
- Botón **Materializar CSV** → `fst.tpm_nodo_estado` + escritura, reutilizando
  `dataset.modal.ConfirmarCargaModal` y `generator.peso_estimado` para el aviso de tamaño.
  Deshabilitado por encima de `DIMS_WARN_THRESHOLD` sin confirmación.

---

## Parte 3 — Execution sobre specs

### 3a. Plumbing (necesario antes que la UI)

**`src/io/manager.py`**

- `listar_redes()` hoy hace `REDES_DIR.glob("*.csv")` — una spec sin CSV es invisible.
  Pasa a unir los stems de `*.csv` y `*.spec.json`.
- Añadir `capacidades(nombre) -> dict` (`{"csv": bool, "npy": bool, "spec": bool}`) para
  que la UI anote sin abrir archivos.
- `dims_de_red()` ya deduce del nombre (`N25B → 25`) y cae al CSV; agregar caída a la spec
  (`len(cargar_espec(nombre).padres)`) para redes sin CSV cuyo nombre no siga el patrón.

**`src/iit/strategies/python/sia.py`** — `necesita_espec: ClassVar[dict[str, bool]]`,
poblado en `__init_subclass__` igual que `necesita_mpt` (`sia.py:44-50`). Es el mecanismo
que ya existe para declarar requisitos de entrada; se sigue tal cual.

### 3b. Estrategia que consume la spec

Sin un consumidor, `necesita_espec` no tiene sentido. Estrategia nueva registrada con
`necesita_espec=True` que use `reduccion.reducir_espec` + `zeta_local`.

⚠️ **Acá está el riesgo real del plan.** `SIA` recibe un `System` (tupla de `NCube`, cada
uno con `data` de `2^D`) — construirlo es exactamente la materialización que se quiere
evitar. Hace falta una ruta paralela que no pase por `System`:

- `winner()` sale directo del oráculo local sobre el `SubEspec`. Ya está resuelto.
- `resolver()` reconstruye hoy con `sistema.bipartir(...)` + `distribucion_marginal()` +
  `emd_efecto`. Los equivalentes nativos son baratos y ya existen las piezas:
  `distribucion_marginal` es `tabla[patrón_de_padres(estado)]` por nodo (`O(k)`), y
  `bipartir` es `reduccion.marginalizar_tabla` aplicada por lado. `emd_efecto` no cambia.

Dimensionar esto antes de empezar; puede justificar su propio plan.

### 3c. UI

**`src/tui/run/widgets.py`** — reordenar `card-selectores` a **estrategia → red → patrón**,
y filtrar la lista de redes por lo que la estrategia declara: `necesita_espec` ⇒ sólo redes
con `.spec.json`; `necesita_mpt` ⇒ sólo con `.csv`. Anotar cada entrada con sus
capacidades (`● spec  ○ csv  ✓ npy`).

**`src/tui/run/screen.py:245`** — hoy `tpm = cargar_mpt(prog.dataset)` incondicional. Debe
ramificar: con `necesita_espec`, `cargar_espec` + `reducir_espec` en vez de `cargar_mpt` +
`preparar_subsistema`.

---

## Parte 4 — Dataset gana información

**`src/tui/dataset/screen.py` / `widgets.py`** — `NetworkItem` muestra las capacidades de
la red, y para las que no tienen spec, el veredicto de
**`identificabilidad.diagnosticar`** (DENSA / NO IDENTIFICABLE / IDENTIFICABLE). Responde
en el lugar correcto la pregunta "¿esta red sirve para la ruta nueva?".

Cachear el veredicto: `influencias()` es `O(N·D·2^D)` y a n=20 no se puede recalcular en
cada repintado.

El generador aleatorio de siempre no se toca.

---

## Verificación

```bash
# regresión (test_io.py revienta con Bus error por una carrera preexistente)
uv run pytest tests/ -q --ignore=tests/dataset/test_io.py
# esperado: 4 fallos preexistentes en test_ejecutar.py

uv run pytest tests/tui -q
uv run python -m src.tui.app --no-watch
```

A mano, en la TUI:

1. **Opciones** — programa con `qsw`: aparecen `modo`/`backend`/`k`; cambiar a
   `queyranne` descarta `backend`; el CSV de salida se llama `qsw+modo=exacto` cuando la
   opción difiere del default.
2. **Specification** — crear `n=8 k=3 PBN L=3 anillo`, verificar que el panel marca **no
   separable**; crear `n=8 k=2 aleatoria` y verificar que marca **separable**.
3. **Selector único** — crear una spec de n=30 sin materializar y confirmar que aparece en
   Execution para la estrategia nueva y **no** para `analytic`.
4. **Equivalencia** — materializar el CSV de una spec chica, correr `analytic` y la
   estrategia nueva sobre la misma red, y confirmar en `cli compare` que quedan como dos
   series de la **misma** red con la misma pérdida.

El punto 4 es el que valida la decisión de namespace único; si falla, el diseño está mal.

---

## Fuera de alcance

- Editor de fórmulas CNF dentro de la TUI.
- `Fatal Python error: Bus error` en `tests/dataset/test_io.py` (carrera del
  `optimizar_red` en hilo *fire-and-forget*).
- Traer a esta rama el fix del default de `modo`, que está en `main` (`33d5e55`).
