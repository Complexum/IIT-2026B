# QSW — Stoer-Wagner × Queyranne

| Término | Qué es |
|---|---|
| MAO | Maximum Adjacency Ordering. El corazón de Stoer-Wagner: va agregando el nodo más "pegado" a los ya elegidos. Los dos últimos son el par colgante y dan una candidata de corte. Son los 2.8 ms. |
| Zeta | El precómputo: suma sobre subconjuntos. Deja `sumas[i,m]` lista para cualquier máscara m. Es el 100 % del costo. |
| butterfly / mariposa | El patrón de acceso del Zeta: D pasadas; en la pasada d se suman pares separados 2^d. |
| DRAM | La RAM normal. ~200–273 GB/s. |
| HBM | La memoria de la GPU. ~1 TB/s — de ahí la ventaja. |
| PCIe | El cable/bus que une la GPU con la CPU. Todo lo que entra o sale de la GPU pasa por ahí, a ~12–25 GB/s. Es el peaje. |
| H2D / D2H | Host to Device / Device to Host: las copias CPU→GPU y GPU→CPU por PCIe. |
| bandwidth-bound | El programa espera datos, no hace cuentas. Agregar cores no ayuda si el bus ya está lleno. Es el caso del Zeta. |
| SIMT | En la GPU, grupos de 32 threads (warp) ejecutan la misma instrucción a la vez. Si se desvían, se serializan. |
| fork / spawn | Dos formas de crear procesos: fork clona el proceso actual (macOS acá); spawn arranca un Python nuevo de cero (Linux/el cluster). |
| RawArray | Memoria que varios procesos ven a la vez, sin copiar. Lo que usa `qsw_mul`. |
| Amdahl | La parte que queda serial pone un techo al speedup, por más cores que agregues. |
| oráculo | La función `f(S)` que da el costo de un corte S. |


Híbrido entre el andamiaje de **Stoer-Wagner** (JACM 1997) y el oráculo Zeta exacto de
**Queyranne** (Math. Prog. 1998) que ya usa `qn`.

Una sola estrategia registrada, `qsw`, con dos atributos configurables declarados en
`QSW.opciones` (ver `SIA.opciones` y el README):

- `modo`: `exacto` (default) | `estatico` | `estocastico` — ver §7
- `backend`: `python` (default) | `c` | `auto` — el kernel C hace el **Zeta**, no el MAO
- `k`: `auto` (default) | un entero — sólo lo usa `modo=estocastico`

```bash
cli run execution program-21 --opcion modo=estatico
cli run execution program-24 --opcion backend=c        # requiere clang/build.sh
```

Módulos, una responsabilidad cada uno: `core.py` el algoritmo · `muestreo.py` el
oráculo estimado · `backend.py` la selección y carga del kernel · `code.py` el glue
con `SIA` · `reference.py` el port crudo de Stoer-Wagner. La transformada Zeta vive
en `src/iit/strategies/python/zeta.py`, compartida con `analytic` y la familia `qn`.

---

## 1. La función objetivo es un corte legítimo

Con `f(S) = f_cara(oráculo, alcance = EFFECT-part(S), mecanismo = ACTUAL-part(S))`:

| Propiedad | Por qué |
|---|---|
| `f(∅) = 0` | `m = 0` → `val_a = \|sumas[:,0]\| = \|δ en el pivote\| = 0` |
| `f(V) = 0` | `cmask = 0` → `val_b = 0`, y todos los nodos caen en alcance |
| `f(S) = f(V∖S)` | complementar intercambia `val_a ↔ val_b` **y** la rama del `where`; los dos swaps se cancelan |

Simétrica, anclada y no negativa: es exactamente la clase de función que Queyranne minimiza y de la
que Stoer-Wagner es el caso gráfico. (Verificado en `tests/strategies/test_qsw.py`.)

---

## 2. El puente: SW **es** Queyranne sobre funciones gráficas

Queyranne construye un *legal ordering* eligiendo

$$v_{i+1} = \arg\min_v\; \big[f(A\cup\{v\}) - f(\{v\})\big].$$

Para una `f` de corte de grafo, `f(A∪v) = f(A) + f(v) − 2·w(A,v)`, luego

$$f(A\cup v) - f(v) = f(A) - 2\,w(A,v).$$

`f(A)` es **constante** durante el argmin sobre `v`. Por lo tanto

> minimizar la ganancia de Queyranne ≡ maximizar la adyacencia de Stoer-Wagner.

Son el mismo algoritmo. Eso legitima usar el bookkeeping barato de SW y quedarse con las
candidatas de par colgante de Queyranne.

---

## 3. Qué se roba de Stoer-Wagner

Queyranne consulta el oráculo dentro del MAO: $f(A \cup w)$ para cada `w` restante, en cada paso →
**O(V³)** consultas de cota. Stoer-Wagner no consulta nada: mantiene una matriz densa `W` y
actualiza `key[w] += W[t][w]` en O(1).

Para tener `W` sin que `f` sea gráfica, se ajusta el mejor surrogate de segundo orden:

$$W[u][v] = \tfrac{1}{2}\big(f(\{u\}) + f(\{v\}) - f(\{u,v\})\big)$$

Exacto si `f` es gráfica (`f({u}) = \deg(u)` lo recupera término a término — lo comprueba
`test_seed_recupera_los_pesos_del_grafo`). Si no lo es, puede tener entradas negativas; el
max-adjacency sólo ordena, así que sigue funcionando.

**El error del surrogate no contamina el resultado:** `W` únicamente *guía* la búsqueda. Toda
candidata se re-puntúa con la `f` exacta, y la ganadora final se reconstruye con
`bipartir + distribucion_marginal + emd_efecto` reales.

---

## 4. Los dos modos

Al contraer `s,t → st`, en un grafo real `W[st][v] = W[s][v] + W[t][v]` es exacto. Con `W`
aproximada esa suma deja de serlo y el error se acumula a lo largo de las V contracciones.

| modo | tras contraer | consultas | forma |
|---|---|---|---|
| `modo=estatico` | `W[s] += W[t]` (Stoer-Wagner puro) | O(V²), **1 batch upfront** | 100 % paralelizable (multiproc / CUDA / kernel C) |
| `modo=exacto` (default) | recalcula sólo la fila del supernodo: `W[st][v] = (f(st) + f(v) − f(st ∪ v))/2` | O(V²), en O(V) batches | sin drift |

Mismo orden las dos: sólo nace **un** supernodo por fase, así que refrescar su fila cuesta `V_p`
consultas y `Σ V_p = O(V²)`.

---

## 5. Estructura

```
1. SEED       W y singletons                          1 batch
              (el baseline C_conc de `analytic` —alcance={i}, mecanismo=∅— ES
               el singleton (EFFECT,i): queda cubierto sin caso especial)
2. FASES      V−1 × [ MAO con keys incrementales → par colgante (s,t)
                      → candidata miembros[t] → contracción ]
3. RE-SCORING todas las candidatas con f exacta      1 batch
4. 1-OPT      mover un vértice — vecinos de TODAS las candidatas, 1 batch
5. 2-OPT      mover dos vértices — |C|·C(V,2) máscaras, 1 batch, con tope
6. RECONSTRUIR el ganador con EMD real
```

Codificación de vértices sobre un entero de `V = D + N` bits:

```
bit  0 .. D-1     → (ACTUAL, dims[j])      candidato a mecanismo
bit  D .. D+N-1   → (EFFECT, indices[i])   candidato a alcance
```

Máscara → `(alcance, mecanismo)` es shift + máscara. `qn` paga hoy un `__flatten`/`sorted`
recursivo en cada consulta.

---

## 6. Complejidad: lo que el O(V³) NO dice

`O(V³)` (Queyranne) y `O(V²)` (QSW) cuentan **llamadas al oráculo**, no tiempo.
Construir el oráculo es `Θ(D·N·2^D)`, y ese término domina:

| | n=18 | n=20 | n=22 |
|---|---|---|---|
| Zeta (`preparar_oraculo`) | 0.037 s | 0.178 s | 0.957 s |
| **TOTAL `qsw`** | 0.043 s | 0.197 s | **0.946 s** |

A n=22 el precómputo *es* la corrida entera; la búsqueda son 2.8 ms. Y corre a
**17–19 GB/s**, o sea saturando el ancho de banda de DRAM: está limitado por
memoria, no por cómputo.

Hay además un piso duro de `Ω(N·2^D)`: la entrada misma son N·2^D probabilidades
(cada `NCube` tiene 2^D celdas). **Ningún algoritmo exacto puede ser polinómico
en D** — no se puede ser más rápido que leer la propia entrada. Por eso `qsw` le
gana a `analytic` por un factor constante y no por uno exponencial.

Corolario para las versiones paralelas: siendo bandwidth-bound, más hilos en la
misma máquina compran poco (comparten DRAM). GPU (HBM ~1 TB/s) y MPI entre nodos
—cada uno con su memoria— sí.

**Medido** (`benchmarks/escalado_mul.py`, M4 Pro 12 cores, `qsw_mul`):

| n | tamaño | serial | w=4 | w=8 | w=16 |
|---|---:|---:|---:|---:|---:|
| 16 | 4.2 MB | 0.0062 s | 0.58× | 0.40× | 0.34× |
| 18 | 18.9 MB | 0.0289 s | 1.16× | 1.12× | 0.99× |
| 20 | 83.9 MB | 0.1423 s | 1.61× | 1.94× | 1.99× |

O sea: el corolario es **casi** correcto pero por el motivo equivocado. La curva se
aplana en ~2×, sí, pero no por saturar la DRAM —a 29 GB/s efectivos se está en el
11 % del pico del M4 Pro—. Pesan más el reparto por filas (N=20 sobre 16 workers
deja rebanadas de 2 y de 1) y los 4 cores de eficiencia, que fijan el makespan.
Debajo de ~80 MB la fila entera entra en L2 y la versión serial ya no toca DRAM,
así que repartir sólo agrega overhead: de ahí el umbral `MIN_ELEMS_PARALELO`.

En un EPYC 7302 —16 cores homogéneos, 8 canales DDR4— el reparto debería verse
distinto; correr `escalado_mul.py` ahí es la medición que falta.

### Lo que sí se optimizó

| Cambio | Ganancia medida |
|---|---|
| `hyperfaces` sin gather (`zeta.py`) | 1.3× sobre el Zeta; el gather era el 95 % y asignaba 738 MB de índices a D=22 |
| Kernel C del Zeta (`backend=c`) | **5–10×** sobre el Zeta de numpy, bit-exacto |
| `preparar_ncubos` compartidos | 3× en la reducción de subsistema (9.5 s → 3.2 s en un barrido de 96 filas) |

Medido y **descartado**: radix-4 en numpy (0 %, el slicing strided recorre el
arreglo igual — por eso vive en C) y float32 desde el origen (~4 %).

### Sobre datos reales (N20A + `patron-2`, 96 combinaciones, vía CLI)

Tiempo **del algoritmo**, sin la preparación del subsistema (que cuesta ~3.2 s para todas por
igual, y que antes se incluía en la medición escondiendo más de la mitad de la ventaja):

| | algoritmo | speedup vs `analytic` |
|---|---:|---:|
| `analytic` | 3.211 s | 1.00× |
| `qn` | 3.281 s | 0.98× |
| `qsw` | 2.182 s | 1.47× |
| `qsw+backend=c` | **0.989 s** | **3.25×** |
| `qsw+modo=estocastico` | 4.955 s | 0.65× |

96/96 exacto contra `analytic` en todos los casos.

**Dónde está ahora el cuello de botella:** `reducir_a_subsistema` era el 75 % del
barrido; tras compartir los NCubes bajó, pero sigue siendo la porción mayor. El
barrido de `patron-2` tiene D mediano 12 (48 de 96 combos son D=9–10, o sea 2^10
celdas), así que las estrategias ya son casi gratis ahí. Optimizarlas más no
mueve la aguja en este flujo; el trabajo está en la preparación del subsistema.

---

## 7. `modo=estocastico`: el único modo de romper el 2^D

Estima las medias de cara con K celdas muestreadas en vez de sumar las 2^|A|
(ver `muestreo.py`). Nunca construye la tabla de 2^D caras, así que el costo pasa
de `O(D·N·2^D)` a `O(N·K·D²)`.

**Resultado honesto:** sobre TPMs sintéticas uniformes daba 4/4 exacto con hasta
27× a n=24. Sobre datos **reales** (N20A) fallaba **62–69 % en D=19–20** — las TPM
reales tienen empates que el ruido de muestreo da vuelta; las uniformes no, y por
eso parecían fáciles.

Lo que lo salva: el diagnóstico de margen-contra-ruido resultó **100 % preciso**
(9/9 en la muestra). Cuando dice "confiable", acierta. Así que se usa como
**compuerta**: si el margen entre la ganadora y la segunda no supera 2σ del ruido,
se rehace con el Zeta exacto. Con eso el modo da 96/96 exacto sobre N20A.

El ahorro aparece sólo donde el muestreo alcanza. En `patron-2`/N20A casi nunca
alcanza (D mediano 12, donde ni siquiera muestrea), así que el modo **hoy no paga
en ese barrido**: 9.12 s contra 6.62 s del exacto. Sirve para D grande y uniforme,
y el `!` del calibrador marca cuándo no confiar.

Calibrar K con datos, no a ojo:

```bash
uv run python benchmarks/calibrar_k.py 18 24 4
```

---

## 8. Trabajo pendiente

- **`reducir_a_subsistema`** — es la porción mayor del barrido. `crear_sistema`
  ya no recopia las columnas por fila (`preparar_ncubos`), pero `condicionar` y
  `substraer` siguen construyendo índices por llamada.

- **Muestreo sin materializar el subsistema** — hoy `reducir_a_subsistema` arma
  los N·2^D valores antes de que la estrategia empiece, así que el muestreo evita
  el factor **D** del Zeta pero no el `2^D` de la carga. Para ser realmente
  sub-exponencial habría que muestrear del TPM mapeado (`cargar_mpt` ya usa
  `mmap` cuando existe el sidecar `.npy`).

- ~~**Paralelizar el Zeta**~~ — hecho en la ruta Python: `qsw_mul`
  (multiprocessing sobre las filas, memoria compartida vía `RawArray`) y
  `qsw_cuda` (butterfly entero en un kernel CUDA). Los dos sobrescriben sólo
  `QSW._preparar_oraculo`, así que heredan la búsqueda y dan resultado idéntico
  —96/96 contra `qsw` sobre N20A—. Queda pendiente el lado C: `qsw_zeta` sigue
  listo para `#pragma omp parallel for` sobre el bucle de filas, y sería la misma
  paralelización sin IPC ni memoria compartida.

- **`qsw_cuda` sin validar** — escrito contra el patrón de `analytic_cuda` y con
  la aritmética de índices del kernel verificada en numpy contra `zeta_inplace`
  (`tests/strategies/test_qsw_cuda.py`, todos los pivotes hasta D=8), pero nunca
  ejecutado: no hay NVIDIA en la máquina de desarrollo.

- **`analytic_cuda` casi no usa la GPU** — calcula `hyperfaces` en el host y sólo
  manda la reducción por máscara, que ya era barata. Y la columna `gpu_mem_mb` del
  CSV es 0.0 siempre: `pynvml` no es dependencia del proyecto, así que
  `NVML_AVAILABLE` es False y nunca se puebla.

- ~~**`queyranne/code.py:111`**~~ — corregido: usaba `max f(A∪w)` en vez de
  `min f(A∪w) − f(w)`. El término `−f({v})` es justamente el que hace coincidir la
  regla con max-adjacency (§2), así que omitirlo perdía la garantía de par
  colgante. Pasó de **370/384 a 383/384** contra `phi`.

- ~~**Reparación local insuficiente**~~ — corregido en dos pasos. El 1-opt sólo
  miraba vecinos de la ganadora; ampliarlo a los de **todas** las candidatas
  recuperó 3 de las 5 instancias fallidas de N5B. Las otras 2 necesitaban 2-opt:
  en N5B[36] el 1-opt se queda en 0.109449 contra 0.037566 del óptimo, y ningún
  movimiento simple cierra esa brecha. Con las dos, `qsw` da **384/384 = 100 %**.

<!-- NOTAS DE INVESTIGACIÓN — 2026-08-21

  Tolerancia de comparación contra `phi`
  --------------------------------------
  Usar 1e-6 da lecturas falsas. En N10B seis instancias aparecen como desacuerdo
  con una desviación de exactamente 1.00e-06: es la resolución de la acumulación
  float32 del oráculo, no una partición distinta. Los desacuerdos reales están 3-5
  órdenes de magnitud más arriba (2.3e-04 el de `queyranne`, 7.2e-02 el de `qsw`
  que motivó el 2-opt). Comparar a 1e-5 separa las dos poblaciones limpio.
  A 1e-6 los tres métodos daban 98.2 % y quedaban mezclados dos fallos aritméticos
  con uno algorítmico.

  El conteo O(V²) ya no aplica al total
  -------------------------------------
  El 2-opt agrega O(V⁴) consultas como post-pass. La BÚSQUEDA (`generar_candidatas`)
  sigue en O(V²) — que es la propiedad que aporta el híbrido — pero el total no.
  `test_conteo_de_oraculo_es_cuadratico` mide ahora `generar_candidatas` sola; si se
  mide `stoer_wagner_queyranne` entero el test falla, y con razón.
  Costo medido del 2-opt: 9.7 ms contra 173 ms del Zeta a V=40 (5 % del total),
  3.7 ms a V=30. El tope `MAX_MASCARAS_2OPT = 200_000` existe porque a V≈100 serían
  ~10⁸ máscaras.

  Descartado como causa (no volver a investigarlo)
  ------------------------------------------------
  Ante una sospecha de que el Zeta o sus optimizaciones estuvieran mal: `analytic`
  es exhaustivo sobre EL MISMO oráculo y coincide con `phi` 96/96 con max|Δ| = 0
  en N5B. Si el Zeta, la eliminación del gather, las coordenadas delta o
  `preparar_ncubos` estuvieran rotos, `analytic` fallaría igual. Es el experimento
  decisivo y es barato: correr `analytic` sobre una red chica con `phi` disponible.

  Medir tiempos en lotes separados falsea
  ---------------------------------------
  N5B y N10A dieron 0.217 s y 0.210 s corridos sueltos contra 0.038 s y 0.068 s
  dentro del lote: el arranque de sesión (imports, primeras asignaciones en frío)
  se cuela en el barrido. Comparar siempre dentro de una misma corrida.
-->
