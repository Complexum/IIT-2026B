# SWQ — Stoer-Wagner × Queyranne

Híbrido entre el andamiaje de **Stoer-Wagner** (JACM 1997) y el oráculo Zeta exacto de
**Queyranne** (Math. Prog. 1998) que ya usa `qn`. Estrategias registradas: `swq` y `swq_static`.

---

## 1. La función objetivo es un corte legítimo

Con `f(S) = f_cara(oráculo, alcance = EFFECT-part(S), mecanismo = ACTUAL-part(S))`:

| Propiedad | Por qué |
|---|---|
| `f(∅) = 0` | `m = 0` → `val_a = \|sumas[:,0]\| = \|δ en el pivote\| = 0` |
| `f(V) = 0` | `cmask = 0` → `val_b = 0`, y todos los nodos caen en alcance |
| `f(S) = f(V∖S)` | complementar intercambia `val_a ↔ val_b` **y** la rama del `where`; los dos swaps se cancelan |

Simétrica, anclada y no negativa: es exactamente la clase de función que Queyranne minimiza y de la
que Stoer-Wagner es el caso gráfico. (Verificado en `tests/strategies/test_swq.py`.)

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

Queyranne consulta el oráculo dentro del MAO: `f(A∪w)` para cada `w` restante, en cada paso →
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
| `swq_static` | `W[s] += W[t]` (SW puro) | O(V²), **1 batch upfront** | 100 % paralelizable (multiproc / CUDA / kernel C) |
| `swq` (default) | recalcula sólo la fila del supernodo: `W[st][v] = (f(st) + f(v) − f(st∪v))/2` | O(V²), en O(V) batches | sin drift |

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
4. 1-OPT      mover un vértice de lado (Kernighan-Lin) ≤2 batches
5. RECONSTRUIR el ganador con EMD real
```

Codificación de vértices sobre un entero de `V = D + N` bits:

```
bit  0 .. D-1     → (ACTUAL, dims[j])      candidato a mecanismo
bit  D .. D+N-1   → (EFFECT, indices[i])   candidato a alcance
```

Máscara → `(alcance, mecanismo)` es shift + máscara. `qn` paga hoy un `__flatten`/`sorted`
recursivo en cada consulta.

---

## 6. Complejidad y medición

| Etapa | Costo | Oráculo |
|---|---|---|
| Precómputo Zeta (`hyperfaces`, compartido con `analytic`/`qn`) | `O(D·N·2^D)`, una vez | — |
| Seed de `W` | `O(V²·N)` flops | 1 batch |
| MAO + contracciones | `O(V³)` flops vectorizados | 0 (estático) / `O(V²)` en O(V) batches (exacto) |
| Re-scoring + 1-opt | `O(V·N)` | 1–3 batches |

**Nota:** Stoer-Wagner NO es O(n²) — es `O(V³)` denso, `O(VE + V² log V)` con heap. Lo que baja a
`O(V²)` es el **conteo de consultas al oráculo**.

**Nota 2:** la cota de Queyranne es `O(V³)` consultas, pero el `memoria_bipart` de `qn` colapsa las
repetidas y en la práctica evalúa ~1.3·V² cortes distintos (V=30: 1193 vs V³=27000). La ganancia
real de SWQ no es el conteo sino la **forma**: `qn` paga un round-trip Python→numpy por consulta,
SWQ agrupa las mismas lecturas en O(V) batches.

Medido (V = D+N, TPMs sintéticas, `scratchpad/split2.py`):

| V | zeta (compartido) | búsqueda `swq` | búsqueda `swq_static` | TOT `analytic` | TOT `qn` | TOT `swq` |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.0098 s | 0.0022 s | 0.0008 s | 0.0135 s | 0.0267 s | 0.0108 s |
| 36 | 0.0467 s | 0.0019 s | 0.0010 s | 0.0611 s | 0.0633 s | 0.0420 s |
| 40 | 0.2849 s | 0.0023 s | 0.0012 s | 0.3361 s | 0.2159 s | 0.1885 s |
| 44 | 1.7642 s | 0.0028 s | 0.0014 s | 2.4364 s | 2.0888 s | 1.2417 s |

La búsqueda de SWQ es prácticamente **plana** (2.2 → 2.8 ms de V=32 a V=44). A partir de V≈40 el
costo de `swq` *es* el precómputo Zeta —inherente a los datos, compartido con `analytic` y `qn`—.
`analytic` en cambio sigue enumerando `2^(D−1)` máscaras.

Consultas al oráculo, sistema completo N15A/N15B (D=N=15, V=30):

| | consultas | batches |
|---|---:|---:|
| `qn` | 1193 – 1825 | 1193 – 1825 |
| `swq` | 871 | **29** |
| `swq_static` | 490 | **2** |

**Exactitud:** 96/96 combinaciones de `patron-2` sobre N15A coinciden con `analytic` (exacto),
error relativo máximo 0.00 %, para `swq`, `swq_static`, `qn` y `queyranne`. En el barrido sintético
n = 10…20 tampoco hay ninguna divergencia.

---

## 7. Trabajo pendiente

- **Kernel C** — `src/iit/strategies/clang/swq/code.c` (hoy vacío). El `O(V³)` restante es
  aritmética densa; portarlo junto con la lectura del oráculo (`sumas` es un `float*`):

  ```c
  int swq_solve(const float *sumas, int N, int D, int V,
                const int *vert_kind, const int *vert_slot, int modo,
                uint64_t *out_candidatos, double *out_valores, int *out_n);
  ```

  `V ≤ 64` → `uint64_t`. Compilar a `clang/__cache__/libswq.so`, cargar con `ctypes`, fallback
  silencioso a Python. **Nota de prioridad:** con la búsqueda ya en ~3 ms, el kernel C rinde poco
  ahí; el blanco real es el **precómputo Zeta**, que es donde se va el 99 % del tiempo a V≥40.

- **Paralelización** — `swq_static` deja todo el oráculo en un único batch: es el candidato natural
  para multiprocessing / CUDA. Igual que arriba, el Zeta manda.

- **Zeta in-place** — `analytic.hyperfaces` asigna un segundo arreglo `N·2^D`; hacerlo sobre
  `delta_nd` reduce el pico de memoria a la mitad (3.35 GB → 1.7 GB a D=N=25).

- **`queyranne/code.py:111`** — su regla MAO usa `max f(A∪w)`; Queyranne exige
  `min f(A∪w) − f(w)`. Rompe la garantía de par colgante. Además no usa el oráculo Zeta, y por eso
  es la estrategia más lenta del conjunto (2.27 s a V=40, 10× peor que `swq`).
