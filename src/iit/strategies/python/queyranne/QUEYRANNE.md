# Estrategia Queyranne

Minimización del MIP via algoritmo de Queyranne (1998).  
Complejidad: **O(D³·N)** vs O(2^(D-1)·N) de `analytic`.

---

## El problema

Dado un subsistema con N nodos y D dimensiones activas, encontrar la bipartición `(alcance, mecanismo)` que minimiza el EMD entre la distribución original y la del sistema cortado.

La función objetivo es la **función sustituta**:

```
f(A) = Σ_i  min( |mean_B(i) − pivot_i| ,  |mean_A(i) − pivot_i| )
```

donde:
- `A ⊆ dims` es el subconjunto de dimensiones en un lado del corte
- `B = dims \ A` es el complemento
- `mean_A(i)` = media de `data_i` con dims de A libres y B fijos en el estado pivote
- `mean_B(i)` = media de `data_i` con dims de B libres y A fijos en el estado pivote
- `pivot_i` = valor de distribución del nodo i en el estado inicial

Esta función es **simétrica**: `f(A) = f(B)`, porque el `min` conmuta.

---

## Por qué Queyranne

El algoritmo de **Queyranne (Math. Prog. 1998)** minimiza cualquier función simétrica `f: 2^V → R` en **O(|V|³)** evaluaciones oracle, usando el hecho de que las funciones submodulares simétricas tienen una estructura de "par colgante" (pendant pair) en el ordenamiento de adyacencia máxima.

Para funciones **no submodulares** (como la nuestra), el algoritmo no tiene garantía teórica de exactitud, pero Kitazono, Kanai & Oizumi (*Entropy* 2018) demuestran empíricamente **~97–100% de precisión** para funciones análogas (Φ_SI, Φ_G). Se añade un pre-pass de singletons para cubrir el caso de falla más común.

---

## Algoritmo paso a paso

### Fase 1 — Pre-pass de singletons (O(D·N))

Verifica los D singletons `f({d})` para cada dimensión d. En funciones no submodulares, el MAO puede no producirlos como pendant values; evaluarlos explícitamente cubre la mayoría de los casos de falla.

```
Para cada d en {0, ..., D-1}:
    val = f({d})
    Si val < mejor: mejor ← val, best_mask ← {d}
```

### Fase 2 — Loop de Queyranne (O(D³·N))

Trabaja con **supernodos**: inicialmente cada dimensión es su propio supernodo `{d}`. En cada iteración:

1. **MAO (Maximum Adjacency Ordering):** ordena los supernodos actuales por ganancia marginal de f.
   - Empezar con A = ∅ y `key[v] = f({v})` para cada supernodo v.
   - Repetir: elegir `u = argmax key[v]`, añadir u a A, actualizar `key[w] = f(A ∪ {w})` para todos los w restantes.
   - Los últimos dos en el orden son el **par colgante (s, t)**.

2. **Candidato:** `pendant_val = s_key = f(V \ {t}) = f({t})` por simetría. Si `pendant_val < mejor`: actualizar.

3. **Contracción:** fusionar s y t en un supernodo `s ∪ t`. Reducir |V| en 1.

Tras D−1 iteraciones, se han evaluado D−1 biparticiones candidatas.

```
V ← [{1<<d} para d en range(D)]

Mientras |V| > 1:
    (s, t, pendant_val) ← MAO(V, f)
    Si pendant_val < mejor: mejor ← pendant_val, best_mask ← t
    V ← V \ {s, t} ∪ {s | t}          # contracción bit a bit

Retornar (mejor, best_mask)
```

### Fase 3 — Derivar (alcance, mecanismo) desde best_mask (O(N))

Con `best_mask_A` = conjunto de dims en el lado A ganador:

```
Para cada nodo i:
    cost_alcance    = |mean_B(i) − pivot_i|   # si nodo i va a alcance, guarda A → ≈ mean_B
    cost_no_alcance = |mean_A(i) − pivot_i|   # si nodo i no va, guarda B → ≈ mean_A
    nodo_en_alcance[i] = (cost_alcance ≤ cost_no_alcance)

alcance   = {nodo i : nodo_en_alcance[i]}
mecanismo = {dim d  : bit d en best_mask_A}
```

---

## Oracle lazy con cache

El oracle `f(mask)` se evalúa **de forma lazy**: solo se computa cuando es pedido, y el resultado se guarda en caché. Al computar `f(mask_A)`, se obtiene `(mean_A, mean_B)` gratis también para `f(mask_B)` (por simetría), guardando ambos.

Durante el MAO solo se consultan O(D²) masks únicos por iteración × (D−1) iteraciones = **O(D³) masks totales**, en lugar de los 2^D del precompute completo de `analytic_optimized`.

Cada evaluación cuesta O(N · 2^k) con k = popcount(mask). Promedio k ≈ D/2.

---

## Complejidades comparadas

| Estrategia | Particiones evaluadas | Coste por evaluación | Total |
|---|---|---|---|
| `analytic` (naive) | O(2^(D-1)) | O(N · 2^k), k variable | O(N · 4^(D/2)) |
| `analytic_optimized` | O(2^(D-1)) | O(N) con precompute | O(N · 2^D) + precompute O(N · 3^D) |
| **`queyranne`** | **O(D−1) pendants + D singletons** | **O(N · 2^(D/2)) lazy** | **O(N · D³ · 2^(D/2))** |

Para D=18 (N=20 nodos):

| Estrategia | Tiempo medido |
|---|---|
| `analytic` | ~20 s |
| `queyranne` | ~415 ms |
| **Speedup** | **~49×** |

El speedup crece aproximadamente como `2^D / D³`: para D=10 → 5×, D=13 → 11×, D=16 → 31×, D=18 → 49×.

---

## Limitaciones

- **No garantiza el mínimo global** sobre todos los 2^(m+n) pares `(alcance, mecanismo)`. Igual que `analytic`, solo explora biparticiones basadas en subconjuntos de dimensiones.
- **No submodular:** la función sustituta no está probada submodular para este oracle. El pre-pass de singletons mitiga la principal causa de falla, pero no garantiza exactitud al 100% para casos muy no-submodulares.
- Para verificar exactitud absoluta, comparar contra `force` (BruteForce) en sistemas pequeños.

---

## Referencias

- Queyranne, M. (1998). *Minimizing symmetric submodular functions.* Mathematical Programming, 82(1–2), 3–12.
- Kitazono, J., Kanai, R., & Oizumi, M. (2018). *Efficient algorithms for searching the minimum information partition in integrated information theory.* Entropy, 20(3), 173.
- Hidaka, S., & Oizumi, M. (2018). *Fast and exact search for the partition with minimal information loss.* PLOS ONE, 13(9).
