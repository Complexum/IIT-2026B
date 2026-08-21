# Representaciones de entrada: romper el `2^D` de la lectura

Estado a 2026-08-21, rama `IIT3-Input-representations`.

Este documento existe para retomar el trabajo sin contexto
previo. Explica **qué problema se ataca, por qué la representación actual lo causa,
y qué se construyó**.

---

## 1. El problema, en una frase

El pipeline trabaja con la TPM **nodo-estado**: una matriz de `2^n × n` donde la
fila es el estado del sistema en `t` y la columna es la probabilidad de que ese
nodo esté encendido en `t+1`.

A n=15 son 32 768 filas. A n=20, un millón. **La complejidad de sólo leer la
entrada ya es exponencial**, así que ninguna mejora en el algoritmo de búsqueda
puede evitarla. Ese es el techo que este trabajo ataca.

La observación central:

> El `2^n` **no viene del problema. Viene de la representación.**

Un nodo con `k` padres se describe con una tabla de verdad de `2^k` entradas.
Materializarla como una columna de `2^n` la replica `2^(n−k)` veces. A n=20 con
k=5 son 32 768 copias idénticas por nodo. La TPM es, en su mayor parte, relleno.

---

## 2. Las tres representaciones

| # | Representación | Forma | ¿Se usa? |
|---|---|---|---|
| 1 | **Especificación (CNF)** | por nodo: fórmula booleana sobre sus padres | **sí — es la fuente** |
| 2 | **TPM nodo-estado** | `2^n × n` | sí, es el CSV de siempre — ruta tradicional y validación |
| 3 | **TPM estado-estado** | `2^n × 2^n` | **no**, nunca se guarda |

La (3) es la matriz que el sistema original define conceptualmente, pero es
`O(4^n)` y no se materializa jamás. Sólo se construye en tests de n≤6 para
demostrar que la (2) es su marginalización fiel (round-trip exacto, `Δ = 0`).

La (2) **no desaparece**: es lo que consumen `analytic`, `qn`, `qsw` y PyPhi, y es
contra lo que se valida todo lo nuevo.

La (1) es lo nuevo. Formato DIMACS, un archivo `.spec.json` junto al CSV:

```json
{
 "1": [[2, 3]],
 "2": [[1], [3]],
 "3": [[1, 2], [-1, -2]]
}
```

- Variable `v > 0` ↔ nodo `v − 1`; literal negativo `-v` = negado.
- Lista interna = **OR** (cláusula). Lista de listas = **AND** (fórmula).
- `[]` = constante 1 (AND vacío). `[[]]` = constante 0 (cláusula vacía).
- **Nada se declara dos veces**: `n` sale de la cantidad de claves, los padres de
  las variables que cada fórmula menciona, la adyacencia de ahí.

Esa red es `A = B∨C ; B = A∧C ; C = A xor B`.

> Las claves son strings porque **JSON exige claves string**. Se convierten con
> `int(k)` al cargar. Alternativa pendiente: usar un array y que el índice sea
> `nodo − 1`, ya que exigimos claves contiguas `1..n`.

---

## 3. Qué necesita el algoritmo (y por qué existe un precómputo)

El algoritmo de búsqueda (Queyranne / QSW) sólo habla con un **oráculo**: dada una
bipartición `S`, devolver su costo.

```
f(S) = Σᵢ |mean_{Fᵢ}(δᵢ)|
```

donde `δᵢ = Hᵢ − pᵢ` es el hipercubo del nodo `i` menos su valor en el estado
pivote, y `Fᵢ` es una **cara**: se dejan variar libremente las dims del mecanismo
(o su complemento, según si `i` está en el alcance) y el resto queda fijo en el
pivote. La media es sobre las `2^|libres|` celdas de esa cara.

**Por qué hay un precómputo.** Calcular una cara a mano cuesta `2^|libres|`
lecturas. La búsqueda pide muchas caras. La **transformada Zeta** (suma sobre
subconjuntos, patrón *butterfly*) calcula **todas** las `2^D` sumas de cara de una
vez en `D · 2^D` operaciones, en vez de `3^D` sumándolas por separado. Después,
cada consulta del oráculo es **una lectura de tabla**.

Ese es el trato actual: se paga `Θ(D · N · 2^D)` una vez, y las consultas salen
gratis. Medido, ese precómputo **es** el 100 % del costo: a D=22 son ~1 s contra
2.8 ms de búsqueda.

---

## 4. La idea: correr la Zeta sobre la tabla de cada nodo

`Hᵢ` **sólo depende de sus padres**. Las dims libres de la cara que no son padres
de `i` no cambian su valor: promediar sobre ellas es promediar una constante.
Entonces

```
mean_A(δᵢ) = mean_{A ∩ padres(i)}(δᵢ)
```

La cara global colapsa a una cara de la **tabla local** del nodo, de tamaño
`2^|A ∩ padres(i)|`. Así que se corre la misma Zeta, pero sobre `2^kᵢ` celdas:

```
Θ(Σᵢ kᵢ · 2^kᵢ)        en vez de        Θ(D · N · 2^D)
```

Es **exacto**, no muestreado ni aproximado.

Concretamente para el caso que mencionaste, n=15 con k=3:

| | celdas | operaciones de la Zeta |
|---|---:|---:|
| TPM materializada | 15 × 32 768 = 491 520 | 15·15·2¹⁵ = **7 372 800** |
| tablas locales | 15 × 8 = 120 | 15·3·8 = **360** |

**20 480× menos.** Y las 32 768 filas que no se pueden pagar simplemente no
existen: son 8 celdas por nodo.

### El piso de lectura también se cae

En análisis previos se afirmó que `Ω(N·2^D)` era un piso para cualquier método
exacto, porque la entrada son `N·2^D` probabilidades. **Eso era una propiedad de
la representación, no del problema.** Con la CNF como entrada, el tamaño de la
entrada es `Σ 2^kᵢ` y el piso baja con ella.

### No hace falta un algoritmo de búsqueda nuevo

`stoer_wagner_queyranne` corre **sin modificar**: sólo cambia con quién habla. Eso
es deliberado — toda la validación de la búsqueda se hereda intacta. Explotar la
estructura del grafo para *podar la búsqueda* (no sólo acelerar el oráculo) es una
idea aparte, no explorada.

---

## 5. Qué se construyó

| archivo | responsabilidad |
|---|---|
| `src/io/cnf.py` | fórmulas CNF: validar, inferir variables, compilar a tabla, tabla → CNF canónica, formatear |
| `src/io/fst.py` | `EspecFST`, derivación de las TPMs, persistencia, generación de redes |
| `src/iit/strategies/python/zeta_local.py` | **el oráculo**: Zeta por nodo, sin materializar nada |
| `src/io/generator.py` | intacto — el generador aleatorio de siempre |

`zeta_local.py` importa exactamente dos cosas del proyecto: `EspecFST` y
`zeta_inplace`. **No conoce la TPM.** `f_cara_batch_local` tiene la misma interfaz
que `qn.oracle.f_cara_batch`, así que es un reemplazo directo.

### Por qué existía un generador que arruinaba todo

`generator.generar_red` sortea **cada celda de la TPM por separado**:

```python
estados = np.random.randint(2, size=(num_estados, dimensiones), dtype=np.int8)
```

Sin grafo, sin aristas, sin función por nodo. Eso produce una función booleana
aleatoria **densa**: medido sobre N20A, los 400 pares (nodo, dim) tienen influencia
en `[0.498, 0.502]` — exactamente 1/2, la firma de una función booleana aleatoria.
Es el peor caso posible para cualquier método que explote estructura, y explica por
qué las redes existentes son incompresibles.

`fst.generar_red_booleana(n, k)` genera desde la estructura y escribe ambas cosas:
el CSV (para la ruta tradicional) y el `.spec.json`.

---

## 6. Validación

**Igualdad exacta con el oráculo actual**, sobre *todos* los cortes (`2^V`
máscaras), *todos* los estados iniciales, red escrita a mano y generadas:

```
red escrita a mano   fan_in=(2,2,2)    64 cortes × 8 estados     max|Δ| = 0.000e+00
n=3  k=1,2,3                           64 cortes × 8 estados     max|Δ| = 0.000e+00
n=6  k=1,2,3,4                       4096 cortes × 64 estados    max|Δ| = 0.000e+00
```

**El pipeline tradicional sobre las redes generadas** coincide con PyPhi:
`phi = analytic = qsw` en los 8 estados × 3 fan-ins (n=3).

**Round-trip de representaciones**: estado-estado → marginalizar → nodo-estado da
`Δ = 0`; las dims que `NCube` ve como efectivas son exactamente `espec.padres`
(prueba de endianness contra el pipeline, no contra el generador).

### Escalado (búsqueda QSW completa sobre el oráculo local, k=3)

| n | V | precómputo | búsqueda | Σkᵢ2^kᵢ | D·N·2^D | la TPM pesaría |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 40 | 0.09 ms | 20 ms | 480 | 4.2×10⁸ | 0.1 GB |
| 30 | 60 | 0.12 ms | 80 ms | 720 | 9.7×10¹¹ | 128.8 GB |
| 40 | 80 | 0.15 ms | 9.8 ms | 960 | 1.8×10¹⁵ | 175 921 GB |
| 60 | 120 | 0.21 ms | 30 ms | 1 440 | 4.2×10²¹ | 2.8×10¹¹ GB |

### Tests

`tests/dataset/test_cnf.py` · `tests/dataset/test_fst.py` ·
`tests/strategies/test_zeta_local.py` — 58 en total, todos verdes.

---

## 7. Problemas abiertos

### 7.1 En el régimen disperso, `φ = 0` — y la causa es topológica

> **Corregido.** La explicación original de esta sección era incorrecta. Decía que el
> cero venía de que `Hᵢ, pᵢ ∈ {0,1}` hacían frecuente que la función fuera constante
> sobre la cara, y que por lo tanto usar probabilidades lo arreglaría. **No lo
> arregla.** Detalle completo en `PBN_Probabilistic_Networks.md` §6.

`φ` mínimo, n=20, mediana de 3 semillas:

| modelo | k=2 | k=3 | k=5 |
|---|---|---|---|
| binario | 0.000000 | 0.000000 | 0.281250 |
| ruido=0.05 | 0.000000 | 0.000000 | 0.253125 |
| ruido=0.20 | 0.000000 | 0.000000 | 0.168750 |
| PBN L=3 | 0.000000 | 0.000000 | 0.011519 |

La causa real: si `A ∩ padres(i) = ∅`, la cara de ese nodo colapsa a la celda del
pivote, donde `δᵢ = 0` **por construcción**. El término se anula sin importar los
valores de la tabla. Entonces `f(S) = 0` en cuanto exista un corte **cerrado** — un
conjunto de dims `A` tal que cada nodo tenga *todos* o *ninguno* de sus padres en `A`.

Verificado construyéndolo explícitamente (n=20, k=2): `|A| = 9/20`, alcance de 11
nodos, `f = 0.0000000000` idéntico en binario, PBN y ruido.

No es un error de cálculo: una red dispersa cuyo grafo se separa limpio **no está
integrada**, y `φ = 0` es la respuesta correcta. El problema es la topología que
genera `elegir_padres`: sorteando padres uniformemente con `k` chico casi siempre
aparece un subconjunto cerrado.

**La salida es topológica, no probabilística**: generar grafos dispersos pero **no
separables** (anillo con atajos), donde la clausura de cualquier conjunto de padres
sea todo el grafo.

### 7.2 Las redes existentes no tienen spec — medido

`src/io/identificabilidad.py` responde si una spec compacta es siquiera encontrable
para una TPM dada, distinguiendo dos maneras de que no lo sea:

    uv run python -m src.io.identificabilidad N10A N15A N20A

```
N10A  n=10
  influencia por (nodo,dim): min=0.4531 mediana=0.5000 max=0.5566
  piso de ruido: 0.0000  [determinista (sin muestreo)]
  fan-in estimado por nodo: min=10 max=10
  -> DENSA — algún nodo depende de las n dims; sin spec compacta

N15A  n=15
  influencia por (nodo,dim): min=0.0419 mediana=0.0432 max=0.0442
  piso de ruido: 0.1265  [≈20 muestras/estado]   ratio mediana/ruido = 0.34
  fan-in estimado por nodo: min=0 max=0
  -> NO IDENTIFICABLE — toda la señal está bajo el ruido de muestreo

N20A  n=20
  influencia por (nodo,dim): min=0.4982 mediana=0.5000 max=0.5021
  piso de ruido: 0.0000  [determinista (sin muestreo)]
  fan-in estimado por nodo: min=20 max=20
  -> DENSA — algún nodo depende de las n dims; sin spec compacta
```

**N10A y N20A** son deterministas con influencia 1/2 en cada par (nodo, dim): la
firma de una función booleana aleatoria densa, que es exactamente lo que produce
`generator.generar_red` sorteando cada celda por separado. No hay compresión posible
**por conteo** — una función booleana aleatoria de `n` variables necesita `2^n` bits.
Ninguna técnica lo evade, tampoco una red neuronal.

**N15A** es distinto y peor: es dato medido, y sus dependencias son *más chicas que
su propio error de estimación*. No es un problema de algoritmo sino de datos.

Y no es que el diagnóstico sea laxo: una red `k=2` determinista medida con sólo 8
muestras por estado **sigue** dando IDENTIFICABLE, porque sus influencias (~0.5–1.0)
superan el piso (~0.2). Las dependencias fuertes sobreviven al ruido pesado. Las de
N15A son genuinamente **débiles**, no sólo ruidosas.

**Conclusión operativa:** para las redes viejas hay que ir a buscar la fuente del
sistema, no invertir la matriz. Para sistemas reales eso siempre existe — la TPM sale
de muestrear un FST, y el FST *es* la spec.

### 7.3 Falta condicionar y substraer

`preparar_oraculo_local` asume el sistema completo (`dims = indices = 0..n−1`), que
es lo que dan los `Params("...", "1"*n, "1"*n, "1"*n)` de los tests. Para enchufarlo
a `reducir_a_subsistema` hay que aplicar las dos operaciones sobre la tabla local:
condicionar es un slice, marginalizar es un promedio sobre un eje — ambas `O(2^kᵢ)`.
**No implementado.**

### 7.4 Todavía no es una estrategia registrada

El oráculo existe y está validado, pero no hay una clase `SIA` que lo use. El patrón
a seguir es el de `qsw_mul` / `qsw_cuda`: sobrescribir sólo `_preparar_oraculo`.

### 7.5 `Fatal Python error: Bus error` en `tests/dataset/test_io.py`

Preexistente y ajeno a este trabajo, pero es una bomba. `generar_red` lanza
`optimizar_red` en un hilo *fire-and-forget* que escribe un `.npy`; otro test lo
mmapea a medio escribir → SIGBUS (`N25A.npy` pesa 6.7 GB). `fst.generar_red_booleana`
copió el mismo patrón, así que hereda la carrera.

### 7.6 Deuda menor

- `fst.py` duplica de `generator.py`: `REDES_DIR`, `_optimizar_executor`, y la
  lógica de sufijo + `np.savetxt`. Se arregla con un `escribir_red()` compartido.
- Máscaras: a n≥32 las de `V = 2n` bits no entran en `int64`. Hay que partir
  mecanismo/alcance **en Python antes** de bajar a numpy, como hace
  `qsw/code.py:__winner_zeta`.
- El fix del default de `modo` en QSW está en `main` (commit `33d5e55`) pero **no en
  esta rama**, que salió de `5d1e628`.

---

## 8. Para retomar

El orden que tiene sentido:

1. **Lo continuo** (§7.1) — sin eso el régimen disperso es trivial y no hay nada
   que medir.
2. **Condicionar y substraer sobre la tabla local** (§7.3) — sin eso no se enchufa
   al pipeline real.
3. **Estrategia registrada** (§7.4) — para que entre a `cli run` y `cli compare`.
4. **Benchmark honesto**: redes generadas desde spec, comparando la ruta local
   contra `analytic`/`qsw` sobre la misma red, a los n donde ambas corren.
