# Redes probabilísticas (PBN): implementación y resultado

Estado a 2026-08-21, rama `IIT3-Input-representations`.
Continúa `Input_Representations.md`, que cubre la representación CNF y el oráculo
por Zeta local. **Corrige su §7.1** — ver la sección 6 de acá.

---

## 1. El problema que motivó esto

Las redes booleanas deterministas sólo producen TPMs con valores en `{0, 1}`. Los
datos reales no: N15A tiene 114 valores distintos (`1/68`, `1/47`, `1/46`, …), que
son frecuencias empíricas. Hacía falta que la especificación compacta pudiera
generar probabilidades sin dejar de ser compacta.

Además había una hipótesis a probar: que el régimen disperso binario diera `φ = 0`
**por ser binario**, y que las probabilidades lo arreglaran. **Esa hipótesis era
falsa** (sección 6).

---

## 2. El modelo: Probabilistic Boolean Networks

Shmulevich, Dougherty, Kim y Zhang, *Bioinformatics* 18(2):261–274 (2002).

Cada nodo tiene un **conjunto** de funciones booleanas y en cada paso elige una al
azar con una probabilidad de selección. La probabilidad de encendido es la mezcla
convexa:

```
P(nodo i = 1 | padres) = Σ_l  c_l · f_l(padres)        con  Σ_l c_l = 1
```

Es literalmente "a veces la operación da 1 y a veces 0". Y encaja sin fricción: el
resultado sigue siendo una tabla de `2^k` floats, que es lo que `EspecFST.tablas` ya
guardaba.

Casos particulares útiles:

| caso | forma |
|---|---|
| determinista | una sola fórmula con `c = 1` |
| volteo de salida (ruido-ε) | `[(1−ε, f), (ε, ¬f)]` → `P = ε + (1−2ε)·f` |
| mezcla general | `L` fórmulas con pesos aleatorios |

---

## 3. Formato

Deterministas y probabilísticos conviven en el mismo archivo:

```json
{
 "1": [[2, 3]],
 "2": [[0.7, [[1], [3]]], [0.3, [[1, 2]]]],
 "3": [[1, 2], [-1, -2]]
}
```

- Nodo 1: `B ∨ C`, siempre.
- Nodo 2: `A ∧ C` el 70 % de las veces, `A ∨ B` el 30 %.
- Nodo 3: `(A ∨ B) ∧ (¬A ∨ ¬B)` = `A xor B`, siempre.

### La regla de discriminación

Es el único punto ambiguo del formato, así que vive en un solo lugar
(`pbn.es_pbn`) y está testeada:

> Una **cláusula** es una lista plana de enteros. Una **entrada PBN** es un par
> `[probabilidad, lista_de_cláusulas]`, cuyo segundo elemento es una lista.

```
[[2, 3]]                          → determinista (una cláusula)
[[1], [3]]                        → determinista (dos cláusulas unitarias)
[[0.7, [[1], [3]]], [0.3, [[2]]]] → PBN
[]                                → determinista, constante 1
[[]]                              → determinista, constante 0
```

No hay colisión: `spec[0][1]` es un entero en una cláusula y una lista en un par PBN.

### Padres

Se infieren como la **unión** de las variables de todas las fórmulas del nodo. Cada
fórmula se evalúa sobre esa unión (`cnf.evaluar` acepta un superconjunto de sus
propias variables). Nada se declara dos veces.

---

## 4. Implementación

| archivo | responsabilidad |
|---|---|
| `src/io/cnf.py` | fórmulas CNF. Se le agregó `depende_de_todas(tabla, k)` — propiedad de una tabla, compartida |
| `src/io/pbn.py` | **nuevo**. `es_pbn`, `compilar`, `generar_formulas`. Sólo sabe de fórmulas de un nodo |
| `src/io/fst.py` | `EspecFST`, TPMs, persistencia. Compone: `elegir_padres` + `pbn.generar_formulas` |
| `src/iit/strategies/python/zeta_local.py` | **sin cambios** |

### Por qué `pbn.py` no importa `fst.py`

`pbn.generar_formulas` **recibe la topología** en vez de elegirla. La selección de
padres vive en `fst.elegir_padres` y la comparten el generador binario y el PBN. Así
no hay ciclo de importación y no se duplica la topología.

### Lo que NO hubo que tocar: el oráculo

La derivación del oráculo local es

```
mean_A(δᵢ) = mean_{A ∩ padres(i)}(δᵢ)
```

y **sólo usa que `Hᵢ` depende únicamente de sus padres**. Nunca usó que fuera
binaria. Verificado antes de escribir una línea de PBN: pasándole tablas con floats
en `(0,1)`, `max|Δ| = 1.5e-07` sobre 1024 cortes × 32 estados — el piso de float32
del oráculo de referencia, que acumula en float32 mientras el local usa float64.

### Fan-in estricto sobre la mezcla

Una tabla aleatoria puede salir constante; su CNF canónica es `[]` o `[[]]`, que no
mencionan variables, y el nodo queda con **fan-in 0**. Con `ruido=0.1` eso apareció
de inmediato (`fan_in = (2,2,2,2,0)`).

Lo que se controla es la **mezcla** `Σ c_l f_l`, porque es la que ve el oráculo, no
las fórmulas sueltas. `pbn.generar_formulas` re-sortea hasta que la mezcla dependa de
sus `k` padres.

---

## 5. Qué se verificó

### Cruce contra `analytic` sobre la TPM materializada — 16/16

Misma red, dos rutas: oráculo local desde la spec, contra `analytic` exhaustivo
sobre la TPM. n=8, `k ∈ {2,3}`, cuatro modelos, dos estados.

```
modelo       k   estado   local        analytic     ok
binario      3   11111111 0.250000     0.250000     ok
ruido=0.05   3   11111111 0.225000     0.225000     ok
ruido=0.20   3   11111111 0.150000     0.150000     ok
PBN L=3      2   00000000 0.000495     0.000495     ok
PBN L=3      3   11111111 0.127127     0.127127     ok
```

**16 de 16.** El oráculo local sigue siendo exacto con tablas no binarias.

### Persistencia

Una red mixta (nodos deterministas y PBN en el mismo archivo) sobrevive
guardar → cargar con fórmulas **idénticas** y tablas idénticas. La CNF escrita a
mano se conserva; no se reescribe como canónica.

### Tests

`tests/dataset/test_cnf.py` · `tests/dataset/test_fst.py` ·
`tests/strategies/test_zeta_local.py`.

---

## 6. El resultado negativo, y la corrección a `Input_Representations.md` §7.1

### Lo que decía ese documento (incorrecto)

> `mean_A(δᵢ) = mean_A(Hᵢ) − pᵢ`. Con `Hᵢ, pᵢ ∈ {0,1}` el término es 0 **sii la
> función del nodo es constante sobre esa cara**. […] Con probabilidades en `(0,1)`
> la constancia exacta pasa a ser de medida cero.

De ahí se seguía que PBN arreglaría la trivialidad. **No la arregla.**

### Lo medido

`φ` mínimo sobre el oráculo local, n=20, mediana de 3 semillas:

| modelo | k=2 | k=3 | k=5 |
|---|---|---|---|
| binario | 0.000000 | 0.000000 | 0.281250 |
| ruido=0.05 | 0.000000 | 0.000000 | 0.253125 |
| ruido=0.20 | 0.000000 | 0.000000 | 0.168750 |
| PBN L=3 | 0.000000 | 0.000000 | 0.011519 |

Con `k` chico da 0 **en todos los modelos**. Y en k=5 el PBN da *menos* φ que el
binario, no más: mezclar funciones vuelve al nodo menos determinista y por lo tanto
**menos** integrado.

### La causa real: es topológica, no de los valores

Si `A ∩ padres(i) = ∅`, la cara de ese nodo colapsa a la celda del pivote, donde
`δᵢ = 0` **por construcción**. El término se anula sin importar qué tenga la tabla.

Entonces `f(S) = 0` en cuanto exista un corte **cerrado**: un conjunto de dims `A`
tal que cada nodo tenga *todos* o *ninguno* de sus padres en `A`. Se pone en el
alcance a los nodos con todos sus padres en `A` y afuera a los demás.

Verificado construyendo el corte explícitamente en una red de n=20, k=2:

```
binario : |A| = 9/20  |alcance| = 11  ->  f = 0.0000000000
PBN L=3 : |A| = 9/20  |alcance| = 11  ->  f = 0.0000000000
ruido .2: |A| = 9/20  |alcance| = 11  ->  f = 0.0000000000
```

Idéntico en los tres. **Las probabilidades no pueden tocar esto** porque el cero no
viene de los valores.

### Y no es un error: es la respuesta correcta de IIT

Una red dispersa cuyo grafo de dependencias se separa limpio **no está integrada**.
`φ = 0` es lo que corresponde. El problema no es el cálculo, es la **topología** que
genera `elegir_padres`: al sortear padres uniformemente con `k` chico, casi siempre
aparece un subconjunto cerrado.

---

## 7. La tensión, y qué sigue

El régimen donde el oráculo local gana (`k` chico) es justo donde la respuesta es
trivial. Pero la salida ahora está identificada, y **no es probabilística**:

> Generar topologías **dispersas pero no separables**: que la clausura de cualquier
> conjunto de padres sea todo el grafo, de modo que no exista corte cerrado propio.

Un anillo con atajos (cada nodo toma como padre a su vecino más `k−1` al azar) lo
consigue con `k` chico. Eso es una perilla de `elegir_padres`, no de `pbn.py`.

El chequeo es barato y ya está escrito: buscar la clausura de cada conjunto de padres
y verificar que ninguna sea un subconjunto propio. Conviene volverlo:

1. una **validación del generador** — rechazar o marcar redes separables;
2. un **test** — una topología de anillo no debe tener corte cerrado propio;
3. una **columna del benchmark** — reportar si la red admite corte cerrado, para que
   un `φ = 0` se lea como propiedad de la red y no como sospecha del algoritmo.

### Pendiente heredado de `Input_Representations.md`

- §7.3 condicionar / substraer sobre la tabla local — sin eso no se enchufa al
  pipeline real.
- §7.4 registrar una estrategia `SIA` que use el oráculo local.
- §7.5 `Fatal Python error: Bus error` en `tests/dataset/test_io.py`.
- El fix del default de `modo` está en `main` (`33d5e55`), no en esta rama.

---

## 8. Reproducir

```bash
uv run python benchmarks/pbn_no_trivial.py 20 8
uv run pytest tests/dataset/test_cnf.py tests/dataset/test_fst.py \
              tests/strategies/test_zeta_local.py -q
```
