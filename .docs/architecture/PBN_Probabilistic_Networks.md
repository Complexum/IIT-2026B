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

## 6. El régimen disperso: tres mecanismos, tres contramedidas

Este es el resultado principal, y corrige dos explicaciones previas que eran
incompletas (`Input_Representations.md` §7.1 original, y la primera versión de esta
sección).

`φ = 0` en el régimen disperso no tiene **una** causa sino tres, y sólo aplicando las
tres contramedidas a la vez el régimen deja de ser trivial.

### Mecanismo 1 — corte cerrado (topológico)

Si `A ∩ padres(i) = ∅`, la cara del nodo `i` colapsa a la celda del pivote, donde
`δᵢ = 0` **por construcción**. Basta entonces con que exista un `A` donde cada nodo
tenga *todos* o *ninguno* de sus padres.

No depende de los valores: verificado con el mismo corte sobre tres redes (n=20,
k=2), `|A| = 9/20`, alcance de 11 nodos:

```
binario : f = 0.0000000000
PBN L=3 : f = 0.0000000000
ruido .2: f = 0.0000000000
```

**Ni las probabilidades ni el ruido lo tocan.** Y sorteando padres uniformemente casi
siempre aparece: n=20, `aleatoria`, **20/20** semillas separables con k=2, 11/20 con
k=3, 3/20 con k=5.

→ **Contramedida: `topologia="anillo"`.** Fija `{i−1, i}` como columna vertebral, así
la clausura de cualquier dim recorre el ciclo entero. Medido: **0/20 separables** con
k=2, 3 y 5.

### Mecanismo 2 — corte de una sola dim con alcance vacío

Con `A = {d}` y alcance `∅`:

```
mean_{d}(δᵢ) = ( δᵢ(pivote) + δᵢ(pivote ⊕ eᵈ) ) / 2
```

y el primer término es 0 por construcción. Con tablas **binarias** el segundo vale 0
o ±½, así que se anula exacto en cuanto voltear esa dim no cambia la salida de sus
hijos en el estado pivote — probabilidad ~¼ por hijo. Con k=2 cada dim tiene 2 hijos,
y la búsqueda prueba las `D` dims: con n=12 la chance de que alguna funcione es ~97 %.

Con tablas **probabilísticas** ese segundo término es un real genérico y la
cancelación exacta pasa a ser de medida cero. **Acá sí aplica el argumento
probabilístico** — al término de volteo simple, no a la constancia sobre la cara como
decía la versión anterior.

→ **Contramedida: PBN.**

### Mecanismo 3 — el estado pivote

Todas las mediciones previas usaban el estado **todo-ceros**, donde muchos nodos
resultan insensibles a un volteo simple. Eso enmascaraba los otros dos.

→ **Contramedida: barrer estados iniciales.**

### Medido

Fracción de estados con `φ = 0`, n=14, 24 estados al azar:

| modelo | topología | k | separable | ceros | mediana φ | max φ |
|---|---|---|---|---|---|---|
| binario | aleatoria | 2 | sí | 24/24 | 0.000000 | 0.000000 |
| binario | anillo | 3 | no | 23/24 | 0.000000 | 0.250000 |
| ruido=0.20 | anillo | 3 | no | 23/24 | 0.000000 | 0.150000 |
| PBN L=3 | aleatoria | 2 | sí | 24/24 | 0.000000 | 0.000000 |
| PBN L=3 | aleatoria | 3 | no | 1/24 | 0.006105 | 0.110471 |
| PBN L=3 | anillo | 2 | no | 13/24 | 0.000000 | 0.019374 |
| **PBN L=3** | **anillo** | **3** | **no** | **1/24** | **0.026536** | **0.057752** |

Lecturas:

- **El binario no se salva con nada.** Con anillo y k=3 sigue dando 23/24 ceros: el
  mecanismo 2 lo domina.
- **El PBN solo tampoco alcanza** si la topología es separable (24/24 con k=2).
- **Los dos juntos, con k=3, funcionan**: mediana 0.0265, y el anillo cuadruplica la
  mediana respecto de `aleatoria` (0.0265 vs 0.0061).
- `k=2` sigue siendo marginal aun con todo aplicado (13/24 ceros): hay pocos hijos por
  dim y el término de volteo simple tiene pocos sumandos.

### Y sigue sin ser un error

Una red que se separa limpio, o en cuyo estado actual una dim no afecta a nadie, **no
está integrada**. `φ = 0` es la respuesta correcta de IIT. Lo que hacía falta no era
arreglar el cálculo sino **generar redes que no fueran triviales**.

---

## 7. Dónde queda esto

El régimen donde el oráculo local gana ya **no** es trivial: `PBN L=3` + `anillo` +
`k=3` da `φ > 0` en 23 de 24 estados. Eso desbloquea medir el oráculo local contra la
ruta tradicional sobre redes que no son degeneradas.

`src/io/topologia.py` expone `es_separable(padres)` para que un `φ = 0` se lea como
propiedad conocida de la red y no como sospecha del algoritmo. Conviene que el
benchmark lo reporte siempre, como ya hace.

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
