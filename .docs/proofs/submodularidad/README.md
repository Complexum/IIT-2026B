# ¿Es submodular el objetivo EMD de QSW?

**No, y está demostrado.** Hay dos contraejemplos explícitos, mecanizados en Lean 4 en
[`demo/`](demo/) y evaluados contra el código que corre en [`numerico.py`](numerico.py).

```
submodularidad/
├── README.md      este documento — generalidades, dónde afecta, conclusiones
├── numerico.py    evidencia empírica reproducible, sobre el `f_cara` real
└── demo/          la demostración formal en Lean 4 + mathlib
```

---

## 1. Generalidades

### La función

El MIP es una bipartición `S ⊆ V = 𝒟 ⊎ ℐ`: las `D` dimensiones activas `𝒟` (candidatas a
*mecanismo*, en `t`) y los `N` índices `ℐ` (candidatos a *alcance*, en `t+1`). Escribiendo
`A = mecanismo(S)`, `B = alcance(S)`, `δᵢ = Hᵢ − pᵢ` para la normalización con signo del
hipercubo `i` contra su pivote, y `δ̄ᵢ(A)` para la media de `δᵢ` sobre la cara donde las
dimensiones de `A` corren libres, el objetivo es

```
f(S) = Σ_{i ∉ B} |δ̄ᵢ(A)|  +  Σ_{i ∈ B} |δ̄ᵢ(Aᶜ)|
```

que es exactamente `f_cara` de `src/iit/strategies/python/qsw/oracle.py:56-83`: en el código
las medias se leen de una tabla Zeta precomputada, pero son estas.

### La propiedad

`f` es **submodular** si `f(S ∪ T) + f(S ∩ T) ≤ f(S) + f(T)` para todo par de cortes. Dicho
de otro modo: el retorno de mover un vértice de lado nunca crece al agrandar el conjunto —
rendimientos decrecientes.

### Por qué importa

Es la hipótesis **exacta** bajo la cual el algoritmo de Queyranne garantiza el óptimo. Sin
ella, el par colgante que produce el ordenamiento de adyacencia máxima no tiene por qué ser
un corte mínimo, y lo que devuelve el algoritmo es una candidata, no un certificado. QSW
hereda esa dependencia: usa la maquinaria de Queyranne con el andamiaje de Stoer-Wagner.

---

## 2. Dónde afecta

La afirmación «simétrica y anclada pero no submodular» aparece en cuatro lugares del
repositorio, en todos como precaución sin prueba. Con esta demostración pasan a poder citar
un teorema:

| Archivo | Línea | Qué dice hoy |
|---|---|---|
| `.docs/papers/qsw/layout/03-background.tex` | 27–29 | «symmetric and anchored … but not submodular» |
| `.docs/papers/qsw/layout/07-conclusions.tex` | 39 | «symmetric and anchored, but not submodular» |
| `src/iit/strategies/python/qsw/info.md` | 126 | «no se demostró submodular» |
| `src/iit/strategies/python/queyranne/QUEYRANNE.md` | 124 | «no está probada submodular» |

La simetría sí tenía respaldo: Proposición `prop:symmetric` en `04-methods.tex:283` y el test
`tests/strategies/test_qsw.py:test_f_es_simetrica_y_anclada`. La no-submodularidad no tenía
ninguno de los dos. Ese es el hueco que cierra este directorio.

Afecta además a la lectura de dos piezas de código:

- **`src/iit/strategies/python/qsw/core.py:220-244`** — las pasadas de reparación 1-opt y
  2-opt dejan de ser un parche empírico y pasan a ser la consecuencia esperada. Si el
  objetivo no es submodular, la búsqueda *tiene* que fallar en algunas instancias, y la
  reparación es lo que cierra la brecha (384/384 contra PyPhi).
- **`src/iit/strategies/python/queyranne/code.py:93`** — el pre-pass de singletons responde
  al mismo problema por otra vía.

### El contraste con la literatura citada

El paper cita `hidaka2018fast`, donde el objetivo es información mutua. Ahí sí hay prueba, y
es corta: `I(S; Sᶜ) = H(S) + H(Sᶜ) − H(V)`, con `H` submodular y la submodularidad estable
bajo complemento, luego `I` es submodular y simétrica. **El EMD no tiene esa estructura.**
Por eso la garantía no se transfiere de un objetivo al otro, y conviene decirlo explícito en
vez de dejar que el lector suponga que sí.

---

## 3. Conclusiones

### 3.1 `f` no es submodular — `demo/QSW/NoSubmodular.lean`

Contraejemplo mínimo: `D = 2` dimensiones `{a, b}`, `N = 1` hipercubo `z`. Con
`δ = (0, 0, 0, 1)` sobre las caras `(∅, {a}, {b}, {a,b})` — **el nodo XOR**, realizable por la
TPM `p(z=1 | ab) = (1, 1, 1, 0)` con pivote `00`, vía `δ(x) = p(00) − p(x)`:

| `S`    | `∅` | `{a}` | `{b}` | `{a,b}` | `{z}` | `{a,z}` | `{b,z}` | `{a,b,z}` |
|--------|-----|-------|-------|---------|-------|---------|---------|-----------|
| `f(S)` | 0   | 0     | 0     | **¼**   | ¼     | 0       | 0       | 0         |

Con `S = {a}`, `T = {b}`:

```
f(S ∪ T) + f(S ∩ T) = ¼ + 0 = ¼   >   0 = 0 + 0 = f(S) + f(T)
```

Cada dimensión por separado no pierde nada; las dos juntas pierden ¼. Es retorno **creciente**,
justo lo que la submodularidad prohíbe. Teorema: `QSW.xor_no_es_submodular`.

Segundo contraejemplo, esta vez en un par **cruzado** dimensión-índice, con
`δ = (0, 0, ½, −½)` (TPM `p(z=1 | ab) = (½, ½, 0, 1)`, pivote `00`):

```
f({a,z}) + f(∅) = ¼   >   0 = f({a}) + f({z})
```

Teorema: `QSW.cruzado_no_es_submodular`. Importa que exista: dice que el problema no se
arregla tratando las dos capas temporales por separado.

### 3.2 `f` sí es modular en el alcance — `demo/QSW/Modular.lean`

Fijado el mecanismo `A`, cada hipercubo aporta una de dos constantes y nada más:

```
f(S) = Σᵢ aᵢ  +  Σ_{i ∈ B} (bᵢ − aᵢ),     aᵢ = |δ̄ᵢ(A)|,  bᵢ = |δ̄ᵢ(Aᶜ)|
```

Una suma de elecciones independientes es modular, así que entre cortes con el mismo mecanismo
la desigualdad se cumple **con igualdad**. Teorema: `QSW.f_modular_en_alcance`, con el
corolario `QSW.segunda_diferencia_indices` para pares de índices.

Esto localiza el problema: **toda la no-submodularidad vive en la capa de dimensiones y en el
cruce entre capas.** La capa de alcance es inocente.

### 3.3 La causa estructural: interacciones de orden ≥ 2

Escribiendo `δ` en base de Möbius, `δ(x) = Σ_{T ⊆ x} c_T`, la media de cara resulta

```
δ̄(A) = Σ_{T ⊆ A} c_T · 2^(−|T|)
```

es decir, una función de conjunto cuyos coeficientes de Möbius son `c_T · 2^(−|T|)`. Su
segunda diferencia en `(A, u, v)` es entonces

```
Δ²(A; u, v) = Σ_{R ⊆ A} c_{R ∪ {u,v}} · 2^(−|R|−2)
```

y la submodularidad en la capa de dimensiones equivale a que esas sumas sean `≤ 0`: **ninguna
interacción de orden ≥ 2 positiva**. El XOR pone `c_{a,b} = +1` y rompe.

La ironía vale decirla: lo que rompe la submodularidad son exactamente las interacciones de
orden alto, que es lo que IIT se propone medir. No es mala suerte con el EMD; es el objeto de
estudio peleándose con la herramienta.

### 3.4 Qué tan frecuente es — `numerico.py`

Sobre **N15A** (`D = N = 15`, `V = 30`), 20 000 segundas diferencias aleatorias:

```
violaciones: 6301/20000 = 31.5 %      peor brecha: 0.395791
por tipo de par:  dim-dim 1790    idx-idx 0    cruzado 4511
```

Tres lecturas:

1. **31.5 % no es patológico, es el caso típico.** La no-submodularidad no es un caso de
   borde construido: es la situación normal en una red real.
2. **`idx-idx 0` no es ruido**, es §3.2. Cero violaciones en 20 000 muestras porque el teorema
   dice que ahí la desigualdad es igualdad.
3. **La mayoría de las violaciones son cruzadas** (4511 de 6301), que es el caso del segundo
   contraejemplo.

### 3.5 Lo que esto *no* dice

No dice que QSW dé resultados malos. Da 384/384 contra PyPhi. Dice que esa exactitud proviene
de la búsqueda **más la reparación 1-opt/2-opt**, verificada empíricamente, y no de una
garantía teórica. Es la posición honesta y además la fuerte: el paper ya la sostiene, y ahora
con un teorema detrás.

---

## 4. Lo que no está formalizado

Para que el documento no prometa de más:

- **§3.3, la caracterización de Möbius para `D` arbitrario** no está mecanizada. Está
  verificada numéricamente (`numerico.py`, bloque 4): forzando `c_T ≤ 0` para `|T| ≥ 2` la
  peor brecha cae a `−6.3e−05` (submodular), con `c_T` libre sube a `+4.8e−01`, y con el valor
  absoluto vuelve a romperse (`+1.9e+00`) aunque se cumpla la condición. Formalizarla exigiría
  inversión de Möbius sobre `Finset.powerset`; es un trabajo aparte.
- **§3.4, la tasa del 31.5 %** es una medición, no un teorema. Semilla fija, reproducible.

Lo que sí está mecanizado sin huecos: §3.1 y §3.2, más la simetría y el anclaje de `f`.

---

## 5. Cómo reproducir

```sh
# La demostración formal (ver demo/README.md para el detalle)
cd demo && lake exe cache get && lake build

# La evidencia numérica, desde la raíz del repositorio
python .docs/proofs/submodularidad/numerico.py           # N15A por defecto
python .docs/proofs/submodularidad/numerico.py N20A
```

`numerico.py` evalúa los dos contraejemplos con el `f_cara` **real** de `oracle.py`, no con
una reimplementación: es el puente entre el Lean y el código que corre. Los ocho valores de la
tabla de §3.1 aparecen en su salida y también como `example`s comprobados en
`demo/QSW/NoSubmodular.lean`, así que los tres artefactos —documento, prueba y código— dicen
el mismo número.
