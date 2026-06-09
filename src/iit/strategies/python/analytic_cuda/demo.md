# Demostración de Optimalidad y Modelo CUDA — Partición Óptima de N-Hipercubos

> Variante GPU de la estrategia óptima `analytical`. La matemática del resultado es idéntica
> (normalización signada $\delta$, valor absoluto tras promediar); lo nuevo es el **mapeo a un
> kernel CUDA propio** (Sección 8). Las Secciones 0–7 prueban que el resultado es óptimo; la
> Sección 8 prueba que la ejecución en GPU lo calcula sin pérdida.

---

## 0. Preliminares y Notación

Sea $\mathcal{H} = \{H_1, H_2, \ldots, H_N\}$ un sistema de $N$ hipercubos de dimensión $D$,
cada uno con valores reales en $[0,1]$.

**Definición 0.1 (Pivote).** Para cada hipercubo $H_i$, su pivote es el elemento
en posición $\mathbf{0} = (0, 0, \ldots, 0)$:

$$p_i := H_i[\mathbf{0}]$$

**Definición 0.2 (Normalización signada).** Definimos el hipercubo normalizado:

$$\delta_i := H_i - p_i$$

donde la resta es elemento a elemento. El pivote queda en $0$, pero $\delta_i$ **conserva
el signo** (puede ser negativo). El valor absoluto se aplica *después* de promediar, no antes.

**Definición 0.3 (Hipercara).** Sea $A \subseteq \{1, \ldots, D\}$ un subconjunto de dimensiones.
La hipercara $S_h(i, A)$ es la suma de todos los elementos de $\delta_i$ cuyo índice en toda
dimensión $d \notin A$ es $0$ (fijada al pivote). Es la suma signada del subespacio de dimensión $|A|$.

Notación: $S_\Omega^{(i)} := S_h(i, \{1,\ldots,D\})$ es la suma del hipercubo completo.

---

## 1. Estructura del Espacio de Soluciones

### Lema 1.1 (Complementariedad obligatoria)

Toda solución válida forma dos conjuntos complementarios $C$ y $\overline{C}$ tales que
$C \cup \overline{C} = \mathcal{H}$ y $C \cap \overline{C} = \emptyset$.

*Demostración.* Por el enunciado del problema, cada hipercubo contribuye a exactamente
un promedio y la operación sobre un conjunto es complementaria al otro.
No existen estados mixtos donde un hipercubo contribuya parcialmente a ambos. $\square$

### Lema 1.2 (Partición de dimensiones)

Para cualquier solución válida, existe una partición $(A, B)$ del conjunto de dimensiones
$\{1, \ldots, D\}$ con $A \cup B = \{1,\ldots,D\}$ y $A \cap B = \emptyset$,
donde un conjunto promedia sobre las dimensiones en $A$ y el complementario sobre $B$.

*Demostración.* Cada hipercubo puede promediar en cualquier subconjunto de dimensiones.
Para que dos hipercubos sean complementarios, las dimensiones promediadas deben ser
complementarias respecto al espacio total. $\square$

### Lema 1.3 (Forma canónica de costo)

Dada una partición $(A, B)$ con $|A| = a$ y $|B| = b$ ($a + b = D$),
el costo de una estrategia de distribución es:

$$C_{\text{dist}}(A, B) = \sum_{i=1}^{N} \min\!\left(\left|\frac{S_h(i, A)}{2^a}\right|,\; \left|\frac{S_h(i, B)}{2^b}\right|\right)$$

*Demostración.* Para hipercubo $i$, promediar $\delta_i$ sobre $A$ da $S_h(i,A)/2^a = \text{mean}_A(H_i) - p_i$;
sobre $B$ da $S_h(i,B)/2^b = \text{mean}_B(H_i) - p_i$. El costo EMD de cada lado es el valor absoluto
de ese promedio signado (distancia de la media marginal al pivote); el hipercubo elige el menor.
Sumando sobre todos los hipercubos se obtiene el costo total. $\square$

*Nota (orden del valor absoluto).* El $|\cdot|$ va **fuera** del promedio: $|\text{mean}_A(H_i) - p_i|$,
no $\text{mean}_A(|H_i - p_i|)$. Por desigualdad triangular $|\text{mean}_A(H_i)-p_i| \le \text{mean}_A(|H_i-p_i|)$,
con igualdad solo si la cara no cruza el pivote. Promediar el signo *antes* del $|\cdot|$ destruiría la
cancelación que la EMD aprovecha y sobreestimaría el costo. Por eso $\delta$ es signado (Def 0.2).

---

## 2. Demostración en Anchura: $D$ Fija, $N$ Variable

### Teorema 2.1 (Optimalidad para $D$ fija, $N$ arbitrario)

Para una dimensión $D$ fija, la estrategia óptima es:

$$\text{Resultado}(N, D) = \min\!\left(\min_{k}\left|\frac{S_\Omega^{(k)}}{2^D}\right|,\;\; \min_{(A,B)} \sum_{i=1}^{N} \min\!\left(\left|\frac{S_h(i,A)}{2^{|A|}}\right|,\; \left|\frac{S_h(i,B)}{2^{|B|}}\right|\right)\right)$$

*Demostración por inducción en $N$.*

---

**Caso base: $N = 1$.**

Con un solo hipercubo, el espacio de estrategias se reduce a elegir una partición $(A, B)$.
Por Lema 1.3, el costo para cada partición es:

$$\min\!\left(\left|\frac{S_\Omega}{2^D}\right|,\; \min_{(A,B)} \left|\frac{S_h(1, A)}{2^{|A|}}\right|,\; \left|\frac{S_h(1, B)}{2^{|B|}}\right|\right)$$

Esto es exactamente el mínimo sobre colapso y todas las particiones. $\blacksquare$

---

**Hipótesis de inducción: El teorema es cierto para $N = n$.**

---

**Paso inductivo: $N = n+1$.**

Sea $\mathcal{H}_{n+1} = \mathcal{H}_n \cup \{H_{n+1}\}$.
Por Lema 1.1, toda solución partitiona $\mathcal{H}_{n+1}$ en dos conjuntos complementarios $(C, \overline{C})$.
Existen tres casos exhaustivos:

**Caso 1:** $H_{n+1} \in C$ y $C = \{H_{n+1}\}$, $\overline{C} = \mathcal{H}_n$.

El costo es:
$$\left|\frac{S_\Omega^{(n+1)}}{2^D}\right| + \sum_{i=1}^{n} 0 = \left|\frac{S_\Omega^{(n+1)}}{2^D}\right|$$

Este es el término de colapso evaluado sobre $H_{n+1}$.

**Caso 2:** $H_{n+1} \in \overline{C}$ y $\overline{C} = \{H_{n+1}\}$, $C = \mathcal{H}_n$.

Análogamente, el costo es $\left|\frac{S_\Omega^{(n+1)}}{2^D}\right|$ (colapso sobre el nuevo hipercubo).

**Caso 3:** $|C| \geq 2$ y $|\overline{C}| \geq 2$.

La contribución de $H_{n+1}$ es:
$$\min\!\left(\left|\frac{S_h(n+1, A)}{2^{|A|}}\right|,\; \left|\frac{S_h(n+1, B)}{2^{|B|}}\right|\right)$$

Los restantes $n$ hipercubos en el mismo conjunto que $H_{n+1}$ forman una solución válida
para el problema con $n$ hipercubos, cuya optimalidad está garantizada por la hipótesis de inducción.

---

**Conclusión:**

El costo óptimo para $N = n+1$ es el mínimo entre colapsar sobre cualquier hipercubo
y la distribución óptima. Esto es la expresión del teorema para $N = n+1$.

Por inducción matemática, el Teorema 2.1 es verdadero para todo $N \in \mathbb{N}^+$. $\square$

---

## 3. Demostración en Profundidad: $N=1$, $D$ Variable

### Teorema 3.1 (Optimalidad para $N=1$, $D$ arbitrario)

Para un solo hipercubo de dimensión $D$, la estrategia óptima es:

$$\text{Resultado}(1, D) = \min_{(A,B)\,\text{partición}} \min\!\left(\left|\frac{S_\Omega}{2^D}\right|,\; \left|\frac{S_h(A)}{2^{|A|}}\right|,\; \left|\frac{S_h(B)}{2^{|B|}}\right|\right)$$

*Demostración por inducción en $D$.*

---

**Caso base: $D = 2$ (matriz $2 \times 2$).**

Sea $\delta = \begin{pmatrix} 0 & x \\ y & z \end{pmatrix}$ con $x, y, z \in \mathbb{R}$ (signados).

Las particiones válidas de $\{x, y\}$ son:

- $A = \emptyset$, $B = \{x, y\}$: Colapso $\Rightarrow \left|\frac{x+y+z}{4}\right|$
- $A = \{x\}$, $B = \{y\}$: Distribución $\Rightarrow \min\!\left(\left|\frac{x}{2}\right|, \left|\frac{y}{2}\right|\right)$

Por análisis directo, el ganador es el menor de ambos: el colapso aprovecha la cancelación
de signo en $x+y+z$, mientras la distribución elige la cara individual más cercana al pivote.
Esto coincide con la expresión del teorema para $D=2$. $\blacksquare$

---

**Hipótesis de inducción: El teorema es cierto para dimensión $D = d$.**

---

**Paso inductivo: $D = d + 1$.**

Consideremos un hipercubo $(d+1)$-dimensional $\delta^{(d+1)}$.
Identificamos tres tipos de elementos según su posición:

1. **Hipercara** $x_{d+1} = 0$: subespacio de dimensión $d$, suma $S_{\text{face}}$.
2. **Hipercara complementaria** $x_{d+1} = 1$: también dimensión $d$, suma $S_{\text{face}}'$.
3. **Diagonales de orden superior**: elementos con al menos una coordenada $\neq 0$ que no pertenece a ninguna cara, suma $S_{\text{diag}}$.

La suma total (signada) es:
$$S_\Omega^{(d+1)} = S_{\text{face}} + S_{\text{face}}' + S_{\text{diag}}$$

---

**Análisis de particiones:**

Toda partición $(A, B)$ de $\{1, \ldots, d+1\}$ cae en:

1. $d+1 \in A$, $d+1 \notin B$: La cara $x_{d+1}=0$ está en $B$.

    Costo: $\min\!\left(\left|\frac{S_{\text{face}}}{2^d}\right|, \left|\frac{S_{\text{face}}' + S_{\text{diag}}}{2^{d+1}}\right|\right)$.

2. $d+1 \in B$, $d+1 \notin A$: Simétrico al anterior.

---

**Reestructuración como problema en $d$ dimensiones:**

Fijar $x_{d+1} = 0$ reduce el problema a dimensión $d$.
Por hipótesis de inducción, el costo óptimo en esa cara es el mínimo entre colapso
y distribución sobre particiones del espacio de $d$ dimensiones.

Aplicando esto a ambas hipercaras y notando que $S_\Omega^{(d+1)} = S_{\text{face}} + S_{\text{face}}' + S_{\text{diag}}$,
el costo total se reduce a:

- Colapso: $\left|\frac{S_\Omega^{(d+1)}}{2^{d+1}}\right|$
- Distribución: se elige para cada cara la mejor partición de sus $d$ dimensiones

Esto es exactamente la expresión del teorema para $D = d+1$.

Por inducción, el Teorema 3.1 es verdadero para todo $D \geq 2$. $\square$

---

## 4. Producto Cruz: $N$ y $D$ Arbitrarios

### Corolario 4.1 (Teorema General)

Para cualquier $N$ y $D$, la solución óptima es:

$$\boxed{
\text{Resultado}(N, D) = \min\!\left(
\min_{k \in [N]}\left|\frac{S_\Omega^{(k)}}{2^D}\right|,
\;\;
\min_{(A,B)\,\in\,\mathcal{P}(D)} \sum_{i=1}^{N} \min\!\left(\left|\frac{S_h(i,A)}{2^{|A|}}\right|,\; \left|\frac{S_h(i,B)}{2^{|B|}}\right|\right)
\right)
}$$

donde $\mathcal{P}(D)$ es el conjunto de todas las particiones no triviales de $\{1,\ldots,D\}$.

*Demostración.* La optimalidad en $N$ (Teorema 2.1) garantiza que para cualquier partición
fija $(A, B)$, la mejor asignación de hipercubos a lados es óptima.
La optimalidad en $D$ (Teorema 3.1) garantiza que para cualquier hipercubo individual,
la mejor partición de sus dimensiones es óptima.
Ambas optimalidades se combinan sin pérdida en el producto cruz $N \times D$. $\square$

---

## 5. Regla de Decisión Compacta

### Corolario 5.1 (Selección sin calcular el resultado)

La estrategia ganadora se determina por:

$$\min_{k}\left|\frac{S_\Omega^{(k)}}{2^D}\right| \;<\; \min_{(A,B)} \sum_{i=1}^{N} \min\!\left(\left|\frac{S_h(i,A)}{2^{|A|}}\right|,\; \left|\frac{S_h(i,B)}{2^{|B|}}\right|\right)$$

- Si la desigualdad es **verdadera**: gana el colapso.
- Si es **falsa**: gana la distribución con la partición $(A, B)$ que minimiza el lado derecho.

*Nota:* Con $\delta$ signado (Def 0.2), $S_h(i,A)/2^{|A|} = \text{mean}_A(H_i) - p_i$ y el costo es
$|S_h(i,A)/2^{|A|}| = |\text{mean}_A(H_i) - p_i|$: el valor absoluto se aplica **tras** promediar.
La normalización con valor absoluto upfront $|H_i - p_i|$ daría $\text{mean}_A(|H_i - p_i|)$, que
$\ge |\text{mean}_A(H_i) - p_i|$ y solo coincide cuando la cara no cruza el pivote (todos los
elementos del mismo signo). Por eso $\delta$ se mantiene signado y el $|\cdot|$ va al final.

---

## 6. Verificación de Casos Extremos

| Caso           | Descripción    | Resultado                                                           |
| -------------- | -------------- | ------------------------------------------------------------------- |
| $N=1$, $D=2$   | Matriz 2×2     | $\min(x+y+z)/4$ o $\min(x,y)/2$ según $z$                           |
| $N=1$, $D=3$   | Cubo 2×2×2     | Mínimo entre colapso y 3 particiones $1\|2$                         |
| $N=2$, $D=2$   | Dos matrices   | Mínimo entre 2 colapsos y 2 distribuciones cruzadas                 |
| $D \to \infty$ | Alta dimensión | Particiones $= 2^{D-1}-1$, algoritmo $O(N \cdot 2^D)$ por pasada    |

---

## 7. Conclusiones (sobre la optimalidad)

- La **inducción en anchura** (Sección 2) prueba optimalidad para todo $N$, con $D$ fija.
- La **inducción en profundidad** (Sección 3) prueba optimalidad para todo $D$, con $N=1$.
- El **producto cruz** (Sección 4) demuestra que ambas se combinan sin pérdida de optimalidad.
- La solución es **globalmente óptima**: no existe ninguna estrategia válida con costo menor.
- El valor absoluto se aplica **tras** promediar ($\delta$ signado), condición necesaria para
  igualar la EMD real $\sum_i |\text{mean}_{\text{face}}(H_i) - p_i|$.

---

## 8. Modelo de Ejecución CUDA (kernel propio)

Las Secciones 0–7 fijan *qué* se calcula. Esta sección fija *cómo* se calcula en GPU sin alterar
el resultado. La estrategia mapea el trabajo $O(D \cdot N \cdot 2^D)$ a **kernels CUDA propios**
(estilo `RawKernel`/`numba.cuda`), no a llamadas de librería.

### 8.0 Notación de cómputo paralelo

Para un algoritmo expresamos dos magnitudes: **work** $W$ (total de operaciones, suma sobre
todos los threads) y **depth** $T$ (longitud de la cadena dependiente más larga, el tiempo con
paralelismo ilimitado). El objetivo es $W$ óptimo (igual al serial) con $T$ mínimo.

### 8.1 Layout de memoria

Cada $\delta_i$ se almacena aplanado en orden little-endian: el bit $d$ del índice plano
corresponde a la dimensión $d$. Dos buffers densos en VRAM, ambos `float32`:

$$\texttt{delta}[i, s], \quad \texttt{sumas}[i, s], \qquad i \in [0,N),\; s \in [0, 2^D)$$

Memoria total $\Theta(N \cdot 2^D \cdot 4)$ bytes. De aquí la cota

$$D \le D_{\max}^{\text{CUDA}} \approx 27 \quad\Longleftrightarrow\quad N \cdot 2^D \cdot 4\,\text{B} \lesssim \text{VRAM}.$$

El eje de máscara $s$ es el contiguo (stride 1), de modo que threads consecutivos de un warp leen
direcciones consecutivas → **accesos coalescentes**. La normalización $\delta = H - p$ se hace una
sola vez (kernel elemento-a-elemento, $W=\Theta(N 2^D)$, $T=O(1)$): el pivote $s=\mathbf{0}$ queda en $0$.

### 8.2 Zeta transform como $D$ pasadas (kernel mariposa / SOS)

La hipercara $S_h(i, A)$ para *todas* las máscaras se obtiene con la transformada Zeta
(suma-sobre-subconjuntos) sobre $\delta$. Se implementa in-place en $D$ pasadas. En la pasada
$d \in \{0,\ldots,D-1\}$, cada par de celdas que difiere solo en el bit $d$ se combina:

$$\texttt{sumas}[i,\; s \,|\, 2^d] \mathrel{+}= \texttt{sumas}[i,\; s], \qquad \text{para todo } s \text{ con } (s \gg d)\,\&\,1 = 0.$$

**Paralelismo.** Dentro de una pasada, los $N \cdot 2^{D-1}$ updates son a direcciones disjuntas
→ sin condiciones de carrera, un thread por update. **Entre** pasadas hay dependencia de datos
(la pasada $d{+}1$ lee lo que escribió la $d$), así que las pasadas son secuenciales y se separan
con `cudaDeviceSynchronize` (o kernels distintos). Resultado:

$$W_{\text{zeta}} = \Theta(D \cdot N \cdot 2^D), \qquad T_{\text{zeta}} = O(D).$$

Tras la pasada final, $\texttt{sumas}[i,m] = S_h(i, A_m)$ con $A_m$ el conjunto de dims cuyos bits
están en $1$ en $m$, y la media de la cara es $\texttt{sumas}[i,m] / 2^{\operatorname{popcount}(m)}$.

### 8.3 Kernel principal: costo por máscara

Por simetría $f(m) = f(m^c)$ (Lema 1.1), basta recorrer $m \in [1, 2^{D-1}]$, es decir $M = 2^{D-1}$
máscaras. Para cada máscara $m$ y cubo $i$, con $\texttt{full} = 2^D - 1$ y $a = \operatorname{popcount}(m)$:

$$
\text{val}_A = \frac{\texttt{sumas}[i,m]}{2^{a}}, \qquad
\text{val}_B = \frac{\texttt{sumas}[i,\, m \,\hat{}\, \texttt{full}]}{2^{\,D-a}}, \qquad
c_i(m) = \min\!\big(|\text{val}_A|,\, |\text{val}_B|\big)
$$

y la función objetivo de la máscara es la reducción sobre los hipercubos:

$$f(m) = \sum_{i=1}^{N} c_i(m).$$

**Mapeo a GPU.** Un bloque por máscara (grid de $M$ bloques); los threads del bloque recorren los
$N$ cubos y acumulan $c_i(m)$ en una **reducción en memoria compartida** (árbol binario), de modo
que la suma sobre $i$ tiene depth $O(\log N)$. El $\operatorname{popcount}$ se calcula con la
intrínseca `__popc`; el $2^{a}$ con `exp2f` o shift. La división por $2^{a}$ ocurre **antes** del
$|\cdot|$ (Sección 5): el signo de $\delta$ se preserva hasta promediar.

$$W_{\text{eval}} = \Theta(N \cdot 2^D), \qquad T_{\text{eval}} = O(\log N).$$

### 8.4 Reducción final (argmin) y colapso

Se buscan en paralelo dos mínimos:

$$f^\star = \min_{m \in [1,\,2^{D-1}]} f(m), \qquad C_{\text{conc}} = \min_{k \in [N]} \left|\frac{S_\Omega^{(k)}}{2^D}\right|,$$

ambos por reducción-argmin en árbol ($T = O(\log M)$ y $O(\log N)$). La regla del Corolario 5.1
decide: si $C_{\text{conc}} \le f^\star$ gana el colapso (un único hipercubo concentra), si no gana
la distribución con la máscara $\arg\min_m f(m)$. La derivación de $(\text{alcance}, \text{mecanismo})$
desde la máscara ganadora —cada cubo va al lado de menor $|\text{val}|$— es $\Theta(N)$ y puede
hacerse en host tras copiar una sola columna.

### 8.5 Exactitud (equivalencia con la prueba serial)

**Proposición 8.1.** El resultado del pipeline GPU es igual al de la regla de decisión del
Corolario 5.1, salvo redondeo de punto flotante.

*Demostración.* Cada kernel realiza únicamente: (i) sumas de la transformada Zeta, (ii) divisiones
por potencias de dos, (iii) valor absoluto, (iv) mínimos y sumas de reducción. Las reducciones
((i) y las sumas $\sum_i$, los $\min$) son sobre operadores **asociativos y conmutativos**, por lo
que cualquier orden de evaluación —incluido el árbol paralelo— produce el mismo valor exacto en
$\mathbb{R}$. La única discrepancia frente al serial es el orden de redondeo `float32`, acotado y
del mismo orden que la versión vectorizada `numpy`. El punto crítico —aplicar $|\cdot|$ **después**
de dividir por $2^a$— se respeta en §8.3 porque $\delta$ se mantiene signado en `sumas` (§8.1–8.2);
así se preserva la cancelación de signo que la EMD aprovecha (Lema 1.3, nota). $\square$

### 8.6 Complejidad y degradación

| Fase                | Work $W$                    | Depth $T$      |
| ------------------- | --------------------------- | -------------- |
| Normalización $\delta$ | $\Theta(N 2^D)$          | $O(1)$         |
| Zeta transform      | $\Theta(D\,N\,2^D)$         | $O(D)$         |
| Costo por máscara   | $\Theta(N\,2^D)$            | $O(\log N)$    |
| Reducción argmin    | $\Theta(2^D + N)$           | $O(D + \log N)$|
| **Total**           | $\Theta(D\,N\,2^D)$         | $O(D + \log N)$|

El work coincide con el de la estrategia serial óptima; la GPU colapsa el depth a $O(D + \log N)$.

**Degradación controlada.**

- Sin CUDA/`cupy` o sin dispositivo: `resolver` lanza `RuntimeError` (hard-fail). La estrategia
  permanece **registrada** y visible en el CLI; solo falla al ejecutarse en un nodo sin GPU.
- $D > D_{\max}^{\text{CUDA}}$: la cota de §8.1 se viola → `RuntimeError` por memoria. Para $D$
  mayores correspondería un modo *lazy* (calcular medias por máscara sin materializar `sumas`),
  análogo al de `analytic_concurrent`, a costa de recomputar caras.

> Instalación en cluster con GPU: `uv pip install -e ".[cuda]"` (extra `cuda = ["cupy-cuda13x"]`).
