# Demostración de Optimalidad — Partición Óptima de N-Hipercubos

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
No existen estados mixtos donde un hipercubo contribute parcialmente a ambos. $\square$

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
para el problema con $n$ hipercubos,cuya optimalidad está garantizada por la hipótesis de inducción.

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
| $D \to \infty$ | Alta dimensión | Particiones $= 2^{D-1}-1$, algoritmo $O(N \cdot 4^D)$ para $D$ fijo |

---

## 7. Conclusiones

- La **inducción en anchura** (Sección 2) prueba optimalidad para todo $N$, con $D$ fija.
- La **inducción en profundidad** (Sección 3) prueba optimalidad para todo $D$, con $N=1$.
- El **producto cruz** (Sección 4) demuestra que ambas se combinan sin pérdida de optimalidad.
- La solución es **globalmente óptima**: no existe ninguna estrategia válida con costo menor.
- El valor absoluto se aplica **tras** promediar ($\delta$ signado), condición necesaria para
  igualar la EMD real $\sum_i |\text{mean}_{\text{face}}(H_i) - p_i|$.