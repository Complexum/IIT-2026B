# Complejidad Computacional y Transformada Zeta

## Conexión con la Demostración

El documento `demo.md` establece que la solución óptima requiere evaluar, para cada hipercubo $i$ y cada subconjunto $A \subseteq \{1,\ldots,D\}$, la suma de hipercara:

$$S_h(i, A) = \sum_{\substack{u \in \{0,1\}^D \\ \text{supp}(u) \subseteq A}} \Delta_i[u]$$

Esto es, la suma de todos los elementos de $\Delta_i$ cuyas coordenadas activas (no nulas) están contenidas en $A$.

El espacio de búsqueda es $N \cdot 2^D$ hipercubos-escala, donde cada hipercara requiere computar una suma sobre $2^{|A|}$ elementos. El problema se reduce a: **calcular eficientemente todas las $2^D$ sumas de subconjuntos para un tensor de $2^D$ elementos**.

---

## 1. El Lema Fundacional: Transformada Zeta sobre el Retículo de Subconjuntos

### Definición 1.1 (Función sobre el retículo de subconjuntos)

Sea $U = \{1, \ldots, D\}$. Una función $f: 2^U \to \mathbb{R}$ asigna un valor real a cada subconjunto de $U$. Representamos $f$ como un array indexado por masks binarios de $D$ bits.

### Definición 1.2 (Transformada Zeta)

La **transformada Zeta** de $f$ es la función $Z[f]: 2^U \to \mathbb{R}$ definida por:

$$Z[f](S) = \sum_{T \subseteq S} f(T)$$

para todo $S \subseteq U$.

### Lema 1.1 (Hipercara como transformada Zeta)

Sea $\Delta: 2^U \to \mathbb{R}$ la función que asigna a cada subconjunto $T$ el valor $\Delta[T]$ (el elemento del hipercubo en las coordenadas indicadas por $T$, con coordenadas fuera de $T$ fijadas en $0$). Entonces:

$$S_h(A) = Z[\Delta](A)$$

para todo $A \subseteq U$.

*Demostración.* Por Definición 1.2:

$$Z[\Delta](A) = \sum_{T \subseteq A} \Delta[T]$$

Sumar $\Delta[T]$ sobre todos los $T \subseteq A$ es exactamente sumar los valores del subespacio donde las coordenadas activas pertenecen a $A$, es decir, la hipercara definida por $A$. $\square$

---

## 2. Algoritmo de la Transformada Zeta

### Lema 2.1 (Propiedad de la transformada Zeta)

Sea $f: 2^U \to \mathbb{R}$. Para cualquier $d \in U$, definamos la operación de **pasada zeta** sobre la dimensión $d$:

$$f'_{S}(x) = f_{S}(x) + f_{S \setminus \{d\}}(x)$$

para todo subconjunto $S$ que contiene a $d$ y todo valor de las coordenadas restantes.

Interpretación: para cada par de subconjuntos que difieren solo en $d$, acumulamos el valor del subconjunto más pequeño en el más grande.

### Lema 2.2 (Pasada zeta computa la suma sobre subconjuntos)

Después de aplicar la pasada zeta sobre todas las $D$ dimensiones, el valor en cada máscara $S$ es:

$$\sum_{T \subseteq S} f(T)$$

*Demostración por inducción sobre $D$.*

**Caso base $D=1$:** Con una sola dimensión, hay dos máscaras: $\emptyset$ y $\{1\}$. La pasada zeta acumula $f(\emptyset) + f(\{1\})$ en la posición $\{1\}$. Esto coincide con $Z[f](\{1\}) = \sum_{T \subseteq \{1\}} f(T) = f(\emptyset) + f(\{1\})$. $\blacksquare$

**Hipótesis de inducción:** La propiedad se cumple para $D = d$.

**Paso inductivo para $D = d+1$:** Consideramos los subconjuntos como pares $(T, b)$ donde $T \subseteq \{1,\ldots,d\}$ y $b \in \{0,1\}$ indica la presencia de la dimensión $d+1$. La pasada zeta sobre la dimensión $d+1$ acumula los valores de $T$ (con $b=0$) en $(T, b=1)$. Por hipótesis de inducción, después de las primeras $d$ pasadas, cada posición $(T, 0)$ ya contiene $\sum_{U \subseteq T} f(U, 0)$. Después de la pasada $d+1$, la posición $(T, 1)$ contiene:

$$\sum_{U \subseteq T} f(U, 0) + f(T, 1) = \sum_{U \subseteq T} f(U, 0) + f(U \cup \{d+1\})$$

para todo $U \subseteq T$, que es exactamente $\sum_{W \subseteq (T \cup \{d+1\})} f(W)$. $\square$

### Teorema 2.1 (Complejidad de la transformada zeta)

La transformada Zeta de una función $f: 2^U \to \mathbb{R}$ se computa en exactamente $D$ pasadas, cada una procesando $2^{D-1}$ elementos. La complejidad total es $O(D \cdot 2^D)$.

*Demostración.* Cada pasada zeta sobre la dimensión $d$ opera sobre pares de elementos que difieren en el bit $d$. Hay $2^{D-1}$ tales pares, y el acumulado es $O(1)$. Con $D$ pasadas, el costo total es $D \cdot 2^{D-1} = O(D \cdot 2^D)$. $\square$

---

## 3. Aplicación al Problema de N Hipercubos

### Lema 3.1 (Complejidad naive)

El cálculo directo de todas las sumas de hipercara para un hipercubo requiere, para cada máscara $S$, sumar $2^{|S|}$ elementos. El costo total es:

$$\sum_{S \subseteq U} 2^{|S|} = \sum_{k=0}^{D} \binom{D}{k} 2^k = (1+2)^D = 3^D$$

*Demostración.* Cada elemento del hipercubo pertenece a exactamente $2^{|S|}$ máscaras $S$ de tamaño $|S|$, ya que de las $D$ dimensiones, las que están en el elemento pueden estar o no en $S$. Sumando sobre todos los elementos y todas las máscaras se obtiene $3^D$. $\square$

### Teorema 3.1 (Transformada Zeta para N hipercubos)

Para $N$ hipercubos de dimensión $D$, la transformada Zeta computa todas las sumas de hipercara en $O(D \cdot N \cdot 2^D)$ operaciones.

*Demostración.* Cada hipercubo se procesa independientemente. Por Teorema 2.1, un hipercubo requiere $O(D \cdot 2^D)$. Con $N$ hipercubos, el costo total es $O(N \cdot D \cdot 2^D)$. $\square$

### Corolario 3.1 (Speedup asintótico)

El speedup de la transformada Zeta sobre el método naive es:

$$\frac{3^D}{D \cdot 2^D} = \frac{(1.5)^D}{D}$$

Para $D = 18$: $\frac{3^{18}}{18 \cdot 2^{18}} \approx \frac{387M}{4.7M} \approx 82\times$.

---

## 4. Evaluación de Todas las Particiones

### Lema 4.1 (Particiones como máscaras complementarias)

El espacio de particiones no triviales de $\{1,\ldots,D\}$ se indexa por máscaras $m$ con $1 \leq m < 2^{D-1}$, donde cada máscara representa la partición $(m, \overline{m})$.

Hay exactamente $2^{D-1} - 1$ particiones no triviales (excluyendo $m = 0$ y $m = 2^{D-1}$ por simetría).

### Lema 4.2 (Costo de evaluación para una partición fija)

Dada una partición $(A, B)$ con $|A| = a$ y $|B| = b$, y las sumas de hipercara $S_h(A)$ y $S_h(B)$, el costo para hipercubo $i$ es:

$$\min\!\left(\frac{S_h(i, A)}{2^a},\; \frac{S_h(i, B)}{2^b}\right)$$

Esta evaluación requiere solo una división y un min — $O(1)$ por hipercubo.

### Teorema 4.1 (Evaluación completa de todas las particiones)

Con las sumas $S_h(i, m)$ para todo $i \in [N]$ y toda máscara $m$ precomputadas mediante transformada Zeta, la evaluación de todas las $2^{D-1} - 1$ particiones tiene complejidad $O(N \cdot 2^{D-1})$.

*Demostración.* Para cada partición $(m, \overline{m})$:
- Obtenemos $S_h(i, m)$ y $S_h(i, \overline{m})$ de las sumas precomputadas — $O(1)$ por hipercubo.
- Calculamos el min dividido — $O(1)$ por hipercubo.
- Sumamos sobre todos los hipercubos — $O(N)$.

Con $2^{D-1}$ particiones, el costo total es $O(N \cdot 2^{D-1})$. $\square$

---

## 5. Complejidad Total del Algoritmo

### Teorema 5.1 (Complejidad en anchura y profundidad)

El algoritmo completo para resolver el problema óptimo sobre $N$ hipercubos de dimensión $D$ tiene complejidad:

$$\underbrace{O(D \cdot N \cdot 2^D)}_{\text{Transformada Zeta}} + \underbrace{O(N \cdot 2^{D-1})}_{\text{Evaluación de particiones}} + \underbrace{O(N \cdot 2^D)}_{\text{Normalización}} = O(D \cdot N \cdot 2^D)$$

*Desglose:*
1. **Normalización**: Restar el pivote de cada hipercubo — $O(N \cdot 2^D)$.
2. **Transformada Zeta**: Computar todas las sumas de hipercara — $O(D \cdot N \cdot 2^D)$.
3. **Evaluación de particiones**: Para las $2^{D-1}$ particiones, evaluar el costo de distribución — $O(N \cdot 2^{D-1})$.
4. **Selección del mínimo**: Comparar concentración con todas las distribuciones — $O(2^{D-1})$.

Factor dominante: $O(D \cdot N \cdot 2^D)$.

### Corolario 5.1 (Dependencia en $N$ y $D$)

- **En $N$ (anchura)**: El algoritmo es lineal — por cada hipercubo adicional, el trabajo grows by $O(D \cdot 2^D)$.
- **En $D$ (profundidad)**: El algoritmo es exponencial en $D$, pero con base $2$ en la transformada Zeta vs base $3$ del método naive.

---

## 6. Conexión Formal con la Demostración de Optimalidad

### Lema 6.1 (La transformada Zeta computa exactamente las $S_h$ de demo.md)

Sea $\Delta_i$ el hipercubo normalizado del hipercubo $i$. Aplicando la transformada Zeta a $\Delta_i$ se obtiene, para cada máscara $m$:

$$Z[\Delta_i](m) = \sum_{u \subseteq m} \Delta_i[u] = S_h(i, m)$$

Por Lema 1.1, esto es exactamente la suma de la hipercara definida en `demo.md` Definición 0.3.

$\square$

### Lema 6.2 (La evaluación vectorizada preserva la optimalidad)

El algoritmo de evaluación de particiones (Sección 4) computa exactamente:

$$C_{\text{dist}}(A, B) = \sum_{i=1}^{N} \min\!\left(\frac{S_h(i,A)}{2^{|A|}},\; \frac{S_h(i,B)}{2^{|B|}}\right)$$

para cada partición $(A, B)$, sin aproximar, sin heurísticas, y sin descartar ninguna partición.

*Demostración.* La evaluación usa directamente las sumas $S_h(i, A)$ y $S_h(i, B)$ computadas por la transformada Zeta (Lema 6.1). Para cada hipercubo se calcula el min exactamente. La suma es exacta. $\square$

### Teorema 6.1 (El algoritmo encuentra el óptimo global)

El algoritmo descrito en este documento retorna exactamente el valor:

$$\min\!\left(\frac{\min_{k}\, S_\Omega^{(k)}}{2^D},\;\; \min_{(A,B)} \sum_{i=1}^{N} \min\!\left(\frac{S_h(i,A)}{2^{|A|}},\; \frac{S_h(i,B)}{2^{|B|}}\right)\right)$$

que, por el Teorema 2.1 y Teorema 3.1 de `demo.md`, es el mínimo global.

*Demostración.* La concentración se evalúa directamente como $\min_k S_\Omega^{(k)} / 2^D$ — $O(N)$. Las distribuciones se evalúan según Lema 6.2 para todas las particiones. El argmin sobre todos los candidatos es exacto. $\square$

---

## 7. Resumen de Complejidades

| Método | Complejidad | Notas |
|--------|-------------|-------|
| Naive (sumar por máscara) | $O(N \cdot 3^D)$ | Cada elemento计入 $2^{\|S\|}$ máscaras |
| **Transformada Zeta** | $O(D \cdot N \cdot 2^D)$ | $D$ pasadas sobre $N \cdot 2^D$ elementos |
| Evaluación de particiones | $O(N \cdot 2^{D-1})$ | Una por máscara no trivial |
| **Total** | $O(D \cdot N \cdot 2^D)$ | Dominado por la transformada |

| $D$ | $3^D$ | $D \cdot 2^D$ | Speedup |
|-----|-------|---------------|---------|
| 10 | 59K | 10K | 5.9× |
| 15 | 14M | 491K | 28.5× |
| 18 | 387M | 4.7M | 82× |
| 20 | 3.5B | 21M | 166× |

---

## 8. Conclusiones

\begin{enumerate}
\item La **transformada Zeta** computa exactamente todas las sumas de hipercara $S_h(i, A)$ definidas en `demo.md` Definición 0.3.
\item La complejidad $O(D \cdot N \cdot 2^D)$ es óptima para este cómputo, mejorando el naive $O(N \cdot 3^D)$ por un factor $(1.5)^D / D$.
\item La evaluación de todas las particiones es $O(N \cdot 2^{D-1})$ con las sumas precomputadas.
\item El algoritmo completo encuentra el óptimo global sin aproximaciones ni heurísticas, directamente conectado a los teoremas de `demo.md`.
\end{enumerate}