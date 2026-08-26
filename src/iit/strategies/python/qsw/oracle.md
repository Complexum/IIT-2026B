# Oráculo Zeta para QNodes: caras precomputadas

Aceleración del algoritmo Q (`QNodes`) reemplazando la **remarginalización por
llamada** del oráculo por una **única transformada Zeta** que precomputa todas las
caras del hipercubo. La búsqueda submodular queda intacta; solo cambia el costo de
cada consulta del oráculo, de $O(N\cdot 2^{|A|})$ a $O(1)$ amortizado.

Notación: el sistema tiene $D = m + n$ vértices —$m$ dimensiones de *mecanismo*
($t$, `ACTUAL`) y $n$ índices de *alcance* ($t{+}1$, `EFFECT`)—. El espacio de estados
del repertorio de efecto tiene $2^{D}$ celdas.

---

## 1. Dónde está el costo (qn actual)

El MAO (Maximum Adjacency Ordering) en `algorithm` evalúa la ganancia marginal con

$$
g(\delta\mid\Omega) \;=\; \underbrace{f(\delta\cup\Omega)}_{\texttt{emd\_union}}
                      \;-\; \underbrace{f(\delta)}_{\texttt{emd\_delta}},
$$

donde $f(A)$ es el EMD de aislar el grupo $A$. Cada $f(A)$ se computa en
`__emd_grupo` como

$$
f(A) \;=\; \mathrm{EMD}\!\big(\,\texttt{bipartir}(A)\texttt{.distribucion\_marginal}(),\; \rho\,\big),
$$

con $\rho$ la distribución del subsistema. La marginalización
`distribucion_marginal()` **recorre** las celdas de la cara asociada a $A$: costo
$O(N\cdot 2^{|A|})$. El MAO pide $O(D^2)$ ganancias por fase y hay $O(D)$ fases, es
decir $O(D^3)$ llamadas al oráculo, y las caras pedidas a lo largo del orden
**se solapan**: la misma suma parcial se recalcula una y otra vez. Costo total

$$
\boxed{\,T_{\text{qn}} \;=\; O\!\big(D^{3}\cdot N\cdot 2^{D}\big)\,}
\qquad(\text{el factor } 2^{D} \text{ es la remarginalización por llamada}).
$$

`memoria_bipart` cachea claves repetidas, pero no elimina el $2^{D}$: caras
distintas comparten celdas que igual se vuelven a sumar.

---

## 2. La operación: transformada Zeta sobre las caras

**Normalización con signo.** Para cada índice $i\in[N]$ sea el pivote
$p_i = H_i[\mathbf{0}]$ y la representación firmada

$$
\delta_i \;=\; H_i - p_i
\qquad(\text{el valor absoluto se aplica } \textbf{después} \text{ de promediar}).
$$

**Suma de cara.** A cada subconjunto de dimensiones $A\subseteq\{1,\dots,D\}$ (una
*máscara* $m\in\{0,1\}^{D}$) le corresponde la suma sobre el sub-hipercubo que $A$
genera:

$$
S_i(A) \;=\; \sum_{x \,\sqsubseteq\, A} \delta_i[x],
\qquad x \sqsubseteq A \iff \operatorname{supp}(x)\subseteq A .
$$

**Recurrencia Zeta (suma sobre subconjuntos).** Todas las $2^{D}$ caras se obtienen
en un solo barrido dimensión a dimensión:

$$
\text{for } d=1\dots D:\quad
S[i,\,m\,|\,2^{d}] \mathrel{+}= S[i,\,m\,\setminus\,2^{d}],
$$

con costo

$$
\boxed{\,D\cdot N\cdot 2^{\,D-1} \;=\; O\!\big(D\cdot N\cdot 2^{D}\big)\,}\quad(\text{una sola vez}).
$$

**Media de cara en $O(1)$.** Tras el precómputo, el promedio sobre cualquier cara es
una lectura:

$$
\bar\delta_i(A) \;=\; \frac{S_i(A)}{2^{|A|}} .
$$

---

## 3. El oráculo en $O(1)$ amortizado

El EMD de aislar un corte $(A_{\text{alc}},A_{\text{mec}})$ se arma a partir de las
medias de cara precomputadas. Con la identidad de normalización firmada (probada para
la estrategia *analytic*: $\big|\overline{\,\cdot\,}\big|$ tras promediar es exacto
para el EMD del efecto), el costo de aislar el grupo se reduce a

$$
f(A) \;=\; \sum_{i=1}^{N}
\min\!\Big(\big|\bar\delta_i(A)\big|,\ \big|\bar\delta_i(\bar A)\big|\Big),
$$

donde $\bar A$ es la máscara complementaria (`full_mask` XOR $m$). Cada término es una
división y un valor absoluto: **$O(1)$ por índice, $O(N)$ por consulta**, sin
marginalizar. La ganancia del MAO

$$
g(\delta\mid\Omega) = f(\delta\cup\Omega) - f(\delta)
$$

usa esas mismas lecturas. Las $O(D^3)$ consultas del MAO pasan de
$O(D^3\cdot N\cdot 2^{D})$ a

$$
\boxed{\,O\!\big(D^{3}\cdot N\big)\,}.
$$

---

## 4. Complejidad: el cruce

| Variante | Precómputo | Búsqueda | Total |
|---|---|---|---|
| `qn` actual | — | $O(D^{3}N\,2^{D})$ | $O(D^{3}N\,2^{D})$ |
| `qn` + Zeta | $O(D N\,2^{D})$ | $O(D^{3}N)$ | $O\!\big(N\,2^{D}(D + D^{3}/2^{D})\big)\approx O(D N\,2^{D})$ |

Para $D$ grande $D^{3}\ll 2^{D}$, así que domina el precómputo único: el oráculo
acelerado iguala el costo *one-time* de la enumeración exhaustiva de *analytic*, pero
con búsqueda **polinómica** en vez de recorrer las $2^{D-1}$ particiones. De ahí que
**supere** a la evaluación exhaustiva a partir de $D\!\approx\!22$–$25$:
para $N=D=25$, $D^{3}\approx 1.6\times10^{4}$ frente a $2^{D}\approx 3.4\times10^{7}$
(tres órdenes de magnitud). Es la complementariedad que la Discusión deja como trabajo
futuro: el cómputo de caras de *analytic* alimentando la búsqueda submodular de Q.

---

## 5. Cómo conectarlo en `code.py` (esquema)

1. **Precomputar una vez** en `resolver`, antes de `algorithm`: construir
   `sumas[i, m]` con la recurrencia Zeta sobre $\delta = H - p$ (idéntico a la función
   `hyperfaces` de `analytic`).
2. **Sustituir `__emd_grupo`**: en lugar de
   `bipartir(A).distribucion_marginal()`, mapear la clave `(alcance, mecanismo)` a su
   máscara $m$, leer `sumas[:, m]` y `sumas[:, ~m]`, y combinar con la fórmula de
   $f(A)$ — $O(N)$, sin marginalizar. `memoria_bipart` sigue cacheando por clave.
3. **`funcion_submodular`** queda igual: la ganancia se calcula con los $f$ rápidos.
4. **Exactitud.** El oráculo Zeta es exacto donde *analytic* lo es (repertorio de
   efecto, normalización firmada). Como salvaguarda, verificar el MIP final con el
   `emd_efecto` real una sola vez, antes de devolver `Solution` — costo $O(N\,2^{D})$
   amortizado sobre toda la corrida.

> Resultado: la calidad de Q (búsqueda submodular, óptima cuando el objetivo es
> submodular) con el costo de oráculo de *analytic* ($O(1)$ por cara), sin el factor
> $2^{D}$ por llamada que hoy lo lastra.
